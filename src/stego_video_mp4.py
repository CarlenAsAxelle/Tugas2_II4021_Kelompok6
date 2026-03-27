# src/stego_video_mp4.py
import os
import numpy as np
from typing import Optional

from src.video_io_mp4 import (
    read_video_frames, write_video_frames,
    mse_psnr_video
)
from src.stego_lsb import (
    bytes_to_bits, bits_to_bytes,
    capacity_332,
    embed_bits_sequential_332, extract_bits_sequential_332,
    embed_bits_random_332, extract_bits_random_332
)
from src.a51_cipher import a51_encrypt_payload, a51_decrypt_payload


# ─── METADATA HEADER ──────────────────────────────────────────────────────────
HEADER_SIZE    = 32   # bytes
HEADER_BITS    = HEADER_SIZE * 8

FORMAT_FLAG_AVI = 0
FORMAT_FLAG_MP4 = 1


def encode_header(is_text: bool, is_encrypted: bool, is_random: bool,
                  extension: str, filename: str, payload_size: int,
                  is_mp4: bool = False, num_frames: int = 0) -> bytes:
    header = bytearray(HEADER_SIZE)
    header[0] = 1 if is_text else 0
    header[1] = 1 if is_encrypted else 0
    header[2] = 1 if is_random else 0

    ext_bytes = extension.encode('utf-8')[:10]
    header[3] = len(ext_bytes)
    header[4:4 + len(ext_bytes)] = ext_bytes

    fname_bytes = filename.encode('utf-8')[:10]
    header[14] = len(fname_bytes)
    header[15:15 + len(fname_bytes)] = fname_bytes
    
    # FIX: Simpan jumlah frame (max 65535) di reserved bytes
    # Ini penting untuk sinkronisasi RNG pada mode MP4
    if num_frames > 65535:
        print("Warning: num_frames > 65535, header trunction may break random shuffle sync.")
    header[25:27] = (num_frames & 0xFFFF).to_bytes(2, 'big')

    header[27] = FORMAT_FLAG_MP4 if is_mp4 else FORMAT_FLAG_AVI
    header[28:32] = payload_size.to_bytes(4, 'big')
    return bytes(header)


def decode_header(header: bytes) -> dict:
    is_text      = bool(header[0])
    is_encrypted = bool(header[1])
    is_random    = bool(header[2])

    ext_len   = header[3]
    extension = header[4:4 + ext_len].decode('utf-8', errors='replace')

    fname_len = header[14]
    filename  = header[15:15 + fname_len].decode('utf-8', errors='replace')

    num_frames   = int.from_bytes(header[25:27], 'big')
    format_flag  = header[27]
    is_mp4       = (format_flag == FORMAT_FLAG_MP4)
    payload_size = int.from_bytes(header[28:32], 'big')

    return {
        "is_text":      is_text,
        "is_encrypted": is_encrypted,
        "is_random":    is_random,
        "extension":    extension,
        "filename":     filename,
        "is_mp4":       is_mp4,
        "payload_size": payload_size,
        "num_frames":   num_frames
    }


# ─── CAPACITY ─────────────────────────────────────────────────────────────────

def total_capacity_bytes(frames: list) -> int:
    total_bits = sum(capacity_332(f) for f in frames)
    return total_bits // 8


def _calculate_embedded_frame_count(total_bits: int, frame_capacity_bits: int) -> int:
    """
    Calculate how many frames starting from frame 0 will contain embedded data.
    
    Args:
        total_bits: Total bits to embed (header + payload)
        frame_capacity_bits: Bits per frame capacity
    
    Returns:
        Number of frames needed to hold all embedded bits
    """
    if total_bits == 0:
        return 0
    return (total_bits + frame_capacity_bits - 1) // frame_capacity_bits


# ─── LOW-LEVEL BIT EMBEDDING WITH PIXEL OFFSET ────────────────────────────────

def _embed_bits_with_pixel_offset(frames: list, bits: np.ndarray,
                                   pixel_offset: int,
                                   is_random: bool, seed: int) -> list:
    """
    Embed bits ke frames mulai dari pixel_offset (dalam satuan piksel),
    melewati piksel-piksel yang sudah dipakai header.
    """
    result_frames = [f.copy() for f in frames]

    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    total_pixels = len(frames) * pixels_per_frame
    available_pixel_indices = np.arange(pixel_offset, total_pixels)

    if is_random:
        # PENTING: Seed RNG harus konsisten. 
        rng = np.random.default_rng(seed)
        rng.shuffle(available_pixel_indices)

    pixels_needed = int(np.ceil(bits.size / 8))
    if pixels_needed > len(available_pixel_indices):
        raise ValueError("Payload terlalu besar untuk kapasitas yang tersisa")

    bit_idx = 0
    for pix_num in available_pixel_indices[:pixels_needed]:
        frame_idx = pix_num // pixels_per_frame
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = result_frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        # Embed 3-3-2
        for k in range(3):
            if bit_idx < bits.size:
                r = (r & ~(1 << k)) | (int(bits[bit_idx]) << k)
                bit_idx += 1
        for k in range(3):
            if bit_idx < bits.size:
                g = (g & ~(1 << k)) | (int(bits[bit_idx]) << k)
                bit_idx += 1
        for k in range(2):
            if bit_idx < bits.size:
                b = (b & ~(1 << k)) | (int(bits[bit_idx]) << k)
                bit_idx += 1

        result_frames[frame_idx][i, j] = [r & 0xFF, g & 0xFF, b & 0xFF]

        if bit_idx >= bits.size:
            break

    return result_frames


def _extract_bits_with_pixel_offset(frames: list, num_bits: int,
                                     pixel_offset: int,
                                     is_random: bool, seed: int,
                                     original_num_frames: int = 0) -> np.ndarray:
    """
    Ekstrak bits dari frames.
    """
    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    current_num_frames = len(frames)

    # Validasi frame count agar tidak crash memory
    # Jika original_num_frames sangat besar (> 2x current), anggap header corrupt
    if original_num_frames > current_num_frames * 2:
        print(f"⚠️ Warning: Header num_frames ({original_num_frames}) > 2x actual ({current_num_frames}). Ignoring header value.")
        calc_num_frames = current_num_frames
    else:
        calc_num_frames = original_num_frames if original_num_frames > 0 else current_num_frames
    
    total_pixels_calc = calc_num_frames * pixels_per_frame
    
    # ─── MEMORY SAFETY GUARD ───────────────────────────────────────────────
    # Jika total pixels masih terlalu besar, batasi.
    # Misalnya max 2 GB buffer index (250 juta pixel * 8 byte).
    # 250M pixel ~ 120 frames FHD. Jika video lebih panjang, kita butuh strategi lain.
    # Tapi untuk assignment ini asumsi video pendek.
    if total_pixels_calc > 500_000_000: # hard limit ~500M pixels (~4GB RAM usage for index array)
        print(f"⚠️ Warning: Total pixels {total_pixels_calc} too large for safe shuffle. Clamping to actual.")
        calc_num_frames = current_num_frames
        total_pixels_calc = calc_num_frames * pixels_per_frame
    # ───────────────────────────────────────────────────────────────────────

    available_pixel_indices = np.arange(pixel_offset, total_pixels_calc)

    if is_random:
        rng = np.random.default_rng(seed)
        rng.shuffle(available_pixel_indices)

    bits = []
    # Loop hanya sebanyak bit yang dibutuhkan
    for pix_num in available_pixel_indices:
        if len(bits) >= num_bits:
            break
        
        frame_idx = pix_num // pixels_per_frame
        
        if frame_idx >= current_num_frames:
            bits.extend([0]*8) # Frame hilang/dropped
            continue
            
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        for k in range(3):
            bits.append((r >> k) & 1)
        for k in range(3):
            bits.append((g >> k) & 1)
        for k in range(2):
            bits.append((b >> k) & 1)

    return np.array(bits[:num_bits], dtype=np.uint8)


# ─── HEADER EMBED/EXTRACT (selalu sekuensial, mulai pixel 0) ──────────────────

def _embed_header_sequential(frames: list, header: bytes) -> tuple:
    header_bits = bytes_to_bits(header)
    pixels_needed = int(np.ceil(HEADER_BITS / 8))
    
    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    result_frames = [f.copy() for f in frames]

    bit_idx = 0
    for pix_num in range(pixels_needed):
        frame_idx = pix_num // pixels_per_frame
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = result_frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        for k in range(3):
            if bit_idx < HEADER_BITS:
                r = (r & ~(1 << k)) | (int(header_bits[bit_idx]) << k)
                bit_idx += 1
        for k in range(3):
            if bit_idx < HEADER_BITS:
                g = (g & ~(1 << k)) | (int(header_bits[bit_idx]) << k)
                bit_idx += 1
        for k in range(2):
            if bit_idx < HEADER_BITS:
                b = (b & ~(1 << k)) | (int(header_bits[bit_idx]) << k)
                bit_idx += 1

        result_frames[frame_idx][i, j] = [r & 0xFF, g & 0xFF, b & 0xFF]

    return result_frames, pixels_needed


def _extract_header_sequential(frames: list) -> tuple:
    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    pixels_needed = 32

    bits = []
    for pix_num in range(pixels_needed):
        frame_idx = pix_num // pixels_per_frame
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        for k in range(3):
            bits.append((r >> k) & 1)
        for k in range(3):
            bits.append((g >> k) & 1)
        for k in range(2):
            bits.append((b >> k) & 1)

    header_bits = np.array(bits[:HEADER_BITS], dtype=np.uint8)
    header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
    return header_bytes, pixels_needed


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def embed_message(cover_path: str, output_path: str, message: bytes, is_text: bool,
                  extension: str = "", filename: str = "", use_encryption: bool = False,
                  a51_key: Optional[int] = None, use_random: bool = False,
                  stego_key: Optional[int] = None, mp4_crf: int = 0) -> dict:

    frames, fps = read_video_frames(cover_path)

    payload = message
    if use_encryption:
        if a51_key is None: raise ValueError("Wait a51_key")
        payload = a51_encrypt_payload(message, a51_key)

    seed = stego_key if stego_key is not None else 0
    total_cap = total_capacity_bytes(frames)
    needed = HEADER_SIZE + len(payload)
    if needed > total_cap:
        raise ValueError(f"Payload too large: {needed} > {total_cap}")

    header = encode_header(
        is_text=is_text, is_encrypted=use_encryption, is_random=use_random,
        extension=extension, filename=filename, payload_size=len(payload),
        is_mp4=True, num_frames=len(frames)
    )

    stego_frames, pixel_offset = _embed_header_sequential(frames, header)
    payload_bits = bytes_to_bits(payload)
    stego_frames = _embed_bits_with_pixel_offset(
        stego_frames, payload_bits, pixel_offset=pixel_offset,
        is_random=use_random, seed=seed
    )

    # Calculate how many frames contain embedded data for selective encoding
    total_bits_embedded = HEADER_BITS + payload_bits.size
    frame_capacity_bits = capacity_332(frames[0])
    embedded_frame_count = _calculate_embedded_frame_count(total_bits_embedded, frame_capacity_bits)

    write_video_frames(output_path, stego_frames, fps, mp4_crf=mp4_crf,
                      embedded_frame_count=embedded_frame_count,
                      audio_source=cover_path)

    mse_list, psnr_list, mse_avg, psnr_avg = mse_psnr_video(frames, stego_frames)

    return {
        "format": "MP4", "total_capacity_bytes": total_cap,
        "payload_size_bytes": len(payload), "header_size_bytes": HEADER_SIZE,
        "total_embedded_bytes": needed, "mse_avg": mse_avg,
        "mse_list": mse_list,
        "psnr_avg": psnr_avg, "psnr_per_frame": psnr_list,
        "lossless_mp4": (mp4_crf == 0),
        "embedded_frame_count": embedded_frame_count
    }


def extract_message(stego_path: str, a51_key: Optional[int] = None,
                    stego_key: Optional[int] = None) -> dict:

    frames, _ = read_video_frames(stego_path)
    seed = stego_key if stego_key is not None else 0

    header_bytes, pixel_offset = _extract_header_sequential(frames)
    try:
        meta = decode_header(header_bytes)
    except Exception as e:
        print(f"⚠️ Header decode error: {e}. Using safe defaults.")
        meta = {"is_random": False, "is_encrypted": False, "payload_size": 0,
                "num_frames": len(frames), "is_text": False, "extension": "", "filename": ""}

    is_random = meta["is_random"]
    is_encrypted = meta["is_encrypted"]
    payload_size = meta["payload_size"]
    num_frames = meta.get("num_frames", 0)

    # Pre-check payload size sanity
    if payload_size > total_capacity_bytes(frames):
        print(f"⚠️ Warning: Payload size {payload_size} > capacity. Header likely corrupt.")
        payload_size = 0

    if num_frames > 0 and len(frames) > num_frames:
        frames = frames[:num_frames]

    payload_bits = _extract_bits_with_pixel_offset(
        frames, num_bits=payload_size * 8, pixel_offset=pixel_offset,
        is_random=is_random, seed=seed, original_num_frames=num_frames
    )
    payload = bits_to_bytes(payload_bits)[:payload_size]

    if is_encrypted and payload_size > 0:
        if a51_key is None: raise ValueError("a51_key needed")
        payload = a51_decrypt_payload(payload, a51_key)

    return {
        "message": payload, "is_text": meta["is_text"],
        "is_encrypted": is_encrypted, "is_random": is_random,
        "extension": meta["extension"], "filename": meta["filename"],
        "payload_size": payload_size, "format": "MP4",
        "is_mp4": True
    }