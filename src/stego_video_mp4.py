# src/stego_video_mp4.py
import os
import numpy as np
from typing import Optional

from src.video_io_mp4 import (
    read_video_frames, write_video_frames,
    mse_psnr_video
)
from src.stego_lsb_utils import (
    bytes_to_bits, bits_to_bytes,
    get_lsb_functions, get_capacity_fn, get_bits_per_pixel,
    LSB_METHOD_332, LSB_METHOD_111, LSB_METHOD_444, LSB_METHOD_LABELS
)
from src.a51_cipher import a51_encrypt_payload, a51_decrypt_payload


# ─── METADATA HEADER ──────────────────────────────────────────────────────────
HEADER_SIZE    = 64   # bytes (matches AVI header size)
HEADER_BITS    = HEADER_SIZE * 8

# Header layout (64 bytes):
#   [0]      is_text
#   [1]      is_encrypted
#   [2]      is_random
#   [3]      ext_len  (max 10)
#   [4:14]   extension (10 bytes)
#   [14]     fname_len (max 40)
#   [15:55]  filename (40 bytes)
#   [55]     lsb_method (0=3-3-2, 1=1-1-1, 2=4-4-4)
#   [56:58]  num_frames (2 bytes big-endian)
#   [58]     format_flag (0=AVI, 1=MP4)
#   [59]     reserved
#   [60:64]  payload_size (4 bytes big-endian)

_FNAME_MAX = 40
_EXT_MAX   = 10

FORMAT_FLAG_AVI = 0
FORMAT_FLAG_MP4 = 1


def encode_header(is_text: bool, is_encrypted: bool, is_random: bool,
                  extension: str, filename: str, payload_size: int,
                  is_mp4: bool = False, num_frames: int = 0,
                  lsb_method: int = LSB_METHOD_332) -> bytes:
    header = bytearray(HEADER_SIZE)
    header[0] = 1 if is_text else 0
    header[1] = 1 if is_encrypted else 0
    header[2] = 1 if is_random else 0

    ext_bytes = extension.encode('utf-8')[:_EXT_MAX]
    header[3] = len(ext_bytes)
    header[4:4 + len(ext_bytes)] = ext_bytes

    fname_bytes = filename.encode('utf-8')[:_FNAME_MAX]
    header[14] = len(fname_bytes)
    header[15:15 + len(fname_bytes)] = fname_bytes

    header[55] = lsb_method & 0xFF

    # Simpan jumlah frame (max 65535)
    if num_frames > 65535:
        print("Warning: num_frames > 65535, header truncation may break random shuffle sync.")
    header[56:58] = (num_frames & 0xFFFF).to_bytes(2, 'big')

    header[58] = FORMAT_FLAG_MP4 if is_mp4 else FORMAT_FLAG_AVI
    header[60:64] = payload_size.to_bytes(4, 'big')
    return bytes(header)


def decode_header(header: bytes) -> dict:
    is_text      = bool(header[0])
    is_encrypted = bool(header[1])
    is_random    = bool(header[2])

    ext_len   = min(header[3], _EXT_MAX)
    extension = header[4:4 + ext_len].decode('utf-8', errors='replace')

    fname_len = min(header[14], _FNAME_MAX)
    filename  = header[15:15 + fname_len].decode('utf-8', errors='replace')

    lsb_method   = header[55]
    num_frames   = int.from_bytes(header[56:58], 'big')
    format_flag  = header[58]
    is_mp4       = (format_flag == FORMAT_FLAG_MP4)
    payload_size = int.from_bytes(header[60:64], 'big')

    return {
        "is_text":      is_text,
        "is_encrypted": is_encrypted,
        "is_random":    is_random,
        "extension":    extension,
        "filename":     filename,
        "is_mp4":       is_mp4,
        "payload_size": payload_size,
        "num_frames":   num_frames,
        "lsb_method":   lsb_method,
    }


# ─── CAPACITY ─────────────────────────────────────────────────────────────────

def total_capacity_bytes(frames: list, lsb_method: int = LSB_METHOD_332) -> int:
    cap_fn = get_capacity_fn(lsb_method)
    total_bits = sum(cap_fn(f) for f in frames)
    return total_bits // 8


def _calculate_embedded_frame_count(total_bits: int, frame_capacity_bits: int) -> int:
    if total_bits == 0:
        return 0
    return (total_bits + frame_capacity_bits - 1) // frame_capacity_bits


# ─── LOW-LEVEL BIT EMBEDDING WITH PIXEL OFFSET ────────────────────────────────

def _embed_bits_with_pixel_offset(frames: list, bits: np.ndarray,
                                   pixel_offset: int,
                                   is_random: bool, seed: int,
                                   lsb_method: int = LSB_METHOD_332) -> list:
    """
    Embed bits ke frames mulai dari pixel_offset (dalam satuan piksel),
    melewati piksel-piksel yang sudah dipakai header.
    """
    result_frames = [f.copy() for f in frames]
    bpp = get_bits_per_pixel(lsb_method)

    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    total_pixels = len(frames) * pixels_per_frame
    available_pixel_indices = np.arange(pixel_offset, total_pixels)

    if is_random:
        rng = np.random.default_rng(seed)
        rng.shuffle(available_pixel_indices)

    pixels_needed = int(np.ceil(bits.size / bpp))
    if pixels_needed > len(available_pixel_indices):
        raise ValueError("Payload terlalu besar untuk kapasitas yang tersisa")

    bit_idx = 0
    
    # Determine bit masks based on method
    if lsb_method == LSB_METHOD_111:
        r_bits, g_bits, b_bits = 1, 1, 1
    elif lsb_method == LSB_METHOD_444:
        r_bits, g_bits, b_bits = 4, 4, 4
    else:  # 332
        r_bits, g_bits, b_bits = 3, 3, 2

    for pix_num in available_pixel_indices[:pixels_needed]:
        frame_idx = pix_num // pixels_per_frame
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = result_frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        # Embed into R channel
        for k in range(r_bits):
            if bit_idx < bits.size:
                r = (r & ~(1 << k)) | (int(bits[bit_idx]) << k)
                bit_idx += 1
        # Embed into G channel
        for k in range(g_bits):
            if bit_idx < bits.size:
                g = (g & ~(1 << k)) | (int(bits[bit_idx]) << k)
                bit_idx += 1
        # Embed into B channel
        for k in range(b_bits):
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
                                     original_num_frames: int = 0,
                                     lsb_method: int = LSB_METHOD_332) -> np.ndarray:
    """
    Ekstrak bits dari frames.
    """
    bpp = get_bits_per_pixel(lsb_method)
    h, w, _ = frames[0].shape
    pixels_per_frame = h * w
    current_num_frames = len(frames)

    if original_num_frames > current_num_frames * 2:
        print(f"⚠️ Warning: Header num_frames ({original_num_frames}) > 2x actual ({current_num_frames}). Ignoring header value.")
        calc_num_frames = current_num_frames
    else:
        calc_num_frames = original_num_frames if original_num_frames > 0 else current_num_frames
    
    total_pixels_calc = calc_num_frames * pixels_per_frame
    
    if total_pixels_calc > 500_000_000:
        print(f"⚠️ Warning: Total pixels {total_pixels_calc} too large for safe shuffle. Clamping to actual.")
        calc_num_frames = current_num_frames
        total_pixels_calc = calc_num_frames * pixels_per_frame

    available_pixel_indices = np.arange(pixel_offset, total_pixels_calc)

    if is_random:
        rng = np.random.default_rng(seed)
        rng.shuffle(available_pixel_indices)

    # Determine bit extraction pattern based on method
    if lsb_method == LSB_METHOD_111:
        r_bits, g_bits, b_bits = 1, 1, 1
    elif lsb_method == LSB_METHOD_444:
        r_bits, g_bits, b_bits = 4, 4, 4
    else:  # 332
        r_bits, g_bits, b_bits = 3, 3, 2

    bits = []
    for pix_num in available_pixel_indices:
        if len(bits) >= num_bits:
            break
        
        frame_idx = pix_num // pixels_per_frame
        
        if frame_idx >= current_num_frames:
            bits.extend([0] * bpp)
            continue
            
        local_pix = pix_num % pixels_per_frame
        i, j = divmod(local_pix, w)

        frame = frames[frame_idx]
        r, g, b = int(frame[i, j, 0]), int(frame[i, j, 1]), int(frame[i, j, 2])

        for k in range(r_bits):
            bits.append((r >> k) & 1)
        for k in range(g_bits):
            bits.append((g >> k) & 1)
        for k in range(b_bits):
            bits.append((b >> k) & 1)

    return np.array(bits[:num_bits], dtype=np.uint8)


# ─── HEADER EMBED/EXTRACT (selalu sekuensial, mulai pixel 0) ──────────────────
# NOTE: Header always uses 3-3-2 method for consistency —
# this way we can always read the header to discover the payload's method.

def _embed_header_sequential(frames: list, header: bytes) -> tuple:
    header_bits = bytes_to_bits(header)
    pixels_needed = int(np.ceil(HEADER_BITS / 8))  # 8 bits per pixel for 3-3-2
    
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
    pixels_needed = HEADER_SIZE  # HEADER_SIZE bytes * 8 bits / 8 bpp = HEADER_SIZE pixels

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
                  stego_key: Optional[int] = None, mp4_crf: int = 0,
                  lsb_method: int = LSB_METHOD_332) -> dict:

    frames, fps = read_video_frames(cover_path)

    payload = message
    if use_encryption:
        if a51_key is None: raise ValueError("Wait a51_key")
        payload = a51_encrypt_payload(message, a51_key)

    seed = stego_key if stego_key is not None else 0
    total_cap = total_capacity_bytes(frames, lsb_method)
    needed = HEADER_SIZE + len(payload)
    if needed > total_cap:
        raise ValueError(f"Payload too large: {needed} > {total_cap}")

    header = encode_header(
        is_text=is_text, is_encrypted=use_encryption, is_random=use_random,
        extension=extension, filename=filename, payload_size=len(payload),
        is_mp4=True, num_frames=len(frames), lsb_method=lsb_method
    )

    stego_frames, pixel_offset = _embed_header_sequential(frames, header)
    payload_bits = bytes_to_bits(payload)
    stego_frames = _embed_bits_with_pixel_offset(
        stego_frames, payload_bits, pixel_offset=pixel_offset,
        is_random=use_random, seed=seed, lsb_method=lsb_method
    )

    # Calculate how many frames contain embedded data for selective encoding
    total_bits_embedded = HEADER_BITS + payload_bits.size
    cap_fn = get_capacity_fn(lsb_method)
    frame_capacity_bits = cap_fn(frames[0])
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
        "embedded_frame_count": embedded_frame_count,
        "lsb_method": lsb_method,
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
                "num_frames": len(frames), "is_text": False, "extension": "", "filename": "",
                "lsb_method": LSB_METHOD_332}

    is_random = meta["is_random"]
    is_encrypted = meta["is_encrypted"]
    payload_size = meta["payload_size"]
    num_frames = meta.get("num_frames", 0)
    lsb_method = meta.get("lsb_method", LSB_METHOD_332)

    # Validate lsb_method
    if lsb_method not in LSB_METHOD_LABELS:
        print(f"⚠️ Warning: Unknown LSB method {lsb_method}, defaulting to 3-3-2")
        lsb_method = LSB_METHOD_332

    # Pre-check payload size sanity
    if payload_size > total_capacity_bytes(frames, lsb_method):
        print(f"⚠️ Warning: Payload size {payload_size} > capacity. Header likely corrupt.")
        payload_size = 0

    if num_frames > 0 and len(frames) > num_frames:
        frames = frames[:num_frames]

    payload_bits = _extract_bits_with_pixel_offset(
        frames, num_bits=payload_size * 8, pixel_offset=pixel_offset,
        is_random=is_random, seed=seed, original_num_frames=num_frames,
        lsb_method=lsb_method
    )
    payload = bits_to_bytes(payload_bits)[:payload_size]

    if is_encrypted and payload_size > 0:
        if a51_key is None: raise ValueError("a51_key needed")
        payload = a51_decrypt_payload(payload, a51_key)

    method_label = LSB_METHOD_LABELS.get(lsb_method, "unknown")
    return {
        "message": payload, "is_text": meta["is_text"],
        "is_encrypted": is_encrypted, "is_random": is_random,
        "extension": meta["extension"], "filename": meta["filename"],
        "payload_size": payload_size, "format": "MP4",
        "is_mp4": True, "lsb_method": lsb_method,
        "lsb_method_label": method_label,
    }