# src/stego_video.py
import os
import numpy as np
from typing import Optional

from src.video_io import read_video_frames, write_video_frames
from src.stego_lsb import (
    bytes_to_bits, bits_to_bytes,
    capacity_332,
    embed_bits_sequential_332, extract_bits_sequential_332,
    embed_bits_random_332, extract_bits_random_332
)
from src.a51_cipher import a51_encrypt_payload, a51_decrypt_payload


# ─── METADATA HEADER ──────────────────────────────────────────────────────────
# Format header (fixed 32 bytes):
# [0]     is_text       : 1 byte  (1=teks, 0=file)
# [1]     is_encrypted  : 1 byte  (1=ya, 0=tidak)
# [2]     is_random     : 1 byte  (1=acak, 0=sekuensial)
# [3]     ext_len       : 1 byte  (panjang string ekstensi)
# [4:14]  extension     : 10 bytes (ekstensi file, misal ".pdf\x00...")
# [14:18] filename_len  : 4 bytes (panjang nama file asli)
# [18:28] filename      : 10 bytes (nama file asli, truncated)
# [28:32] payload_size  : 4 bytes (ukuran payload BYTES setelah enkripsi)
HEADER_SIZE = 32  # bytes


def encode_header(is_text: bool, is_encrypted: bool, is_random: bool,
                  extension: str, filename: str, payload_size: int) -> bytes:
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

    payload_size = int.from_bytes(header[28:32], 'big')

    return {
        "is_text":      is_text,
        "is_encrypted": is_encrypted,
        "is_random":    is_random,
        "extension":    extension,
        "filename":     filename,
        "payload_size": payload_size
    }


# ─── CAPACITY ─────────────────────────────────────────────────────────────────

def total_capacity_bytes(frames: list) -> int:
    """Total kapasitas semua frame dalam bytes."""
    total_bits = sum(capacity_332(f) for f in frames)
    return total_bits // 8


# ─── EMBED / EXTRACT ACROSS FRAMES ───────────────────────────────────────────

def _spread_bits_to_frames(frames: list, all_bits: np.ndarray,
                           is_random: bool, seed: int) -> list:
    """Distribusikan bits ke frame-frame secara berurutan."""
    stego_frames = []
    offset = 0

    for frame in frames:
        cap = capacity_332(frame)
        remaining = all_bits.size - offset

        if remaining <= 0:
            stego_frames.append(frame.copy())
            continue

        chunk = all_bits[offset:offset + min(cap, remaining)]
        offset += len(chunk)

        if is_random:
            stego_frame = embed_bits_random_332(frame, chunk, seed=seed)
        else:
            stego_frame = embed_bits_sequential_332(frame, chunk)

        stego_frames.append(stego_frame)

    return stego_frames


def _collect_bits_from_frames(frames: list, total_bits_needed: int,
                               is_random: bool, seed: int) -> np.ndarray:
    """Kumpulkan bits dari frame-frame secara berurutan."""
    collected = []
    remaining = total_bits_needed

    for frame in frames:
        if remaining <= 0:
            break
        cap = capacity_332(frame)
        to_read = min(cap, remaining)

        if is_random:
            bits = extract_bits_random_332(frame, to_read, seed=seed)
        else:
            bits = extract_bits_sequential_332(frame, to_read)

        collected.append(bits)
        remaining -= to_read

    return np.concatenate(collected) if collected else np.array([], dtype=np.uint8)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def embed_message(
    cover_path: str,
    output_path: str,
    message: bytes,
    is_text: bool,
    extension: str = "",
    filename: str = "",
    use_encryption: bool = False,
    a51_key: Optional[int] = None,
    use_random: bool = False,
    stego_key: Optional[int] = None
) -> dict:
    """
    Embed pesan ke video AVI.

    Returns dict berisi info embed: kapasitas, ukuran pesan, PSNR rata-rata.
    """
    frames, fps = read_video_frames(cover_path)

    # Enkripsi jika dipilih
    payload = message
    if use_encryption:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi jika enkripsi diaktifkan")
        payload = a51_encrypt_payload(message, a51_key)

    # Cek kapasitas
    seed = stego_key if stego_key is not None else 0
    total_cap = total_capacity_bytes(frames)
    needed = HEADER_SIZE + len(payload)
    if needed > total_cap:
        raise ValueError(
            f"Pesan terlalu besar: butuh {needed} bytes, "
            f"kapasitas {total_cap} bytes"
        )

    # Buat header
    header = encode_header(
        is_text=is_text,
        is_encrypted=use_encryption,
        is_random=use_random,
        extension=extension,
        filename=filename,
        payload_size=len(payload)
    )

    # Gabung header + payload → bits
    all_data = header + payload
    all_bits = bytes_to_bits(all_data)

    # Embed ke frame-frame
    stego_frames = _spread_bits_to_frames(frames, all_bits, use_random, seed)

    # Tulis stego video
    write_video_frames(output_path, stego_frames, fps)

    # Hitung PSNR rata-rata
    from src.video_io import mse_psnr_video
    _, psnr_list, mse_avg, psnr_avg = mse_psnr_video(frames, stego_frames)

    return {
        "total_capacity_bytes": total_cap,
        "payload_size_bytes":   len(payload),
        "header_size_bytes":    HEADER_SIZE,
        "total_embedded_bytes": needed,
        "mse_avg":              mse_avg,
        "psnr_avg":             psnr_avg,
        "psnr_per_frame":       psnr_list
    }


def extract_message(
    stego_path: str,
    a51_key: Optional[int] = None,
    stego_key: Optional[int] = None
) -> dict:
    """
    Ekstrak pesan dari stego video AVI.

    Returns dict berisi: message (bytes), metadata dari header.
    """
    frames, _ = read_video_frames(stego_path)
    seed = stego_key if stego_key is not None else 0

    # Step 1: baca header dulu (HEADER_SIZE * 8 bit)
    header_bits_needed = HEADER_SIZE * 8
    header_bits = _collect_bits_from_frames(frames, header_bits_needed,
                                             is_random=False, seed=seed)
    header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
    meta = decode_header(header_bytes)

    is_random    = meta["is_random"]
    is_encrypted = meta["is_encrypted"]
    payload_size = meta["payload_size"]

    # Step 2: baca header + payload sekaligus
    total_bits = (HEADER_SIZE + payload_size) * 8
    all_bits = _collect_bits_from_frames(frames, total_bits,
                                          is_random=is_random, seed=seed)
    all_bytes = bits_to_bytes(all_bits)[:HEADER_SIZE + payload_size]
    payload = all_bytes[HEADER_SIZE:]

    # Step 3: dekripsi jika perlu
    if is_encrypted:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi untuk dekripsi")
        payload = a51_decrypt_payload(payload, a51_key)

    return {
        "message":      payload,
        "is_text":      meta["is_text"],
        "is_encrypted": is_encrypted,
        "is_random":    is_random,
        "extension":    meta["extension"],
        "filename":     meta["filename"],
        "payload_size": payload_size
    }


# cara menggunakannya : 
# from src.stego_video import embed_message, extract_message

## EMBED
# result = embed_message(
#     cover_path     = "samples/sample_video.avi",
#     output_path    = "samples/stego_output.avi",
#     message        = open("samples/pesan.txt", "rb").read(),
#     is_text        = True,
#     extension      = ".txt",
#     filename       = "pesan.txt",
#     use_encryption = True,
#     a51_key        = 0x123456789ABCDEF0,
#     use_random     = False,
#     stego_key      = None
# )
# print(f"PSNR: {result['psnr_avg']:.2f} dB")

## EXTRACT
# out = extract_message(
#     stego_path = "samples/stego_output.avi",
#     a51_key    = 0x123456789ABCDEF0
# )
# print(out["message"].decode())
