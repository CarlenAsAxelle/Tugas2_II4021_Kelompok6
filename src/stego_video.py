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

HEADER_SIZE = 64  # bytes (expanded from 32 to fit longer filenames)
# Header layout (64 bytes):
#   [0]      is_text
#   [1]      is_encrypted
#   [2]      is_random
#   [3]      ext_len  (max 10)
#   [4:14]   extension (10 bytes)
#   [14]     fname_len (max 50)
#   [15:65]  filename -- BUT we only have 64 bytes total, so bytes [15:63] = 48 bytes for name
#            (fname stored in [15:63], max 48 chars; fname_len stored in [14])
#   [60:64]  payload_size (4 bytes big-endian)

_FNAME_MAX = 45   # bytes reserved for filename (bytes 15..59)
_EXT_MAX   = 10   # bytes reserved for extension (bytes 4..13)

def encode_header(is_text, is_encrypted, is_random, extension, filename, payload_size):
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
    header[60:64] = payload_size.to_bytes(4, 'big')
    return bytes(header)

def decode_header(header):
    is_text      = bool(header[0])
    is_encrypted = bool(header[1])
    is_random    = bool(header[2])
    ext_len      = header[3]
    extension    = header[4:4 + min(ext_len, _EXT_MAX)].decode('utf-8', errors='replace')
    fname_len    = header[14]
    filename     = header[15:15 + min(fname_len, _FNAME_MAX)].decode('utf-8', errors='replace')
    payload_size = int.from_bytes(header[60:64], 'big')
    return dict(is_text=is_text, is_encrypted=is_encrypted, is_random=is_random,
                extension=extension, filename=filename, payload_size=payload_size)

def _is_valid_header(meta, total_cap):
    """
    Validate header to filter out garbage data.
    Check that metadata fields are within reasonable ranges.
    """
    # Payload size should be positive and reasonable
    if meta["payload_size"] <= 0:
        return False
    if meta["payload_size"] > total_cap:
        return False
    
    return True

def total_capacity_bytes(frames):
    if not frames:
        return 0
    return (capacity_332(frames[0]) * len(frames)) // 8

def _spread_bits_to_frames(frames, all_bits, is_random, seed):
    stego_frames  = []
    offset        = 0
    cap_per_frame = capacity_332(frames[0])
    for frame in frames:
        remaining = all_bits.size - offset
        if remaining <= 0:
            stego_frames.append(frame)
            continue
        chunk  = all_bits[offset:offset + min(cap_per_frame, remaining)]
        offset += len(chunk)
        if is_random:
            embedded = embed_bits_random_332(frame, chunk, seed=seed)
        else:
            embedded = embed_bits_sequential_332(frame, chunk)
        stego_frames.append(embedded)
    return stego_frames

def _collect_bits_from_frames(frames, total_bits_needed, is_random, seed):
    out       = np.empty(total_bits_needed, dtype=np.uint8)
    written   = 0
    remaining = total_bits_needed
    for frame in frames:
        if remaining <= 0:
            break
        cap     = capacity_332(frame)
        to_read = min(cap, remaining)
        bits    = (extract_bits_random_332(frame, to_read, seed=seed)
                   if is_random else
                   extract_bits_sequential_332(frame, to_read))
        out[written:written + bits.size] = bits
        written   += bits.size
        remaining -= bits.size
    return out[:written]

def embed_message(cover_path, output_path, message, is_text,
                  extension="", filename="",
                  use_encryption=False, a51_key=None,
                  use_random=False, stego_key=None):
    frames, fps = read_video_frames(cover_path)

    payload = message
    if use_encryption:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi jika enkripsi diaktifkan")
        payload = a51_encrypt_payload(message, a51_key)

    seed      = stego_key if stego_key is not None else 0
    total_cap = total_capacity_bytes(frames)
    needed    = HEADER_SIZE + len(payload)
    if needed > total_cap:
        raise ValueError(f"Pesan terlalu besar: butuh {needed} bytes, kapasitas {total_cap} bytes")

    header   = encode_header(is_text, use_encryption, use_random, extension, filename, len(payload))
    all_bits = bytes_to_bits(header + payload)

    # Header + payload embedded together with the SAME mode.
    stego_frames = _spread_bits_to_frames(frames, all_bits, use_random, seed)

    # Calculate how many frames contain embedded data for selective encoding
    cap_per_frame = capacity_332(frames[0])
    total_bits_embedded = all_bits.size
    embedded_frame_count = (total_bits_embedded + cap_per_frame - 1) // cap_per_frame

    write_video_frames(output_path, stego_frames, fps,
                      embedded_frame_count=embedded_frame_count,
                      audio_source=cover_path)

    from src.video_io import mse_psnr_video
    mse_list, psnr_list, mse_avg, psnr_avg = mse_psnr_video(frames, stego_frames)

    return dict(total_capacity_bytes=total_cap, payload_size_bytes=len(payload),
                header_size_bytes=HEADER_SIZE, total_embedded_bytes=needed,
                mse_avg=mse_avg, mse_list=mse_list,
                psnr_avg=psnr_avg, psnr_per_frame=psnr_list,
                embedded_frame_count=embedded_frame_count)

def extract_message(stego_path, a51_key=None, stego_key=None):
    """
    Extract message with explicit mode awareness.
    - If stego_key is provided → try random mode
    - Otherwise → try sequential mode only
    """
    frames, _ = read_video_frames(stego_path)
    total_cap = total_capacity_bytes(frames)
    
    is_random_mode = None
    seed = None
    meta = None
    
    # If stego_key provided, use random mode explicitly
    if stego_key is not None:
        try:
            header_bits  = _collect_bits_from_frames(frames, HEADER_SIZE * 8, True, stego_key)
            header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
            test_meta    = decode_header(header_bytes)
            
            # Validate header
            if _is_valid_header(test_meta, total_cap):
                is_random_mode = True
                seed = stego_key
                meta = test_meta
        except Exception as e:
            pass
    else:
        # No stego_key → try sequential mode only
        try:
            header_bits  = _collect_bits_from_frames(frames, HEADER_SIZE * 8, False, 0)
            header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
            test_meta    = decode_header(header_bytes)
            
            # Validate header
            if _is_valid_header(test_meta, total_cap):
                is_random_mode = False
                seed = 0
                meta = test_meta
        except Exception as e:
            pass
    
    if meta is None or is_random_mode is None:
        mode_hint = f"(tried random mode with key {stego_key})" if stego_key is not None else "(tried sequential mode)"
        raise ValueError(
            f"Header tidak valid {mode_hint}. "
            "Pastikan: 1) File berisi embedded message yang valid, 2) Mode sesuai"
        )

    is_encrypted = meta["is_encrypted"]
    payload_size = meta["payload_size"]

    # Read header + payload with the correct mode
    total_bits = (HEADER_SIZE + payload_size) * 8
    all_bits   = _collect_bits_from_frames(frames, total_bits, is_random_mode, seed)
    all_bytes  = bits_to_bytes(all_bits)[:HEADER_SIZE + payload_size]
    payload    = all_bytes[HEADER_SIZE:]

    # Decrypt if needed
    if is_encrypted:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi untuk dekripsi")
        payload = a51_decrypt_payload(payload, a51_key)

    return dict(message=payload, is_text=meta["is_text"],
                is_encrypted=is_encrypted, is_random=meta["is_random"],
                extension=meta["extension"], filename=meta["filename"],
                payload_size=payload_size)