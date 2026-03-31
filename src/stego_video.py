# src/stego_video.py
import os
import numpy as np
from typing import Optional

from src.video_io import read_video_frames, write_video_frames
from src.stego_lsb_utils import (
    bytes_to_bits, bits_to_bytes,
    get_lsb_functions, get_capacity_fn, get_bits_per_pixel,
    LSB_METHOD_332, LSB_METHOD_111, LSB_METHOD_444, LSB_METHOD_LABELS
)
from src.a51_cipher import a51_encrypt_payload, a51_decrypt_payload

HEADER_SIZE = 64  # bytes (expanded from 32 to fit longer filenames)
# Header layout (64 bytes):
#   [0]      is_text
#   [1]      is_encrypted
#   [2]      is_random
#   [3]      ext_len  (max 10)
#   [4:14]   extension (10 bytes)
#   [14]     fname_len (max 45)
#   [15:59]  filename (max 45 bytes)
#   [59]     lsb_method (0=3-3-2, 1=1-1-1, 2=4-4-4)
#   [60:64]  payload_size (4 bytes big-endian)

_FNAME_MAX = 45   # bytes reserved for filename (bytes 15..59)
_EXT_MAX   = 10   # bytes reserved for extension (bytes 4..13)

def encode_header(is_text, is_encrypted, is_random, extension, filename,
                  payload_size, lsb_method=LSB_METHOD_332):
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
    header[59] = lsb_method & 0xFF
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
    lsb_method   = header[59]
    payload_size = int.from_bytes(header[60:64], 'big')
    return dict(is_text=is_text, is_encrypted=is_encrypted, is_random=is_random,
                extension=extension, filename=filename, payload_size=payload_size,
                lsb_method=lsb_method)

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
    # LSB method should be known
    if meta["lsb_method"] not in LSB_METHOD_LABELS:
        return False
    
    return True

def total_capacity_bytes(frames, lsb_method=LSB_METHOD_332):
    if not frames:
        return 0
    cap_fn = get_capacity_fn(lsb_method)
    return (cap_fn(frames[0]) * len(frames)) // 8

def _spread_bits_to_frames(frames, all_bits, is_random, seed, lsb_method):
    cap_fn, embed_seq, _, embed_rand, _ = get_lsb_functions(lsb_method)
    stego_frames  = []
    offset        = 0
    cap_per_frame = cap_fn(frames[0])
    for frame in frames:
        remaining = all_bits.size - offset
        if remaining <= 0:
            stego_frames.append(frame)
            continue
        chunk  = all_bits[offset:offset + min(cap_per_frame, remaining)]
        offset += len(chunk)
        if is_random:
            embedded = embed_rand(frame, chunk, seed=seed)
        else:
            embedded = embed_seq(frame, chunk)
        stego_frames.append(embedded)
    return stego_frames

def _collect_bits_from_frames(frames, total_bits_needed, is_random, seed, lsb_method):
    cap_fn, _, extract_seq, _, extract_rand = get_lsb_functions(lsb_method)
    out       = np.empty(total_bits_needed, dtype=np.uint8)
    written   = 0
    remaining = total_bits_needed
    for frame in frames:
        if remaining <= 0:
            break
        cap     = cap_fn(frame)
        to_read = min(cap, remaining)
        bits    = (extract_rand(frame, to_read, seed=seed)
                   if is_random else
                   extract_seq(frame, to_read))
        out[written:written + bits.size] = bits
        written   += bits.size
        remaining -= bits.size
    return out[:written]

def embed_message(cover_path, output_path, message, is_text,
                  extension="", filename="",
                  use_encryption=False, a51_key=None,
                  use_random=False, stego_key=None,
                  lsb_method=LSB_METHOD_332):
    frames, fps = read_video_frames(cover_path)

    payload = message
    if use_encryption:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi jika enkripsi diaktifkan")
        payload = a51_encrypt_payload(message, a51_key)

    seed      = stego_key if stego_key is not None else 0
    total_cap = total_capacity_bytes(frames, lsb_method)
    needed    = HEADER_SIZE + len(payload)
    if needed > total_cap:
        raise ValueError(f"Pesan terlalu besar: butuh {needed} bytes, kapasitas {total_cap} bytes")

    header   = encode_header(is_text, use_encryption, use_random, extension,
                             filename, len(payload), lsb_method=lsb_method)
    all_bits = bytes_to_bits(header + payload)

    # Header + payload embedded together with the SAME mode.
    stego_frames = _spread_bits_to_frames(frames, all_bits, use_random, seed, lsb_method)

    # Calculate how many frames contain embedded data for selective encoding
    cap_fn = get_capacity_fn(lsb_method)
    cap_per_frame = cap_fn(frames[0])
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
                embedded_frame_count=embedded_frame_count,
                lsb_method=lsb_method)

def extract_message(stego_path, a51_key=None, stego_key=None):
    """
    Extract message with explicit mode awareness.
    - If stego_key is provided → try random mode
    - Otherwise → try sequential mode only
    
    The LSB method is auto-detected from the header.
    We try each method until we find a valid header.
    """
    frames, _ = read_video_frames(stego_path)
    
    is_random_mode = None
    seed = None
    meta = None
    detected_method = None
    
    # Try each LSB method to find the correct one
    methods_to_try = [LSB_METHOD_332, LSB_METHOD_111, LSB_METHOD_444]
    
    for try_method in methods_to_try:
        total_cap = total_capacity_bytes(frames, try_method)
        
        if stego_key is not None:
            # stego_key provided → try random mode
            try:
                header_bits  = _collect_bits_from_frames(frames, HEADER_SIZE * 8, True, stego_key, try_method)
                header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
                test_meta    = decode_header(header_bytes)
                
                if (_is_valid_header(test_meta, total_cap) and 
                    test_meta.get("lsb_method", LSB_METHOD_332) == try_method):
                    is_random_mode = True
                    seed = stego_key
                    meta = test_meta
                    detected_method = try_method
                    break
            except Exception:
                pass
        else:
            # No stego_key → try sequential mode only
            try:
                header_bits  = _collect_bits_from_frames(frames, HEADER_SIZE * 8, False, 0, try_method)
                header_bytes = bits_to_bytes(header_bits)[:HEADER_SIZE]
                test_meta    = decode_header(header_bytes)
                
                if (_is_valid_header(test_meta, total_cap) and
                    test_meta.get("lsb_method", LSB_METHOD_332) == try_method):
                    is_random_mode = False
                    seed = 0
                    meta = test_meta
                    detected_method = try_method
                    break
            except Exception:
                pass
    
    if meta is None or is_random_mode is None:
        mode_hint = f"(tried random mode with key {stego_key})" if stego_key is not None else "(tried sequential mode)"
        raise ValueError(
            f"Header tidak valid {mode_hint}. "
            "Pastikan: 1) File berisi embedded message yang valid, 2) Mode sesuai"
        )

    is_encrypted = meta["is_encrypted"]
    payload_size = meta["payload_size"]

    # Read header + payload with the correct mode and method
    total_bits = (HEADER_SIZE + payload_size) * 8
    all_bits   = _collect_bits_from_frames(frames, total_bits, is_random_mode, seed, detected_method)
    all_bytes  = bits_to_bytes(all_bits)[:HEADER_SIZE + payload_size]
    payload    = all_bytes[HEADER_SIZE:]

    # Decrypt if needed
    if is_encrypted:
        if a51_key is None:
            raise ValueError("a51_key wajib diisi untuk dekripsi")
        payload = a51_decrypt_payload(payload, a51_key)

    method_label = LSB_METHOD_LABELS.get(detected_method, "unknown")
    return dict(message=payload, is_text=meta["is_text"],
                is_encrypted=is_encrypted, is_random=meta["is_random"],
                extension=meta["extension"], filename=meta["filename"],
                payload_size=payload_size, lsb_method=detected_method,
                lsb_method_label=method_label)