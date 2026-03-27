# src/stego_lsb_444.py
# 4-4-4 LSB steganography: 4 bits in R, 4 bits in G, 4 bits in B = 12 bits/pixel.
# Highest capacity but most visual distortion.
import numpy as np
from src.stego_lsb_utils import pixel_indices_random


def capacity_444(frame: np.ndarray) -> int:
    h, w, _ = frame.shape
    return h * w * 12  # 4+4+4 = 12 bit per piksel


# ─── INTERNAL VECTORIZED HELPERS ──────────────────────────────────────────────

def _embed_444_vectorized(pixels: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """
    Embed bits into a flat array of pixels (shape: [N, 3]) using 4-4-4 scheme.
    4 bits in R[0:3], 4 bits in G[0:3], 4 bits in B[0:3].
    Fully vectorized.
    """
    n_pixels = pixels.shape[0]
    n_bits   = bits.size

    # Pad bits to full pixel boundary (multiple of 12)
    full_bits = n_pixels * 12
    if n_bits < full_bits:
        padded = np.zeros(full_bits, dtype=np.uint8)
        padded[:n_bits] = bits
        bits = padded

    # Reshape bits → [N, 12]: columns 0-3 → R, 4-7 → G, 8-11 → B
    bits_2d = bits[:full_bits].reshape(n_pixels, 12)

    result = pixels.copy().astype(np.uint8)

    # R channel: embed 4 bits (bits 0,1,2,3) into bits 0,1,2,3 of R
    r = result[:, 0].astype(np.int32)
    r = (r & ~0b1111) | (bits_2d[:, 0] | (bits_2d[:, 1] << 1) |
                          (bits_2d[:, 2] << 2) | (bits_2d[:, 3] << 3))
    result[:, 0] = r.astype(np.uint8)

    # G channel: embed 4 bits (bits 4,5,6,7) into bits 0,1,2,3 of G
    g = result[:, 1].astype(np.int32)
    g = (g & ~0b1111) | (bits_2d[:, 4] | (bits_2d[:, 5] << 1) |
                          (bits_2d[:, 6] << 2) | (bits_2d[:, 7] << 3))
    result[:, 1] = g.astype(np.uint8)

    # B channel: embed 4 bits (bits 8,9,10,11) into bits 0,1,2,3 of B
    b = result[:, 2].astype(np.int32)
    b = (b & ~0b1111) | (bits_2d[:, 8] | (bits_2d[:, 9] << 1) |
                          (bits_2d[:, 10] << 2) | (bits_2d[:, 11] << 3))
    result[:, 2] = b.astype(np.uint8)

    return result


def _extract_444_vectorized(pixels: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Extract bits from a flat pixel array (shape: [N, 3]) using 4-4-4 scheme.
    Fully vectorized.
    """
    n_pixels = pixels.shape[0]
    bits_2d  = np.empty((n_pixels, 12), dtype=np.uint8)

    r = pixels[:, 0].astype(np.uint8)
    g = pixels[:, 1].astype(np.uint8)
    b = pixels[:, 2].astype(np.uint8)

    bits_2d[:, 0]  = (r >> 0) & 1
    bits_2d[:, 1]  = (r >> 1) & 1
    bits_2d[:, 2]  = (r >> 2) & 1
    bits_2d[:, 3]  = (r >> 3) & 1

    bits_2d[:, 4]  = (g >> 0) & 1
    bits_2d[:, 5]  = (g >> 1) & 1
    bits_2d[:, 6]  = (g >> 2) & 1
    bits_2d[:, 7]  = (g >> 3) & 1

    bits_2d[:, 8]  = (b >> 0) & 1
    bits_2d[:, 9]  = (b >> 1) & 1
    bits_2d[:, 10] = (b >> 2) & 1
    bits_2d[:, 11] = (b >> 3) & 1

    return bits_2d.ravel()[:num_bits]


# ─── PUBLIC: SEQUENTIAL ───────────────────────────────────────────────────────

def embed_bits_sequential_444(frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_444(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    pixels = frame.reshape(-1, 3)
    n_pixels_needed = int(np.ceil(bits.size / 12))

    stego_pixels = pixels.copy()
    stego_pixels[:n_pixels_needed] = _embed_444_vectorized(
        pixels[:n_pixels_needed], bits
    )
    return stego_pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_sequential_444(frame: np.ndarray, num_bits: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_444(frame)
    if num_bits > cap:
        raise ValueError(f"Meminta {num_bits} bit, tapi kapasitas frame {cap}")

    n_pixels_needed = int(np.ceil(num_bits / 12))
    pixels = frame.reshape(-1, 3)[:n_pixels_needed]
    return _extract_444_vectorized(pixels, num_bits)


# ─── PUBLIC: RANDOM ───────────────────────────────────────────────────────────

def embed_bits_random_444(frame: np.ndarray, bits: np.ndarray, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_444(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    n_pixels_needed = int(np.ceil(bits.size / 12))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3).copy()
    pixels[pix_idx] = _embed_444_vectorized(pixels[pix_idx], bits)
    return pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_random_444(frame: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    n_pixels_needed = int(np.ceil(num_bits / 12))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3)[pix_idx]
    return _extract_444_vectorized(pixels, num_bits)
