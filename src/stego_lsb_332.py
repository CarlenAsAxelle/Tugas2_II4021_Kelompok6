# src/stego_lsb_332.py
# 3-3-2 LSB steganography: 3 bits in R, 3 bits in G, 2 bits in B = 8 bits/pixel.
import numpy as np
from src.stego_lsb_utils import pixel_indices_random


def capacity_332(frame: np.ndarray) -> int:
    h, w, _ = frame.shape
    return h * w * 8  # 3+3+2 = 8 bit per piksel


# ─── INTERNAL VECTORIZED HELPERS ──────────────────────────────────────────────

def _embed_332_vectorized(pixels: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """
    Embed bits into a flat array of pixels (shape: [N, 3]) using 3-3-2 scheme.
    Fully vectorized — no Python-level loops over pixels.
    """
    n_pixels = pixels.shape[0]
    n_bits   = bits.size

    # Pad bits to full pixel boundary (multiple of 8)
    full_bits = n_pixels * 8
    if n_bits < full_bits:
        padded = np.zeros(full_bits, dtype=np.uint8)
        padded[:n_bits] = bits
        bits = padded

    # Reshape bits → [N, 8]: columns 0-2 → R (3 LSBs), 3-5 → G (3 LSBs), 6-7 → B (2 LSBs)
    bits_2d = bits[:full_bits].reshape(n_pixels, 8)

    result = pixels.copy().astype(np.uint8)

    # R channel: embed 3 bits (bits 0,1,2) into bits 0,1,2 of R
    r = result[:, 0].astype(np.int32)
    r = (r & ~0b111) | (bits_2d[:, 0] | (bits_2d[:, 1] << 1) | (bits_2d[:, 2] << 2))
    result[:, 0] = r.astype(np.uint8)

    # G channel: embed 3 bits (bits 3,4,5) into bits 0,1,2 of G
    g = result[:, 1].astype(np.int32)
    g = (g & ~0b111) | (bits_2d[:, 3] | (bits_2d[:, 4] << 1) | (bits_2d[:, 5] << 2))
    result[:, 1] = g.astype(np.uint8)

    # B channel: embed 2 bits (bits 6,7) into bits 0,1 of B
    b = result[:, 2].astype(np.int32)
    b = (b & ~0b11) | (bits_2d[:, 6] | (bits_2d[:, 7] << 1))
    result[:, 2] = b.astype(np.uint8)

    return result


def _extract_332_vectorized(pixels: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Extract bits from a flat pixel array (shape: [N, 3]) using 3-3-2 scheme.
    Fully vectorized — no Python-level loops.
    """
    n_pixels = pixels.shape[0]
    bits_2d  = np.empty((n_pixels, 8), dtype=np.uint8)

    r = pixels[:, 0].astype(np.uint8)
    g = pixels[:, 1].astype(np.uint8)
    b = pixels[:, 2].astype(np.uint8)

    bits_2d[:, 0] = (r >> 0) & 1
    bits_2d[:, 1] = (r >> 1) & 1
    bits_2d[:, 2] = (r >> 2) & 1

    bits_2d[:, 3] = (g >> 0) & 1
    bits_2d[:, 4] = (g >> 1) & 1
    bits_2d[:, 5] = (g >> 2) & 1

    bits_2d[:, 6] = (b >> 0) & 1
    bits_2d[:, 7] = (b >> 1) & 1

    return bits_2d.ravel()[:num_bits]


# ─── PUBLIC: SEQUENTIAL ───────────────────────────────────────────────────────

def embed_bits_sequential_332(frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    pixels = frame.reshape(-1, 3)          # [H*W, 3]
    n_pixels_needed = int(np.ceil(bits.size / 8))

    stego_pixels = pixels.copy()
    stego_pixels[:n_pixels_needed] = _embed_332_vectorized(
        pixels[:n_pixels_needed], bits
    )
    return stego_pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_sequential_332(frame: np.ndarray, num_bits: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if num_bits > cap:
        raise ValueError(f"Meminta {num_bits} bit, tapi kapasitas frame {cap}")

    n_pixels_needed = int(np.ceil(num_bits / 8))
    pixels = frame.reshape(-1, 3)[:n_pixels_needed]
    return _extract_332_vectorized(pixels, num_bits)


# ─── PUBLIC: RANDOM ───────────────────────────────────────────────────────────

def embed_bits_random_332(frame: np.ndarray, bits: np.ndarray, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    n_pixels_needed = int(np.ceil(bits.size / 8))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3).copy()
    pixels[pix_idx] = _embed_332_vectorized(pixels[pix_idx], bits)
    return pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_random_332(frame: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    n_pixels_needed = int(np.ceil(num_bits / 8))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3)[pix_idx]
    return _extract_332_vectorized(pixels, num_bits)
