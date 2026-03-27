# src/stego_lsb_111.py
# 1-1-1 LSB steganography: 1 bit in R, 1 bit in G, 1 bit in B = 3 bits/pixel.
# Lowest capacity but most subtle — hardest to detect visually.
import numpy as np
from src.stego_lsb_utils import pixel_indices_random


def capacity_111(frame: np.ndarray) -> int:
    h, w, _ = frame.shape
    return h * w * 3  # 1+1+1 = 3 bit per piksel


# ─── INTERNAL VECTORIZED HELPERS ──────────────────────────────────────────────

def _embed_111_vectorized(pixels: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """
    Embed bits into a flat array of pixels (shape: [N, 3]) using 1-1-1 scheme.
    1 bit in R LSB, 1 bit in G LSB, 1 bit in B LSB.
    Fully vectorized.
    """
    n_pixels = pixels.shape[0]
    n_bits   = bits.size

    # Pad bits to full pixel boundary (multiple of 3)
    full_bits = n_pixels * 3
    if n_bits < full_bits:
        padded = np.zeros(full_bits, dtype=np.uint8)
        padded[:n_bits] = bits
        bits = padded

    # Reshape bits → [N, 3]: column 0 → R, 1 → G, 2 → B
    bits_2d = bits[:full_bits].reshape(n_pixels, 3)

    result = pixels.copy().astype(np.uint8)

    # R channel: embed 1 bit into LSB
    r = result[:, 0].astype(np.int32)
    r = (r & ~1) | bits_2d[:, 0]
    result[:, 0] = r.astype(np.uint8)

    # G channel: embed 1 bit into LSB
    g = result[:, 1].astype(np.int32)
    g = (g & ~1) | bits_2d[:, 1]
    result[:, 1] = g.astype(np.uint8)

    # B channel: embed 1 bit into LSB
    b = result[:, 2].astype(np.int32)
    b = (b & ~1) | bits_2d[:, 2]
    result[:, 2] = b.astype(np.uint8)

    return result


def _extract_111_vectorized(pixels: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Extract bits from a flat pixel array (shape: [N, 3]) using 1-1-1 scheme.
    Fully vectorized.
    """
    n_pixels = pixels.shape[0]
    bits_2d  = np.empty((n_pixels, 3), dtype=np.uint8)

    bits_2d[:, 0] = pixels[:, 0] & 1  # R LSB
    bits_2d[:, 1] = pixels[:, 1] & 1  # G LSB
    bits_2d[:, 2] = pixels[:, 2] & 1  # B LSB

    return bits_2d.ravel()[:num_bits]


# ─── PUBLIC: SEQUENTIAL ───────────────────────────────────────────────────────

def embed_bits_sequential_111(frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_111(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    pixels = frame.reshape(-1, 3)
    n_pixels_needed = int(np.ceil(bits.size / 3))

    stego_pixels = pixels.copy()
    stego_pixels[:n_pixels_needed] = _embed_111_vectorized(
        pixels[:n_pixels_needed], bits
    )
    return stego_pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_sequential_111(frame: np.ndarray, num_bits: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_111(frame)
    if num_bits > cap:
        raise ValueError(f"Meminta {num_bits} bit, tapi kapasitas frame {cap}")

    n_pixels_needed = int(np.ceil(num_bits / 3))
    pixels = frame.reshape(-1, 3)[:n_pixels_needed]
    return _extract_111_vectorized(pixels, num_bits)


# ─── PUBLIC: RANDOM ───────────────────────────────────────────────────────────

def embed_bits_random_111(frame: np.ndarray, bits: np.ndarray, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_111(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    n_pixels_needed = int(np.ceil(bits.size / 3))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3).copy()
    pixels[pix_idx] = _embed_111_vectorized(pixels[pix_idx], bits)
    return pixels.reshape(h, w, 3).astype(np.uint8)


def extract_bits_random_111(frame: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    n_pixels_needed = int(np.ceil(num_bits / 3))
    pix_idx = pixel_indices_random(h, w, seed)[:n_pixels_needed]

    pixels = frame.reshape(-1, 3)[pix_idx]
    return _extract_111_vectorized(pixels, num_bits)
