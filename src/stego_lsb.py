# src/stego_lsb.py
import numpy as np
from typing import Iterable, Tuple

def bytes_to_bits(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)

def bits_to_bytes(bits: Iterable[int]) -> bytes:
    bits = np.array(bits, dtype=np.uint8)
    # padding ke kelipatan 8
    if bits.size % 8 != 0:
        pad_len = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def capacity_332(frame: np.ndarray) -> int:
    """
    Kapasitas bit untuk satu frame dengan skema 3-3-2.
    """
    h, w, _ = frame.shape
    return h * w * 8  # 3+3+2 = 8 bit per piksel

def _embed_channel(channel_val: int, bits: np.ndarray, offset: int, n: int):
    """Embed n bit mulai dari offset ke channel_val (dari LSB ke atas)."""
    val = channel_val
    for k in range(n):
        if offset + k >= bits.size:
            break
        bit = int(bits[offset + k])
        val = (val & ~(1 << k)) | (bit << k)
    return val & 0xFF

def _extract_channel(channel_val: int, n: int) -> list:
    """Ekstrak n bit dari channel_val (dari LSB ke atas)."""
    return [(channel_val >> k) & 1 for k in range(n)]

def embed_bits_sequential_332(frame: np.ndarray, bits: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    stego = frame.copy().astype(np.uint8)
    idx = 0

    for i in range(h):
        for j in range(w):
            if idx >= bits.size:
                return stego

            r = int(stego[i, j, 0])
            g = int(stego[i, j, 1])
            b = int(stego[i, j, 2])

            r = _embed_channel(r, bits, idx,     3)
            g = _embed_channel(g, bits, idx + 3, 3)
            b = _embed_channel(b, bits, idx + 6, 2)

            stego[i, j] = [r, g, b]
            idx += 8  # 3+3+2 per piksel

    return stego


def extract_bits_sequential_332(frame: np.ndarray, num_bits: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if num_bits > cap:
        raise ValueError(f"Meminta {num_bits} bit, tapi kapasitas frame {cap}")

    bits = []

    for i in range(h):
        for j in range(w):
            if len(bits) >= num_bits:
                break

            r = int(frame[i, j, 0])
            g = int(frame[i, j, 1])
            b = int(frame[i, j, 2])

            bits.extend(_extract_channel(r, 3))
            bits.extend(_extract_channel(g, 3))
            bits.extend(_extract_channel(b, 2))

        if len(bits) >= num_bits:
            break

    return np.array(bits[:num_bits], dtype=np.uint8)

def pixel_indices_random(h: int, w: int, seed: int) -> np.ndarray:
    """
    Generate urutan indeks piksel acak (flattened index) untuk mode acak. [file:1]
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(h * w)
    rng.shuffle(idx)
    return idx

def embed_bits_random_332(frame: np.ndarray, bits: np.ndarray, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    cap = capacity_332(frame)
    if bits.size > cap:
        raise ValueError(f"Payload terlalu besar: {bits.size} > {cap}")

    stego = frame.copy().astype(np.uint8)
    pix_idx = pixel_indices_random(h, w, seed)

    idx = 0
    for flat in pix_idx:
        if idx >= bits.size:
            break
        i, j = divmod(int(flat), w)

        r = int(stego[i, j, 0])
        g = int(stego[i, j, 1])
        b = int(stego[i, j, 2])

        r = _embed_channel(r, bits, idx,     3)
        g = _embed_channel(g, bits, idx + 3, 3)
        b = _embed_channel(b, bits, idx + 6, 2)

        stego[i, j] = [r, g, b]
        idx += 8

    return stego


def extract_bits_random_332(frame: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w, _ = frame.shape
    pix_idx = pixel_indices_random(h, w, seed)

    bits = []
    for flat in pix_idx:
        if len(bits) >= num_bits:
            break
        i, j = divmod(int(flat), w)

        r = int(frame[i, j, 0])
        g = int(frame[i, j, 1])
        b = int(frame[i, j, 2])

        bits.extend(_extract_channel(r, 3))
        bits.extend(_extract_channel(g, 3))
        bits.extend(_extract_channel(b, 2))

    return np.array(bits[:num_bits], dtype=np.uint8)