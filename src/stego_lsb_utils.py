# src/stego_lsb_utils.py
# Shared utilities for all LSB steganography methods.
import numpy as np
from typing import Iterable


def bytes_to_bits(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    bits = np.array(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        pad_len = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def pixel_indices_random(h: int, w: int, seed: int) -> np.ndarray:
    """
    Generate urutan indeks piksel acak (flattened index) untuk mode acak.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(h * w)
    rng.shuffle(idx)
    return idx


# ─── LSB METHOD REGISTRY ─────────────────────────────────────────────────────
# Each method is identified by an integer stored in the header.
# Method ID → string label mapping:
LSB_METHOD_332 = 0
LSB_METHOD_111 = 1
LSB_METHOD_444 = 2

LSB_METHOD_LABELS = {
    LSB_METHOD_332: "3-3-2",
    LSB_METHOD_111: "1-1-1",
    LSB_METHOD_444: "4-4-4",
}

LSB_LABEL_TO_ID = {v: k for k, v in LSB_METHOD_LABELS.items()}


def get_lsb_functions(method_id: int):
    """
    Return (capacity_fn, embed_seq_fn, extract_seq_fn, embed_rand_fn, extract_rand_fn)
    for the given LSB method ID.
    """
    if method_id == LSB_METHOD_332:
        from src.stego_lsb_332 import (
            capacity_332, embed_bits_sequential_332, extract_bits_sequential_332,
            embed_bits_random_332, extract_bits_random_332
        )
        return (capacity_332, embed_bits_sequential_332, extract_bits_sequential_332,
                embed_bits_random_332, extract_bits_random_332)

    elif method_id == LSB_METHOD_111:
        from src.stego_lsb_111 import (
            capacity_111, embed_bits_sequential_111, extract_bits_sequential_111,
            embed_bits_random_111, extract_bits_random_111
        )
        return (capacity_111, embed_bits_sequential_111, extract_bits_sequential_111,
                embed_bits_random_111, extract_bits_random_111)

    elif method_id == LSB_METHOD_444:
        from src.stego_lsb_444 import (
            capacity_444, embed_bits_sequential_444, extract_bits_sequential_444,
            embed_bits_random_444, extract_bits_random_444
        )
        return (capacity_444, embed_bits_sequential_444, extract_bits_sequential_444,
                embed_bits_random_444, extract_bits_random_444)

    else:
        raise ValueError(f"Unknown LSB method ID: {method_id}")


def get_capacity_fn(method_id: int):
    """Return just the capacity function for the given method."""
    return get_lsb_functions(method_id)[0]


def get_bits_per_pixel(method_id: int) -> int:
    """Return how many bits are embedded per pixel for the given method."""
    if method_id == LSB_METHOD_332:
        return 8   # 3+3+2
    elif method_id == LSB_METHOD_111:
        return 3   # 1+1+1
    elif method_id == LSB_METHOD_444:
        return 12  # 4+4+4
    else:
        raise ValueError(f"Unknown LSB method ID: {method_id}")
