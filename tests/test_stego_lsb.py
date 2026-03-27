# tests/test_stego_lsb.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.stego_lsb_utils import bytes_to_bits, bits_to_bytes, pixel_indices_random
from src.stego_lsb_332 import (
    capacity_332,
    embed_bits_sequential_332, extract_bits_sequential_332,
    embed_bits_random_332, extract_bits_random_332,
)
from src.video_io import read_video_frames

SAMPLE_VIDEO = "samples/sample_video.avi"

# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def real_frame():
    """Frame pertama dari sample_video.avi."""
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip(f"Sample video tidak ada: {SAMPLE_VIDEO}")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    return frames[0]

@pytest.fixture
def sample_message():
    return b"Pesan rahasia Tugas 2 II4021 Kelompok 6 ITB!"

@pytest.fixture
def sample_message_file():
    """Baca dari samples/pesan.txt."""
    path = "samples/pesan.txt"
    if not os.path.exists(path):
        pytest.skip(f"pesan.txt tidak ada: {path}")
    with open(path, 'rb') as f:
        return f.read()

# ─── UTILS ────────────────────────────────────────────────────────────────────

def test_bytes_to_bits_and_back(sample_message):
    bits = bytes_to_bits(sample_message)
    assert len(bits) == len(sample_message) * 8
    assert set(bits).issubset({0, 1})

    recovered = bits_to_bytes(bits)[:len(sample_message)]
    assert recovered == sample_message

def test_bits_to_bytes_padding():
    """Padding ke kelipatan 8 tidak error."""
    bits = np.array([1, 0, 1], dtype=np.uint8)
    result = bits_to_bytes(bits)
    assert isinstance(result, bytes)
    assert len(result) == 1  # 3 bit → padding jadi 8 bit → 1 byte

# ─── CAPACITY ─────────────────────────────────────────────────────────────────

def test_capacity_332(real_frame):
    h, w, _ = real_frame.shape
    cap = capacity_332(real_frame)
    assert cap == h * w * 8
    print(f"\nKapasitas frame {w}x{h}: {cap} bit = {cap // 8} bytes = {cap // 8 // 1024} KB")

# ─── SEQUENTIAL EMBED/EXTRACT ─────────────────────────────────────────────────

def test_sequential_roundtrip_short_message(real_frame, sample_message):
    """Embed → extract pesan pendek, harus identical."""
    bits = bytes_to_bits(sample_message)

    stego = embed_bits_sequential_332(real_frame, bits)
    extracted_bits = extract_bits_sequential_332(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]

    md5_orig = hashlib.md5(sample_message).hexdigest()
    md5_recv = hashlib.md5(recovered).hexdigest()
    print(f"\nMD5 original:  {md5_orig}")
    print(f"MD5 recovered: {md5_recv}")

    assert recovered == sample_message

def test_sequential_roundtrip_file(real_frame, sample_message_file):
    """Embed → extract isi pesan.txt, harus identical."""
    bits = bytes_to_bits(sample_message_file)
    cap = capacity_332(real_frame)
    assert bits.size <= cap, f"pesan.txt terlalu besar untuk 1 frame: {bits.size} > {cap}"

    stego = embed_bits_sequential_332(real_frame, bits)
    extracted_bits = extract_bits_sequential_332(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message_file)]

    assert recovered == sample_message_file
    print(f"\npesan.txt roundtrip OK ({len(sample_message_file)} bytes)")

def test_sequential_stego_frame_diff(real_frame, sample_message):
    """Frame asli dan stego harus beda (embed terjadi), tapi sangat mirip."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_332(real_frame, bits)

    assert not np.array_equal(real_frame, stego), "Stego frame identik dengan cover (embed tidak terjadi?)"

    diff = np.abs(real_frame.astype(int) - stego.astype(int))
    max_diff = diff.max()

    # Skema 3-3-2: max diff per channel = 7 (R/G, 3 bit) atau 3 (B, 2 bit)
    assert max_diff <= 7, f"Perbedaan piksel > 7 (tidak wajar untuk 3-3-2): max_diff={max_diff}"
    print(f"\nMax pixel diff: {max_diff} (wajar untuk skema 3-3-2, max=7)")

def test_sequential_capacity_exceeded(real_frame):
    """Embed melebihi kapasitas harus raise ValueError."""
    cap = capacity_332(real_frame)
    oversized_bits = np.ones(cap + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="Payload terlalu besar"):
        embed_bits_sequential_332(real_frame, oversized_bits)

# ─── RANDOM EMBED/EXTRACT ─────────────────────────────────────────────────────

def test_random_roundtrip(real_frame, sample_message):
    """Embed → extract acak dengan seed yang sama harus identical."""
    seed = 42
    bits = bytes_to_bits(sample_message)

    stego = embed_bits_random_332(real_frame, bits, seed=seed)
    extracted_bits = extract_bits_random_332(stego, len(bits), seed=seed)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]

    assert recovered == sample_message
    print(f"\nRandom roundtrip OK (seed={seed})")

def test_random_wrong_seed_fails(real_frame, sample_message):
    """Extract dengan seed salah tidak boleh recover pesan yang sama."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_random_332(real_frame, bits, seed=42)

    extracted_bits = extract_bits_random_332(stego, len(bits), seed=999)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]

    assert recovered != sample_message
    print(f"\nWrong seed correctly fails to recover message")

def test_random_vs_sequential_differ(real_frame, sample_message):
    """Stego hasil random dan sequential harus berbeda (urutan piksel beda)."""
    bits = bytes_to_bits(sample_message)
    stego_seq = embed_bits_sequential_332(real_frame, bits)
    stego_rnd = embed_bits_random_332(real_frame, bits, seed=42)

    assert not np.array_equal(stego_seq, stego_rnd)
    print(f"\nSequential dan random menghasilkan stego berbeda")

def test_pixel_indices_random_reproducible():
    """Seed yang sama harus selalu hasilkan urutan yang sama."""
    idx1 = pixel_indices_random(480, 640, seed=42)
    idx2 = pixel_indices_random(480, 640, seed=42)
    assert np.array_equal(idx1, idx2)

    idx3 = pixel_indices_random(480, 640, seed=99)
    assert not np.array_equal(idx1, idx3)
    print(f"\nSeed reproducible OK")

# ─── OUTPUT SAVE ──────────────────────────────────────────────────────────────

def test_save_stego_output(real_frame, sample_message):
    """Simpan frame stego ke tests/output/ untuk inspeksi visual."""
    import cv2
    os.makedirs("tests_output", exist_ok=True)

    bits = bytes_to_bits(sample_message)

    stego_seq = embed_bits_sequential_332(real_frame, bits)
    stego_rnd = embed_bits_random_332(real_frame, bits, seed=42)

    cv2.imwrite("tests_output/cover_frame.png", real_frame)
    cv2.imwrite("tests_output/stego_sequential_frame.png", stego_seq)
    cv2.imwrite("tests_output/stego_random_frame.png", stego_rnd)

    diff_seq = cv2.absdiff(real_frame, stego_seq)
    cv2.imwrite("tests_output/diff_sequential.png", diff_seq * 50)  # amplify diff

    print(f"\n  Saved to tests_output/")
    print(f"   cover_frame.png")
    print(f"   stego_sequential_frame.png")
    print(f"   stego_random_frame.png")
    print(f"   diff_sequential.png (amplified x50)")

    assert os.path.exists("tests_output/stego_sequential_frame.png")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
