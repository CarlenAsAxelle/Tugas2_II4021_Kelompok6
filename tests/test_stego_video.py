# tests/test_stego_video.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.stego_video import embed_message, extract_message, total_capacity_bytes
from src.video_io import read_video_frames

SAMPLE_VIDEO = "samples/sample_video.avi"
OUTPUT_VIDEO  = "tests_output/stego_test_output.avi"
PESAN_PATH    = "samples/pesan.txt"
A51_KEY       = 0x123456789ABCDEF0
STEGO_KEY     = 42

@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs("tests_output", exist_ok=True)

@pytest.fixture
def pesan():
    if not os.path.exists(PESAN_PATH):
        pytest.skip(f"pesan.txt tidak ada: {PESAN_PATH}")
    with open(PESAN_PATH, 'rb') as f:
        return f.read()

# ─── 1. KAPASITAS ─────────────────────────────────────────────────────────────

def test_capacity():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames)
    assert cap > 0
    print(f"\nKapasitas video: {cap} bytes = {cap // 1024} KB")

# ─── 2. EMBED + EXTRACT (no encryption, sequential) ───────────────────────────

def test_roundtrip_plain_sequential(pesan):
    result = embed_message(
        cover_path     = SAMPLE_VIDEO,
        output_path    = OUTPUT_VIDEO,
        message        = pesan,
        is_text        = True,
        extension      = ".txt",
        filename       = "pesan.txt",
        use_encryption = False,
        use_random     = False
    )
    print(f"\nPSNR: {result['psnr_avg']:.2f} dB | MSE: {result['mse_avg']:.4f}")
    assert result['psnr_avg'] > 30, "PSNR terlalu rendah"

    out = extract_message(stego_path=OUTPUT_VIDEO)

    assert out["message"] == pesan
    assert out["is_text"] == True
    assert out["is_encrypted"] == False
    assert out["is_random"] == False
    assert out["filename"] == "pesan.txt"

    md5_orig = hashlib.md5(pesan).hexdigest()
    md5_recv = hashlib.md5(out["message"]).hexdigest()
    print(f"MD5 original:  {md5_orig}")
    print(f"MD5 recovered: {md5_recv}")
    assert md5_orig == md5_recv

# ─── 3. EMBED + EXTRACT (with encryption, sequential) ────────────────────────

def test_roundtrip_encrypted_sequential(pesan):
    result = embed_message(
        cover_path     = SAMPLE_VIDEO,
        output_path    = OUTPUT_VIDEO,
        message        = pesan,
        is_text        = True,
        extension      = ".txt",
        filename       = "pesan.txt",
        use_encryption = True,
        a51_key        = A51_KEY,
        use_random     = False
    )
    print(f"\nPSNR: {result['psnr_avg']:.2f} dB")

    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=A51_KEY)

    assert out["message"] == pesan
    assert out["is_encrypted"] == True
    print("Encrypted sequential roundtrip OK")

# ─── 4. EMBED + EXTRACT (no encryption, random) ───────────────────────────────

def test_roundtrip_plain_random(pesan):
    result = embed_message(
        cover_path     = SAMPLE_VIDEO,
        output_path    = OUTPUT_VIDEO,
        message        = pesan,
        is_text        = True,
        extension      = ".txt",
        filename       = "pesan.txt",
        use_encryption = False,
        use_random     = True,
        stego_key      = STEGO_KEY
    )
    print(f"\nPSNR: {result['psnr_avg']:.2f} dB")

    out = extract_message(stego_path=OUTPUT_VIDEO, stego_key=STEGO_KEY)

    assert out["message"] == pesan
    assert out["is_random"] == True
    print("Plain random roundtrip OK")

# ─── 5. EMBED + EXTRACT (encrypted + random) ──────────────────────────────────

def test_roundtrip_encrypted_random(pesan):
    result = embed_message(
        cover_path     = SAMPLE_VIDEO,
        output_path    = OUTPUT_VIDEO,
        message        = pesan,
        is_text        = True,
        extension      = ".txt",
        filename       = "pesan.txt",
        use_encryption = True,
        a51_key        = A51_KEY,
        use_random     = True,
        stego_key      = STEGO_KEY
    )
    print(f"\nPSNR: {result['psnr_avg']:.2f} dB")

    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=A51_KEY, stego_key=STEGO_KEY)

    assert out["message"] == pesan
    assert out["is_encrypted"] == True
    assert out["is_random"] == True
    print("Encrypted random roundtrip OK")

# ─── 6. KAPASITAS TERLAMPAUI ──────────────────────────────────────────────────

def test_capacity_exceeded():
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames)
    oversized = b"X" * (cap + 1)

    with pytest.raises(ValueError, match="Pesan terlalu besar"):
        embed_message(
            cover_path  = SAMPLE_VIDEO,
            output_path = OUTPUT_VIDEO,
            message     = oversized,
            is_text     = True,
            use_random  = False
        )
    print(f"\nKapasitas exceeded correctly rejected ({cap} bytes)")

# ─── 7. WRONG KEY FAILS ───────────────────────────────────────────────────────

def test_wrong_a51_key_fails(pesan):
    embed_message(
        cover_path     = SAMPLE_VIDEO,
        output_path    = OUTPUT_VIDEO,
        message        = pesan,
        is_text        = True,
        use_encryption = True,
        a51_key        = A51_KEY,
        use_random     = False
    )

    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=0xDEADBEEFDEADBEEF)
    assert out["message"] != pesan
    print("Wrong A5/1 key correctly fails to recover message")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
