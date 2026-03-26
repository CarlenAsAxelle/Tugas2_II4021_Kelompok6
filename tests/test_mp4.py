# tests/test_mp4.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.stego_video_mp4 import embed_message, extract_message

def test_mp4_roundtrip():
    # Load pesan
    pesan_path = "samples/pesan.txt"
    if not os.path.exists(pesan_path):
        print(f"⚠️  {pesan_path} tidak ada, buat dulu!")
        return

    with open(pesan_path, 'rb') as f:
        original = f.read()

    print(f"Original ({len(original)} bytes): {original[:50]}...")

    # Cek video input
    cover_path = "samples/blackpink.mp4"
    if not os.path.exists(cover_path):
        print(f"⚠️  {cover_path} tidak ada, taruh dulu video MP4-nya!")
        return

    os.makedirs("tests_output", exist_ok=True)
    output_path = "tests_output/stego_blackpink.mp4"

    A51_KEY   = 0x123456789ABCDEF0
    STEGO_KEY = 42

    # ── EMBED ──
    print("\n[1] EMBED pesan ke video MP4...")
    print(f"    Cover  : {cover_path}")
    print(f"    Output : {output_path}")
    print(f"    Enkripsi A5/1 : YA  (key={hex(A51_KEY)})")
    print(f"    Mode         : ACAK (seed={STEGO_KEY})")

    try:
        result = embed_message(
            cover_path     = cover_path,
            output_path    = output_path,
            message        = original,
            is_text        = True,
            extension      = ".txt",
            filename       = "pesan.txt",
            use_encryption = True,
            a51_key        = A51_KEY,
            use_random     = True,
            stego_key      = STEGO_KEY,
            mp4_crf        = 0
        )
        print(f"\n    ✓ Embed berhasil!")
        print(f"    Format         : {result['format']}")
        print(f"    Kapasitas      : {result['total_capacity_bytes']:,} bytes")
        print(f"    Pesan+header   : {result['total_embedded_bytes']:,} bytes")
        print(f"    MSE rata-rata  : {result['mse_avg']:.4f}")
        print(f"    PSNR rata-rata : {result['psnr_avg']:.2f} dB")
        print(f"    Lossless MP4   : {result['lossless_mp4']}")
    except Exception as e:
        print(f"\n    ✗ Embed GAGAL: {e}")
        return

    # ── EXTRACT ──
    print("\n[2] EXTRACT pesan dari stego video...")
    print(f"    Stego  : {output_path}")

    try:
        out = extract_message(
            stego_path = output_path,
            a51_key    = A51_KEY,
            stego_key  = STEGO_KEY
        )
        extracted = out["message"]
        print(f"\n    ✓ Extract berhasil!")
        print(f"    Pesan hasil : {extracted[:50]}...")
        print(f"    Format      : {out['format']}")
        print(f"    Terenkripsi : {out['is_encrypted']}")
        print(f"    Mode acak   : {out['is_random']}")
    except Exception as e:
        print(f"\n    ✗ Extract GAGAL: {e}")
        return

    # ── SAVE OUTPUT ──
    with open("tests_output/test_output_mp4_extracted.txt", 'wb') as f:
        f.write(extracted)

    # ── MD5 VERIFICATION ──
    orig_md5 = hashlib.md5(original).hexdigest()
    extr_md5 = hashlib.md5(extracted).hexdigest()

    print(f"\n[3] VERIFIKASI integritas pesan...")
    print(f"    MD5 original  : {orig_md5}")
    print(f"    MD5 extracted : {extr_md5}")

    assert extracted == original, "EXTRACT GAGAL — pesan tidak sama!"
    print("    ✓ PESAN IDENTIK — steganografi MP4 berhasil sempurna!")
    print("\nOutput tersimpan: tests_output/")

if __name__ == "__main__":
    test_mp4_roundtrip()
