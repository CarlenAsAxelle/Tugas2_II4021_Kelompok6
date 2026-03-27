# tests/test_a51_cipher.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.a51_cipher import A51, a51_encrypt_payload, a51_decrypt_payload
from src.stego_lsb_utils import bytes_to_bits, bits_to_bytes

def test_a51_roundtrip():
    """
    Test enkripsi → dekripsi A5/1 perfect recovery.
    """
    # Load pesan
    pesan_path = "samples/pesan.txt"
    if not os.path.exists(pesan_path):
        print(f"⚠️  {pesan_path} tidak ada, buat dulu!")
        return
    
    with open(pesan_path, 'rb') as f:
        original = f.read()
    
    print(f"Original ({len(original)} bytes): {original[:50]}...")
    
    # Test key 64-bit (contoh)
    key_64bit = 0x123456789ABCDEF0
    
    # Encrypt → Decrypt
    ciphertext = a51_encrypt_payload(original, key_64bit)
    decrypted = a51_decrypt_payload(ciphertext, key_64bit)
    
    # SAVE OUTPUT
    os.makedirs("tests_output", exist_ok=True)
    
    with open("tests_output/test_output_pesan_encrypted.txt", 'wb') as f:
        f.write(ciphertext)
    
    with open("tests_output/test_output_pesan_decrypted.txt", 'wb') as f:
        f.write(decrypted)
    
    # MD5 verification
    orig_md5 = hashlib.md5(original).hexdigest()
    decr_md5 = hashlib.md5(decrypted).hexdigest()
    
    print(f"MD5 original:  {orig_md5}")
    print(f"MD5 decrypted: {decr_md5}")
    assert decrypted == original, "DECRYPT GAGAL!"
    
    print("Output tersimpan: tests_output/")
    
    # Test stream cipher property
    keystream1 = a51_encrypt_payload(b"A", key_64bit)
    keystream2 = a51_encrypt_payload(b"B", key_64bit)
    xor_result = bytes(k1 ^ k2 for k1, k2 in zip(keystream1, keystream2))
    print(f"Keystream XOR test: {xor_result[:10].hex()}")

if __name__ == "__main__":
    test_a51_roundtrip()
