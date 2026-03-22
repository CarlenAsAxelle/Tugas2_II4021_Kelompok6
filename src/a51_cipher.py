# src/a51_cipher.py
from dataclasses import dataclass
from typing import List

@dataclass
class LFSR:
    size: int
    taps: List[int]  # posisi bit yang di-XOR (0 = LSB / atau sesuaikan konvensi)
    reg: int         # state saat ini (disimpan sebagai integer)

    def clock(self, control_bit: int = None, majority_bit: int = None) -> int:
        """
        Clock LFSR satu kali, return output bit.
        Jika control_bit/majority_bit dipakai (A5/1), tambahkan logika dulu di luar. 
        """
        out = self.reg & 1
        feedback = 0
        for t in self.taps:
            feedback ^= (self.reg >> t) & 1

        self.reg >>= 1
        self.reg |= (feedback << (self.size - 1))
        self.reg &= (1 << self.size) - 1
        return out

def majority(x: int, y: int, z: int) -> int:
    return 1 if (x + y + z) >= 2 else 0

class A51:
    """
    Implementasi A5/1 stream cipher (kasar, disesuaikan dengan spesifikasi kuliah).
    """
    def __init__(self, key_64bit: int):
        # TODO: inisialisasi 3 LFSR dengan ukuran dan taps yang benar
        # Contoh placeholder (bukan nilai final!)
        self.r1 = LFSR(size=19, taps=[13, 16, 17, 18], reg=0)
        self.r2 = LFSR(size=22, taps=[20, 21], reg=0)
        self.r3 = LFSR(size=23, taps=[7, 20, 21, 22], reg=0)

        self._init_with_key(key_64bit)

    def _init_with_key(self, key_64bit: int):
        """
        Inisialisasi internal state dari key 64-bit (ikuti prosedur A5/1). 
        """
        # TODO: isi sesuai algoritma A5/1 versi tugas
        self.r1.reg = key_64bit & ((1 << 19) - 1)
        self.r2.reg = (key_64bit >> 19) & ((1 << 22) - 1)
        self.r3.reg = (key_64bit >> (19 + 22)) & ((1 << 23) - 1)

    def keystream_bit(self) -> int:
        """
        Hasilkan 1 bit keystream.
        Di A5/1 asli, pakai majority clocking pada bit kontrol tertentu (mis: bit 8,10,10).
        """
        # TODO: majority-based clocking
        b1 = self.r1.clock()
        b2 = self.r2.clock()
        b3 = self.r3.clock()
        return b1 ^ b2 ^ b3

    def keystream(self, n_bits: int) -> bytes:
        result = bytearray()
        byte_val = 0
        for i in range(n_bits):
            bit = self.keystream_bit()
            byte_val = (byte_val << 1) | bit
            if (i + 1) % 8 == 0:
                result.append(byte_val)
                byte_val = 0
        return bytes(result)


    def encrypt(self, data: bytes) -> bytes:
        """
        XOR data dengan keystream (stream cipher).
        """
        n_bits = len(data) * 8
        ks = self.keystream(n_bits)
        return bytes(d ^ k for d, k in zip(data, ks))

    def decrypt(self, data: bytes) -> bytes:
        """
        Sama dengan encrypt, karena stream cipher.
        """
        return self.encrypt(data)

def a51_encrypt_payload(payload: bytes, key_64bit: int) -> bytes:
    """
    Enkripsi payload dengan A5/1 menggunakan key 64-bit.
    Stream cipher: XOR payload dengan keystream. [file:1]
    """
    cipher = A51(key_64bit)
    return cipher.encrypt(payload)


def a51_decrypt_payload(ciphertext: bytes, key_64bit: int) -> bytes:
    """
    Dekripsi ciphertext dengan A5/1 (identik dengan encrypt). [file:1]
    """
    cipher = A51(key_64bit)
    return cipher.decrypt(ciphertext)