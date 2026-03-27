# 🎬 Video Steganography with A5/1 Encryption

> **Tugas 2 — II4021 Kriptografi dan Koding**  
> Kelompok 6

Aplikasi steganografi video yang menyembunyikan pesan (teks maupun file) ke dalam video AVI/MP4 menggunakan metode **Modified LSB 3-3-2** dengan enkripsi opsional **A5/1 stream cipher**.

---

## ✨ Fitur Utama

| Fitur                        | Deskripsi                                                                      |
|------------------------------|--------------------------------------------------------------------------------|
| **Steganografi Video**       | Embed pesan teks atau file biner ke dalam video AVI dan MP4                    |
| **Modified LSB 3-3-2**       | Menyisipkan 8 bit per piksel (3 bit di R, 3 bit di G, 2 bit di B)              |
| **Enkripsi A5/1**            | Stream cipher berbasis GSM A5/1 dengan kunci 64-bit                            |
| **Mode Sequential & Random** | Penyisipan piksel berurutan atau acak (dengan stego key)                       |
| **Selective Encoding**       | Frame berisi data = lossless, frame kosong = lossy → ukuran file lebih kecil   |
| **Analisis Kualitas**        | MSE, PSNR per-frame, dan histogram warna cover vs stego                        |
| **GUI Dashboard**            | Antarmuka Tkinter dengan preview video, kapasitas bar, dan chart               |

---

## 📁 Struktur Proyek

```
Tugas2_II4021_Kelompok6/
├── GUI.py                      # Aplikasi GUI utama (Tkinter)
├── requirements.txt            # Dependensi Python
├── src/
│   ├── a51_cipher.py           # Implementasi A5/1 stream cipher
│   ├── stego_lsb.py            # LSB 3-3-2 embedding/extraction (vectorized)
│   ├── stego_video.py          # Steganografi video — AVI
│   ├── stego_video_mp4.py      # Steganografi video — MP4
│   ├── video_io.py             # Video I/O — AVI (FFV1 via ffmpeg)
│   └── video_io_mp4.py         # Video I/O — MP4 (libx264rgb via ffmpeg)
├── tests/
│   ├── test_a51_cipher.py      # Unit test A5/1
│   ├── test_stego_lsb.py       # Unit test LSB 3-3-2
│   ├── test_stego_video.py     # Unit test steganografi AVI
│   ├── test_mp4.py             # Unit test steganografi MP4
│   └── test_video_io.py        # Unit test video I/O
├── samples/                    # Video sample untuk testing
└── report/                     # Laporan
```

---

## 🚀 Instalasi & Menjalankan

### Prasyarat

- **Python** 3.9+
- **ffmpeg** — harus tersedia di PATH

#### Install ffmpeg

| OS      | Perintah                                                                        |
|---------|---------------------------------------------------------------------------------|
| Windows | Download dari [ffmpeg.org](https://ffmpeg.org/download.html), tambahkan ke PATH |
| Linux   | `sudo apt install ffmpeg`                                                       |
| macOS   | `brew install ffmpeg`                                                           |

### Setup

```bash
# Clone repository
git clone https://github.com/<username>/Tugas2_II4021_Kelompok6.git
cd Tugas2_II4021_Kelompok6

# Install dependensi
pip install -r requirements.txt
```

### Menjalankan GUI

```bash
python GUI.py
```

### Menjalankan Tests

```bash
pytest tests/ -v
```

---

## 🔧 Cara Penggunaan

### Embed Pesan

1. Buka tab **📥 Embed**
2. Klik **Select Cover Video** — pilih file `.avi` atau `.mp4`
3. Masukkan pesan teks di text box, **atau** klik **Select File to Embed** untuk menyisipkan file biner
4. (Opsional) Aktifkan **A5/1 Encryption** dan masukkan kunci 64-bit, contoh: `0x123456789ABCDEF0`
5. (Opsional) Aktifkan **Random Pixel Order** dan masukkan stego key (integer)
6. Klik **🔒 Embed Message** → pilih lokasi output
7. File kunci otomatis disimpan sebagai `*_keys.txt` di samping output video

### Extract Pesan

1. Buka tab **📤 Extract**
2. Klik **Select Stego Video** — pilih video yang sudah di-embed
3. Masukkan **A5/1 Key** jika pesan dienkripsi
4. Masukkan **Stego Key** jika mode random digunakan
5. Klik **🔓 Extract Message**
6. Pesan teks ditampilkan langsung; file biner akan diminta lokasi Save As

---

## 🏗️ Arsitektur

### Pipeline

```
                    ┌─────────────┐
    Pesan/File ───▶ │  A5/1       │ ───▶ Ciphertext (opsional)
                    │  Encrypt    │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  LSB 3-3-2  │ ───▶ Embed ke frame video
                    │  Embedding  │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Selective  │ ───▶ Output video (AVI/MP4)
                    │  Encoding   │
                    └─────────────┘
```

### Modified LSB 3-3-2

Setiap piksel menyimpan **8 bit** data:

| Channel | Bits Disimpan|    Mask   |
|---------|--------------|-----------|
| Red     | 3 LSBs       | `& 0b111` |
| Green   | 3 LSBs       | `& 0b111` |
| Blue    | 2 LSBs       | `& 0b11`  |

Kapasitas per frame = `width × height × 8 bits`

### Selective Encoding (Pengurangan Ukuran File)

Untuk menghindari bloat pada file output:

- **AVI**: Frame yang mengandung data di-encode lossless (FFV1), frame lainnya melalui JPEG round-trip (quality 92) sebelum encode FFV1 → mengurangi entropi signifikan
- **MP4**: Frame data di-encode lossless (libx264rgb CRF 0), frame lainnya di-encode lossy (libx264 CRF 23), lalu di-concat

### A5/1 Stream Cipher

Implementasi sesuai spesifikasi GSM:
- 3 LFSR (19-bit, 22-bit, 23-bit) dengan majority clocking
- Kunci 64-bit, block size 228 bit
- XOR keystream dengan plaintext

---

## 📊 Metrik Kualitas

Setelah embedding, aplikasi menghitung:

- **MSE (Mean Squared Error)** — rata-rata error per piksel antara cover dan stego
- **PSNR (Peak Signal-to-Noise Ratio)** — kualitas visual dalam dB (>40 dB = imperceptible)
- **Color Histogram** — perbandingan distribusi warna R/G/B cover vs stego

---

## 📦 Dependensi

| Package                 | Fungsi                                   |
|-------------------------|------------------------------------------|
| `opencv-python` ≥ 4.8.0 | Baca/tulis frame video, image processing |
| `numpy` ≥ 1.24.0        | Operasi array dan bit manipulation       |
| `matplotlib`            | Visualisasi histogram warna              |
| `Pillow`                | Konversi frame untuk preview di Tkinter  |
| `ffmpeg` (system)       | Encoding/decoding video lossless         |

---

## 🧪 Testing

```bash
# Jalankan semua test
pytest tests/ -v

# Jalankan test spesifik
pytest tests/test_a51_cipher.py -v
pytest tests/test_stego_lsb.py -v
pytest tests/test_stego_video.py -v
pytest tests/test_mp4.py -v
```

---

## 👥 Kelompok 6

| NIM      | Nama                      |
|----------|---------------------------|
| 18223017 | Carlen Asadel Axelle      |
| —        | —                         |
| —        | —                         |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan tugas kuliah **II4021 Kriptografi dan Koding** — Institut Teknologi Bandung.