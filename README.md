# Video Steganography with A5/1 Encryption

> **Tugas 2 — II4021 Kriptografi dan Koding**  
> Kelompok 6

Aplikasi steganografi video yang menyembunyikan pesan (teks maupun file) ke dalam video AVI/MP4 menggunakan metode **Modified LSB** (1-1-1, 3-3-2, 4-4-4) dengan enkripsi opsional **A5/1 stream cipher**.

---

## Fitur Utama

| Fitur                        | Deskripsi                                                                      |
|------------------------------|--------------------------------------------------------------------------------|
| **Steganografi Video**       | Embed pesan teks atau file biner ke dalam video AVI dan MP4                    |
| **3 Metode LSB**             | 1-1-1 (3 bit/px), 3-3-2 (8 bit/px), 4-4-4 (12 bit/px)                        |
| **Enkripsi A5/1**            | Stream cipher berbasis GSM A5/1 dengan kunci 64-bit                            |
| **Mode Sequential & Random** | Penyisipan piksel berurutan atau acak (dengan stego key)                       |
| **Selective Encoding**       | Frame berisi data = lossless, frame kosong = lossy → ukuran file lebih kecil   |
| **Analisis Kualitas**        | MSE, PSNR per-frame, dan histogram warna cover vs stego                        |
| **GUI Dashboard**            | Antarmuka Tkinter dengan preview video, kapasitas bar, dan chart               |

---

## Struktur Proyek

```
Tugas2_II4021_Kelompok6/
├── GUI.py                      # Aplikasi GUI utama (Tkinter)
├── requirements.txt            # Dependensi Python
├── src/
│   ├── a51_cipher.py           # Implementasi A5/1 stream cipher
│   ├── stego_lsb_utils.py      # Shared utilities & method registry
│   ├── stego_lsb_111.py        # LSB 1-1-1 embedding/extraction
│   ├── stego_lsb_332.py        # LSB 3-3-2 embedding/extraction
│   ├── stego_lsb_444.py        # LSB 4-4-4 embedding/extraction
│   ├── stego_video.py          # Steganografi video — AVI
│   ├── stego_video_mp4.py      # Steganografi video — MP4
│   ├── video_io.py             # Video I/O — AVI (FFV1 via ffmpeg)
│   └── video_io_mp4.py         # Video I/O — MP4 (libx264rgb via ffmpeg)
├── tests/
│   ├── test_a51_cipher.py      # Unit test A5/1
│   ├── test_stego_lsb.py       # Unit test LSB (semua metode)
│   ├── test_stego_video.py     # Unit test steganografi AVI
│   ├── test_mp4.py             # Unit test steganografi MP4
│   └── test_video_io.py        # Unit test video I/O
├── samples/                    # Video sample untuk testing
└── report/                     # Laporan
```

---

## Instalasi & Menjalankan

### Prasyarat

- **Python** 3.9+
- **ffmpeg** — harus tersedia di PATH
- **opencv-python** 4.8.0+
- **numpy** 1.24.0+
- **pytest** 7.0+
- **matplotlib** 3.5.0+
- **PyQt5** 5.15.0+
- **ffmpeg-python** 0.2.1+

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

## Cara Penggunaan

### Embed Pesan

1. Buka tab **Embed**
2. Klik **Select Cover Video** — pilih file `.avi` atau `.mp4`
3. Pilih **LSB Method** dari dropdown: `1-1-1`, `3-3-2`, atau `4-4-4`
4. Masukkan pesan teks di text box, **atau** klik **Select File to Embed** untuk menyisipkan file biner
5. (Opsional) Aktifkan **A5/1 Encryption** dan masukkan kunci 64-bit, contoh: `0x123456789ABCDEF0`
6. (Opsional) Aktifkan **Random Pixel Order** dan masukkan stego key (integer)
7. Klik **Embed Message** → pilih lokasi output
8. File kunci otomatis disimpan sebagai `*_keys.txt` di samping output video

### Extract Pesan

1. Buka tab **Extract**
2. Klik **Select Stego Video** — pilih video yang sudah di-embed
3. Masukkan **A5/1 Key** jika pesan dienkripsi
4. Masukkan **Stego Key** jika mode random digunakan
5. Klik **Extract Message**
6. Pesan teks ditampilkan langsung; file biner akan diminta lokasi Save As
7. Metode LSB yang digunakan akan terdeteksi secara otomatis dari header

---

## Arsitektur

### Tech Stack 
python 3.9+, OpenCV, NumPy, Matplotlib, Pillow, ffmpeg (system), ffmpeg-python

### Metode LSB

Setiap metode menyisipkan jumlah bit yang berbeda per piksel:

| Metode  | R (LSBs) | G (LSBs) | B (LSBs) | Bit/Piksel | Karakteristik                    |
|---------|----------|----------|----------|------------|----------------------------------|
| **1-1-1** | 1      | 1        | 1        | 3          | Distorsi minimal, kapasitas kecil |
| **3-3-2** | 3      | 3        | 2        | 8          | Keseimbangan kapasitas & stealth  |
| **4-4-4** | 4      | 4        | 4        | 12         | Kapasitas maksimal, distorsi besar|

**Kapasitas per frame** (1920×1080):

| Metode  | Per Frame   | Per 30 Frame |
|---------|-------------|--------------|
| 1-1-1   | ~778 KB     | ~22.8 MB     |
| 3-3-2   | ~2.07 MB    | ~60.7 MB     |
| 4-4-4   | ~3.11 MB    | ~91.1 MB     |

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

## Metrik Kualitas

Setelah embedding, aplikasi menghitung:

- **MSE (Mean Squared Error)** — rata-rata error per piksel antara cover dan stego
- **PSNR (Peak Signal-to-Noise Ratio)** — kualitas visual dalam dB (>40 dB = imperceptible)
- **Color Histogram** — perbandingan distribusi warna R/G/B cover vs stego

---

## Dependensi

| Package                 | Fungsi                                   |
|-------------------------|------------------------------------------|
| `opencv-python` ≥ 4.8.0 | Baca/tulis frame video, image processing |
| `numpy` ≥ 1.24.0        | Operasi array dan bit manipulation       |
| `matplotlib`            | Visualisasi histogram warna              |
| `Pillow`                | Konversi frame untuk preview di Tkinter  |
| `ffmpeg` (system)       | Encoding/decoding video lossless         |

---

## Testing

```bash
# Jalankan semua test
pytest tests/ -v

# Jalankan test spesifik
pytest tests/test_a51_cipher.py -v
pytest tests/test_stego_lsb.py -v
pytest tests/test_stego_video.py -v
pytest tests/test_mp4.py -v
pytest tests/test_video_io.py -v
```

---

## Kelompok 6

| NIM      | Nama                                     |
|----------|------------------------------------------|
| 18223011 | Samuel Chris Michael Bagasta Simanjuntak |
| 18223017 | Carlen Asadel Axelle                     |
| 18223092 | Gabriela Jennifer Sandy                  |

---

## Lisensi

Proyek ini dibuat untuk keperluan tugas kuliah **II4021 Kriptografi dan Koding** — Institut Teknologi Bandung.