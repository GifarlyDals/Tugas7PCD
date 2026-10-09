# Prototype Verifikasi Ijazah (OCR Nomor + Deteksi Tanda Tangan)

Input: citra ijazah (`.jpg`). Output:

```
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## 1. Struktur Folder

```
ijazah_verifier/
├── verifikasi_ijazah.py   # pipeline utama (CLI 1 citra)
├── evaluasi.py            # benchmark CER semua metode enhancement + uji tanda tangan
├── requirements.txt
├── input/                 # 9 citra uji (01_HighQuality ... 09_CombinedDegradation)
└── output/                # hasil CSV, grafik, dan citra debug tiap tahap
```

## 2. Cara Menjalankan

**Prasyarat:** Python 3.9+ dan program Tesseract OCR terpasang.

```bash
# Linux (Debian/Ubuntu)
sudo apt install tesseract-ocr
# macOS:   brew install tesseract
# Windows: pasang installer UB-Mannheim, lalu tambahkan folder Tesseract ke PATH
#          (atau set pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe")

pip install -r requirements.txt
```

**Verifikasi satu citra:**

```bash
python verifikasi_ijazah.py input/01_HighQuality_Enhanced.jpg
python verifikasi_ijazah.py input/09_CombinedDegradation.jpg --method bilateral --debug
```

Opsi `--debug` menyimpan citra setiap tahap (grayscale, ROI nomor, hasil threshold, morfologi, dst.) ke `output/<nama_citra>/`.
Opsi `--method` memilih enhancement area nomor: `none, hist_eq, clahe, gaussian, median, bilateral, nlmeans, unsharp, illum_norm, clahe_sharpen, clahe_denoise_sharpen, illum_clahe_sharpen` (default `bilateral`).

**Benchmark CER semua metode (±5 menit karena stress test):**

```bash
python evaluasi.py                 # benchmark + stress test
python evaluasi.py --no-stress     # hanya 9 citra asli (±1 menit)
python evaluasi.py --gt 571012022000056   # ground truth nomor ijazah
```

Keluaran di `output/`: `hasil_cer.csv`, `ringkasan_cer.csv`, `hasil_cer_stress.csv`, `ringkasan_cer_stress.csv`, `hasil_verifikasi.csv`, `grafik_cer.png`.

## 3. Metode yang Digunakan

| Tahap | Metode | Alasan |
|---|---|---|
| Orientasi | Normalisasi lebar (1823 px), lalu coba rotasi 0/90/180/270°, dipilih yang OCR-nya paling banyak menemukan kata kunci (`universitas`, `ijazah`, `rektor`, ...) | Citra uji tersimpan miring 90°; ukuran citra 08 berbeda dari lainnya |
| Grayscale | `cv2.cvtColor` BGR→Gray | Mengurangi dimensi; menghilangkan efek *color shift/warm tint* |
| Pemisahan area | ROI relatif (rasio terhadap lebar/tinggi citra tegak): area nomor di kiri-bawah, area tanda tangan di antara kata "Rektor" dan nama "Prof. Ari Kuncoro" | ROI relatif kebal terhadap perbedaan resolusi |
| **Area nomor** | Crop → upscale 3× (bicubic) → **enhancement** (dibandingkan 12 metode) → padding → **Tesseract OCR** (`--psm 7`, whitelist angka 0-9) | Whitelist mencegah huruf/simbol salah baca |
| **Area tanda tangan** | Crop → koreksi iluminasi (`illum_norm`: bagi dengan estimasi latar) → Gaussian blur → **thresholding Otsu** + batas gelap ketat (`0.6 × median`) → **morfologi** opening 3×3 (buang bintik/pola latar guilloche) lalu closing 9×9 (sambung goresan putus) → *connected components* | Tanda tangan = goresan tinta gelap, kontinu, ukuran besar; pola latar = bintik kecil tersebar |
| Keputusan TTD | `PRESENT` bila ada komponen dengan area ≥ 250 px, rentang (bbox) ≥ 60 px, dan rasio tinta ≥ 0,4 % dari ROI | Aturan sederhana & bisa dijelaskan |

Catatan: pada ijazah UI tanda tangan yang dideteksi adalah **Rektor** (kolom kiri, di atas "Prof. Ari Kuncoro"). Tanda tangan Dekan dapat dideteksi dengan mengganti `ROI_TTD_REKTOR` ke koordinat kolom kanan.

## 4. Hasil Evaluasi

Ground truth nomor ijazah: `571012022000056` (15 digit). CER = jarak Levenshtein / panjang ground truth.

### 4.1 Verifikasi pada 9 citra uji (metode `bilateral`)

- **Nomor ijazah: 9/9 terbaca benar** (CER = 0).
- **Tanda tangan:** `PRESENT` pada 9/9 citra.
- **Uji negatif:** tanda tangan dihapus secara sintetis (ROI diisi warna latar) → `ABSENT` pada 9/9 citra (rasio tinta 0,0000 vs 0,0067–0,0333 bila ada tanda tangan). Citra `02_LowContrast` adalah kasus paling tipis (0,0067), masih di atas ambang 0,004.

### 4.2 CER metode enhancement — 9 citra uji asli

| Metode | CER rata-rata | CER terburuk | Citra sempurna |
|---|---|---|---|
| none, gaussian, median, bilateral, nlmeans, unsharp, illum_norm | **0,0000** | 0,0000 | 9/9 |
| clahe, clahe_sharpen, clahe_denoise_sharpen | 0,0074 | 0,0667 | 8/9 |
| illum_clahe_sharpen | 0,1111 | 1,0000 | 8/9 |
| hist_eq | 0,8667 | 1,0000 | 0/9 |

Pada citra uji asli hasilnya **jenuh**: 7 metode sama-sama CER 0, sehingga tidak bisa dibedakan. Karena itu ditambahkan *stress test*.

### 4.3 Stress test (degradasi sintetis tambahan pada 9 citra × 5 jenis = 45 kasus)

CER rata-rata, makin kecil makin baik:

| Metode | Rata-rata | noise σ=35 | blur σ=3 | kontras ×0,25 | resolusi 0,4× | gabungan |
|---|---|---|---|---|---|---|
| **bilateral** | **0,2267** | **0,1259** | 0,0074 | 0,0000 | 0,0000 | 1,0000 |
| none | 0,2711 | 0,4444 | 0,0074 | 0,0000 | 0,0000 | 0,9037 |
| nlmeans | 0,2874 | 0,5185 | 0,0296 | 0,0000 | 0,0000 | 0,8889 |
| illum_norm | 0,3155 | 0,4444 | 0,1333 | 0,0000 | 0,0000 | 1,0000 |
| median | 0,3437 | 0,7111 | 0,0074 | 0,0000 | 0,0000 | 1,0000 |
| gaussian | 0,3600 | 0,7926 | 0,0074 | 0,0000 | 0,0000 | 1,0000 |
| unsharp | 0,3674 | 0,8296 | 0,0074 | 0,0000 | 0,0000 | 1,0000 |
| clahe_sharpen | 0,4015 | 0,9778 | 0,0222 | 0,0074 | 0,0000 | 1,0000 |
| illum_clahe_sharpen | 0,4015 | 1,0000 | 0,1333 | 0,0000 | 0,0000 | 0,8741 |
| clahe_denoise_sharpen | 0,4030 | 0,9852 | 0,0593 | 0,0000 | 0,0000 | 0,9704 |
| clahe | 0,4059 | 1,0000 | 0,0222 | 0,0074 | 0,0000 | 1,0000 |
| hist_eq | 0,8667 | 0,8963 | 0,6667 | 0,7926 | 0,9778 | 1,0000 |

## 5. Jawaban: Metode Enhancement Mana yang Paling Efektif Berdasarkan CER?

**Jawaban: bilateral filter** (`cv2.bilateralFilter(g, 7, 50, 50)`) — CER rata-rata terendah pada stress test (**0,2267**), dan tetap CER 0 pada seluruh 9 citra uji asli.

Alasan:

1. **Sumber kesalahan OCR terbesar adalah noise.** Pada noise σ=35, bilateral menurunkan CER dari 0,4444 (tanpa enhancement) menjadi 0,1259, sedangkan median/Gaussian/unsharp justru lebih buruk daripada tanpa enhancement. Bilateral meredam noise tetapi **mempertahankan tepi** angka; Gaussian dan median ikut mengaburkan tepi goresan tipis huruf serif sehingga Tesseract salah membedakan angka (1/7, 0/6, dst.).
2. **Metode berbasis kontras malah merugikan.** `hist_eq` global (CER 0,87) menghancurkan citra karena latar guilloche ikut dikontraskan; `clahe` dan turunannya memperkuat noise dan tekstur latar (CER ≈ 0,40 di stress test, dan 0,0667 pada satu citra asli).
3. **Citra asli sudah cukup bersih.** Karena `none` pun CER 0 pada 9 citra asli, enhancement yang agresif tidak diperlukan; enhancement hanya bernilai bila ada noise, sehingga pilihan terbaik adalah yang paling "aman" (bilateral).