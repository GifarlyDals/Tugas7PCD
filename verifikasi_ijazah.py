#!/usr/bin/env python3
"""
Prototype Verifikasi Ijazah
===========================
Pipeline:
    Citra -> Grayscale -> Image Enhancement -> +-> Area Nomor  -> Enhancement -> OCR
                                               +-> Area TTD    -> Threshold -> Morfologi -> Deteksi TTD
                                                                     -> Hasil Verifikasi

Pemakaian:
    python verifikasi_ijazah.py input/ijazah_001.jpg
    python verifikasi_ijazah.py input/ijazah_001.jpg --method clahe_denoise_sharpen --debug
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np
import pytesseract

STD_WIDTH = 1823            # lebar standar (portrait) -> semua citra dinormalisasi
DEFAULT_METHOD = "bilateral"   # terbaik pada stress test (lihat README)

# ROI relatif (x0, x1, y0, y1) terhadap citra yang sudah tegak (landscape) & dinormalisasi.
# Nilai dikalibrasi dari tata letak ijazah UI (foto landscape 2576x1823).
ROI_NOMOR = (0.1585, 0.300, 0.895, 0.950)     # angka setelah "Nomor ijazah:"
ROI_TTD_REKTOR = (0.125, 0.400, 0.695, 0.825)  # antara "Rektor" dan "Prof. ..."


# ---------------------------------------------------------------- 1. Preprocessing
def load_and_orient(path):
    """Baca citra, normalisasi ukuran, dan putar hingga teks tegak (auto-orientasi)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if min(h, w) != STD_WIDTH:
        s = STD_WIDTH / min(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    rots = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    # Tata letak ijazah landscape: kandidat yang lebarnya > tingginya dicoba dulu.
    best, best_score = None, -1
    for r in rots:
        cand = img if r is None else cv2.rotate(img, r)
        if cand.shape[1] < cand.shape[0]:
            continue
        small = cv2.resize(cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
        txt = pytesseract.image_to_string(small, config="--psm 11").lower()
        score = sum(txt.count(k) for k in ("universitas", "ijazah", "komputer", "rektor", "dekan"))
        if score > best_score:
            best, best_score = cand, score
    # Jika semua kandidat landscape gagal (citra sangat rusak), pakai rotasi 90 CW sebagai default.
    if best is None or best_score <= 0:
        best = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if img.shape[1] < img.shape[0] else img
    return best


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- 2. Enhancement
def _clahe(g, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def _unsharp(g, sigma=1.5, amount=1.5):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


def _illum_norm(g):
    """Koreksi iluminasi/warna: bagi dengan estimasi latar (closing + blur)."""
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    bg = cv2.GaussianBlur(bg, (0, 0), 21)
    out = cv2.divide(g, bg, scale=255)
    return out


METHODS = {
    "none":                  lambda g: g,
    "hist_eq":               lambda g: cv2.equalizeHist(g),
    "clahe":                 lambda g: _clahe(g),
    "gaussian":              lambda g: cv2.GaussianBlur(g, (3, 3), 0),
    "median":                lambda g: cv2.medianBlur(g, 3),
    "bilateral":             lambda g: cv2.bilateralFilter(g, 7, 50, 50),
    "nlmeans":               lambda g: cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21),
    "unsharp":               lambda g: _unsharp(g),
    "illum_norm":            lambda g: _illum_norm(g),
    "clahe_sharpen":         lambda g: _unsharp(_clahe(g)),
    "clahe_denoise_sharpen": lambda g: _unsharp(_clahe(cv2.fastNlMeansDenoising(g, None, h=7, templateWindowSize=7, searchWindowSize=21))),
    "illum_clahe_sharpen":   lambda g: _unsharp(_clahe(_illum_norm(g))),
}


def enhance(gray, method=DEFAULT_METHOD):
    return METHODS[method](gray)


# ---------------------------------------------------------------- 3. ROI helper
def crop_roi(img, roi):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


# ---------------------------------------------------------------- 4. OCR nomor ijazah
def ocr_nomor(gray_full, method=DEFAULT_METHOD, debug=None):
    """Area nomor -> enhancement -> OCR (Tesseract, whitelist digit)."""
    roi = crop_roi(gray_full, ROI_NOMOR)
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    roi = enhance(roi, method)
    roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    if debug is not None:
        debug["nomor_roi"] = roi
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(roi, config=cfg)
    return re.sub(r"\D", "", text)


# ---------------------------------------------------------------- 5. Deteksi tanda tangan
def detect_signature(gray_full, roi=ROI_TTD_REKTOR, method="illum_norm", debug=None,
                     min_ink=0.004, min_cc_area=250, min_extent=60):
    """
    Area TTD -> (enhancement) -> thresholding (Otsu) -> morfologi -> analisis komponen.
    Tanda tangan = goresan tinta gelap, kontinu, berukuran besar (bounding box lebar).
    Latar guilloche/noise = bintik kecil, tersebar -> dibuang oleh opening + filter area.
    """
    crop = crop_roi(gray_full, roi)
    g = enhance(crop, method)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Threshold: Otsu, lalu pastikan hanya piksel benar-benar gelap (tinta) yang lolos
    otsu, binar = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_level = min(otsu, 0.6 * float(np.median(g)))        # batas ketat -> hindari pola latar
    _, binar = cv2.threshold(g, ink_level, 255, cv2.THRESH_BINARY_INV)

    # Morfologi: opening (hapus bintik) -> closing (sambung goresan putus)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clean = cv2.morphologyEx(binar, cv2.MORPH_OPEN, k_open)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    keep = np.zeros_like(clean)
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_cc_area and max(w, h) >= min_extent:
            keep[lab == i] = 255
            if best is None or a > best[4]:
                best = (x, y, w, h, a)
    ink_ratio = float(np.count_nonzero(keep)) / keep.size
    present = best is not None and ink_ratio >= min_ink

    if debug is not None:
        debug.update(ttd_crop=crop, ttd_enh=g, ttd_bin=binar, ttd_morph=clean, ttd_final=keep)
    return {
        "present": bool(present),
        "ink_ratio": round(ink_ratio, 5),
        "largest_cc_area": int(best[4]) if best else 0,
        "bbox": [int(v) for v in best[:4]] if best else None,
    }


# ---------------------------------------------------------------- 6. Pipeline utama
def verifikasi(path, method=DEFAULT_METHOD, debug_dir=None):
    dbg = {} if debug_dir else None
    img = load_and_orient(path)
    gray = to_gray(img)
    enhanced = enhance(gray, "clahe")         # enhancement global (sebelum pemisahan area)
    nomor = ocr_nomor(gray, method, dbg)      # area nomor memakai gray asli + enhancement khusus
    ttd = detect_signature(gray, debug=dbg)
    hasil = {"file": os.path.basename(path), "nomor_ijazah": nomor,
             "tanda_tangan": "PRESENT" if ttd["present"] else "ABSENT", "detail_ttd": ttd}
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "1_oriented.jpg"), img)
        cv2.imwrite(os.path.join(debug_dir, "2_gray.jpg"), gray)
        cv2.imwrite(os.path.join(debug_dir, "3_enhanced_global.jpg"), enhanced)
        for k, v in dbg.items():
            cv2.imwrite(os.path.join(debug_dir, f"{k}.png"), v)
    return hasil


def main():
    ap = argparse.ArgumentParser(description="Verifikasi nomor ijazah (OCR) & tanda tangan")
    ap.add_argument("image")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=list(METHODS))
    ap.add_argument("--debug", action="store_true", help="simpan citra tiap tahap ke output/<nama>/")
    a = ap.parse_args()
    dd = os.path.join("output", os.path.splitext(os.path.basename(a.image))[0]) if a.debug else None
    r = verifikasi(a.image, a.method, dd)
    print(f"Nomor Ijazah : {r['nomor_ijazah']}")
    print(f"Tanda Tangan : {r['tanda_tangan']}")


if __name__ == "__main__":
    sys.exit(main())
