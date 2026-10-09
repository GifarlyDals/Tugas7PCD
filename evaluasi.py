#!/usr/bin/env python3
"""
Evaluasi metode enhancement berdasarkan CER (Character Error Rate) + uji deteksi tanda tangan.

    python evaluasi.py                       # semua citra di input/
    python evaluasi.py --gt 571012022000056  # ganti ground-truth nomor ijazah

Keluaran: output/hasil_cer.csv, output/ringkasan_cer.csv, output/hasil_verifikasi.csv, output/grafik_cer.png
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

import verifikasi_ijazah as V

GT_DEFAULT = "571012022000056"


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    return levenshtein(pred, gt) / len(gt)


def blank_signature(gray, roi=V.ROI_TTD_REKTOR):
    """Uji negatif: hapus tanda tangan (isi ROI dengan warna latar median) -> harus ABSENT."""
    out = gray.copy()
    h, w = out.shape
    x0, x1, y0, y1 = roi
    ys, ye, xs, xe = int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w)
    reg = out[ys:ye, xs:xe]
    bg = cv2.GaussianBlur(cv2.medianBlur(reg, 31), (0, 0), 5)
    out[ys:ye, xs:xe] = bg
    return out


# ---------------------------------------------------------------- Stress test
def degrade(g, kind, seed=0):
    """Degradasi sintetis tambahan (lebih berat dari citra uji) pada citra gray penuh."""
    rng = np.random.default_rng(seed)
    f = g.astype(np.float32)
    if kind == "noise_s35":
        f = f + rng.normal(0, 35, f.shape)
    elif kind == "blur_s3":
        f = cv2.GaussianBlur(f, (0, 0), 3)
    elif kind == "kontras_0.25":
        f = (f - f.mean()) * 0.25 + f.mean()
    elif kind == "resolusi_0.4x":
        h, w = g.shape
        f = cv2.resize(cv2.resize(f, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "gabungan":
        f = cv2.GaussianBlur(f, (0, 0), 2)
        f = (f - f.mean()) * 0.4 + f.mean() * 0.8
        f = f + rng.normal(0, 20, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


STRESS_KINDS = ["noise_s35", "blur_s3", "kontras_0.25", "resolusi_0.4x", "gabungan"]


def run_stress(grays, gt, out_dir):
    rows = []
    for name, g in grays.items():
        for kind in STRESS_KINDS:
            gd = degrade(g, kind)
            for m in V.METHODS:
                pred = V.ocr_nomor(gd, m)
                rows.append({"citra": name, "degradasi": kind, "metode": m, "prediksi": pred, "cer": round(cer(pred, gt), 4)})
        print(f"[STRESS] {name} selesai")
    with open(os.path.join(out_dir, "hasil_cer_stress.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["citra", "degradasi", "metode", "prediksi", "cer"])
        w.writeheader(); w.writerows(rows)
    summ = []
    for m in V.METHODS:
        c = [r["cer"] for r in rows if r["metode"] == m]
        per = {k: round(float(np.mean([r["cer"] for r in rows if r["metode"] == m and r["degradasi"] == k])), 4) for k in STRESS_KINDS}
        summ.append({"metode": m, "cer_rata2": round(float(np.mean(c)), 4), "cer_terburuk": max(c), **per})
    summ.sort(key=lambda r: r["cer_rata2"])
    with open(os.path.join(out_dir, "ringkasan_cer_stress.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    print("\n=== STRESS TEST: CER rata-rata (9 citra x 5 degradasi) ===")
    print(f"{'metode':26s}{'rata2':>8s}" + "".join(f"{k:>15s}" for k in STRESS_KINDS))
    for r in summ:
        print(f"{r['metode']:26s}{r['cer_rata2']:8.4f}" + "".join(f"{r[k]:15.4f}" for k in STRESS_KINDS))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--output", default="output")
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--no-stress", action="store_true", help="lewati stress test")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, "*.jpg")) + glob.glob(os.path.join(a.input, "*.png")))
    methods = list(V.METHODS)

    rows, grays = [], {}
    for f in files:
        name = os.path.basename(f)
        g = V.to_gray(V.load_and_orient(f))
        grays[name] = g
        for m in methods:
            pred = V.ocr_nomor(g, m)
            rows.append({"citra": name, "metode": m, "prediksi": pred, "cer": round(cer(pred, a.gt), 4)})
        print(f"[CER] {name} selesai")

    with open(os.path.join(a.output, "hasil_cer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["citra", "metode", "prediksi", "cer"])
        w.writeheader(); w.writerows(rows)

    # Ringkasan per metode: rata-rata CER, jumlah citra CER=0, CER terburuk
    summ = []
    for m in methods:
        c = [r["cer"] for r in rows if r["metode"] == m]
        summ.append({"metode": m, "cer_rata2": round(float(np.mean(c)), 4),
                     "cer_median": round(float(np.median(c)), 4),
                     "cer_terburuk": round(float(np.max(c)), 4),
                     "citra_sempurna": f"{sum(1 for x in c if x == 0)}/{len(c)}"})
    summ.sort(key=lambda r: (r["cer_rata2"], r["cer_terburuk"]))
    with open(os.path.join(a.output, "ringkasan_cer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    print("\n=== Ranking metode enhancement (CER rata-rata, makin kecil makin baik) ===")
    print(f"{'metode':26s}{'rata2':>8s}{'median':>8s}{'terburuk':>10s}{'sempurna':>10s}")
    for r in summ:
        print(f"{r['metode']:26s}{r['cer_rata2']:8.4f}{r['cer_median']:8.4f}{r['cer_terburuk']:10.4f}{r['citra_sempurna']:>10s}")

    # Tabel citra x metode
    print("\n=== CER per citra (kolom: 5 metode terbaik) ===")
    top = [r["metode"] for r in summ[:5]]
    print(f"{'citra':36s}" + "".join(f"{t[:14]:>15s}" for t in top))
    for f in files:
        n = os.path.basename(f)
        print(f"{n:36s}" + "".join(f"{[r['cer'] for r in rows if r['citra']==n and r['metode']==t][0]:15.4f}" for t in top))

    # Verifikasi lengkap dengan metode terbaik + uji tanda tangan (positif & negatif)
    best = V.DEFAULT_METHOD
    ver = []
    for f in files:
        n = os.path.basename(f)
        g = grays[n]
        pos = V.detect_signature(g)
        neg = V.detect_signature(blank_signature(g))
        ver.append({"citra": n, "nomor_ijazah": V.ocr_nomor(g, best),
                    "ttd_asli": "PRESENT" if pos["present"] else "ABSENT",
                    "ink_asli": pos["ink_ratio"],
                    "ttd_dihapus(uji_negatif)": "PRESENT" if neg["present"] else "ABSENT",
                    "ink_dihapus": neg["ink_ratio"]})
    with open(os.path.join(a.output, "hasil_verifikasi.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ver[0])); w.writeheader(); w.writerows(ver)
    print(f"\n=== Hasil verifikasi (metode OCR: {best}) ===")
    for v in ver:
        print(f"{v['citra']:36s} nomor={v['nomor_ijazah']:18s} TTD asli={v['ttd_asli']:8s}(ink {v['ink_asli']:.4f})  "
              f"TTD dihapus={v['ttd_dihapus(uji_negatif)']:8s}(ink {v['ink_dihapus']:.4f})")
    tp = sum(v["ttd_asli"] == "PRESENT" for v in ver)
    tn = sum(v["ttd_dihapus(uji_negatif)"] == "ABSENT" for v in ver)
    print(f"\nAkurasi deteksi TTD: positif {tp}/{len(ver)}, negatif {tn}/{len(ver)}")

    if not a.no_stress:
        run_stress(grays, a.gt, a.output)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([r["metode"] for r in summ][::-1], [r["cer_rata2"] for r in summ][::-1], color="#3b6ea5")
        ax.set_xlabel("CER rata-rata (lebih kecil lebih baik)"); ax.set_title("Perbandingan metode enhancement")
        plt.tight_layout(); plt.savefig(os.path.join(a.output, "grafik_cer.png"), dpi=130)
    except Exception as e:  # matplotlib opsional
        print("grafik dilewati:", e)


if __name__ == "__main__":
    main()
