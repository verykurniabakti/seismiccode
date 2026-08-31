# -*- coding: utf-8 -*-
"""
Ablation study: KDE Re-Embedding varian "noise-only".

Perbedaan dari evaluate_kde_re_embedding_all.py (skenario "kde_reembed"):
  - Referensi kelas GEMPA ("le")  -> tetap dari sumber luas (UUSS + STEAD),
    SAMA seperti skenario "baseline" (tidak dipersempit ke data Indonesia).
  - Referensi kelas NOISE          -> tetap di-re-embed dari data latih
    Indonesia (TRAIN_JSON), seperti skenario "kde_reembed" asli.

Tujuan: menguji hipotesis bahwa mempersempit kerapatan referensi GEMPA ke
4.127 event Indonesia (seperti pada skenario "kde_reembed" asli) adalah
penyebab anjloknya Recall pada pengujian STEAD Unseen -- bukan karena
"model statis sulit beradaptasi" secara umum.

Skenario yang dijalankan:
  - kde_reembed_noise_only              -> uji pada Indonesia blind test
  - retention_stead_unseen_noise_only   -> uji pada STEAD Unseen (jika file tersedia)

Boilerplate ekstraksi embedding, potong model ke layer laten, cek kebocoran
data, dan pembangunan KDE (dengan jitter epsilon reproducible, dan fix
numpy >=2.4 pada inferensi) sudah dipindah ke Library/kde_reembedding.py
(dipakai bersama oleh skrip ini, evaluate_kde_re_embedding.py, dan
evaluate_kde_re_embedding_all.py) -- skrip ini hanya berisi konfigurasi
path & susunan referensi campuran (le=luas, noise=Indonesia) yang spesifik
untuk studi ablasi ini.
"""

import json
import os

import numpy as np

from Library.kde_reembedding import (
    NpEncoder,
    build_embeddings,
    assert_disjoint,
    load_frozen_extractor,
    KDEReEmbedder,
)
from Library.utils import plot_confusion

# ==========================================
# KONFIGURASI PATH (identik dengan evaluate_kde_re_embedding_all.py)
# ==========================================
BASE_DATA = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia"
TRAIN_JSON = os.path.join(BASE_DATA, "indonesia_train_data.json")
TEST_JSON = os.path.join(BASE_DATA, "indonesia_test_data.json")

UUSS_JSON = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ UUSS 3C_ test n2222 r100/UUSS 3C data, test n2222 r100.json"
STEAD_JSON = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json"
STEAD_UNSEEN_JSON = '/Volumes/Extreme SSD/stream_stead/data_stead/stead_sample_5000/STEAD_5000_3C_20260719_062844.json'

BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/Code & Figure demo"
MODEL_PRETRAINED = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")

OUTPUT_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/output_eval_noise_only_ablation"
CHANNEL = "Z"
RANDOM_SEED = 42

# test_path -> nama skenario
TEST_TARGETS = {
    "kde_reembed_noise_only": TEST_JSON,
    # retention_uuss_noise_only dan retention_stead_noise_only DIHAPUS:
    # referensi "le" (luas) pada varian ini sudah memakai SELURUH data
    # UUSS+STEAD, sehingga menguji ulang pada UUSS/STEAD akan menguji
    # model dengan data yang sama persis dengan referensinya sendiri
    # (kebocoran data, terbukti trivial/leaky -- assert_disjoint menangkapnya).
    "retention_stead_unseen_noise_only": STEAD_UNSEEN_JSON,
}

SCENARIO_DISPLAY_TITLES = {
    "kde_reembed_noise_only": "KDE Re-Embedding (noise-only) - Indonesia",
    "retention_uuss_noise_only": "KDE Re-Embedding (noise-only) - UUSS Retention",
    "retention_stead_noise_only": "KDE Re-Embedding (noise-only) - STEAD Retention",
    "retention_stead_unseen_noise_only": "KDE Re-Embedding (noise-only) - STEAD Unseen",
}


def run_all():
    if not os.path.exists(MODEL_PRETRAINED):
        print(f"[FATAL] Model pretrained tidak ditemukan: {MODEL_PRETRAINED}")
        return {}

    model = load_frozen_extractor(MODEL_PRETRAINED, latent_dim=32)

    # --- Referensi GEMPA ("le"): tetap luas, dari UUSS + STEAD (sama seperti "baseline") ---
    print("\n" + "=" * 70)
    print("Membangun referensi GEMPA (luas, UUSS+STEAD) -- TIDAK dipersempit ke Indonesia")
    print("=" * 70)
    ref_u, _ = build_embeddings(UUSS_JSON, model, channel=CHANNEL, label="ref-le-uuss")
    ref_s, _ = build_embeddings(STEAD_JSON, model, channel=CHANNEL, label="ref-le-stead")
    le_ref_wide = ref_u["le"] + ref_s["le"]
    print(f"[INFO] Total vektor referensi GEMPA (luas): {len(le_ref_wide)}")

    # --- Referensi NOISE: tetap di-re-embed dari data latih Indonesia ---
    print("\n" + "=" * 70)
    print("Membangun referensi NOISE dari data latih Indonesia (re-embedding)")
    print("=" * 70)
    ref_indo, _ = build_embeddings(TRAIN_JSON, model, channel=CHANNEL, label="ref-noise-indonesia")
    noise_ref_indo = ref_indo["noise"]
    print(f"[INFO] Total vektor referensi NOISE (Indonesia): {len(noise_ref_indo)}")

    print("\n[INFO] Membangun KDE referensi campuran (le=luas, noise=Indonesia), dengan jitter epsilon...")
    ref_mixed = {"noise": noise_ref_indo, "le": le_ref_wide}
    kde = KDEReEmbedder(choose_pdf="Kernel", jitter=True, seed=RANDOM_SEED).fit(ref_mixed)

    # Cek disjoint terhadap keseluruhan kunci referensi yang dipakai (le + noise)
    ref_all_keys = {"keys": ref_u["keys"] + ref_s["keys"] + ref_indo["keys"]}

    hasil = {}
    for name, test_path in TEST_TARGETS.items():
        print("\n" + "=" * 70)
        print(f"SKENARIO: {name}  ->  uji: {test_path}")
        print("=" * 70)

        if not os.path.exists(test_path):
            print(f"[LEWATI] Berkas uji tidak ditemukan (kemungkinan drive eksternal belum terhubung): {test_path}")
            continue

        test, skipped = build_embeddings(test_path, model, channel=CHANNEL, label="uji")
        if len(test["keys"]) == 0:
            print(f"[LEWATI] Tidak ada data uji valid untuk {name}")
            continue

        assert_disjoint(ref_all_keys, test)

        print("[INFO] Inferensi pada himpunan uji...")
        eval_hasil = kde.evaluate(test)
        cnf, metrics = eval_hasil["confusion_matrix"], eval_hasil["metrics"]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.join(OUTPUT_DIR, f"{name}_{CHANNEL}")
        with open(f"{stem}_metrics.json", "w") as f:
            json.dump(
                {
                    "skenario": name,
                    "model": MODEL_PRETRAINED,
                    "ref_le": "UUSS+STEAD (luas, tidak dipersempit)",
                    "ref_noise": "Indonesia train (re-embedded)",
                    "berkas_uji": test_path,
                    "n_ref_le": len(le_ref_wide),
                    "n_ref_noise": len(noise_ref_indo),
                    "n_uji": len(test["keys"]),
                    "confusion_matrix": cnf,
                    "metrics": metrics,
                },
                f, indent=2, cls=NpEncoder
            )

        display_name = SCENARIO_DISPLAY_TITLES.get(name, name)
        fig = plot_confusion(
            title=f"{display_name} (Component {CHANNEL})\nn = {len(test['keys'])} events/class",
            true_labels=["NO", "LE"],
            matrix=cnf,
            metrics=metrics,
        )
        fig.savefig(f"{stem}_confusion.png", bbox_inches="tight", dpi=300)
        print(f"OK {name} selesai -- Tersimpan: {stem}_confusion.png")

        hasil[name] = (cnf, metrics)

    if hasil:
        print("\n" + "=" * 70)
        print("RINGKASAN (TNR / Recall / Presisi kelas LE, dalam persen)")
        print("=" * 70)
        print(f"{'skenario':34s} {'TNR':>7s} {'Recall':>8s} {'Presisi':>9s} {'FP':>7s} {'FN':>7s}")
        for s, (cnf, met) in hasil.items():
            tnr = met["True negative rate"][0] * 100
            rec = met["True positive rate"][1] * 100
            ppv = met["Positive predictive value"][1] * 100
            fp = int(cnf[0][1])
            fn = int(cnf[1][0])
            print(f"{s:34s} {tnr:7.2f} {rec:8.2f} {ppv:9.2f} {fp:7d} {fn:7d}")

    return hasil


if __name__ == "__main__":
    run_all()
