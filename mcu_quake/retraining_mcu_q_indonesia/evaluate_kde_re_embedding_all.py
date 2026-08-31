# -*- coding: utf-8 -*-
"""
Evaluasi KDE Re-Embedding dengan pemisahan latih/uji yang ketat.
Termasuk ekstraksi otomatis Embedding 32D untuk mikrokontroler.

Boilerplate ekstraksi embedding, potong model ke layer laten, cek kebocoran
data, dan pembangunan KDE (dengan jitter epsilon reproducible) sudah
dipindah ke Library/kde_reembedding.py (dipakai bersama oleh skrip ini,
evaluate_kde_re_embedding.py, dan evaluate_kde_reembed_noise_only.py) --
skrip ini hanya berisi konfigurasi path & 10 skenario yang spesifik untuknya.
"""

import glob
import json
import os
import argparse

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
# KONFIGURASI PATH
# ==========================================
BASE_DATA = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia"
TRAIN_JSON = os.path.join(BASE_DATA, "indonesia_train_data.json")
TEST_JSON = os.path.join(BASE_DATA, "indonesia_test_data.json")

UUSS_JSON = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ UUSS 3C_ test n2222 r100/UUSS 3C data, test n2222 r100.json"
STEAD_JSON = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json"
# STEAD Unseen: versi terkoreksi penuh (windowing [P-1s,P+6s]/[P-8s,P-1s],
# bandpass Butterworth 4-orde zero-phase 1-20Hz, filter SNR>=3.0dB) --
# menggantikan generator lama stead_sample_5000 yang belum difilter bandpass
# maupun SNR, dan salah windowing ([P,P+7s]).
_STEAD_UNSEEN_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/stead_unseen_self_reference"
STEAD_UNSEEN_JSON = sorted(glob.glob(os.path.join(_STEAD_UNSEEN_DIR, "STEAD_UNSEEN_bandpass_added_3C_*.json")))[-1]

BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/Code & Figure demo"
MODEL_PRETRAINED = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
MODEL_RETRAINED = "output_models/frozen_extractor_indonesia_Z.keras"

OUTPUT_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/output_eval_03"
CHANNEL = "Z"
RANDOM_SEED = 42

SCENARIOS = {
    "baseline":               (MODEL_PRETRAINED, "source",    TEST_JSON),
    "kde_reembed":            (MODEL_PRETRAINED, TRAIN_JSON,  TEST_JSON),
    "retrained":              (MODEL_RETRAINED,  TRAIN_JSON,  TEST_JSON),
    "retention_uuss":           (MODEL_PRETRAINED, TRAIN_JSON,  UUSS_JSON),
    "retention_stead":          (MODEL_PRETRAINED, TRAIN_JSON,  STEAD_JSON),
    "retention_uuss_retrain":   (MODEL_RETRAINED,  TRAIN_JSON,  UUSS_JSON),
    "retention_stead_retrain":  (MODEL_RETRAINED,  TRAIN_JSON,  STEAD_JSON),
    "baseline_stead_unseen":          (MODEL_PRETRAINED, "source",    STEAD_UNSEEN_JSON),
    "retention_stead_unseen":         (MODEL_PRETRAINED, TRAIN_JSON,  STEAD_UNSEEN_JSON),
    "retention_stead_unseen_retrain": (MODEL_RETRAINED,  TRAIN_JSON,  STEAD_UNSEEN_JSON),
}

# Judul gambar yang rapi untuk publikasi (bahasa Inggris, tanpa nama variabel
# mentah). Dipakai menggantikan `name` (key SCENARIOS) di judul plot_confusion.
SCENARIO_DISPLAY_TITLES = {
    "baseline": "Baseline (Static Model)",
    "kde_reembed": "KDE Re-Embedding",
    "retrained": "Full Retraining",
    "retention_uuss": "KDE Re-Embedding — UUSS Retention",
    "retention_stead": "KDE Re-Embedding — STEAD Retention",
    "retention_uuss_retrain": "Full Retraining — UUSS Retention",
    "retention_stead_retrain": "Full Retraining — STEAD Retention",
    "baseline_stead_unseen": "Baseline (Static Model) — STEAD Unseen",
    "retention_stead_unseen": "KDE Re-Embedding — STEAD Unseen",
    "retention_stead_unseen_retrain": "Full Retraining — STEAD Unseen",
}


# ==========================================
# EVALUASI SATU SKENARIO
# ==========================================
def run_scenario(name):
    model_path, ref_source, test_path = SCENARIOS[name]

    print("\n" + "=" * 70)
    print(f"SKENARIO: {name}")
    print("=" * 70)

    if ref_source != "source" and not os.path.exists(ref_source):
        return None, None
    if not os.path.exists(test_path):
        return None, None
    if not os.path.exists(model_path):
        return None, None

    # Potong model klasifikasi penuh agar mengembalikan vektor laten 32D
    model = load_frozen_extractor(model_path, latent_dim=32)

    if ref_source == "source":
        try:
            ref_u, _ = build_embeddings(UUSS_JSON, model, channel=CHANNEL, label="ref-uuss")
            ref_s, _ = build_embeddings(STEAD_JSON, model, channel=CHANNEL, label="ref-stead")
            ref = {
                "keys": ref_u["keys"] + ref_s["keys"],
                "noise": ref_u["noise"] + ref_s["noise"],
                "le": ref_u["le"] + ref_s["le"],
            }
        except FileNotFoundError:
            return None, None
    else:
        ref, _ = build_embeddings(ref_source, model, channel=CHANNEL, label="referensi")

    test, skipped = build_embeddings(test_path, model, channel=CHANNEL, label="uji")
    if len(ref["keys"]) == 0 or len(test["keys"]) == 0:
        return None, None

    assert_disjoint(ref, test)

    # Simpan embedding referensi "kde_reembed" ke JSON untuk firmware ESP32-S3
    if name == "kde_reembed":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        emb_out_path = os.path.join(OUTPUT_DIR, "Embedding data, Z.json")
        with open(emb_out_path, "w") as f_emb:
            json.dump({"noise": ref["noise"], "le": ref["le"]}, f_emb, cls=NpEncoder)
        print(f"[INFO] ✅ Vektor KDE berhasil disimpan ke:\n       {emb_out_path}")

    print("[INFO] Membangun KDE dari himpunan referensi (dengan jitter epsilon reproducible)...")
    kde = KDEReEmbedder(choose_pdf="Kernel", jitter=True, seed=RANDOM_SEED).fit(ref)

    print("[INFO] Inferensi pada himpunan uji...")
    hasil = kde.evaluate(test)
    cnf, metrics = hasil["confusion_matrix"], hasil["metrics"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, f"{name}_{CHANNEL}")

    n_le, n_noise = len(test["le"]), len(test["noise"])
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(
            {
                "skenario": name,
                "model": model_path,
                "sumber_referensi": ref_source,
                "berkas_uji": test_path,
                "n_referensi": len(ref["keys"]),
                "n_uji": len(test["keys"]),
                "n_le": n_le,
                "n_noise": n_noise,
                "dimensi_laten": int(np.asarray(test["le"][0]).shape[-1]),
                "confusion_matrix": cnf,
                "metrics": metrics,
            },
            f, indent=2, cls=NpEncoder
        )

    display_name = SCENARIO_DISPLAY_TITLES.get(name, name)
    fig = plot_confusion(
        title=f"{display_name} (1C)\nn_le={n_le}, n_noise={n_noise}",
        true_labels=["NO", "LE"],
        matrix=cnf,
        metrics=metrics,
    )
    fig.savefig(f"{stem}_confusion.png", bbox_inches="tight", dpi=300)
    print(f"\n✅ {name} selesai — Tersimpan: {stem}_confusion.png")

    return cnf, metrics


# ==========================================
# EKSEKUSI
# ==========================================
def run_all(skenario="all"):
    targets = list(SCENARIOS) if skenario == "all" else [skenario]
    hasil = {}
    for s in targets:
        cnf, met = run_scenario(s)
        if cnf is not None:
            hasil[s] = (cnf, met)

    if hasil:
        print("\n" + "=" * 70)
        print("RINGKASAN (TNR / Recall / Presisi kelas LE, dalam persen)")
        print("=" * 70)
        print(f"{'skenario':26s} {'TNR':>7s} {'Recall':>8s} {'Presisi':>9s} {'FP':>7s} {'FN':>7s}")
        for s, (cnf, met) in hasil.items():
            tnr = met["True negative rate"][0] * 100
            rec = met["True positive rate"][1] * 100
            ppv = met["Positive predictive value"][1] * 100
            fp = int(cnf[0][1])
            fn = int(cnf[1][0])
            print(f"{s:26s} {tnr:7.2f} {rec:8.2f} {ppv:9.2f} {fp:7d} {fn:7d}")
    return hasil


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        run_all("all")
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("skenario", nargs="?", default="all", choices=list(SCENARIOS) + ["all"])
        run_all(ap.parse_args().skenario)
