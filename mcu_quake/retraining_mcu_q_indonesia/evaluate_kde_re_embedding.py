# -*- coding: utf-8 -*-
"""
Evaluasi KDE Re-Embedding dengan pemisahan latih/uji yang ketat.

Aturan yang dijaga skrip ini:
  - KDE HANYA dibangun dari kejadian pada berkas latih.
  - Inferensi HANYA dijalankan pada kejadian di berkas uji.
  - Tidak ada satu pun event_id yang muncul di kedua sisi (diverifikasi otomatis).

Versi ringkas/contoh dasar. Boilerplate ekstraksi embedding, cek kebocoran
data, dan pembangunan KDE sudah dipindah ke Library/kde_reembedding.py
(dipakai bersama oleh skrip ini, evaluate_kde_re_embedding_all.py, dan
evaluate_kde_reembed_noise_only.py) -- skrip ini hanya berisi konfigurasi
path & susunan skenario yang spesifik untuknya.

Catatan: berkas ini sebelumnya tersimpan dalam format notebook (JSON)
berekstensi .py yang tidak bisa dijalankan langsung dari command line;
sekarang sudah jadi berkas Python biasa.
"""

import argparse
import os

import tensorflow as tf

from Library.kde_reembedding import (
    build_embeddings,
    assert_disjoint,
    KDEReEmbedder,
)
from Library.utils import calc_confusion_metrics, plot_confusion

# ==========================================
# KONFIGURASI
# ==========================================
BASE_DATA = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia"
TRAIN_JSON = os.path.join(BASE_DATA, "indonesia_train_data.json")
TEST_JSON = os.path.join(BASE_DATA, "indonesia_test_data.json")

# Himpunan domain sumber (opsional, untuk skenario baseline & uji retensi)
UUSS_JSON = "/path/ke/uuss_data.json"
STEAD_JSON = "/path/ke/stead_data.json"

MODEL_PRETRAINED = "/Volumes/Local Disk/.../Pre-trained model/MCU-Quake 5-20"
MODEL_RETRAINED = "output_models/frozen_extractor_indonesia_Z.keras"

OUTPUT_DIR = "output_eval"
CHANNEL = "Z"

SCENARIOS = {
    # nama            model              sumber KDE     himpunan uji
    "baseline":       (MODEL_PRETRAINED, "source",      TEST_JSON),
    "kde_reembed":    (MODEL_PRETRAINED, TRAIN_JSON,    TEST_JSON),
    "retrained":      (MODEL_RETRAINED,  TRAIN_JSON,    TEST_JSON),
    "retensi_uuss":   (MODEL_PRETRAINED, TRAIN_JSON,    UUSS_JSON),
    "retensi_stead":  (MODEL_PRETRAINED, TRAIN_JSON,    STEAD_JSON),
}


# ==========================================
# EVALUASI SATU SKENARIO
# ==========================================
def run_scenario(name):
    model_path, ref_source, test_path = SCENARIOS[name]

    print("=" * 70)
    print(f"SKENARIO: {name}")
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)

    if ref_source == "source":
        # Baseline: kerapatan referensi dibangun dari domain sumber, bukan Indonesia
        ref_u, _ = build_embeddings(UUSS_JSON, model, channel=CHANNEL, label="ref-uuss")
        ref_s, _ = build_embeddings(STEAD_JSON, model, channel=CHANNEL, label="ref-stead")
        ref = {
            "keys": ref_u["keys"] + ref_s["keys"],
            "noise": ref_u["noise"] + ref_s["noise"],
            "le": ref_u["le"] + ref_s["le"],
        }
    else:
        ref, _ = build_embeddings(ref_source, model, channel=CHANNEL, label="referensi")

    test, skipped = build_embeddings(test_path, model, channel=CHANNEL, label="uji")
    assert_disjoint(ref, test)

    print("[INFO] Membangun KDE dari himpunan referensi...")
    kde = KDEReEmbedder(choose_pdf="Kernel").fit(ref)

    print("[INFO] Inferensi pada himpunan uji...")
    hasil = kde.evaluate(test)
    cnf, metrics = hasil["confusion_matrix"], hasil["metrics"]

    print("\n--- METRIK ---")
    calc_confusion_metrics(hasil["true_labels"], hasil["pred_labels"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, f"{name}_{CHANNEL}")

    import json
    from Library.kde_reembedding import NpEncoder
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(
            {
                "skenario": name,
                "model": model_path,
                "sumber_referensi": ref_source,
                "berkas_uji": test_path,
                "n_referensi": len(ref["keys"]),
                "n_uji": len(test["keys"]),
                "n_dilewati": len(skipped),
                "confusion_matrix": cnf,
                "metrics": metrics,
            },
            f, indent=2, cls=NpEncoder,
        )

    fig = plot_confusion(
        title=f"{name} ({CHANNEL}-Axis)\nUji: {len(test['keys'])} kejadian/kelas",
        true_labels=["Noise", "Local Eq."],
        matrix=cnf,
        metrics=metrics,
    )
    fig.savefig(f"{stem}_confusion.png", bbox_inches="tight", dpi=300)
    print(f"\n✅ {name} selesai — {stem}_confusion.png")

    return cnf, metrics


# ==========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("skenario", nargs="?", default="all", choices=list(SCENARIOS) + ["all"])
    args = ap.parse_args()

    targets = list(SCENARIOS) if args.skenario == "all" else [args.skenario]
    for s in targets:
        run_scenario(s)
