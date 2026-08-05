# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Pastikan Library/utils.py sudah menggunakan versi bersih yang kita bahas sebelumnya
from Library.utils import (
    latent_codes_1D, 
    embedding_PDFs_1D, 
    infer_1C_PDFs, 
    calc_confusion_metrics, 
    plot_confusion
)

# ==========================================
# KONFIGURASI PATH
# ==========================================
JSON_PATH = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia/extracted_data_3c_sesi_clean.json'
MODEL_PATH = "output_models/frozen_extractor_indonesia_Z.keras"
OUTPUT_DIR = "output_eval"

INPUT_SIZE = 700
CHANNEL = "Z"
NOISE_CHANNEL = "Z_noise"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    print("1. Memuat Dataset dan Model Frozen Extractor...")
    with open(JSON_PATH, "r") as f:
        data_indonesia = json.load(f)
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("2. Mengekstraksi Koordinat Ruang Laten (Proyeksi 1D)...")
    embeddings_dict = {"noise": [], "le": []}
    
    keys = list(data_indonesia.keys())
    
    for key in keys:
        # Ekstraksi Sinyal Gempa (Local Earthquake / le)
        sig_le = np.array(data_indonesia[key][CHANNEL][:INPUT_SIZE])
        code_le = latent_codes_1D(sig_le, model)
        embeddings_dict["le"].append(code_le)
        
        # Ekstraksi Sinyal Derau Maritim (noise)
        sig_noise = np.array(data_indonesia[key][NOISE_CHANNEL][-INPUT_SIZE:])
        code_noise = latent_codes_1D(sig_noise, model)
        embeddings_dict["noise"].append(code_noise)

    # Simpan titik embedding baru
    embedding_path = os.path.join(OUTPUT_DIR, f"Embedding_data_{CHANNEL}_Indonesia.json")
    with open(embedding_path, 'w') as f:
        # Konversi array Numpy ke list Python standar agar bisa di-serialize JSON
        json.dump({k: [x.tolist() for x in v] for k, v in embeddings_dict.items()}, f, indent=4)

    print("3. Menghitung Probability Density Function (KDE)...")
    # Membangun fungsi probabilitas statistik (KDE) dari ruang laten yang baru
    pdfs_1d = embedding_PDFs_1D(embeddings_dict, source_list=["noise", "le"])

    print("4. Melakukan Inferensi dan Evaluasi Klasifikasi...")
    true_labels = []
    pred_labels = []
    
    # Inferensi kelas "le" (Ground truth: 1)
    for emb in embeddings_dict["le"]:
        infer_type, _, _ = infer_1C_PDFs(emb, pdfs_1d, choose_pdf="Kernel")
        true_labels.append(1)
        pred_labels.append(infer_type)
        
    # Inferensi kelas "noise" (Ground truth: 0)
    for emb in embeddings_dict["noise"]:
        infer_type, _, _ = infer_1C_PDFs(emb, pdfs_1d, choose_pdf="Kernel")
        true_labels.append(0)
        pred_labels.append(infer_type)

    print("\n--- METRIK PERFORMANSI ---")
    cnf_matrix, metrics = calc_confusion_metrics(true_labels, pred_labels)
    
    print("\n5. Mencetak Matriks Kebingungan (Confusion Matrix)...")
    fig = plot_confusion(
        title=f"MCU-Quake Retrain ({CHANNEL}-Axis)\nDataset: Indonesia",
        true_labels=["Noise", "Local Eq."], 
        matrix=cnf_matrix, 
        metrics=metrics
    )
    
    fig_path = os.path.join(OUTPUT_DIR, f"Confusion_Matrix_Retrained_{CHANNEL}.png")
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"✅ Evaluasi Selesai! Matriks tersimpan di: {fig_path}")

if __name__ == "__main__":
    main()