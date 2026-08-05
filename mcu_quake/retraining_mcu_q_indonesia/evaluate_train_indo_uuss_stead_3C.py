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
MODEL_PATH = "output_models/frozen_extractor_indonesia_Z.keras"
OUTPUT_DIR = "output_eval"

INPUT_SIZE = 700
CHANNEL = "Z"
NOISE_CHANNEL = "Z_noise"

# Dictionary untuk menyimpan daftar dataset yang akan dieksekusi sekaligus
DATASETS = {
    "Indonesia": '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia/indonesia_test_data.json',
    "UUSS": '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ UUSS 3C_ test n2222 r100/UUSS 3C data, test n2222 r100.json',
    "STEAD": '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json'
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def evaluate_dataset(dataset_name, json_path, model):
    print(f"\n{'='*60}")
    print(f"🔄 MEMULAI EVALUASI UNTUK DATASET: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"1. Memuat Dataset {dataset_name}...")
    try:
        with open(json_path, "r") as f:
            data_json = json.load(f)
    except Exception as e:
        print(f"❌ Gagal memuat JSON {dataset_name}. Lewati... Error: {e}")
        return
        
    print("2. Mengekstraksi Koordinat Ruang Laten (Proyeksi 1D)...")
    embeddings_dict = {"noise": [], "le": []}
    keys = list(data_json.keys())
    
    for key in keys:
        try:
            # Ekstraksi Sinyal Gempa (Local Earthquake / le)
            sig_le = np.array(data_json[key][CHANNEL][:INPUT_SIZE])
            code_le = latent_codes_1D(sig_le, model)
            embeddings_dict["le"].append(code_le)
            
            # Ekstraksi Sinyal Derau Maritim (noise)
            sig_noise = np.array(data_json[key][NOISE_CHANNEL][-INPUT_SIZE:])
            code_noise = latent_codes_1D(sig_noise, model)
            embeddings_dict["noise"].append(code_noise)
        except KeyError as e:
            print(f"\n❌ ERROR STRUKTUR DATA: Kunci {e} tidak ditemukan pada ID '{key}'.")
            print(f"Pastikan JSON '{dataset_name}' memiliki struktur kunci '{CHANNEL}' dan '{NOISE_CHANNEL}'. Lewati dataset ini...")
            return

    # Simpan titik embedding baru secara dinamis
    embedding_path = os.path.join(OUTPUT_DIR, f"Embedding_data_{CHANNEL}_{dataset_name}.json")
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

    print(f"\n--- METRIK PERFORMANSI: {dataset_name} ---")
    cnf_matrix, metrics = calc_confusion_metrics(true_labels, pred_labels)
    
    print(f"\n5. Mencetak Matriks Kebingungan untuk {dataset_name}...")
    fig = plot_confusion(
        title=f"MCU-Quake Retrained\nDataset: {dataset_name}",
        true_labels=["NO", "LE"], # <--- LABEL DIUBAH DI SINI
        matrix=cnf_matrix, 
        metrics=metrics
    )
    
    fig_path = os.path.join(OUTPUT_DIR, f"Confusion_Matrix_{CHANNEL}_{dataset_name}.png")
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig) # SANGAT PENTING: Mencegah gambar bertumpuk saat looping dataset
    print(f"✅ Evaluasi {dataset_name} Selesai! Matriks tersimpan di: {fig_path}")

def main():
    print("MENGINISIALISASI EVALUASI MULTI-DATASET")
    # Memuat model cukup SATU KALI agar menghemat memori dan waktu komputasi
    print("Memuat Model Frozen Extractor...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    for name, path in DATASETS.items():
        evaluate_dataset(name, path, model)
        
    print("\n🎉 SEMUA EVALUASI DATASET TELAH SELESAI!")

if __name__ == "__main__":
    main()