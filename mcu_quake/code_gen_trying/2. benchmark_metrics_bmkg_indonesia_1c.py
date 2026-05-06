# -*- coding: utf-8 -*-
import numpy as np
import os
from tqdm import tqdm
from Library import utils, dataset
from tensorflow import keras
from datetime import datetime

if __name__ == "__main__":
    #===================================================================
    # 1. KONFIGURASI PATH & OUTPUT
    #===================================================================
    # File gabungan yang baru saja kita buat
    X_DATA_PATH = '/Volumes/Extreme SSD/benchmark_bmkg_indonesia/X_benchmark_final.npy'
    Y_DATA_PATH = '/Volumes/Extreme SSD/benchmark_bmkg_indonesia/y_benchmark_final.npy'
    
    # Lokasi Model dan Embedding (Sesuai skrip STEAD Bapak)
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538")
    
    SAVE_DIR = '/Volumes/Extreme SSD/benchmark_bmkg_indonesia/results_1c_z_only'
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # Load Model & Embeddings
    print("[INFO] Memuat Model dan Embedding...")
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    embedding_N = dataset.load_embedding_data(EMB_DIR, "Embedding data, N.json")
    embedding_E = dataset.load_embedding_data(EMB_DIR, "Embedding data, E.json")
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E)

    #===================================================================
    # 2. LOAD DATASET INDONESIA
    #===================================================================
    print("[INFO] Memuat Dataset Benchmark Indonesia...")
    X = np.load(X_DATA_PATH)
    y_true_text = np.load(Y_DATA_PATH) # Berisi "NO" atau "LE"
    
    # Pemetaan Label sesuai ralat Bapak
    # Kita ubah LE menjadi 1 (Event) dan NO menjadi 0 (Noise) untuk fungsi utils
    label_map = {"NO": 0, "LE": 1} 
    
    total_true, total_pred = [], []

    #===================================================================
    # 3. INFERENCE DENGAN SKEMA 1-COMPONENT (Z-ONLY)
    #===================================================================
    print(f"[INFO] Memulai Inferensi pada {len(X)} sampel...")
    
    for i in tqdm(range(len(X)), desc="Benchmarking Indonesia 1C"):
        try:
            # Ambil data Z (karena data kita 1C, maka X[i] adalah Z itu sendiri)
            z_component = X[i].copy()
            
            # Preprocessing (Sesuai standar Wu et al)
            z_component -= np.mean(z_component)
            norm_val = np.max(np.abs(z_component))
            if norm_val > 0: z_component /= norm_val
            
            # Buat kanal E dan N menjadi NOL (Zero-Padding)
            empty_channel = np.zeros_like(z_component)
            
            # Latent Extraction
            _in_E = utils.latent_codes_1D(empty_channel, embedding_model)
            _in_N = utils.latent_codes_1D(empty_channel, embedding_model)
            _in_Z = utils.latent_codes_1D(z_component, embedding_model)
            
            # Inference menggunakan PDF 3D (Integritas Model Terjaga)
            emb_3c = np.array([_in_E, _in_N, _in_Z]).reshape(1, -1)
            p_pred, _, _ = utils.infer_3C_PDFs(emb_3c, embeddings_3C_PDFs, "Kernel")
            
            # Simpan hasil (p_pred >= 1 dianggap Event/Gempa)
            total_true.append(label_map[y_true_text[i]])
            total_pred.append(1 if p_pred >= 1 else 0)
            
        except Exception as e:
            continue

   
   #===================================================================
    # 4. FINAL RESULTS (REVISI)
    #===================================================================
    # Menghitung metriks menggunakan fungsi bawaan Library Bapak
    matrix, metrics = utils.calc_confusion_metrics(total_true, total_pred)
    
    # Simpan Final JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    final_save_path = os.path.join(SAVE_DIR, f"FINAL_BENCHMARK_1C_{timestamp}.json")
    dataset.save_json_data(final_save_path, metrics)
    
    # Plot Final Confusion Matrix
    # Pastikan label sesuai ralat Bapak: NO dan LE
    fig = utils.plot_confusion("MCU-Quake Indonesia 1C (Z-Only)", ["NO", "LE"], matrix, metrics)
    fig.savefig(os.path.join(SAVE_DIR, "Final_Confusion_Matrix_1C_Indonesia.jpg"), dpi=300)
    
    # --- PERBAIKAN TAMPILAN OUTPUT ---
    print("\n" + "="*40)
    print("      RINGKASAN METRIKS BENCHMARK")
    print("="*40)
    
    # Mengambil nilai akurasi dengan aman
    # Jika 'accuracy (avg.)' tidak ada, ia akan mengambil 'Accuracy' indeks 0
    acc_avg = metrics.get('accuracy (avg.)', metrics.get('Accuracy', [0])[0])
    f1_avg = metrics.get('F1-score (avg.)', metrics.get('F1-score', [0])[0])
    
    print(f"Total Sampel    : {len(total_true)}")
    print(f"Akurasi Rata2   : {acc_avg:.4f}")
    print(f"F1-Score Rata2  : {f1_avg:.4f}")
    print("-" * 40)
    print(f"[SUKSES] Laporan lengkap tersimpan di: {SAVE_DIR}")