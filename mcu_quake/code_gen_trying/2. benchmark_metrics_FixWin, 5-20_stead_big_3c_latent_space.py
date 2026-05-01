# -*- coding: utf-8 -*-
import h5py
import pandas as pd
import numpy as np
import os
import gc  # Modul untuk manajemen memori
from tqdm import tqdm
from Library import utils, dataset
from tensorflow import keras
from datetime import datetime

if __name__ == "__main__":
    #===================================================================
    # 1. KONFIGURASI PATH & OUTPUT
    #===================================================================
    CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
    HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
    SAVE_DIR = '/Volumes/Extreme SSD/mcu_quake_big_stead_output_3c_latent_space'
   
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538")

    # Load Model & Embeddings
    print("[INFO] Memuat Model dan Data Embedding Reference...")
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    embedding_N = dataset.load_embedding_data(EMB_DIR, "Embedding data, N.json")
    embedding_E = dataset.load_embedding_data(EMB_DIR, "Embedding data, E.json")
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E)

    # ===================================================================
    # 2. LOAD METADATA & INISIALISASI
    # ===================================================================
    print("[INFO] Membaca metadata STEAD (1.2M lines)...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df['trace_category'].isin(['earthquake_local', 'noise'])]

    total_true, total_pred = [], []
    num_points = 700 
    checkpoint_step = 100000 

    # --- KONFIGURASI EMBEDDING ---
    all_latent_codes = [] 
    all_true_labels = []  
    limit_embedding = 100000 # Tetap dibatasi agar RAM stabil
    # --------------------------------

    # ===================================================================
    # 3. BIG DATA STREAMING INFERENCE
    # ===================================================================
    with h5py.File(HDF5_PATH, 'r') as f:
        data_group = f['data']
        
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="STEAD 1.2M 3C")):
            try:
                trace_id = row['trace_name']
                category = row['trace_category']
                wave_data = data_group[trace_id][()]
                
                # Prosedur Alignment
                if category == 'earthquake_local':
                    p_arrival = int(row['p_arrival_sample'])
                    start_idx = p_arrival - 50
                    end_idx = start_idx + num_points
                    if end_idx > 6000 or start_idx < 0: continue
                    sample_3c = wave_data[start_idx:end_idx, :]
                    true_label = 1
                else:
                    sample_3c = wave_data[:num_points, :]
                    true_label = 0
                
                # Preprocessing
                sample_3c -= np.mean(sample_3c, axis=0)
                norm_val = np.max(np.abs(sample_3c))
                if norm_val > 0: sample_3c /= norm_val
                
                # Latent Extraction
                _in_E = utils.latent_codes_1D(sample_3c[:, 0], embedding_model)
                _in_N = utils.latent_codes_1D(sample_3c[:, 1], embedding_model)
                _in_Z = utils.latent_codes_1D(sample_3c[:, 2], embedding_model)
                
                # Simpan Embedding (Menggunakan .item() agar efisien)
                if len(all_latent_codes) < limit_embedding:
                    avg_latent = (_in_E.item() + _in_N.item() + _in_Z.item()) / 3.0
                    all_latent_codes.append(avg_latent)
                    all_true_labels.append(true_label)

                # Inference
                emb_3c = np.array([_in_E, _in_N, _in_Z]).reshape(1, -1)
                p_pred, _, _ = utils.infer_3C_PDFs(emb_3c, embeddings_3C_PDFs, "Kernel")
                
                total_true.append(true_label)
                total_pred.append(1 if p_pred >= 1 else 0)

                # Checkpoint Saving & RAM Cleanup
                if (i + 1) % checkpoint_step == 0:
                    temp_matrix, temp_metrics = utils.calc_confusion_metrics(total_true, total_pred)
                    temp_metrics['latent_codes'] = all_latent_codes
                    temp_metrics['true_labels'] = all_true_labels
                    
                    dataset.save_json_data(os.path.join(SAVE_DIR, f"checkpoint_{i+1}.json"), temp_metrics)
                    
                    # --- PROSES PEMBERSIHAN MEMORI ---
                    del temp_matrix, temp_metrics # Hapus variabel sementara
                    gc.collect()                  # Paksa bersihkan RAM
                    # ---------------------------------
                    
            except Exception:
                continue

    # ===================================================================
    # 4. FINAL RESULTS
    # ===================================================================
    matrix, metrics = utils.calc_confusion_metrics(total_true, total_pred)

    metrics['latent_codes'] = all_latent_codes
    metrics['true_labels'] = all_true_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset.save_json_data(os.path.join(SAVE_DIR, f"FINAL_STEAD_1.2M_3C_{timestamp}.json"), metrics)

    print(f"\n[SUKSES] Akurasi Akhir: {metrics.get('accuracy (avg.)')}")