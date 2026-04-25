import sys
import os
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import json
from tqdm import tqdm

# 1. SETUP PATH LIBRARY
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Library import utils

# ===================================================================
# KONFIGURASI PATH DATA, MODEL, & OUTPUT
# ===================================================================
CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'

OUTPUT_DIR = '/Volumes/Extreme SSD/mcu_quake_big_stead_output_normalisasi'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint_results_stead_full.csv')

BASE_LIB = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code'
MODEL_PATH = os.path.join(BASE_LIB, "Pre-trained model", "MCU-Quake 5-20")
EMBEDDING_DIR = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/code_gen/Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538'

# Parameter Teknis
INPUT_WIN = 7
SAMPLING_RATE = 100
num_points = int(INPUT_WIN * SAMPLING_RATE) # 700
BATCH_REFRESH = 100000 

def load_embedding_json(path, file_name):
    full_path = os.path.join(path, file_name)
    with open(full_path, 'r') as f:
        data = json.load(f)
    return data

def run_benchmark_with_checkpoint():
    # --- A. LOAD METADATA ---
    print(f"[INFO] Memuat metadata CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # --- B. LOAD DATA TRAINING & BUILD PDF (KDE) ---
    print(f"[INFO] Memuat embedding training dari JSON...")
    train_Z = load_embedding_json(EMBEDDING_DIR, "Embedding data, Z.json")
    train_N = load_embedding_json(EMBEDDING_DIR, "Embedding data, N.json")
    train_E = load_embedding_json(EMBEDDING_DIR, "Embedding data, E.json")

    print(f"[INFO] Membangun objek PDF/KDE Referensi (3-Channel)...")
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(train_Z, train_N, train_E)

    # --- C. LOGIKA RESUME ---
    start_index = 0
    if os.path.exists(CHECKPOINT_PATH):
        df_checkpoint = pd.read_csv(CHECKPOINT_PATH)
        start_index = len(df_checkpoint)
        print(f"[RESUME] Melanjutkan dari index {start_index}...")
    else:
        pd.DataFrame(columns=['trace_name', 'y_true', 'y_pred']).to_csv(CHECKPOINT_PATH, index=False)

    # --- D. LOAD MODEL ---
    print(f"[INFO] Memuat model MCU-Quake...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    results_buffer = []

    # --- E. PROSES UTAMA ---
    with h5py.File(HDF5_PATH, 'r') as f:
        dataset = f['data']
        
        for i in tqdm(range(start_index, len(df)), desc="Overall Progress"):
            row = df.iloc[i]
            trace_name = row['trace_name']
            
            try:
                # 1. Ambil data mentah (7 detik pertama)
                data_hdf5 = dataset[trace_name]
                z_data = data_hdf5[:num_points, 0]
                n_data = data_hdf5[:num_points, 1]
                e_data = data_hdf5[:num_points, 2]

                # 2. NORMALISASI INPUT (Z-Score) - KUNCI REPLIKASI
                z_norm = (z_data - np.mean(z_data)) / (np.std(z_data) + 1e-6)
                n_norm = (n_data - np.mean(n_data)) / (np.std(n_data) + 1e-6)
                e_norm = (e_data - np.mean(e_data)) / (np.std(e_data) + 1e-6)

                # 3. FEATURE EXTRACTION (32D)
                z_latent = utils.latent_codes_1D(z_norm, embedding_model)
                n_latent = utils.latent_codes_1D(n_norm, embedding_model)
                e_latent = utils.latent_codes_1D(e_norm, embedding_model)

                # 4. REDUKSI DIMENSI (Averaging)
                _feat_z = np.mean(z_latent)
                _feat_n = np.mean(n_latent)
                _feat_e = np.mean(e_latent)

                # 5. GABUNGKAN KE 3D (E, N, Z)
                input_embeddings_3C = np.array([[_feat_e, _feat_n, _feat_z]])

                # 6. KLASIFIKASI DENGAN KDE
                y_pred, _, _ = utils.infer_3C_PDFs(
                    input_embeddings_3C, 
                    embeddings_3C_PDFs, 
                    choose_pdf="Kernel"
                )

                # Mapping label asli untuk evaluasi
                y_true = 2 if 'earthquake' in str(row['trace_category']).lower() else 0

                results_buffer.append({'trace_name': trace_name, 'y_true': y_true, 'y_pred': y_pred})

                # --- BATCH SAVE & REFRESH ---
                if (i + 1) % BATCH_REFRESH == 0 or (i + 1) == len(df):
                    pd.DataFrame(results_buffer).to_csv(CHECKPOINT_PATH, mode='a', header=False, index=False)
                    results_buffer = [] 
                    gc.collect()
                    tf.keras.backend.clear_session()
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

            except Exception as e:
                with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as err_file:
                    err_file.write(f"Error pada index {i} ({trace_name}): {str(e)}\n")
                continue

    print(f"\n[SELESAI] Hasil akhir: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    run_benchmark_with_checkpoint()