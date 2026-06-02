# -*- coding: utf-8 -*-
import sys
import os
import gc
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.stats import gaussian_kde 

# ==============================================================================
# PERTAHANAN MURNI CPU & ANTI-LEAK
# ==============================================================================
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') 
from tensorflow import keras
from tensorflow.keras import backend as K

# --- PATH UTAMA ---
BASE_REP = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Code & Figure demo'
if BASE_REP not in sys.path:
    sys.path.append(BASE_REP)

from Library import utils, dataset

def bandpass_filter_1c(data, lowcut=5.0, highcut=20.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

if __name__ == "__main__":
    CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
    HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
    
    # PATH JSON ZHI GENG (Blacklist)
    ZHI_GENG_JSON = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json'
    
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")

    # --------------------------------------------------------------------------
    # FASE 0: LOAD MODEL & BANGUN KDE MANDIRI
    # --------------------------------------------------------------------------
    print("[INFO] Memuat Model dan Membangun Kurva PDF 1D Mandiri...")
    with tf.device('/CPU:0'):
        embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    
    available_keys = list(embedding_Z.keys())
    k_noise = next((k for k in available_keys if k.lower() in ['noise', 'no']), available_keys[0])
    k_le = next((k for k in available_keys if k.lower() in ['le', 'earthquake', 'eq']), available_keys[1] if len(available_keys)>1 else available_keys[0])
    k_qb = next((k for k in available_keys if k.lower() in ['qb', 'quarry']), None)

    kde_noise = gaussian_kde(np.array(embedding_Z[k_noise]).T)
    kde_le = gaussian_kde(np.array(embedding_Z[k_le]).T)
    kde_qb = gaussian_kde(np.array(embedding_Z[k_qb]).T) if k_qb else None
    
    # --------------------------------------------------------------------------
    # FASE 1: EKSTRAKSI DAFTAR HITAM (ANTI-JOIN)
    # --------------------------------------------------------------------------
    print("\n[INFO] Membaca rekam jejak Zhi Geng dari JSON...")
    with open(ZHI_GENG_JSON, 'r') as f:
        zhi_geng_data = json.load(f)
    
    if isinstance(zhi_geng_data, dict):
        zhi_geng_traces = set(zhi_geng_data.keys())
    else:
        zhi_geng_traces = set(zhi_geng_data)
        
    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    df_filtered = df_raw[df_raw['trace_category'].isin(['earthquake_local', 'noise'])]
    df_unseen = df_filtered[~df_filtered['trace_name'].isin(zhi_geng_traces)]
    print(f"[INFO] Tersisa {len(df_unseen):,} data STEAD yang 100% UNSEEN!")

    # --------------------------------------------------------------------------
    # FASE 2: RANDOM SAMPLING 100.000 DATA SEIMBANG
    # --------------------------------------------------------------------------
    df_eq = df_unseen[df_unseen['trace_category'] == 'earthquake_local']
    df_noise = df_unseen[df_unseen['trace_category'] == 'noise']
    
    n_samples = 50000 # TARGET BARU: 50.000 per kelas
    df_eq_sample = df_eq.sample(n=min(n_samples, len(df_eq)), random_state=42)
    df_noise_sample = df_noise.sample(n=min(n_samples, len(df_noise)), random_state=42)
    
    df = pd.concat([df_eq_sample, df_noise_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Total Data Uji Acak: {len(df):,} (Gempa: {len(df_eq_sample):,}, Noise: {len(df_noise_sample):,})")
    
    # --------------------------------------------------------------------------
    # FASE 3: INFERENSI TERMINAL (0-BYTE MEMORY)
    # --------------------------------------------------------------------------
    num_points = 700 
    BUFFER_SIZE = 128
    buffer_waves = []
    buffer_labels = []

    TP, TN, FP, FN = 0, 0, 0, 0

    print(f"\n🚀 MEMULAI EKSEKUSI UNSEEN DATA (1C) UNTUK {len(df):,} SAMPEL...")
    with h5py.File(HDF5_PATH, 'r') as f_h5:
        data_group = f_h5['data']
        
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="STEAD Unseen 100K")):
            try:
                trace_id = row['trace_name']
                category = row['trace_category']
                
                if trace_id not in data_group:
                    continue
                    
                wave_data = data_group[trace_id][()]
                
                if category == 'earthquake_local':
                    p_arrival = int(row['p_arrival_sample'])
                    start_idx = p_arrival - 50
                    end_idx = start_idx + num_points
                    if end_idx > 6000 or start_idx < 0: continue
                    z_component = wave_data[start_idx:end_idx, 2] 
                    true_label = 1
                else:
                    z_component = wave_data[:num_points, 2] 
                    true_label = 0
                
                z_component = bandpass_filter_1c(z_component)
                z_component -= np.mean(z_component)
                norm_val = np.max(np.abs(z_component))
                if norm_val > 0: 
                    z_component /= norm_val
                
                buffer_waves.append(z_component)
                buffer_labels.append(true_label)
                
                if len(buffer_waves) == BUFFER_SIZE:
                    batch_input = np.array(buffer_waves).astype(np.float32).reshape(-1, num_points, 1)
                    
                    with tf.device('/CPU:0'):
                        embeddings = embedding_model.predict_on_batch(batch_input)
                    
                    emb_T = embeddings.T 
                    like_noise = kde_noise.pdf(emb_T)
                    like_le = kde_le.pdf(emb_T)
                    like_qb = kde_qb.pdf(emb_T) if kde_qb else np.zeros_like(like_noise)
                    
                    likelihoods = np.vstack([like_noise, like_qb, like_le])
                    preds = np.argmax(likelihoods, axis=0) 
                    preds[preds > 0] = 1 
                    
                    for t_lbl, p_lbl in zip(buffer_labels, preds):
                        if t_lbl == 1 and p_lbl == 1: TP += 1
                        elif t_lbl == 0 and p_lbl == 0: TN += 1
                        elif t_lbl == 0 and p_lbl == 1: FP += 1
                        elif t_lbl == 1 and p_lbl == 0: FN += 1
                    
                    buffer_waves.clear()
                    buffer_labels.clear()
                    del batch_input, embeddings, emb_T, likelihoods, preds
                    
                # Hapus cache setiap 5.000 iterasi agar terminal tidak terlalu penuh
                if (i + 1) % 5000 == 0:
                    print(f"\n[BACKUP {i+1}] TP:{TP} | TN:{TN} | FP:{FP} | FN:{FN}")
                    gc.collect() 
                    K.clear_session()
                    
            except Exception as e:
                continue

        # Sisa Buffer
        if len(buffer_waves) > 0:
            batch_input = np.array(buffer_waves).astype(np.float32).reshape(-1, num_points, 1)
            with tf.device('/CPU:0'):
                embeddings = embedding_model.predict_on_batch(batch_input)
            
            emb_T = embeddings.T 
            like_noise = kde_noise.pdf(emb_T)
            like_le = kde_le.pdf(emb_T)
            like_qb = kde_qb.pdf(emb_T) if kde_qb else np.zeros_like(like_noise)
            
            likelihoods = np.vstack([like_noise, like_qb, like_le])
            preds = np.argmax(likelihoods, axis=0) 
            preds[preds > 0] = 1 
            
            for t_lbl, p_lbl in zip(buffer_labels, preds):
                if t_lbl == 1 and p_lbl == 1: TP += 1
                elif t_lbl == 0 and p_lbl == 0: TN += 1
                elif t_lbl == 0 and p_lbl == 1: FP += 1
                elif t_lbl == 1 and p_lbl == 0: FN += 1

    # ==========================================================================
    # KALKULASI AKHIR MATEMATIS
    # ==========================================================================
    total_data = TP + TN + FP + FN
    akurasi = (TP + TN) / total_data if total_data > 0 else 0
    recall_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    spesifisitas_tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    presisi_ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (presisi_ppv * recall_tpr) / (presisi_ppv + recall_tpr) if (presisi_ppv + recall_tpr) > 0 else 0

    print("\n=======================================================")
    print(" HASIL EVALUASI ZERO DATA LEAKAGE (STEAD UNSEEN 100K 1C)")
    print("=======================================================")
    print(f"Total Data Terditeksi : {total_data:,} kejadian")
    print(f"True Positives (TP)   : {TP:,}")
    print(f"True Negatives (TN)   : {TN:,}")
    print(f"False Positives (FP)  : {FP:,}")
    print(f"False Negatives (FN)  : {FN:,}")
    print("-------------------------------------------------------")
    print(f"Akurasi Global        : {akurasi:.4f} ({(akurasi*100):.2f}%)")
    print(f"Recall (TPR)          : {recall_tpr:.4f} ({(recall_tpr*100):.2f}%)")
    print(f"Spesifisitas (TNR)    : {spesifisitas_tnr:.4f} ({(spesifisitas_tnr*100):.2f}%)")
    print(f"Presisi (PPV)         : {presisi_ppv:.4f} ({(presisi_ppv*100):.2f}%)")
    print(f"F1-Score              : {f1_score:.4f} ({(f1_score*100):.2f}%)")
    print("=======================================================")