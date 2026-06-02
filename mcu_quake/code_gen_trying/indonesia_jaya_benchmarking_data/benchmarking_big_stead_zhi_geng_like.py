# -*- coding: utf-8 -*-
import sys
import os
import gc
import traceback
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.signal import butter, filtfilt

# ==============================================================================
# 1. PERTAHANAN ANTI-LEAK UNTUK MACBOOK M3 PRO
# ==============================================================================
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') 

from tensorflow import keras
from tensorflow.keras import backend as K
# ==============================================================================

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
    SAVE_DIR = '/Volumes/Extreme SSD/stream_stead/output'
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # --------------------------------------------------------------------------
    # FITUR BARU: LOG SSD UNTUK MENCEGAH RAM JEBOL
    # --------------------------------------------------------------------------
    TIMESTAMP_START = datetime.now().strftime("%Y%m%d_%H%M")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, f"safe_predictions_log_{TIMESTAMP_START}.csv")
    
    # Buat file CSV kosong dan tulis header-nya
    with open(LOG_CSV_PATH, 'w') as f:
        f.write("true_label,pred_label\n")
    # --------------------------------------------------------------------------

    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")

    print("[INFO] Memuat Model dan Kurva PDF 1D (MODE CPU MURNI)...")
    try:
        with tf.device('/CPU:0'):
            embedding_model = keras.models.load_model(filepath=MODEL_PATH)
        embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
        embeddings_Z_PDFs = utils.embedding_PDFs_1D(embedding_Z)
    except Exception as e:
        print(f"❌ Gagal memuat Model/JSON: {e}")
        sys.exit(1)

    print("[INFO] Membaca metadata STEAD...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df['trace_category'].isin(['earthquake_local', 'noise'])]
    
    num_points = 700 
    BUFFER_SIZE = 256 
    
    buffer_waves = []
    buffer_labels = []

    print("\n🚀 MEMULAI EKSEKUSI STREAM-TO-DISK DENGAN CPU...")
    with h5py.File(HDF5_PATH, 'r') as f_h5:
        data_group = f_h5['data']
        
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="STEAD 1.2M 1C")):
            try:
                trace_id = row['trace_name']
                category = row['trace_category']
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
                
                z_component = bandpass_filter_1c(z_component, lowcut=5.0, highcut=20.0, fs=100.0)
                z_component -= np.mean(z_component)
                norm_val = np.max(np.abs(z_component))
                if norm_val > 0: 
                    z_component /= norm_val
                
                buffer_waves.append(z_component)
                buffer_labels.append(true_label)
                
                # JIKA BUFFER PENUH, EKSEKUSI SEKALIGUS
                if len(buffer_waves) == BUFFER_SIZE:
                    batch_input = np.array(buffer_waves).astype(np.float32).reshape(-1, num_points, 1)
                    
                    with tf.device('/CPU:0'):
                        embeddings = embedding_model.predict_on_batch(batch_input)
                    
                    emb_T = embeddings.T 
                    like_noise = embeddings_Z_PDFs["noise"].pdf(emb_T)
                    like_le = embeddings_Z_PDFs["le"].pdf(emb_T)
                    like_qb = embeddings_Z_PDFs["qb"].pdf(emb_T) if "qb" in embeddings_Z_PDFs else np.zeros_like(like_noise)
                    
                    likelihoods = np.vstack([like_noise, like_qb, like_le])
                    preds = np.argmax(likelihoods, axis=0) 
                    preds[preds > 0] = 1 
                    
                    # ----------------------------------------------------------
                    # TULIS LANGSUNG KE SSD (ANTI RAM JEBOL)
                    # ----------------------------------------------------------
                    with open(LOG_CSV_PATH, 'a') as f_out:
                        for t_lbl, p_lbl in zip(buffer_labels, preds):
                            f_out.write(f"{t_lbl},{p_lbl}\n")
                    # ----------------------------------------------------------
                    
                    buffer_waves.clear()
                    buffer_labels.clear()
                    del batch_input, embeddings, emb_T, likelihoods, preds
                    
                # Garbage Collection Berkala (tanpa perlu menyimpan json sementara lagi karena CSV sudah aman)
                if (i + 1) % 100000 == 0:
                    gc.collect() 
                    K.clear_session()
                    
            except Exception:
                continue

        # Eksekusi sisa buffer di akhir
        if len(buffer_waves) > 0:
            batch_input = np.array(buffer_waves).astype(np.float32).reshape(-1, num_points, 1)
            with tf.device('/CPU:0'):
                embeddings = embedding_model.predict_on_batch(batch_input)
            emb_T = embeddings.T 
            like_noise = embeddings_Z_PDFs["noise"].pdf(emb_T)
            like_le = embeddings_Z_PDFs["le"].pdf(emb_T)
            like_qb = embeddings_Z_PDFs["qb"].pdf(emb_T) if "qb" in embeddings_Z_PDFs else np.zeros_like(like_noise)
            likelihoods = np.vstack([like_noise, like_qb, like_le])
            preds = np.argmax(likelihoods, axis=0) 
            preds[preds > 0] = 1 
            
            with open(LOG_CSV_PATH, 'a') as f_out:
                for t_lbl, p_lbl in zip(buffer_labels, preds):
                    f_out.write(f"{t_lbl},{p_lbl}\n")

    # ==========================================================================
    # TAHAP AKHIR: MEMBACA DARI SSD LALU MENGHITUNG METRIK
    # ==========================================================================
    print("\n[INFO] Mengumpulkan data dari SSD untuk metrik akhir...")
    df_results = pd.read_csv(LOG_CSV_PATH)
    
    # Kalkulasi matriks menggunakan library bawaan
    matrix, metrics = utils.calc_confusion_metrics(df_results['true_label'].tolist(), df_results['pred_label'].tolist())
    
    dataset.save_json_data(os.path.join(SAVE_DIR, f"FINAL_STEAD_1.2M_1C_{TIMESTAMP_START}.json"), metrics)
    fig = utils.plot_confusion("MCU_5-20 STEAD 1.2M 1C (Z-Only)", ["NO", "LE"], matrix, metrics)
    fig.savefig(os.path.join(SAVE_DIR, f"Final_Confusion_Matrix_1C_{TIMESTAMP_START}.jpg"), dpi=300)
    
    print(f"\n[SUKSES] Evaluasi Big Data STEAD 1C Selesai Tanpa Kebocoran Memori!")
    print(f"Akurasi Akhir: {metrics.get('Accuracy (avg.)')}")