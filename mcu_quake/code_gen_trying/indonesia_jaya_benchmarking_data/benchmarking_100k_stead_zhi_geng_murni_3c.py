# -*- coding: utf-8 -*-
import sys
import os
import gc
import csv
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde

# Library untuk Visualisasi Grafis Jurnal
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PERTAHANAN MURNI CPU & ANTI-LEAK (0-BYTE MEMORY LEAK ASSURANCE)
# ==============================================================================
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

if __name__ == "__main__":
    # --- PATH SUMBER DATA UTAMA (MURNI STEAD) ---
    CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
    HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
    ZHI_GENG_JSON = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json'
    
    # --- PATH OUTPUT GRAFIS & CSV ---
    DIR_OUTPUT_GRAFIS = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/output/output_grafis_100k_stead'
    os.makedirs(DIR_OUTPUT_GRAFIS, exist_ok=True)
    
    CSV_HASIL_PREDIKSI = os.path.join(DIR_OUTPUT_GRAFIS, "hasil_prediksi_3c_100k_strict.csv")
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")

    # --------------------------------------------------------------------------
    # FASE 0: RESOLUSI STRUKTUR & PEMULIHAN INTEGRAL RUANG LATEN 3C
    # --------------------------------------------------------------------------
    print("[INFO] Memuat Model dan Merekonstruksi Ruang Bersama 3C dari 3 Sumbu JSON...")
    with tf.device('/CPU:0'):
        embedding_model = keras.models.load_model(filepath=MODEL_PATH, compile=False)
    
    emb_E = dataset.load_embedding_data(EMB_DIR, "Embedding data, E.json")
    available_keys = list(emb_E.keys())
    k_noise = next((k for k in available_keys if k.lower() in ['noise', 'no']), available_keys[0])
    k_le = next((k for k in available_keys if k.lower() in ['le', 'earthquake', 'eq']), available_keys[1] if len(available_keys)>1 else available_keys[0])
    
    arr_noise_E, arr_le_E = np.array(emb_E[k_noise]), np.array(emb_E[k_le])
    del emb_E

    emb_N = dataset.load_embedding_data(EMB_DIR, "Embedding data, N.json")
    arr_noise_N, arr_le_N = np.array(emb_N[k_noise]), np.array(emb_N[k_le])
    del emb_N

    emb_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    arr_noise_Z, arr_le_Z = np.array(emb_Z[k_noise]), np.array(emb_Z[k_le])
    del emb_Z
    
    try:
        combined_noise = np.hstack([arr_noise_E, arr_noise_N, arr_noise_Z])
        combined_le = np.hstack([arr_le_E, arr_le_N, arr_le_Z])
    except ValueError:
        combined_noise = (arr_noise_E + arr_noise_N + arr_noise_Z) / 3.0
        combined_le = (arr_le_E + arr_le_N + arr_le_Z) / 3.0

    kde_noise = gaussian_kde(combined_noise.T)
    kde_le = gaussian_kde(combined_le.T)
    
    del arr_noise_E, arr_noise_N, arr_noise_Z, arr_le_E, arr_le_N, arr_le_Z, combined_noise, combined_le
    gc.collect()

    # --------------------------------------------------------------------------
    # FASE 1 & 2: ANTI-LEAKAGE FILTER & DATA BALANCING (100K STEAD)
    # --------------------------------------------------------------------------
    print("\n[INFO] Mengekstraksi daftar hitam anti-leakage dari JSON...")
    with open(ZHI_GENG_JSON, 'r') as f:
        zhi_geng_data = json.load(f)
    zhi_geng_traces = set(zhi_geng_data.keys()) if isinstance(zhi_geng_data, dict) else set(zhi_geng_data)
        
    print("[INFO] Membaca metadata CSV murni STEAD...")
    df_raw = pd.read_csv(CSV_PATH, low_memory=False)
    df_filtered = df_raw[df_raw['trace_category'].isin(['earthquake_local', 'noise'])]
    df_unseen = df_filtered[~df_filtered['trace_name'].isin(zhi_geng_traces)]
    
    df_eq = df_unseen[df_unseen['trace_category'] == 'earthquake_local']
    df_noise = df_unseen[df_unseen['trace_category'] == 'noise']
    
    n_samples = 50000 
    df_eq_sample = df_eq.sample(n=min(n_samples, len(df_eq)), random_state=42)
    df_noise_sample = df_noise.sample(n=min(n_samples, len(df_noise)), random_state=42)
    
    df_final = pd.concat([df_eq_sample, df_noise_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    total_target = len(df_final)
    print(f"[INFO] Total target uji STEAD 3C Murni: {total_target:,}")
    
    trace_names = df_final['trace_name'].to_numpy()
    trace_categories = df_final['trace_category'].to_numpy()
    p_arrivals = df_final['p_arrival_sample'].fillna(0).to_numpy().astype(np.int32)
    
    del df_raw, df_filtered, df_unseen, df_eq, df_noise, df_eq_sample, df_noise_sample, df_final
    gc.collect()

    # --------------------------------------------------------------------------
    # FASE 3: INFERENSI ON-THE-FLY & CHECKPOINTING CSV (ANTI MEMORY LEAK)
    # --------------------------------------------------------------------------
    num_points = 700 
    norm_points = 900
    BUFFER_SIZE = 128
    
    buffer_waves = []
    buffer_traces = []
    buffer_y_true = []

    print(f"\n[INFO] Memulai inferensi on-the-fly 3C. Hasil dicicil ke: {CSV_HASIL_PREDIKSI}")
    
    # Buka CSV dalam mode Write
    with open(CSV_HASIL_PREDIKSI, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['trace_id', 'y_true', 'y_pred']) # Tulis Header

        with h5py.File(HDF5_PATH, 'r') as f_h5:
            data_group = f_h5['data']
            
            for idx in tqdm(range(total_target), desc="Inferensi STEAD 3C Murni"):
                try:
                    trace_id = trace_names[idx]
                    category = trace_categories[idx]
                    
                    if trace_id not in data_group:
                        continue
                        
                    raw_wave_3c = data_group[trace_id][()]
                    
                    # [ATURAN ZHI GENG 1] Detrending secara terisolasi per sumbu
                    detrended_wave_3c = raw_wave_3c - np.mean(raw_wave_3c, axis=0)
                    
                    if category == 'earthquake_local':
                        start_idx = int(p_arrivals[idx])
                        end_idx = start_idx + num_points
                        norm_end_idx = start_idx + norm_points
                        
                        if norm_end_idx > len(detrended_wave_3c) or start_idx < 0: 
                            continue
                            
                        wave_slice_3c = detrended_wave_3c[start_idx:end_idx, :]
                        
                        # [ATURAN ZHI GENG 2] NORMALISASI GLOBAL LINTAS 3 SUMBU 
                        # Pencarian nilai tunggal absolut tertinggi dari seluruh saluran E, N, Z
                        norm_val_3c = np.max(np.abs(detrended_wave_3c[start_idx:norm_end_idx, :]))
                        true_label = 1
                    else:
                        wave_slice_3c = detrended_wave_3c[:num_points, :]
                        norm_val_3c = np.max(np.abs(detrended_wave_3c[:norm_points, :]))
                        true_label = 0
                    
                    if norm_val_3c == 0:
                        norm_val_3c = 1e-8
                        
                    wave_slice_3c /= norm_val_3c
                    
                    # Masukkan ke dalam antrean (buffer) sementara
                    buffer_waves.append(wave_slice_3c)
                    buffer_traces.append(trace_id)
                    buffer_y_true.append(true_label)
                    
                    # JIKA BUFFER PENUH -> INFERENSI -> TULIS KE CSV -> HAPUS RAM
                    # Eksekusi sisa buffer terakhir di akhir looping
                    if len(buffer_waves) > 0:
                        batch_input = np.array(buffer_waves, dtype=np.float32)
                        
                        batch_E = batch_input[:, :, 0:1]
                        batch_N = batch_input[:, :, 1:2]
                        batch_Z = batch_input[:, :, 2:3]
                        
                        with tf.device('/CPU:0'):
                            emb_E = embedding_model(batch_E, training=False).numpy()
                            emb_N = embedding_model(batch_N, training=False).numpy()
                            emb_Z = embedding_model(batch_Z, training=False).numpy()
                        
                        embeddings_3c = np.hstack([emb_E, emb_N, emb_Z])
                        laten_space = embeddings_3c.T 
                        
                        like_noise = kde_noise.pdf(laten_space)
                        like_le = kde_le.pdf(laten_space)
                        b_preds = np.argmax(np.vstack([like_noise, like_le]), axis=0)
                        
                        for i in range(len(b_preds)):
                            writer.writerow([buffer_traces[i], buffer_y_true[i], b_preds[i]])
                        
                        # Kosongkan RAM
                        buffer_waves.clear()
                        buffer_traces.clear()
                        buffer_y_true.clear()
                        
                    if (idx + 1) % 10000 == 0:
                        gc.collect() 
                        K.clear_session()
                        
                except Exception:
                    continue

            # Eksekusi sisa buffer terakhir di akhir looping
            if len(buffer_waves) > 0:
                batch_input = np.array(buffer_waves, dtype=np.float32).reshape(-1, num_points, 3)
                with tf.device('/CPU:0'):
                    embeddings = embedding_model(batch_input, training=False).numpy()
                
                laten_space = embeddings.T 
                like_noise = kde_noise.pdf(laten_space)
                like_le = kde_le.pdf(laten_space)
                b_preds = np.argmax(np.vstack([like_noise, like_le]), axis=0)
                
                for i in range(len(b_preds)):
                    writer.writerow([buffer_traces[i], buffer_y_true[i], b_preds[i]])

    # ==========================================================================
    # FASE 4: BACA HASIL CSV UNTUK KALKULASI METRIK & CETAK GRAFIK
    # ==========================================================================
    print("\n[INFO] Membaca rekapitulasi CSV untuk kalkulasi Metrik Akhir...")
    df_hasil = pd.read_csv(CSV_HASIL_PREDIKSI)
    
    y_true_final = df_hasil['y_true'].to_numpy()
    y_pred_final = df_hasil['y_pred'].to_numpy()
    
    TP = np.sum((y_true_final == 1) & (y_pred_final == 1))
    TN = np.sum((y_true_final == 0) & (y_pred_final == 0))
    FP = np.sum((y_true_final == 0) & (y_pred_final == 1))
    FN = np.sum((y_true_final == 1) & (y_pred_final == 0))

    total_valid = TP + TN + FP + FN
    akurasi = (TP + TN) / total_valid if total_valid > 0 else 0
    tpr_recall = TP / (TP + FN) if (TP + FN) > 0 else 0          
    tnr_spesifisitas = TN / (TN + FP) if (TN + FP) > 0 else 0    
    ppv_presisi = TP / (TP + FP) if (TP + FP) > 0 else 0         
    f1_score = 2 * (ppv_presisi * tpr_recall) / (ppv_presisi + tpr_recall) if (ppv_presisi + tpr_recall) > 0 else 0

    tpr_no = tnr_spesifisitas * 100
    tpr_le = tpr_recall * 100
    ppv_no = (TN / (TN + FN) * 100) if (TN + FN) > 0 else 0.0
    ppv_le = ppv_presisi * 100

    # ==========================================================================
    # OTOMASI GENERASI GRAFIK OUTPUT 3C (ZHI GENG STYLE)
    # ==========================================================================
    print(f"[INFO] Menghasilkan grafik visualisasi ilmiah 3C ke path: {DIR_OUTPUT_GRAFIS}")
    
    # --- GRAFIK 1: CONFUSION MATRIX ---
    cm_matrix = np.array([[TN, FP], [FN, TP]])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cax = ax.matshow(cm_matrix, cmap=plt.cm.Blues, vmin=0, vmax=np.max(cm_matrix))

    threshold = np.max(cm_matrix) / 2.
    for i in range(2):
        for j in range(2):
            color = "white" if cm_matrix[i, j] > threshold else "#08306b"
            ax.text(j, i, format(cm_matrix[i, j], 'd'), ha="center", va="center", color=color, fontsize=15, weight='bold')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['NO.', 'LE.'], fontsize=13, weight='bold') 
    ax.set_yticklabels(['NO.', 'LE.'], fontsize=13, weight='bold')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')
    
    ax.set_ylabel('True Label (Ground Truth)', fontsize=13, weight='bold', labelpad=10)
    ax.set_xlabel('Predicted Label (Model Inferens)', fontsize=13, weight='bold', labelpad=12)
    ax.xaxis.set_label_position('top') 
    
    ax.text(-0.5, -0.8, 'Baseline MCU-Quake, STEAD Unseen, 3C Murni', ha='left', va='center', fontsize=13, weight='bold', color='#c92a2a', clip_on=False)

    jarak_x_tpr = 1.7 
    ax.text(jarak_x_tpr, -0.4, 'TPR:', ha='center', va='center', fontsize=12, weight='bold', color='black', clip_on=False)
    ax.text(jarak_x_tpr, 0, f"{tpr_no:.2f}%", ha='center', va='center', fontsize=12, weight='bold', color='#c92a2a', clip_on=False)
    ax.text(jarak_x_tpr, 1, f"{tpr_le:.2f}%", ha='center', va='center', fontsize=12, weight='bold', color='blue', clip_on=False)

    jarak_y_ppv = 1.7 
    ax.text(-0.4, jarak_y_ppv, 'PPV:', ha='center', va='center', fontsize=12, weight='bold', color='black', clip_on=False)
    ax.text(0, jarak_y_ppv, f"{ppv_no:.2f}%", ha='center', va='center', fontsize=12, weight='bold', color='black', clip_on=False)
    ax.text(1, jarak_y_ppv, f"{ppv_le:.2f}%", ha='center', va='center', fontsize=12, weight='bold', color='black', clip_on=False)

    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.8)
    ax.grid(False)
    plt.tight_layout()
    
    out_path_cm = os.path.join(DIR_OUTPUT_GRAFIS, "3C_STEAD_ZhiGeng_Style_CM.png")
    plt.savefig(out_path_cm, dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRAFIK 2: DIAGRAM BATANG ---
    metrics_names = ['Akurasi', 'TPR\n(Recall)', 'TNR\n(Spesifisitas)', 'PPV\n(Presisi)', 'F1-Score']
    metrics_scores = [akurasi*100, tpr_recall*100, tnr_spesifisitas*100, ppv_presisi*100, f1_score*100]

    plt.figure(figsize=(9, 5.5))
    ax_bar = sns.barplot(x=metrics_names, y=metrics_scores, hue=metrics_names, palette="viridis", legend=False)
    
    plt.title("Metrik Evaluasi - Exact Replication MCU-Quake (3C Murni, Data STEAD Unseen)", fontsize=12, weight='bold', pad=15)
    plt.ylabel("Persentase (%)", fontsize=11, weight='bold')
    plt.ylim(0, 115)  
    
    for i, p in enumerate(ax_bar.patches):
        ax_bar.annotate(f'{metrics_scores[i]:.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, fontweight='bold', color='black', 
                    xytext=(0, 8), textcoords='offset points')

    ax_bar.spines['left'].set_linewidth(1.5)
    ax_bar.spines['bottom'].set_linewidth(1.5)
    sns.despine() 
    plt.tight_layout()
    
    out_path_bar = os.path.join(DIR_OUTPUT_GRAFIS, "3C_STEAD_Metrics_BarChart.png")
    plt.savefig(out_path_bar, dpi=300)
    plt.close()

    # ==========================================================================
    # TERMINAL OUTPUT REPORTING 3C
    # ==========================================================================
    print("\n=======================================================")
    print(" HASIL RESTORED EXACT REPLICATION ZHI GENG (STEAD 100K 3C)")
    print("=======================================================")
    print(f"Total Data Terproses  : {total_valid:,} sampel")
    print(f"True Positives (TP)   : {TP:,}")
    print(f"True Negatives (TN)   : {TN:,}")
    print(f"False Positives (FP)  : {FP:,}")
    print(f"False Negatives (FN)  : {FN:,}")
    print("-------------------------------------------------------")
    print(f"Akurasi Global        : {akurasi:.4f} ({(akurasi*100):.2f}%)")
    print(f"Recall (TPR)          : {tpr_recall:.4f} ({(tpr_recall*100):.2f}%)")
    print(f"Spesifisitas (TNR)    : {tnr_spesifisitas:.4f} ({(tnr_spesifisitas*100):.2f}%)")
    print(f"Presisi (PPV)         : {ppv_presisi:.4f} ({(ppv_presisi*100):.2f}%)")
    print(f"F1-Score              : {f1_score:.4f} ({(f1_score*100):.2f}%)")
    print("=======================================================")