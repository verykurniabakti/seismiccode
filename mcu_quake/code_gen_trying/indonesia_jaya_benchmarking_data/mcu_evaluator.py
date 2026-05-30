# -*- coding: utf-8 -*-
import os, sys, h5py, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# 1. KONFIGURASI PATH
# ==============================================================================
BASE_PATH = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code'
CONFIG_DIR = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Code & Figure demo'
MODEL_PATH = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Code & Figure demo/Pre-trained model/MCU-Quake 5-20'
EMB_DIR = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mcquake_ori_file/Code & Figure demo/Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909'
HDF5_PATH = '/Volumes/Local Disk/Code_Git/S3_code/seismic/waveform_indonesia_usgs_bmkg_katalog/hdf5_output/dataset_indonesia.hdf5'

# Setup GPU (Dilakukan di awal untuk menghindari RuntimeError)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# Import Modul Lokal
sys.path.append(BASE_PATH)
sys.path.append(CONFIG_DIR)
from Library import utils, dataset

# Load Model (Global)
model = keras.models.load_model(MODEL_PATH, compile=False)
emb_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
pdfs_Z = utils.embedding_PDFs_1D(emb_Z)

# ==============================================================================
# 2. FUNGSI KERJA & VISUALISASI
# ==============================================================================
def process_single_event(dataset_path):
    with h5py.File(HDF5_PATH, 'r') as hf:
        data = hf[dataset_path][:].astype(np.float32).flatten()
        
    triggers = []
    step, win = 100, 700
    for i in range(0, len(data) - win + 1, step):
        window = data[i:i+win]
        norm_w = window / (np.max(np.abs(window)) + 1e-6)
        emb = utils.latent_codes_1D(norm_w.reshape(1, -1), model)
        if utils.infer_1C_PDFs(emb, pdfs_Z, "Kernel")[0] == 0:
            triggers.append(i / 100.0)
            
    is_tp = any(55.0 <= t <= 75.0 for t in triggers)
    
    # Visualisasi otomatis untuk 0.1% sampel data agar disertasi Anda memiliki bukti visual
    if is_tp and np.random.rand() < 0.001: 
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(data))/100.0, data, 'k', alpha=0.5, label='Seismic Signal')
        for t in triggers: plt.axvline(t, color='red', linestyle='--', alpha=0.3)
        plt.axvspan(55.0, 75.0, color='yellow', alpha=0.2, label='Target Window')
        plt.title(f"Visualisasi Deteksi: {dataset_path}")
        plt.legend()
        plt.savefig(f"plot_sample_{dataset_path.replace('/','_')}.png")
        plt.close()
        
    return {'true': 1, 'pred': (1 if is_tp else 0)}

# ==============================================================================
# 3. PROSES UTAMA
# ==============================================================================
def run_evaluation():
    all_datasets = []
    with h5py.File(HDF5_PATH, 'r') as hf:
        hf.visititems(lambda name, obj: all_datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
    
    print(f"🚀 Memproses {len(all_datasets)} dataset...")
    
    results = []
    # Menggunakan ThreadPool untuk efisiensi RAM di Mac M3 Pro
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_single_event, all_datasets), total=len(all_datasets)):
            results.append(res)
            
    y_true = [r['true'] for r in results]
    y_pred = [r['pred'] for r in results]
    
    # Laporan Akhir
    print("\n=== LAPORAN EVALUASI ===")
    print(classification_report(y_true, y_pred))
    
    # Plot Matriks Konfusi
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("MCU-Quake 1C-Z Continuous Evaluation")
    plt.savefig('final_evaluation_report.png', dpi=300)
    print("✅ Selesai. Laporan dan plot tersimpan.")

if __name__ == "__main__":
    run_evaluation()