import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from Library import utils
import os
from scipy import signal

# --- KONFIGURASI ---
CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
MODEL_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Pre-trained model/MCU-Quake 5-20"

def run_peak_norm_test():
    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    emb_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # 2. Ambil Sampel Seimbang (500 Noise, 500 Earthquake)
    df = pd.read_csv(CSV_PATH)
    noise_df = df[df['trace_category'] == 'noise'].sample(500, random_state=42)
    quake_df = df[df['trace_category'].str.contains('earthquake', na=False)].sample(500, random_state=42)
    test_df = pd.concat([noise_df, quake_df])

    results = []
    print("[INFO] Mengekstraksi fitur dengan Peak Normalization (1.000 sampel)...")

    with h5py.File(HDF5_PATH, 'r') as f:
        dataset = f['data']
        for _, row in test_df.iterrows():
            t_name = row['trace_name']
            y_true = 2 if "earthquake" in str(row['trace_category']).lower() else 0
            
            try:
                # Ambil data 7 detik (700 points)
                raw = dataset[t_name][:700]
                
                # --- PRAPEMROSESAN SESUAI ARTIKEL ---
                # A. Detrending
                z_detrend = signal.detrend(raw[:, 0])
                
                # B. Peak Normalization (x / |max|)
                # Ditambah epsilon 1e-6 untuk menghindari pembagian dengan nol
                z_norm = z_detrend / (np.max(np.abs(z_detrend)) + 1e-6)
                
                # C. Extract Latent Code
                latent_val = np.mean(utils.latent_codes_1D(z_norm, emb_model))
                
                results.append({'y_true': y_true, 'latent_val': latent_val})
            except:
                continue

    res_df = pd.DataFrame(results)
    
    # 3. Analisis Distribusi Fitur
    mean_noise = res_df[res_df['y_true']==0]['latent_val'].mean()
    mean_quake = res_df[res_df['y_true']==2]['latent_val'].mean()
    
    print("\n" + "="*50)
    print("      HASIL UJI PEAK NORMALIZATION")
    print("="*50)
    print(f"Rata-rata Latent NOISE      : {mean_noise:.4f} (Target Asli: -4.58)")
    print(f"Rata-rata Latent EARTHQUAKE : {mean_quake:.4f} (Target Asli: +1.04)")
    print("-" * 50)
    
    # Cek apakah sudah ada pemisahan (Zero-crossing)
    if mean_noise < 0 < mean_quake:
        print("STATUS: SUKSES! Peak Normalization memisahkan Noise dan Quake.")
    elif mean_noise < mean_quake:
        print("STATUS: ADA PEMISAHAN, tapi mungkin perlu sedikit offset.")
    else:
        print("STATUS: MASIH TERCAMPUR. Perlu evaluasi prapemrosesan lebih lanjut.")
    print("="*50)

if __name__ == "__main__":
    run_peak_norm_test()