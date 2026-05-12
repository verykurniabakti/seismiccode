import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from Library import utils
import os

# --- KONFIGURASI ---
CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
MODEL_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Pre-trained model/MCU-Quake 5-20"

def run_calibration():
    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # 2. Ambil Sampel Seimbang (500 Noise, 500 Earthquake)
    df = pd.read_csv(CSV_PATH)
    noise_df = df[df['trace_category'] == 'noise'].sample(500)
    quake_df = df[df['trace_category'].str.contains('earthquake', na=False)].sample(500)
    test_df = pd.concat([noise_df, quake_df])

    results = []
    print("[INFO] Mengekstraksi fitur latent untuk 1.000 sampel...")

    with h5py.File(HDF5_PATH, 'r') as f:
        for _, row in test_df.iterrows():
            trace_name = row['trace_name']
            y_true = 2 if "earthquake" in row['trace_category'] else 0
            
            # Ambil & Normalisasi Input
            data = f['data'][trace_name][:700, 0] # Ambil Z-channel saja untuk hitung stats
            z_norm = (data - np.mean(data)) / (np.std(data) + 1e-6)
            
            # Extract Raw Latent Mean
            raw_val = np.mean(utils.latent_codes_1D(z_norm, embedding_model))
            results.append({'y_true': y_true, 'raw_val': raw_val})

    res_df = pd.DataFrame(results)
    
    # 3. Hitung Parameter Mapping
    curr_n = res_df[res_df['y_true']==0]['raw_val'].mean()
    curr_q = res_df[res_df['y_true']==2]['raw_val'].mean()
    
    # Target dari JSON Asli
    target_n = -4.589
    target_q = 1.049
    
    # Rumus Linear: target = (raw - offset) * multiplier
    multiplier = (target_q - target_n) / (curr_q - curr_n)
    offset = curr_n - (target_n / multiplier)

    print("\n" + "="*50)
    print("      HASIL KALIBRASI LATENT SPACE")
    print("="*50)
    print(f"Pusat Noise Saat Ini : {curr_n:.4f}  -> Target: {target_n}")
    print(f"Pusat Quake Saat Ini : {curr_q:.4f}  -> Target: {target_q}")
    print("-" * 50)
    print(f"ANGKA PENGGESER (OFFSET)   : {offset:.6f}")
    print(f"ANGKA PENGALI (MULTIPLIER) : {multiplier:.6f}")
    print("="*50)
    print("\n[RUMUS FINAL UNTUK DISERTASI]:")
    print(f"f_calibrated = (f_raw - {offset:.4f}) * {multiplier:.4f}")

if __name__ == "__main__":
    run_calibration()