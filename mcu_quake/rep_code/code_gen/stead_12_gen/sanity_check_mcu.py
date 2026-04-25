import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from Library import utils
import os

# --- CONFIG ---
HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
MODEL_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Pre-trained model/MCU-Quake 5-20"

def run_sanity_check():
    # 1. Load Model
    print("[1] Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # 2. Ambil Sampel Data (50 Noise, 50 Earthquake)
    print("[2] Selecting 100 Samples...")
    df = pd.read_csv(CSV_PATH)
    noise_samples = df[df['trace_category'] == 'noise'].head(50)
    quake_samples = df[df['trace_category'].str.contains('earthquake', na=False)].head(50)
    test_df = pd.concat([noise_samples, quake_samples])

    results = []

    with h5py.File(HDF5_PATH, 'r') as f:
        for _, row in test_df.iterrows():
            trace_name = row['trace_name']
            category = "Earthquake" if "earthquake" in row['trace_category'] else "Noise"
            
            data = f['data'][trace_name]
            # Ambil komponen Z (indeks 0) - 700 sampel
            z_raw = data[:700, 0]

            # --- SKENARIO A: TANPA NORMALISASI (Cara lama) ---
            z_latent_old = utils.latent_codes_1D(z_raw, embedding_model)
            val_old = np.mean(z_latent_old)

            # --- SKENARIO B: DENGAN NORMALISASI (StandardScaler style) ---
            z_norm = (z_raw - np.mean(z_raw)) / (np.std(z_raw) + 1e-6)
            z_latent_new = utils.latent_codes_1D(z_norm, embedding_model)
            val_new = np.mean(z_latent_new)

            results.append({
                'Category': category,
                'Val_Old': val_old,
                'Val_New': val_new
            })

    # 3. Analisis Hasil
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("                HASIL SANITY CHECK (Z-CHANNEL)")
    print("="*60)
    print("TARGET REFERENSI JSON ASLI:")
    print(" - Noise      : ~ -4.5")
    print(" - Earthquake : ~ +1.0")
    print("-" * 60)
    
    summary = res_df.groupby('Category').agg(['mean', 'std'])
    print(summary)
    print("="*60)

    # Kesimpulan otomatis
    avg_quake_new = res_df[res_df['Category'] == 'Earthquake']['Val_New'].mean()
    if abs(avg_quake_new - 1.0) < abs(res_df[res_df['Category'] == 'Earthquake']['Val_Old'].mean() - 1.0):
        print("\n[KESIMPULAN] Normalisasi MEMPERBAIKI distribusi data!")
        print("Silakan gunakan normalisasi input untuk run 6 jam berikutnya.")
    else:
        print("\n[KESIMPULAN] Normalisasi tidak memberikan efek signifikan atau justru memperburuk.")

if __name__ == "__main__":
    run_sanity_check()