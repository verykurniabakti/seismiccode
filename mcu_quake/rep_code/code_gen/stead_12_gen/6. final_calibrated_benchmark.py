import sys, os, h5py, gc, json
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Library import utils

# --- KONFIGURASI PATH ---
CSV_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.csv'
HDF5_PATH = '/Volumes/Extreme SSD/stream_stead/data_stead/merge.hdf5'
OUTPUT_DIR = '/Volumes/Extreme SSD/mcu_quake_bis_stead_output_normalisasi_final'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'final_calibrated_results.csv')
MODEL_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Pre-trained model/MCU-Quake 5-20"
EMBEDDING_DIR = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/code_gen/Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538'

# --- PARAMETER KALIBRASI HASIL HITUNG PAK VERY ---
OFFSET = 0.285578
MULTIPLIER = 268.476980

def run_final_benchmark():
    print(f"[INFO] Memulai Benchmark Final di: {OUTPUT_DIR}")
    df = pd.read_csv(CSV_PATH)
    
    # Load Referensi KDE
    train_Z = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, Z.json")))
    train_N = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, N.json")))
    train_E = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, E.json")))
    pdfs = utils.embedding_PDFs_3D(train_Z, train_N, train_E)

    start_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_idx = len(pd.read_csv(CHECKPOINT_PATH))
        print(f"[RESUME] Melanjutkan dari indeks {start_idx}...")
    else:
        pd.DataFrame(columns=['trace_name', 'y_true', 'y_pred', 'lat_e', 'lat_n', 'lat_z']).to_csv(CHECKPOINT_PATH, index=False)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    emb_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    buffer = []
    with h5py.File(HDF5_PATH, 'r') as f:
        dataset = f['data']
        for i in tqdm(range(start_idx, len(df)), desc="Final Calibrated Run"):
            row = df.iloc[i]
            t_name = row['trace_name']
            try:
                # 1. Normalisasi Input Z-Score
                raw = dataset[t_name][:700]
                z_n = (raw[:,0] - np.mean(raw[:,0])) / (np.std(raw[:,0]) + 1e-6)
                n_n = (raw[:,1] - np.mean(raw[:,1])) / (np.std(raw[:,1]) + 1e-6)
                e_n = (raw[:,2] - np.mean(raw[:,2])) / (np.std(raw[:,2]) + 1e-6)

                # 2. Extract Latent & Apply Calibration
                f_z = (np.mean(utils.latent_codes_1D(z_n, emb_model)) - OFFSET) * MULTIPLIER
                f_n = (np.mean(utils.latent_codes_1D(n_n, emb_model)) - OFFSET) * MULTIPLIER
                f_e = (np.mean(utils.latent_codes_1D(e_n, emb_model)) - OFFSET) * MULTIPLIER

                # 3. Predict via KDE
                emb_3c = np.array([[f_e, f_n, f_z]])
                y_p, _, _ = utils.infer_3C_PDFs(emb_3c, pdfs, choose_pdf="Kernel")
                
                y_t = 2 if 'earthquake' in str(row['trace_category']).lower() else 0
                buffer.append([t_name, y_t, y_p, f_e, f_n, f_z])

                # Batch Save tiap 10.000 data
                if (i + 1) % 10000 == 0 or (i + 1) == len(df):
                    pd.DataFrame(buffer).to_csv(CHECKPOINT_PATH, mode='a', header=False, index=False)
                    buffer = []
                    # Refresh Memory
                    gc.collect()
                    tf.keras.backend.clear_session()
            except:
                continue

    print(f"\n[SELESAI] Hasil akhir tersimpan di: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    run_final_benchmark()