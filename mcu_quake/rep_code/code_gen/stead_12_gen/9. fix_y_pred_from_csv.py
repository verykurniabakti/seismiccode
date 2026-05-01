import pandas as pd
import numpy as np
import os, json
from Library import utils
from sklearn.metrics import classification_report

# --- KONFIGURASI ---
CSV_INPUT = '/Volumes/Extreme SSD/mcu_quake_bis_stead_output_normalisasi_final/final_calibrated_results.csv'
EMBEDDING_DIR = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/code_gen/Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538'

def run_sweep():
    print("[1/3] Membaca file CSV (1.2jt baris)...")
    df = pd.read_csv(CSV_INPUT)
    
    print("[2/3] Loading referensi KDE...")
    train_Z = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, Z.json")))
    train_N = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, N.json")))
    train_E = json.load(open(os.path.join(EMBEDDING_DIR, "Embedding data, E.json")))
    pdfs = utils.embedding_PDFs_3D(train_Z, train_N, train_E)

    y_true = df['y_true'].values
    lat_e_raw = df['lat_e'].values
    lat_n_raw = df['lat_n'].values
    lat_z_raw = df['lat_z'].values

    # Daftar shift yang akan diuji
    shifts_to_test = [5.5, 6.0, 6.5, 7.0, 7.5]
    
    print("[3/3] Memulai proses Sweep Optimization...")
    
    for s in shifts_to_test:
        print(f"\n" + "="*40)
        print(f" TESTING SHIFT: +{s}")
        print("="*40)
        
        new_preds = []
        # Shift data secara batch
        lat_e = lat_e_raw + s
        lat_n = lat_n_raw + s
        lat_z = lat_z_raw + s
        
        # Inferensi ulang
        for i in range(len(df)):
            input_3c = np.array([[lat_e[i], lat_n[i], lat_z[i]]])
            y_p, _, _ = utils.infer_3C_PDFs(input_3c, pdfs, choose_pdf="Kernel")
            new_preds.append(y_p)
            
            if (i+1) % 250000 == 0:
                print(f"   > Progres: {((i+1)/len(df)*100):.1f}%")

        print("\n[HASIL METRIK]")
        print(classification_report(y_true, new_preds, target_names=['Noise (0)', 'Earthquake (2)']))
        
        # Simpan sementara jika ini adalah hasil yang Bapak inginkan
        # (Opsional: Simpan ke CSV jika F1-score memuaskan)

if __name__ == "__main__":
    run_sweep()