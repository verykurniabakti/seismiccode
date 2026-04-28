import json
import numpy as np
import os
from tqdm import tqdm

def peak_normalize(data):
    # Sesuai standar Golden Demo: Max Absolut = 1.0
    if data is None or len(data) == 0:
        return []
    data = np.array(data)
    data = data - np.mean(data) # Detrend/Zero-mean
    max_val = np.max(np.abs(data))
    if max_val > 1e-9: # Menghindari pembagian nol jika sinyal flat
        data = data / max_val
    return data.tolist()

def process_ukraine_1C_to_golden(input_path, output_path):
    print(f"[INFO] Membuka file JSON Ukraine (1-Component)...")
    try:
        with open(input_path, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membuka file: {e}")
        return
    
    new_dataset = {}
    
    print("[INFO] Sinkronisasi ke Golden Standard (Peak-Norm Z Only)...")
    for record_key in tqdm(dataset, desc="Processing"):
        record = dataset[record_key]
        
        # Ambil data Z (Satu-satunya komponen yang tersedia sesuai audit)
        z_raw = record.get('Z')
        z_noise_raw = record.get('Z_noise')
        
        # Validasi: Pastikan data Z tidak kosong dan memiliki panjang yang cukup (misal 700 titik)
        if z_raw is None or len(z_raw) < 100:
            continue
            
        # Terapkan Peak Normalization (Resep Golden Demo)
        z_norm = peak_normalize(z_raw)
        z_noise_norm = peak_normalize(z_noise_raw)
        
        # Buat record baru agar metadata lainnya tetap terjaga (ML, origin time, dll)
        new_record = record.copy()
        new_dataset[record_key] = new_record
        
        # Update hanya bagian sinyal dengan yang sudah ternormalisasi
        new_dataset[record_key]['Z'] = z_norm
        new_dataset[record_key]['Z_noise'] = z_noise_norm
        new_dataset[record_key]['component_count'] = 1 # Flag untuk identifikasi di skrip benchmark

    # Buat folder output jika belum ada (antisipasi folder di SSD belum dibuat)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[INFO] Menyimpan hasil ke: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_dataset, f, indent=4)
    
    print(f"[SUCCESS] Ukraine 1C Golden Standard selesai. Total data: {len(new_dataset)}")

if __name__ == "__main__":
    # Path Input
    INPUT_JSON = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/Benchmark_ Ukraine 3C_ test n1335 r100 sdb8-956/Ukraine data, test n1335 r100 sdb8-956.json"
    
    # Path Output ke SSD Extreme
    OUTPUT_JSON = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo/Ukraine_1C_golden_standard.json"
    
    if os.path.exists(INPUT_JSON):
        process_ukraine_1C_to_golden(INPUT_JSON, OUTPUT_JSON)
    else:
        print(f"[ERROR] Path input tidak ditemukan: {INPUT_JSON}")