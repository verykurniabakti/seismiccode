#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKSTRAKSI WAVEFORM KE JSON + SINKRONISASI KATALOG (GABUNGAN) - VERSI DIPERBAIKI
- 1C dan 3C (langsung dihasilkan bersamaan)
- Preprocessing: detrend, resample 100 Hz, normalisasi max 9 detik setelah P
- STA/LTA untuk P-wave picking
- Filter kualitas: travel time 0-60 detik, SNR ≥ 1.5, tidak ada NaN/Inf
- Parallel processing dengan resume
- Output: 2 file JSON (1C dan 3C) dalam satu proses
"""

import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from obspy import read
from obspy.signal.trigger import recursive_sta_lta
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime

# =============================================
# 1. KONFIGURASI
# =============================================

WAVEFORM_DIR = '/Volumes/Extreme SSD/unduhan_juli_indonesia_sesi_4'
CATALOG_CSV = "/Volumes/Extreme SSD/katalog/hybrid_catalog_filtered.csv"
OUTPUT_JSON_1C = '/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_1c_4_final.json'
OUTPUT_JSON_3C = "/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_3c_4_final.json"
LOG_FILE = "extract_waveforms_both.log"

# Parameter MCU-Quake
SAMPLE_RATE = 100.0
SIG_DURATION = 7.0
NOISE_DURATION = 7.0
NORM_WINDOW = 9.0

# Parameter STA/LTA
STA_WIN = 1.0
LTA_WIN = 10.0
TRIGGER_THRESHOLD = 2.5

# Filter kualitas
MAX_TRAVEL_TIME = 60.0
MIN_TRAVEL_TIME = 0.0
MIN_SNR = 1.5  # SNR minimal untuk lolos filter

# Parallel
MAX_WORKERS = 8
MIN_FILE_SIZE_BYTES = 1024
MAX_EVENTS = 15000  # Set angka untuk testing, None untuk semua

# =============================================
# 2. SETUP LOGGING
# =============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================
# 3. BACA KATALOG
# =============================================

def read_catalog(csv_path):
    """Baca katalog dan buat dictionary timestamp -> metadata."""
    logger.info(f"📂 Membaca katalog: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Deteksi kolom
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'datetime' in col.lower() or 'origin' in col.lower():
            time_col = col
            break
    if time_col is None:
        raise ValueError("Kolom waktu tidak ditemukan.")
    
    lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
    mag_col = next((col for col in df.columns if 'mag' in col.lower()), None)
    
    # Parsing datetime
    try:
        df['datetime'] = pd.to_datetime(df[time_col], utc=True, format='ISO8601')
    except:
        df['datetime'] = pd.to_datetime(df[time_col], utc=True, infer_datetime_format=True)
    
    df['timestamp_key'] = df['datetime'].dt.strftime("%Y%m%d_%H%M%S")
    df = df.drop_duplicates(subset=['timestamp_key'], keep='first')
    
    # Buat dictionary
    catalog_dict = {}
    for _, row in df.iterrows():
        key = row['timestamp_key']
        catalog_dict[key] = {
            'origin_time': row['datetime'].isoformat(),
            'latitude': row[lat_col],
            'longitude': row[lon_col],
            'magnitude': row[mag_col] if mag_col else None
        }
    
    logger.info(f"✅ Katalog dimuat: {len(catalog_dict)} timestamp unik")
    return catalog_dict


# =============================================
# 4. FUNGSI PREPROCESSING & QC (VERSI TERPADU)
# =============================================

def is_trace_valid(trace):
    """Periksa apakah trace memiliki data yang valid."""
    data = trace.data
    return data is not None and len(data) > 0 and not np.isnan(data).any() and not np.isinf(data).any() and not np.all(data == 0)

def extract_component_windows(trace, p_time):
    """Ekstraksi sinyal dengan filter anti zero-padding TERPADU."""
    try:
        if not is_trace_valid(trace):
            return None, None
            
        tr_signal = trace.copy().trim(p_time, p_time + SIG_DURATION)
        tr_noise = trace.copy().trim(p_time - NOISE_DURATION, p_time)
        
        # --- FILTER ANTI ZERO-PADDING (Ditempatkan di sini agar membuang data cacat lebih awal) ---
        target_len_raw = int(trace.stats.sampling_rate * SIG_DURATION)
        if len(tr_signal.data) < target_len_raw * 0.95 or len(tr_noise.data) < target_len_raw * 0.95:
            return None, None
            
        # Detrend
        tr_signal.detrend('simple')
        tr_noise.detrend('simple')
        
        # Resample
        if tr_signal.stats.sampling_rate != SAMPLE_RATE: tr_signal.resample(SAMPLE_RATE)
        if tr_noise.stats.sampling_rate != SAMPLE_RATE: tr_noise.resample(SAMPLE_RATE)
        
        # Normalisasi
        tr_norm = trace.copy().trim(p_time, p_time + NORM_WINDOW)
        max_val = np.max(np.abs(tr_norm.data)) if len(tr_norm.data) > 0 else np.max(np.abs(tr_signal.data))
        max_val = max_val if max_val > 1e-6 else 1.0
            
        signal_data = (tr_signal.data / max_val).tolist()
        noise_data = (tr_noise.data / max_val).tolist()
        
        # Fixed length to 700
        target_len = int(SAMPLE_RATE * SIG_DURATION)
        def fix(d): return d[:target_len] if len(d) > target_len else d + [0.0]*(target_len-len(d))
            
        return fix(signal_data), fix(noise_data)
    except Exception as e:
        logger.debug(f"Error dalam ekstraksi: {e}")
        return None, None

def pick_p_arrival(trace):
    """Deteksi P-wave dengan STA/LTA Recursive."""
    try:
        sr = trace.stats.sampling_rate
        sta_n = int(STA_WIN * sr)
        lta_n = int(LTA_WIN * sr)
        if len(trace.data) < lta_n + sta_n: return trace.stats.starttime + 5.0
        
        cft = recursive_sta_lta(trace.data, sta_n, lta_n)
        trigger_indices = np.where(cft > TRIGGER_THRESHOLD)[0]
        if len(trigger_indices) > 0:
            return trace.stats.starttime + (trigger_indices[0] / sr)
            
        # Fallback
        max_idx = np.argmax(np.abs(trace.data[:int(15*sr)]))
        return trace.stats.starttime + (max_idx / sr) if max_idx > 0 else trace.stats.starttime + 5.0
    except: return trace.stats.starttime + 5.0


def compute_snr(signal, noise):
    """Hitung Signal-to-Noise Ratio."""
    if len(noise) == 0 or np.std(noise) < 1e-9:
        return 0
    peak_signal = np.max(np.abs(signal))
    std_noise = np.std(noise)
    return peak_signal / std_noise if std_noise > 0 else 0


def extract_timestamp_from_filename(filename):
    """Ekstrak timestamp YYYYMMDD_HHMMSS dari nama file."""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    match = re.search(r'(\d{14})', filename)
    if match:
        ts = match.group(1)
        return f"{ts[:8]}_{ts[8:]}"
    return None

def process_file(file_path, catalog_dict):
    """
    Proses satu file .mseed.
    Return dict jika berhasil dan lolos filter, None jika gagal.
    """
    try:
        st = read(str(file_path))
        if len(st) == 0:
            return None
        
        trace_z = trace_n = trace_e = None
        for tr in st:
            ch = tr.stats.channel
            if ch.endswith('Z'):
                trace_z = tr
            elif ch.endswith('N'):
                trace_n = tr
            elif ch.endswith('E'):
                trace_e = tr
        
        if trace_z is None:
            return None
        
        # Periksa trace Z valid
        if not is_trace_valid(trace_z):
            return None
        
        # P-pick
        p_time = pick_p_arrival(trace_z)
        z_signal, z_noise = extract_component_windows(trace_z, p_time)
        if z_signal is None:
            return None
        
        # Ekstrak timestamp dari nama file
        file_stem = file_path.stem
        timestamp_key = extract_timestamp_from_filename(file_stem)
        
        # Ambil metadata dari katalog
        origin_time = None
        latitude = None
        longitude = None
        magnitude = None
        if timestamp_key and timestamp_key in catalog_dict:
            cat = catalog_dict[timestamp_key]
            origin_time = cat['origin_time']
            latitude = cat['latitude']
            longitude = cat['longitude']
            magnitude = cat['magnitude']
        
        # ===== FILTER KUALITAS =====
        travel_time = None
        if origin_time and p_time:
            try:
                if hasattr(p_time, 'isoformat'):
                    p_str = p_time.isoformat()
                else:
                    p_str = str(p_time)
                p_dt = datetime.fromisoformat(p_str.replace('Z', '+00:00'))
                o_dt = datetime.fromisoformat(origin_time.replace('Z', '+00:00'))
                travel_time = (p_dt - o_dt).total_seconds()
                
                # Filter travel time
                if travel_time < MIN_TRAVEL_TIME or travel_time > MAX_TRAVEL_TIME:
                    logger.debug(f"Event {file_stem}: travel_time={travel_time:.2f}s (out of range)")
                    return None
            except Exception as e:
                logger.debug(f"Event {file_stem}: gagal hitung travel_time: {e}")
                # Jika tidak bisa hitung travel_time, tetap proses
                pass
        
        # ===== FILTER SNR =====
        # Gunakan Z_signal dan Z_noise untuk menghitung SNR
        signal_arr = np.array(z_signal)
        noise_arr = np.array(z_noise)
        snr = compute_snr(signal_arr, noise_arr)
        if snr < MIN_SNR:
            logger.debug(f"Event {file_stem}: SNR={snr:.2f} < {MIN_SNR}")
            return None
        
        # ===== BENTUK HASIL =====
        result = {
            'event_id': file_stem,
            'network': trace_z.stats.network,
            'station': trace_z.stats.station,
            'p_arrival': str(p_time),
            'file': file_path.name,
            'has_z': True,
            'has_n': False,
            'has_e': False,
            'Z': z_signal,
            'Z_noise': z_noise,
            'metadata': {
                'network': trace_z.stats.network,
                'station': trace_z.stats.station,
                'p_arrival': str(p_time),
                'file': file_path.name,
                'origin_time': origin_time,
                'latitude': latitude,
                'longitude': longitude,
                'magnitude': magnitude,
                'snr': snr,
                'travel_time': travel_time
            }
        }
        
        # Ekstrak N
        if trace_n is not None and is_trace_valid(trace_n):
            n_signal, n_noise = extract_component_windows(trace_n, p_time)
            if n_signal is not None:
                result['has_n'] = True
                result['N'] = n_signal
                result['N_noise'] = n_noise
        
        # Ekstrak E
        if trace_e is not None and is_trace_valid(trace_e):
            e_signal, e_noise = extract_component_windows(trace_e, p_time)
            if e_signal is not None:
                result['has_e'] = True
                result['E'] = e_signal
                result['E_noise'] = e_noise
        
        return result
    except Exception as e:
        logger.debug(f"Error processing {file_path.name}: {e}")
        return None

# =============================================
# 5. MAIN
# =============================================

def main():
    logger.info("="*70)
    logger.info("🚀 EKSTRAKSI + SINKRONISASI (LANGSUNG 1C & 3C) - VERSI DIPERBAIKI")
    logger.info("="*70)
    
    # Buat folder output
    os.makedirs(os.path.dirname(OUTPUT_JSON_1C), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_3C), exist_ok=True)
    
    # Baca katalog
    try:
        catalog_dict = read_catalog(CATALOG_CSV)
    except Exception as e:
        logger.warning(f"⚠️ Gagal membaca katalog: {e}")
        logger.warning("   Melanjutkan tanpa metadata katalog (origin_time dll akan kosong).")
        catalog_dict = {}
    
    # Cari file .mseed
    wave_dir = Path(WAVEFORM_DIR)
    all_files = list(wave_dir.glob("*.mseed"))
    all_files = [f for f in all_files if f.stat().st_size >= MIN_FILE_SIZE_BYTES]
    
    # Batasi jumlah file jika MAX_EVENTS di-set
    if MAX_EVENTS and len(all_files) > MAX_EVENTS:
        all_files = all_files[:MAX_EVENTS]
        logger.info(f"⚠️ Testing: hanya {MAX_EVENTS} file pertama yang diproses.")
    
    logger.info(f"📁 Ditemukan {len(all_files)} file .mseed")
    
    # Load JSON existing (resume)
    data_1c = {}
    data_3c = {}
    if os.path.exists(OUTPUT_JSON_1C):
        with open(OUTPUT_JSON_1C, 'r') as f:
            data_1c = json.load(f)
        logger.info(f"📂 Load JSON 1C existing: {len(data_1c)} entries")
    if os.path.exists(OUTPUT_JSON_3C):
        with open(OUTPUT_JSON_3C, 'r') as f:
            data_3c = json.load(f)
        logger.info(f"📂 Load JSON 3C existing: {len(data_3c)} entries")
    
    # Filter file yang belum diproses
    files_to_process = [f for f in all_files if f.stem not in data_1c]
    logger.info(f"📦 File baru: {len(files_to_process)}")
    
    if len(files_to_process) == 0:
        logger.info("✅ Semua file sudah diproses!")
        logger.info(f"📁 Total 1C: {len(data_1c)}")
        logger.info(f"📁 Total 3C: {len(data_3c)}")
        return
    
    # Proses paralel
    success_1c = 0
    success_3c = 0
    failed = 0
    rejected_snr = 0
    rejected_travel = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, f, catalog_dict): f for f in files_to_process}
        with tqdm(total=len(futures), desc="Ekstraksi", unit="file") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                result = future.result()
                if result:
                    key = result['event_id']
                    # Simpan 1C
                    data_1c[key] = {
                        'type': 'se',
                        'Z': result['Z'],
                        'Z_noise': result['Z_noise'],
                        'metadata': result['metadata']
                    }
                    success_1c += 1
                    # Simpan 3C jika lengkap
                    if result['has_n'] and result['has_e']:
                        data_3c[key] = {
                            'type': 'se',
                            'Z': result['Z'],
                            'N': result['N'],
                            'E': result['E'],
                            'Z_noise': result['Z_noise'],
                            'N_noise': result['N_noise'],
                            'E_noise': result['E_noise'],
                            'metadata': result['metadata']
                        }
                        success_3c += 1
                else:
                    failed += 1
                pbar.update(1)
                
                # Simpan setiap 100 file
                if (success_1c + failed) % 100 == 0:
                    with open(OUTPUT_JSON_1C, 'w') as f:
                        json.dump(data_1c, f, indent=2)
                    with open(OUTPUT_JSON_3C, 'w') as f:
                        json.dump(data_3c, f, indent=2)
    
    # Simpan final
    with open(OUTPUT_JSON_1C, 'w') as f:
        json.dump(data_1c, f, indent=2)
    with open(OUTPUT_JSON_3C, 'w') as f:
        json.dump(data_3c, f, indent=2)
    
    logger.info("="*70)
    logger.info(f"✨ SELESAI!")
    logger.info(f"✅ 1C: {len(data_1c)} entries")
    logger.info(f"✅ 3C: {len(data_3c)} entries")
    logger.info(f"❌ Gagal (total): {failed}")
    logger.info(f"📂 Output 1C: {OUTPUT_JSON_1C}")
    logger.info(f"📂 Output 3C: {OUTPUT_JSON_3C}")
    logger.info("="*70)

if __name__ == "__main__":
    main()