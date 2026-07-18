#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKSTRAKSI WAVEFORM KE JSON (DENGAN QC SNR & PICKING STA/LTA)
- Menggunakan STA/LTA untuk menentukan P-arrival secara akurat.
- Menggunakan filter SNR untuk membuang gempa yang tidak terlihat (terkubur noise).
- Output: 2 file JSON (1C dan 3C) dengan jendela Z dan Z_noise yang valid.
"""

import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime

# =============================================
# 1. KONFIGURASI (SESUAIKAN DENGAN DIREKTORI BAPAK)
# =============================================

WAVEFORM_DIR = '/Volumes/Extreme SSD/unduhan_juli_indonesia_sesi_4'
CATALOG_CSV = "/Volumes/Extreme SSD/katalog/hybrid_catalog_filtered.csv"
OUTPUT_JSON_1C = '/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_1c_PROPER.json'
OUTPUT_JSON_3C = "/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_3c_PROPER.json"
LOG_FILE = "extract_proper.log"

# Parameter MCU-Quake
SAMPLE_RATE = 100.0
SIG_DURATION = 7.0
NOISE_DURATION = 7.0
NORM_WINDOW = 9.0

# Parameter STA/LTA untuk AUTO-PICKING
STA_WIN = 1.0
LTA_WIN = 10.0
TRIGGER_ON = 3.0   # Threshold batas atas trigger
TRIGGER_OFF = 1.5  # Threshold batas bawah trigger

# Filter Kualitas Data (Quality Control)
MIN_SNR = 2.5  # Sinyal gempa harus 2.5x lipat lebih besar dari noise, jika tidak buang.

# Parallel
MAX_WORKERS = 8
MIN_FILE_SIZE_BYTES = 1024
MAX_EVENTS = 30000 

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
    logger.info(f"📂 Membaca katalog: {csv_path}")
    df = pd.read_csv(csv_path)
    
    time_col = next((col for col in df.columns if 'time' in col.lower() or 'datetime' in col.lower()), None)
    if time_col is None:
        raise ValueError("Kolom waktu tidak ditemukan.")
    
    lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
    mag_col = next((col for col in df.columns if 'mag' in col.lower()), None)
    
    df['datetime'] = pd.to_datetime(df[time_col], utc=True, format='ISO8601', errors='coerce')
    df['timestamp_key'] = df['datetime'].dt.strftime("%Y%m%d_%H%M%S")
    df = df.drop_duplicates(subset=['timestamp_key'], keep='first')
    
    catalog_dict = {}
    for _, row in df.iterrows():
        key = row['timestamp_key']
        catalog_dict[key] = {
            'origin_time': row['datetime'].isoformat(),
            'latitude': row[lat_col],
            'longitude': row[lon_col],
            'magnitude': row[mag_col] if mag_col else None
        }
    return catalog_dict

# =============================================
# 4. FUNGSI PREPROCESSING & QC
# =============================================
def extract_component_windows(trace, p_time):
    """Ekstraksi sinyal dan noise dengan pengecekan validitas dan anti zero-padding."""
    try:
        # Periksa trace valid
        if not is_trace_valid(trace):
            return None, None
        
        # Ekstrak sinyal dan noise
        tr_signal = trace.copy().trim(p_time, p_time + SIG_DURATION)
        tr_noise = trace.copy().trim(p_time - NOISE_DURATION, p_time)
        
        # =====================================================================
        # TAMBAHKAN FILTER ANTI ZERO-PADDING DI SINI
        # Jika rekaman asli terpotong (misal karena file mseed habis sebelum 7 detik),
        # maka data hasil trim akan pendek. Kita buang jika kurang dari 95% durasi target.
        # =====================================================================
        target_len_raw = int(trace.stats.sampling_rate * SIG_DURATION)
        
        if len(tr_signal.data) < target_len_raw * 0.95 or len(tr_noise.data) < target_len_raw * 0.95:
            # Data ini cacat karena terpotong di ujung file
            return None, None
            
        # Periksa hasil trim setelah filtering
        if len(tr_signal.data) == 0 or len(tr_noise.data) == 0:
            return None, None
        
        # Detrend
        tr_signal.detrend('simple')
        tr_noise.detrend('simple')
        
def is_trace_valid(trace):
    data = trace.data
    if data is None or len(data) == 0:
        return False
    if np.isnan(data).any() or np.isinf(data).any() or np.all(data == 0):
        return False
    return True

def pick_p_arrival_stalta(trace):
    """Mencari waktu kedatangan gelombang P menggunakan STA/LTA."""
    try:
        sr = trace.stats.sampling_rate
        cft = classic_sta_lta(trace.data, int(STA_WIN * sr), int(LTA_WIN * sr))
        onsets = trigger_onset(cft, TRIGGER_ON, TRIGGER_OFF)
        
        if len(onsets) > 0:
            # Ambil trigger pertama yang terdeteksi
            p_index = onsets[0][0]
            return trace.stats.starttime + (p_index / sr)
        return None
    except Exception:
        return None

def compute_snr(signal, noise):
    """Menghitung perbandingan rasio kekuatan gempa terhadap derau."""
    if len(noise) == 0 or np.std(noise) < 1e-9:
        return 0
    peak_signal = np.max(np.abs(signal))
    std_noise = np.std(noise)
    return peak_signal / std_noise if std_noise > 0 else 0

def extract_component_windows(trace, p_time):
    try:
        tr_signal = trace.copy().trim(p_time, p_time + SIG_DURATION)
        tr_noise = trace.copy().trim(p_time - NOISE_DURATION, p_time)
        
        if len(tr_signal.data) == 0 or len(tr_noise.data) == 0:
            return None, None
        
        tr_signal.detrend('simple')
        tr_noise.detrend('simple')
        
        if tr_signal.stats.sampling_rate != SAMPLE_RATE:
            tr_signal.resample(SAMPLE_RATE)
        if tr_noise.stats.sampling_rate != SAMPLE_RATE:
            tr_noise.resample(SAMPLE_RATE)
        
        # Normalisasi
        tr_norm = trace.copy().trim(p_time, p_time + NORM_WINDOW)
        max_val = np.max(np.abs(tr_norm.data)) if len(tr_norm.data) > 0 else np.max(np.abs(tr_signal.data))
        if np.isnan(max_val) or max_val < 1e-6:
            max_val = 1.0
            
        signal_data = tr_signal.data / max_val
        noise_data = tr_noise.data / max_val
        
        target_len = int(SAMPLE_RATE * SIG_DURATION)
        def fix_length(data):
            if len(data) > target_len: return data[:target_len]
            elif len(data) < target_len: return np.pad(data, (0, target_len - len(data)), 'constant')
            return data
            
        return fix_length(signal_data).tolist(), fix_length(noise_data).tolist()
    except Exception:
        return None, None

def extract_timestamp_from_filename(filename):
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match: return match.group(1)
    match = re.search(r'(\d{14})', filename)
    if match: return f"{match.group(1)[:8]}_{match.group(1)[8:]}"
    return None

def process_file(file_path, catalog_dict):
    try:
        st = read(str(file_path))
        if len(st) == 0: return None
        
        trace_z = next((tr for tr in st if tr.stats.channel.endswith('Z')), None)
        trace_n = next((tr for tr in st if tr.stats.channel.endswith('N')), None)
        trace_e = next((tr for tr in st if tr.stats.channel.endswith('E')), None)
        
        if trace_z is None or not is_trace_valid(trace_z): return None
        
        # 1. Temukan P-Arrival dengan STA/LTA
        p_time = pick_p_arrival_stalta(trace_z)
        if p_time is None:
            return None # Skip jika tidak ada gelombang gempa yang terdeteksi
            
        # 2. Ekstrak Jendela
        z_signal, z_noise = extract_component_windows(trace_z, p_time)
        if z_signal is None: return None
        
        # 3. Hitung SNR dan Terapkan Quality Control (QC)
        snr = compute_snr(np.array(z_signal), np.array(z_noise))
        if snr < MIN_SNR:
            return None # Skip jika gempa tidak lebih besar dari noise (tertimbun noise)

        file_stem = file_path.stem
        timestamp_key = extract_timestamp_from_filename(file_stem)
        
        origin_time, latitude, longitude, magnitude = None, None, None, None
        if timestamp_key and timestamp_key in catalog_dict:
            cat = catalog_dict[timestamp_key]
            origin_time, latitude, longitude, magnitude = cat['origin_time'], cat['latitude'], cat['longitude'], cat['magnitude']

        result = {
            'event_id': file_stem,
            'network': trace_z.stats.network,
            'station': trace_z.stats.station,
            'p_arrival': str(p_time),
            'has_z': True, 'has_n': False, 'has_e': False,
            'Z': z_signal, 'Z_noise': z_noise,
            'metadata': {
                'origin_time': origin_time, 'latitude': latitude, 'longitude': longitude,
                'magnitude': magnitude, 'snr': snr
            }
        }
        
        if trace_n and is_trace_valid(trace_n):
            n_signal, n_noise = extract_component_windows(trace_n, p_time)
            if n_signal is not None:
                result['has_n'], result['N'], result['N_noise'] = True, n_signal, n_noise
                
        if trace_e and is_trace_valid(trace_e):
            e_signal, e_noise = extract_component_windows(trace_e, p_time)
            if e_signal is not None:
                result['has_e'], result['E'], result['E_noise'] = True, e_signal, e_noise
                
        return result
    except Exception:
        return None

# =============================================
# 5. MAIN EXECUTION
# =============================================

def main():
    logger.info("="*70)
    logger.info("🚀 EKSTRAKSI DENGAN QC (SNR & STA/LTA SEBAGAI AUTO-PICKER)")
    logger.info("="*70)
    
    os.makedirs(os.path.dirname(OUTPUT_JSON_1C), exist_ok=True)
    catalog_dict = read_catalog(CATALOG_CSV)
    
    wave_dir = Path(WAVEFORM_DIR)
    all_files = [f for f in wave_dir.glob("*.mseed") if f.stat().st_size >= MIN_FILE_SIZE_BYTES]
    if MAX_EVENTS: all_files = all_files[:MAX_EVENTS]
    
    data_1c, data_3c = {}, {}
    success_1c, success_3c, failed = 0, 0, 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, f, catalog_dict): f for f in all_files}
        with tqdm(total=len(futures), desc="Ekstraksi", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    key = result['event_id']
                    data_1c[key] = {'type': 'se', 'Z': result['Z'], 'Z_noise': result['Z_noise'], 'metadata': result['metadata']}
                    success_1c += 1
                    if result['has_n'] and result['has_e']:
                        data_3c[key] = {
                            'type': 'se', 'Z': result['Z'], 'N': result['N'], 'E': result['E'],
                            'Z_noise': result['Z_noise'], 'N_noise': result['N_noise'], 'E_noise': result['E_noise'],
                            'metadata': result['metadata']
                        }
                        success_3c += 1
                else:
                    failed += 1
                pbar.update(1)

    with open(OUTPUT_JSON_1C, 'w') as f: json.dump(data_1c, f)
    with open(OUTPUT_JSON_3C, 'w') as f: json.dump(data_3c, f)
    
    logger.info(f"✅ Selesai! Data valid: 1C ({success_1c}), 3C ({success_3c}). Data dibuang/gagal: {failed}")

if __name__ == "__main__":
    main()