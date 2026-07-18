#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKSTRAKSI WAVEFORM KE JSON + SINKRONISASI KATALOG (TANPA FILTER SNR & TANPA STA/LTA)
- 1C dan 3C (langsung dihasilkan bersamaan)
- Preprocessing: detrend, resample 100 Hz, normalisasi max 9 detik setelah P
- TANPA STA/LTA: p_time = trace.stats.starttime + 5.0 (window statis)
- TANPA filter SNR (semua data lolos)
- TANPA filter travel time (semua data lolos)
- Parallel processing dengan resume
- Output: 2 file JSON (1C dan 3C)
"""

import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from obspy import read
# STA/LTA tidak digunakan lagi
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
OUTPUT_JSON_1C = '/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_1c_4_NOFILTER.json'
OUTPUT_JSON_3C = "/Volumes/Extreme SSD/json_indonesia_juli_sesi_4/extracted_data_3c_4_NOFILTER.json"
LOG_FILE = "extract_no_filter.log"

# Parameter MCU-Quake
SAMPLE_RATE = 100.0
SIG_DURATION = 7.0
NOISE_DURATION = 7.0
NORM_WINDOW = 9.0

# Parameter STA/LTA (DIMATIKAN)
STA_WIN = 1.0
LTA_WIN = 10.0
TRIGGER_THRESHOLD = 2.5  # tidak digunakan

# Filter kualitas (DIMATIKAN)
MAX_TRAVEL_TIME = 999.0
MIN_TRAVEL_TIME = -999.0
MIN_SNR = -1.0  # semua lolos

# Parallel
MAX_WORKERS = 8
MIN_FILE_SIZE_BYTES = 1024
MAX_EVENTS = 30000  # Set untuk membatasi jumlah file

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
    
    try:
        df['datetime'] = pd.to_datetime(df[time_col], utc=True, format='ISO8601')
    except:
        df['datetime'] = pd.to_datetime(df[time_col], utc=True, infer_datetime_format=True)
    
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
    
    logger.info(f"✅ Katalog dimuat: {len(catalog_dict)} timestamp unik")
    return catalog_dict

# =============================================
# 4. FUNGSI PREPROCESSING (TANPA STA/LTA)
# =============================================

def is_trace_valid(trace):
    """Periksa apakah trace memiliki data yang valid."""
    data = trace.data
    if data is None or len(data) == 0:
        return False
    if np.isnan(data).any() or np.isinf(data).any():
        return False
    if np.all(data == 0):
        return False
    return True

def pick_p_arrival_fixed(trace):
    """P-time selalu di 5 detik setelah starttime (tanpa STA/LTA)."""
    # Ambil 5 detik dari awal trace sebagai 'p_time' (bukan onset)
    return trace.stats.starttime + 5.0

def compute_snr(signal, noise):
    """Hitung SNR, tapi tidak digunakan untuk filter."""
    if len(noise) == 0 or np.std(noise) < 1e-9:
        return 0
    peak_signal = np.max(np.abs(signal))
    std_noise = np.std(noise)
    return peak_signal / std_noise if std_noise > 0 else 0

def extract_component_windows(trace, p_time):
    """Ekstraksi sinyal dan noise dengan p_time fixed."""
    try:
        if not is_trace_valid(trace):
            return None, None
        
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
        
        tr_norm = trace.copy().trim(p_time, p_time + NORM_WINDOW)
        if len(tr_norm.data) > 0:
            max_val = np.max(np.abs(tr_norm.data))
        else:
            max_val = np.max(np.abs(tr_signal.data))
        
        if np.isnan(max_val) or max_val < 1e-6:
            max_val = 1.0
        
        signal_data = tr_signal.data / max_val
        noise_data = tr_noise.data / max_val
        
        if np.isnan(signal_data).any() or np.isnan(noise_data).any():
            return None, None
        
        target_len = int(SAMPLE_RATE * SIG_DURATION)
        def fix_length(data):
            if len(data) > target_len:
                return data[:target_len]
            elif len(data) < target_len:
                return np.pad(data, (0, target_len - len(data)), 'constant')
            return data
        
        return fix_length(signal_data).tolist(), fix_length(noise_data).tolist()
    except Exception as e:
        logger.debug(f"extract_component_windows error: {e}")
        return None, None

def extract_timestamp_from_filename(filename):
    """Ekstrak timestamp dari nama file."""
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
    Proses satu file .mseed tanpa STA/LTA dan tanpa filter.
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
        
        if not is_trace_valid(trace_z):
            return None
        
        # ===== TANPA STA/LTA =====
        p_time = pick_p_arrival_fixed(trace_z)  # fixed 5 detik dari start
        
        z_signal, z_noise = extract_component_windows(trace_z, p_time)
        if z_signal is None:
            return None
        
        file_stem = file_path.stem
        timestamp_key = extract_timestamp_from_filename(file_stem)
        
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
        
        # ===== FILTER DIMATIKAN =====
        # Tidak ada filter travel time (selalu lolos)
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
            except:
                pass
            # Tidak ada penolakan berdasarkan travel_time
        
        # ===== FILTER SNR DIMATIKAN =====
        # SNR dihitung tapi tidak digunakan untuk menolak
        signal_arr = np.array(z_signal)
        noise_arr = np.array(z_noise)
        snr = compute_snr(signal_arr, noise_arr)
        # semua data lolos, apapun SNR-nya
        
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
    logger.info("🚀 EKSTRAKSI (TANPA STA/LTA & TANPA FILTER)")
    logger.info("="*70)
    
    os.makedirs(os.path.dirname(OUTPUT_JSON_1C), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_3C), exist_ok=True)
    
    try:
        catalog_dict = read_catalog(CATALOG_CSV)
    except Exception as e:
        logger.warning(f"⚠️ Gagal membaca katalog: {e}")
        catalog_dict = {}
    
    wave_dir = Path(WAVEFORM_DIR)
    all_files = list(wave_dir.glob("*.mseed"))
    all_files = [f for f in all_files if f.stat().st_size >= MIN_FILE_SIZE_BYTES]
    
    if MAX_EVENTS and len(all_files) > MAX_EVENTS:
        all_files = all_files[:MAX_EVENTS]
        logger.info(f"⚠️ Testing: hanya {MAX_EVENTS} file pertama.")
    
    logger.info(f"📁 Ditemukan {len(all_files)} file .mseed")
    
    # Load existing (resume)
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
    
    files_to_process = [f for f in all_files if f.stem not in data_1c]
    logger.info(f"📦 File baru: {len(files_to_process)}")
    
    if len(files_to_process) == 0:
        logger.info("✅ Semua file sudah diproses!")
        logger.info(f"📁 Total 1C: {len(data_1c)}")
        logger.info(f"📁 Total 3C: {len(data_3c)}")
        return
    
    success_1c = 0
    success_3c = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, f, catalog_dict): f for f in files_to_process}
        with tqdm(total=len(futures), desc="Ekstraksi", unit="file") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                result = future.result()
                if result:
                    key = result['event_id']
                    data_1c[key] = {
                        'type': 'se',
                        'Z': result['Z'],
                        'Z_noise': result['Z_noise'],
                        'metadata': result['metadata']
                    }
                    success_1c += 1
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
                
                if (success_1c + failed) % 100 == 0:
                    with open(OUTPUT_JSON_1C, 'w') as f:
                        json.dump(data_1c, f, indent=2)
                    with open(OUTPUT_JSON_3C, 'w') as f:
                        json.dump(data_3c, f, indent=2)
    
    with open(OUTPUT_JSON_1C, 'w') as f:
        json.dump(data_1c, f, indent=2)
    with open(OUTPUT_JSON_3C, 'w') as f:
        json.dump(data_3c, f, indent=2)
    
    logger.info("="*70)
    logger.info(f"✨ SELESAI!")
    logger.info(f"✅ 1C: {len(data_1c)} entries")
    logger.info(f"✅ 3C: {len(data_3c)} entries")
    logger.info(f"❌ Gagal: {failed}")
    logger.info(f"📂 Output 1C: {OUTPUT_JSON_1C}")
    logger.info(f"📂 Output 3C: {OUTPUT_JSON_3C}")
    logger.info("="*70)

if __name__ == "__main__":
    main()