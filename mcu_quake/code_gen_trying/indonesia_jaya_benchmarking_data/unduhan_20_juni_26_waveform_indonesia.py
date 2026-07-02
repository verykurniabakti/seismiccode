#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOWNLOAD WAVEFORM PRIORITAS GEOFON - PARALLEL VERSION
Mengunduh 3 komponen (BH?/HH?) untuk setiap event di katalog.
Dengan caching stasiun, parallel download, dan resume otomatis.
"""

import os
import sys
import time
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

# =============================================
# KONFIGURASI
# =============================================

#CATALOG_CSV = "/Volumes/Extreme SSD/unduhan_juni_bmkg_usgs/hasilscan/eda_final_output/filtered_events_selected.csv"
CATALOG_CSV = '/Volumes/Extreme SSD/unduhan_juni_bmkg_usgs/hasilscan/eda_final_output/HYBRID_EARTHQUAKE_CATALOG_2001_2024_FIX.csv'
OUTPUT_DIR = "/Volumes/Extreme SSD/unduhan_waveform_geofon_juli"

TIME_BEFORE = 30          # detik sebelum origin
TIME_AFTER = 120          # detik setelah origin
MAX_RADIUS_DEG = 15.0     # radius pencarian stasiun awal
FALLBACK_RADIUS_DEG = 25.0 # radius kedua jika gagal
CHANNELS = ['BHZ','BHN','BHE','HHZ','HHN','HHE','EHZ','EHN','EHE']
LOCATION = "*"

# --- Parallel ---
MAX_WORKERS = 6           # jumlah thread paralel (sesuaikan)
MAX_EVENTS = 25000         # None untuk semua, atau angka untuk testing

MIN_MAGNITUDE = 4.5        # hanya event dengan magnitudo ≥ 4.5
MIN_YEAR = 2004            # hanya event dari 2004 ke atas (kualitas data lebih baik)

# --- Logging ---
LOG_FILE = "download_waveform.log"
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
# FUNGSI BANTUAN CEK UKURAN
# =============================================

def download_event_with_retry(event, output_dir, max_retries=3):
    """Download dengan retry mechanism."""
    for attempt in range(max_retries):
        try:
            result = download_event(event, output_dir)
            if result:
                # Cek ukuran file
                event_id = event['id']
                existing = [f for f in os.listdir(output_dir) if event_id in f and f.endswith('.mseed')]
                if existing:
                    filepath = os.path.join(output_dir, existing[0])
                    if os.path.getsize(filepath) < 1024:  # < 1 KB
                        os.remove(filepath)
                        logger.warning(f"Event {event_id}: File terlalu kecil, dihapus. Retry {attempt+1}/{max_retries}")
                        continue
                return True
        except Exception as e:
            logger.warning(f"Event {event_id}: Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)  # exponential backoff
    return False

def read_catalog(catalog_path):
    df = pd.read_csv(catalog_path)
    # ... (deteksi kolom) ...
    
    # Filter tambahan
    if 'magnitude' in df.columns:
        df = df[df['magnitude'] >= MIN_MAGNITUDE]
        logger.info(f"🔍 Filter magnitudo ≥ {MIN_MAGNITUDE}: {len(df)} event tersisa")
    
    if 'year' in df.columns:
        df = df[df['year'] >= MIN_YEAR]
        logger.info(f"🔍 Filter tahun ≥ {MIN_YEAR}: {len(df)} event tersisa")
    
    return df, lat_col, lon_col
# =============================================
# CACHE STASIUN (agar tidak query ulang)
# =============================================
station_cache = {}

def cache_key(client_name, lat, lon, radius_deg, year):
    """Buat key untuk cache stasiun."""
    return f"{client_name}_{round(lat,2)}_{round(lon,2)}_{radius_deg}_{year}"

def find_station(client, client_name, lat, lon, origin_time, radius_deg, channels):
    """
    Cari stasiun terdekat dengan channel yang diinginkan.
    Return (network, station, distance_deg) atau (None, None, None)
    """
    year = origin_time.year
    key = cache_key(client_name, lat, lon, radius_deg, year)
    
    if key in station_cache:
        return station_cache[key]
    
    try:
        inventory = client.get_stations(
            latitude=lat,
            longitude=lon,
            maxradius=radius_deg,
            level="channel",
            channel=",".join(channels),
            location=LOCATION,
            starttime=origin_time - 60,
            endtime=origin_time + 60
        )
        if not inventory:
            station_cache[key] = (None, None, None)
            return None, None, None

        best_dist = float('inf')
        best_net = None
        best_sta = None
        for network in inventory:
            for station in network:
                if station.latitude is None or station.longitude is None:
                    continue
                dist_m, _, _ = gps2dist_azimuth(lat, lon, station.latitude, station.longitude)
                dist_deg = dist_m / 111000.0
                if dist_deg < best_dist:
                    best_dist = dist_deg
                    best_net = network.code
                    best_sta = station.code
        if best_net and best_sta:
            result = (best_net, best_sta, best_dist)
            station_cache[key] = result
            return result
        else:
            station_cache[key] = (None, None, None)
            return None, None, None
    except Exception as e:
        logger.debug(f"Error finding station: {e}")
        station_cache[key] = (None, None, None)
        return None, None, None

# =============================================
# FUNGSI DOWNLOAD SATU EVENT
# =============================================

def download_event(event, output_dir):
    """
    Unduh waveform untuk satu event.
    Return True jika berhasil, False jika gagal.
    """
    lat = event['lat']
    lon = event['lon']
    origin = event['time']
    event_id = event['id']

    os.makedirs(output_dir, exist_ok=True)

    # Cek apakah file sudah ada (resume)
    existing = [f for f in os.listdir(output_dir) if event_id in f and f.endswith('.mseed')]
    if existing:
        logger.info(f"Event {event_id}: File sudah ada, skip.")
        return True

    # --- 1. Coba GEOFON ---
    client_geofon = Client("GEOFON")
    net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)

    if not net:
        logger.info(f"Event {event_id}: GEOFON radius {MAX_RADIUS_DEG}° gagal, coba {FALLBACK_RADIUS_DEG}°...")
        net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)

    # --- 2. Jika gagal, coba IRIS ---
    client_iris = Client("IRIS")
    if not net:
        logger.info(f"Event {event_id}: GEOFON tidak ditemukan, coba IRIS...")
        net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)
        if not net:
            net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)
    
    # --- 3. Jika tetap gagal ---
    if not net:
        logger.warning(f"Event {event_id}: Tidak ada stasiun dalam radius {FALLBACK_RADIUS_DEG}° (GEOFON/IRIS)")
        return False

    logger.info(f"Event {event_id}: Stasiun {net}.{sta} (jarak {dist:.2f}°)")

    starttime = origin - TIME_BEFORE
    endtime = origin + TIME_AFTER

    # Pilih client terakhir yang berhasil
    if net and client_geofon:
        client = client_geofon
    else:
        client = client_iris

    # --- Unduh data untuk semua channel ---
    stream = None
    for ch in CHANNELS:
        try:
            st = client.get_waveforms(
                network=net,
                station=sta,
                location=LOCATION,
                channel=ch,
                starttime=starttime,
                endtime=endtime
            )
            if st and len(st) > 0:
                if stream is None:
                    stream = st
                else:
                    stream += st
        except Exception as e:
            # Gagal untuk channel ini, lanjutkan
            continue

    if stream is None or len(stream) == 0:
        logger.warning(f"Event {event_id}: Tidak ada data untuk {net}.{sta}")
        return False

    # --- Simpan ---
    filename = f"{net}_{sta}_{event_id}.mseed"
    filepath = os.path.join(output_dir, filename)
    try:
        stream.write(filepath, format="MSEED")
        logger.info(f"Event {event_id}: Berhasil disimpan ke {filename} (trace: {len(stream)})")
        return True
    except Exception as e:
        logger.error(f"Event {event_id}: Gagal menyimpan: {e}")
        return False

# =============================================
# FUNGSI MEMBACA KATALOG
# =============================================

def locate_catalog_file(catalog_path):
    """Cari file katalog di beberapa lokasi."""
    if os.path.exists(catalog_path):
        return catalog_path

    # Cari di direktori script (jika ada)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "filtered_events_selected.csv")
        if os.path.exists(alt_path):
            return alt_path
    except NameError:
        pass  # di Jupyter/REPL

    # Cari di direktori kerja
    cwd_path = os.path.join(os.getcwd(), "filtered_events_selected.csv")
    if os.path.exists(cwd_path):
        return cwd_path

    # Minta input user
    user_input = input("Masukkan path lengkap ke file filtered_events_selected.csv (atau Enter untuk keluar): ").strip()
    if user_input and os.path.exists(user_input):
        return user_input
    else:
        logger.error("File tidak ditemukan. Keluar.")
        sys.exit(1)

def read_catalog(catalog_path):
    """Baca dan validasi katalog."""
    df = pd.read_csv(catalog_path)
    logger.info(f"✅ Total event dalam katalog: {len(df)}")
    logger.info(f"📋 Kolom yang tersedia: {df.columns.tolist()}")

    # Deteksi kolom waktu
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'datetime' in col.lower():
            time_col = col
            break
    if time_col is None:
        logger.error("❌ Kolom waktu tidak ditemukan.")
        sys.exit(1)
    logger.info(f"🕒 Kolom waktu: '{time_col}'")

    # Deteksi kolom latitude
    lat_col = None
    for col in df.columns:
        if 'lat' in col.lower():
            lat_col = col
            break
    if lat_col is None:
        logger.error("❌ Kolom latitude tidak ditemukan.")
        sys.exit(1)
    logger.info(f"🌐 Kolom latitude: '{lat_col}'")

    # Deteksi kolom longitude
    lon_col = None
    for col in df.columns:
        if 'lon' in col.lower():
            lon_col = col
            break
    if lon_col is None:
        logger.error("❌ Kolom longitude tidak ditemukan.")
        sys.exit(1)
    logger.info(f"🌐 Kolom longitude: '{lon_col}'")

    # Konversi datetime
    df['datetime'] = pd.to_datetime(df[time_col], utc=True)
    df = df.dropna(subset=['datetime'])

    return df, lat_col, lon_col

# =============================================
# MAIN
# =============================================

def main():
    logger.info("="*60)
    logger.info("🚀 DOWNLOAD WAVEFORM - PARALLEL (GEOFON/IRIS)")
    logger.info("="*60)

    # 1. Temukan file katalog
    global CATALOG_CSV
    CATALOG_CSV = locate_catalog_file(CATALOG_CSV)
    logger.info(f"📂 Katalog: {CATALOG_CSV}")

    # 2. Baca katalog
    df, lat_col, lon_col = read_catalog(CATALOG_CSV)

    if MAX_EVENTS and len(df) > MAX_EVENTS:
        df = df.head(MAX_EVENTS)
        logger.info(f"⚠️ Hanya {MAX_EVENTS} event pertama yang diproses.")
    else:
        logger.info(f"📦 Total event yang akan diproses: {len(df)}")

    # 3. Siapkan daftar event
    events = []
    for _, row in df.iterrows():
        origin = UTCDateTime(row['datetime'])
        events.append({
            'id': origin.strftime("%Y%m%d_%H%M%S"),
            'time': origin,
            'lat': row[lat_col],
            'lon': row[lon_col]
        })

    logger.info(f"⚡ Parallel workers: {MAX_WORKERS}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")

    # 4. Proses unduhan paralel
    success = 0
    failed = 0
    total = len(events)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_event, ev, OUTPUT_DIR): ev for ev in events}
        with tqdm(total=len(futures), desc="Mengunduh", unit="event") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error pada event: {e}")
                    failed += 1
                pbar.update(1)

    logger.info("="*60)
    logger.info(f"✨ SELESAI! Berhasil: {success}, Gagal: {failed}, Total: {total}")
    logger.info(f"📁 Data disimpan di: {OUTPUT_DIR}")
    logger.info(f"📄 Log tersimpan di: {LOG_FILE}")
    logger.info("="*60)

if __name__ == "__main__":
    main()