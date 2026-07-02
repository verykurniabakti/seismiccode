#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOWNLOAD WAVEFORM UNTUK GEMPA-GEMPA MERUSAK INDONESIA (FIXED)
Menangani error timezone pada perbandingan datetime.
"""

import os
import sys
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =============================================
# 1. KONFIGURASI
# =============================================

CATALOG_CSV = '/Volumes/Extreme SSD/unduhan_juni_bmkg_usgs/hasilscan/eda_final_output/HYBRID_EARTHQUAKE_CATALOG_2001_2024_FIX.csv'
OUTPUT_DIR = "/Volumes/Extreme SSD/unduhan_waveform_major_earthquakes"

TIME_BEFORE = 60          # detik sebelum origin
TIME_AFTER = 300          # detik setelah origin
MAX_RADIUS_DEG = 10.0
FALLBACK_RADIUS_DEG = 20.0
CHANNELS = ['BHZ','BHN','BHE','HHZ','HHN','HHE']
LOCATION = "*"
MAX_WORKERS = 4
LOG_FILE = "download_major.log"

# =============================================
# 2. DAFTAR GEMPA MERUSAK
# =============================================

MAJOR_EARTHQUAKES = [
    (2004, 12, 26, 9.1, "Aceh", "Tsunami Aceh 2004"),
    (2005, 3, 28, 8.6, "Nias", "Gempa Nias 2005"),
    (2006, 5, 27, 6.3, "Yogyakarta", "Gempa Yogyakarta 2006"),
    (2006, 7, 17, 7.7, "Pangandaran", "Tsunami Pangandaran 2006"),
    (2007, 9, 12, 8.5, "Bengkulu", "Gempa Bengkulu 2007"),
    (2009, 9, 30, 7.6, "Padang", "Gempa Padang 2009"),
    (2010, 4, 6, 7.7, "Sumatra", "Gempa Sumatra 2010"),
    (2010, 10, 25, 7.7, "Mentawai", "Tsunami Mentawai 2010"),
    (2012, 4, 11, 8.6, "Aceh", "Gempa Aceh 2012"),
    (2013, 4, 6, 7.2, "Papua", "Gempa Papua 2013"),
    (2016, 12, 7, 6.5, "Pidie Jaya", "Gempa Pidie Jaya 2016"),
    (2018, 8, 5, 7.0, "Lombok", "Gempa Lombok 2018"),
    (2018, 9, 28, 7.5, "Palu", "Tsunami Palu 2018"),
    (2019, 7, 14, 7.3, "Maluku", "Gempa Maluku 2019"),
    (2021, 1, 14, 7.0, "Mamuju", "Gempa Mamuju 2021"),
    (2022, 11, 21, 5.6, "Cianjur", "Gempa Cianjur 2022"),
    (2023, 1, 18, 7.2, "Maluku", "Gempa Maluku 2023"),
    (2023, 11, 8, 7.6, "Maluku", "Gempa Maluku 2023"),
    (2024, 1, 8, 7.0, "Maluku", "Gempa Maluku 2024"),
]

# =============================================
# 3. SETUP LOGGING
# =============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================
# 4. BACA KATALOG & COCOKKAN EVENT (FIXED)
# =============================================

def find_major_events(catalog_path, major_list, tolerance_days=2):
    """
    Cari event di katalog yang cocok dengan daftar gempa merusak.
    FIX: Menggunakan timezone-aware Timestamp.
    """
    df = pd.read_csv(catalog_path)
    # Parse datetime dengan timezone UTC
    df['datetime'] = pd.to_datetime(df['time_utc'], utc=True)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    
    found_events = []
    
    for year, month, day, expected_mag, location, desc in major_list:
        # Buat Timestamp dengan timezone UTC (tz-aware)
        start = pd.Timestamp(f"{year}-{month:02d}-{day:02d}", tz='UTC') - pd.Timedelta(days=tolerance_days)
        end = pd.Timestamp(f"{year}-{month:02d}-{day:02d}", tz='UTC') + pd.Timedelta(days=tolerance_days)
        
        mask = (df['datetime'] >= start) & (df['datetime'] <= end)
        candidates = df[mask].sort_values('magnitude', ascending=False)
        
        if len(candidates) > 0:
            best = candidates.iloc[0]
            found_events.append({
                'datetime': best['datetime'],
                'latitude': best['latitude'],
                'longitude': best['longitude'],
                'magnitude': best['magnitude'],
                'depth': best['depth_km'],
                'source': best['source'],
                'event_id': best.get('event_id', 'unknown'),
                'description': desc,
                'expected_mag': expected_mag,
                'location': location,
                'matched': True
            })
            logger.info(f"✅ Ditemukan: {desc} (M{best['magnitude']:.1f}) pada {best['datetime']}")
        else:
            logger.warning(f"❌ Tidak ditemukan: {desc} (M{expected_mag:.1f})")
            found_events.append({
                'datetime': None,
                'latitude': None,
                'longitude': None,
                'magnitude': None,
                'depth': None,
                'source': None,
                'event_id': None,
                'description': desc,
                'expected_mag': expected_mag,
                'location': location,
                'matched': False
            })
    
    # Filter hanya yang ditemukan
    matched = [e for e in found_events if e['matched']]
    logger.info(f"\n📊 Total gempa merusak ditemukan: {len(matched)} dari {len(major_list)}")
    
    return matched

# =============================================
# 5. FUNGSI DOWNLOAD
# =============================================

station_cache = {}

def cache_key(client_name, lat, lon, radius_deg, year):
    return f"{client_name}_{round(lat,2)}_{round(lon,2)}_{radius_deg}_{year}"

def find_station(client, client_name, lat, lon, origin_time, radius_deg, channels):
    year = origin_time.year
    key = cache_key(client_name, lat, lon, radius_deg, year)
    if key in station_cache:
        return station_cache[key]
    
    try:
        inventory = client.get_stations(
            latitude=lat, longitude=lon, maxradius=radius_deg,
            level="channel", channel=",".join(channels), location=LOCATION,
            starttime=origin_time - 60, endtime=origin_time + 60
        )
        if not inventory:
            station_cache[key] = (None, None, None)
            return None, None, None

        best_dist = float('inf')
        best_net = best_sta = None
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
        result = (best_net, best_sta, best_dist) if best_net else (None, None, None)
        station_cache[key] = result
        return result
    except Exception as e:
        logger.debug(f"Error finding station: {e}")
        station_cache[key] = (None, None, None)
        return None, None, None

def download_event(event_data, output_dir):
    """Download waveform untuk satu event."""
    if not event_data['matched']:
        return False
    
    lat = event_data['latitude']
    lon = event_data['longitude']
    origin = UTCDateTime(event_data['datetime'])
    event_id = event_data['event_id']
    description = event_data['description']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Cek file existing
    existing = [f for f in os.listdir(output_dir) if str(event_id) in f and f.endswith('.mseed')]
    if existing:
        logger.info(f"{description}: File sudah ada, skip.")
        return True
    
    # Cari stasiun
    client_geofon = Client("GEOFON")
    net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)
    
    if not net:
        logger.info(f"{description}: GEOFON radius {MAX_RADIUS_DEG}° gagal, coba {FALLBACK_RADIUS_DEG}°...")
        net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)
    
    client_iris = Client("IRIS")
    if not net:
        logger.info(f"{description}: GEOFON tidak ditemukan, coba IRIS...")
        net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)
        if not net:
            net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)
    
    if not net:
        logger.warning(f"{description}: Tidak ada stasiun dalam radius {FALLBACK_RADIUS_DEG}°")
        return False
    
    logger.info(f"{description}: Stasiun {net}.{sta} (jarak {dist:.2f}°)")
    
    starttime = origin - TIME_BEFORE
    endtime = origin + TIME_AFTER
    client = client_geofon if net and client_geofon else client_iris
    
    # Unduh data
    stream = None
    for ch in CHANNELS:
        try:
            st = client.get_waveforms(
                network=net, station=sta, location=LOCATION,
                channel=ch, starttime=starttime, endtime=endtime
            )
            if st and len(st) > 0:
                stream = st if stream is None else stream + st
        except Exception:
            continue
    
    if stream is None or len(stream) == 0:
        logger.warning(f"{description}: Tidak ada data untuk {net}.{sta}")
        return False
    
    # Simpan dengan nama deskriptif
    filename = f"{description.replace(' ', '_')}_{net}_{sta}_{event_id}.mseed"
    filepath = os.path.join(output_dir, filename)
    try:
        stream.write(filepath, format="MSEED")
        logger.info(f"{description}: Berhasil disimpan ke {filename} ({len(stream)} trace)")
        return True
    except Exception as e:
        logger.error(f"{description}: Gagal menyimpan: {e}")
        return False

# =============================================
# 6. MAIN
# =============================================

def main():
    logger.info("="*70)
    logger.info("🌋 DOWNLOAD WAVEFORM GEMPA MERUSAK INDONESIA")
    logger.info("="*70)
    
    # 1. Cari event di katalog
    logger.info("\n📂 Mencari event di katalog...")
    events = find_major_events(CATALOG_CSV, MAJOR_EARTHQUAKES)
    
    if len(events) == 0:
        logger.error("❌ Tidak ada event yang ditemukan!")
        sys.exit(1)
    
    logger.info(f"\n📦 {len(events)} event akan diunduh.")
    
    # 2. Tampilkan daftar
    print("\n📋 DAFTAR EVENT YANG AKAN DIUNDUH:")
    print("="*70)
    for e in events:
        mag = e['magnitude']
        desc = e['description']
        loc = e['location']
        date = e['datetime'].strftime('%Y-%m-%d %H:%M') if e['datetime'] else 'N/A'
        print(f"  {desc:30} | M{mag:.1f} | {date}")
    print("="*70)
    
    # 3. Download
    logger.info(f"\n🚀 Memulai unduhan {len(events)} event dengan {MAX_WORKERS} worker...")
    
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_event, e, OUTPUT_DIR): e for e in events}
        with tqdm(total=len(futures), desc="Mengunduh", unit="event") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
                pbar.update(1)
    
    logger.info("\n" + "="*70)
    logger.info(f"✨ SELESAI! Berhasil: {success}, Gagal: {failed}, Total: {len(events)}")
    logger.info(f"📁 Data disimpan di: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()