#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOWNLOAD WAVEFORM 3 KOMPONEN (Z, N, E) - OPSI B (TAHUN ≥ 2010)
Dengan tambahan 6 gempa merusak 2004-2009
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================
# 1. KONFIGURASI
# =============================================

# Path katalog input (sesuaikan jika perlu)
CATALOG_PATH = "/Volumes/Extreme SSD/katalog/hybrid_catalog_filtered.csv"
OUTPUT_DIR = "/Volumes/Extreme SSD/unduhan_juli_hybrid_katalog_2026"

# Parameter unduhan
TIME_BEFORE = 30
TIME_AFTER = 120
MAX_RADIUS_DEG = 10.0
FALLBACK_RADIUS_DEG = 20.0

# CHANNELS 3 KOMPONEN (Z, N, E)
CHANNELS = ['BHZ','BHN','BHE','HHZ','HHN','HHE','EHZ','EHN','EHE']
LOCATION = "*"

# Sampling: ambil N event per tahun (tahun ≥ 2010)
EVENTS_PER_YEAR = 1500

# Parallel
MAX_WORKERS = 6

# Logging
LOG_FILE = "download_waveform_3c.log"

# =============================================
# 2. DAFTAR GEMPA MERUSAK 2004-2009
# =============================================

MAJOR_EVENTS_2004_2009 = [
    {"year": 2004, "month": 12, "day": 26, "lat": 3.295, "lon": 95.982, "mag": 9.1, "name": "Aceh_2004"},
    {"year": 2005, "month": 3, "day": 28, "lat": 2.085, "lon": 97.108, "mag": 8.6, "name": "Nias_2005"},
    {"year": 2006, "month": 5, "day": 27, "lat": -7.961, "lon": 110.446, "mag": 6.3, "name": "Yogyakarta_2006"},
    {"year": 2006, "month": 7, "day": 17, "lat": -9.284, "lon": 107.419, "mag": 7.7, "name": "Pangandaran_2006"},
    {"year": 2007, "month": 9, "day": 12, "lat": -4.438, "lon": 101.367, "mag": 8.5, "name": "Bengkulu_2007"},
    {"year": 2009, "month": 9, "day": 30, "lat": -0.800, "lon": 99.880, "mag": 7.6, "name": "Padang_2009"},
]

# =============================================
# 3. SETUP LOGGING
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
# 4. BACA KATALOG & STRATIFIED SAMPLING
# =============================================

def read_and_sample_catalog(catalog_path, events_per_year=1500, min_year=2010):
    """Baca katalog dan ambil sampel terstratifikasi per tahun."""
    df = pd.read_csv(catalog_path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, format='mixed')
    df['year'] = df['datetime'].dt.year
    
    df_filtered = df[df['year'] >= min_year].copy()
    logger.info(f"✅ Event tahun ≥ {min_year}: {len(df_filtered):,}")
    
    sampled = []
    years = sorted(df_filtered['year'].unique())
    
    for year in years:
        subset = df_filtered[df_filtered['year'] == year]
        n = min(events_per_year, len(subset))
        if n > 0:
            sampled_subset = subset.sample(n=n, random_state=42)
            sampled.append(sampled_subset)
            logger.info(f"  Tahun {year}: {len(subset):,} event → ambil {n}")
    
    df_sampled = pd.concat(sampled).reset_index(drop=True)
    logger.info(f"✅ Total event setelah stratified sampling: {len(df_sampled):,}")
    
    return df_sampled

# =============================================
# 5. TAMBAHKAN GEMPA MERUSAK
# =============================================

def add_major_events(major_events):
    events = []
    for ev in major_events:
        dt = pd.Timestamp(f"{ev['year']}-{ev['month']:02d}-{ev['day']:02d}", tz='UTC')
        dt = dt.replace(hour=12, minute=0, second=0)
        events.append({
            'datetime': dt,
            'latitude': ev['lat'],
            'longitude': ev['lon'],
            'magnitude': ev['mag'],
            'depth_km': 30.0,
            'source': 'MAJOR',
            'event_id': ev['name'],
            'year': ev['year'],
            'is_major': True,
            'name': ev['name']
        })
        logger.info(f"  ✅ Ditambahkan: {ev['name']} (M{ev['mag']:.1f})")
    
    return pd.DataFrame(events)

# =============================================
# 6. CACHE STASIUN
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
            latitude=lat, longitude=lon,
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

# =============================================
# 7. DOWNLOAD SATU EVENT
# =============================================

def download_event(event, output_dir):
    lat = event['lat']
    lon = event['lon']
    origin = event['time']
    event_id = event['id']

    os.makedirs(output_dir, exist_ok=True)

    # Resume
    existing = [f for f in os.listdir(output_dir) if event_id in f and f.endswith('.mseed')]
    if existing:
        filepath = os.path.join(output_dir, existing[0])
        if os.path.getsize(filepath) >= 1024:
            return True
        else:
            os.remove(filepath)

    # Cari stasiun
    client_geofon = Client("GEOFON")
    net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)

    if not net:
        logger.info(f"Event {event_id}: GEOFON radius {MAX_RADIUS_DEG}° gagal, coba {FALLBACK_RADIUS_DEG}°...")
        net, sta, dist = find_station(client_geofon, "GEOFON", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)

    client_iris = Client("IRIS")
    if not net:
        logger.info(f"Event {event_id}: GEOFON tidak ditemukan, coba IRIS...")
        net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, MAX_RADIUS_DEG, CHANNELS)
        if not net:
            net, sta, dist = find_station(client_iris, "IRIS", lat, lon, origin, FALLBACK_RADIUS_DEG, CHANNELS)

    if not net:
        logger.warning(f"Event {event_id}: Tidak ada stasiun dalam radius {FALLBACK_RADIUS_DEG}°")
        return False

    logger.info(f"Event {event_id}: Stasiun {net}.{sta} (jarak {dist:.2f}°)")

    starttime = origin - TIME_BEFORE
    endtime = origin + TIME_AFTER
    client = client_geofon if net and client_geofon else client_iris

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
        logger.warning(f"Event {event_id}: Tidak ada data untuk {net}.{sta}")
        return False

    filename = f"{net}_{sta}_{event_id}.mseed"
    filepath = os.path.join(output_dir, filename)
    try:
        stream.write(filepath, format="MSEED")
        logger.info(f"Event {event_id}: Berhasil disimpan ({len(stream)} trace)")
        return True
    except Exception as e:
        logger.error(f"Event {event_id}: Gagal menyimpan: {e}")
        return False

# =============================================
# 8. MAIN
# =============================================

def main():
    logger.info("="*70)
    logger.info("🚀 DOWNLOAD WAVEFORM 3 KOMPONEN (Z, N, E) - OPSI B")
    logger.info("="*70)
    logger.info(f"Radius: {MAX_RADIUS_DEG}° (fallback {FALLBACK_RADIUS_DEG}°)")
    logger.info(f"Parallel workers: {MAX_WORKERS}")
    logger.info(f"Events per year: {EVENTS_PER_YEAR}")

    df_sampled = read_and_sample_catalog(CATALOG_PATH, EVENTS_PER_YEAR, min_year=2010)
    df_major = add_major_events(MAJOR_EVENTS_2004_2009)
    df_all = pd.concat([df_sampled, df_major], ignore_index=True)
    logger.info(f"✅ Total event setelah penambahan major: {len(df_all):,}")

    events = []
    for _, row in df_all.iterrows():
        origin = UTCDateTime(row['datetime'])
        events.append({
            'id': origin.strftime("%Y%m%d_%H%M%S"),
            'time': origin,
            'lat': row['latitude'],
            'lon': row['longitude'],
            'is_major': row.get('is_major', False),
            'name': row.get('name', '')
        })

    logger.info(f"📦 Total event akan diproses: {len(events)}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_event, ev, OUTPUT_DIR): ev for ev in events}
        with tqdm(total=len(futures), desc="Mengunduh", unit="event") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    logger.info("="*70)
    logger.info(f"✨ SELESAI! Berhasil: {success}, Gagal: {failed}, Total: {len(events)}")
    logger.info(f"📁 Data disimpan di: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()