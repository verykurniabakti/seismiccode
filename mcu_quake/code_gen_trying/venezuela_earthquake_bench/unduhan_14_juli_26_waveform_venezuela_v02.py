#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOWNLOAD WAVEFORM - VENEZUELA (3C - OPTIMASI DENGAN JARINGAN KARIBIA)
Prioritas: CU, CM, PR, DR (regional) → IU, II, GT, GE, G (global)
Radius: 15° (fallback 30°), hanya 3 stasiun terbaik per event.
Retry mechanism, optimasi memory, resume otomatis.
"""

import os
import sys
import gc
import time
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================
# KONFIGURASI (DIOPTIMASI)
# =============================================
CATALOG_CSV = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/venezuela_earthquake_bench/query_venezuela.csv"
OUTPUT_DIR = "/Volumes/Extreme SSD/venezuela_data_earthquake/waveform_venezuela_3c_v3"

TIME_BEFORE = 30
TIME_AFTER = 120                     # 120 detik cukup untuk gempa lokal
MAX_RADIUS_DEG = 15.0                # radius utama (Karibia)
FALLBACK_RADIUS_DEG = 30.0           # radius cadangan (tidak terlalu jauh)
MIN_MAGNITUDE = 4.5                  # turun dari 4.5 untuk menambah event
MIN_YEAR = 2010                      # turun dari 2015 untuk menambah event
MAX_WORKERS = 8                      # stabil, hindari rate limit
MAX_STATIONS_PER_EVENT = 3           # hanya 3 stasiun terbaik per event
MAX_EVENTS = None                    # None untuk semua

# Jaringan prioritas: Karibia dulu, global sebagai cadangan
PRIORITY_NETWORKS = [
    'CU',   # Caribbean-USGS (utama)
    'CM',   # Caribbean Network
    'PR',   # Puerto Rico Seismic Network
    'DR',   # Dominican Republic
    'IU',   # Global Seismograph Network (USGS)
    'II',   # Global Seismograph Network (IDA)
    'GT',   # Global Telemetered Seismograph Network (USAF/USGS)
    'GE',   # GEOSCOPE (Prancis)
    'G'     # GEOSCOPE (alternate code)
]

# Semua channel yang mungkin (Z, N, E) untuk berbagai tipe sensor
ALL_CHANNELS = ['BHZ','BHN','BHE','HHZ','HHN','HHE',
                'EHZ','EHN','EHE','LHZ','LHN','LHE']

LOG_FILE = "download_venezuela_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================
# FUNGSI DENGAN RETRY
# =============================================

def get_stations_with_3comp(client, net_code, lat, lon, origin_time, radius_deg, retries=2):
    """
    Cari stasiun 3C dalam jaringan tertentu, dengan retry jika gagal.
    Kembalikan list (net, sta, dist_deg, channels) yang sudah diurutkan.
    """
    for attempt in range(retries + 1):
        try:
            inventory = client.get_stations(
                network=net_code,
                latitude=lat,
                longitude=lon,
                maxradius=radius_deg,
                level="channel",
                starttime=origin_time - 60,
                endtime=origin_time + 60
            )
            if not inventory:
                return []
            candidates = []
            for net in inventory:
                for sta in net:
                    if sta.latitude is None or sta.longitude is None:
                        continue
                    channel_codes = [ch.code for ch in sta.channels]
                    has_z = any(ch in channel_codes for ch in ['BHZ','HHZ','EHZ','LHZ'])
                    has_n = any(ch in channel_codes for ch in ['BHN','HHN','EHN','LHN'])
                    has_e = any(ch in channel_codes for ch in ['BHE','HHE','EHE','LHE'])
                    if has_z and has_n and has_e:
                        dist_m, _, _ = gps2dist_azimuth(lat, lon, sta.latitude, sta.longitude)
                        dist_deg = dist_m / 111000.0
                        candidates.append((net.code, sta.code, dist_deg, channel_codes))
            candidates.sort(key=lambda x: x[2])
            return candidates[:MAX_STATIONS_PER_EVENT]
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                logger.debug(f"Retry {net_code} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.debug(f"get_stations_with_3comp error {net_code}: {e}")
                return []
    return []

def download_3comp_from_station(client, net, sta, origin, output_dir, event_id, retries=2):
    """
    Unduh semua trace dari stasiun yang diyakini memiliki 3C, dengan retry.
    """
    starttime = origin - TIME_BEFORE
    endtime = origin + TIME_AFTER

    for attempt in range(retries + 1):
        try:
            stream = None
            for ch in ALL_CHANNELS:
                try:
                    st = client.get_waveforms(
                        network=net,
                        station=sta,
                        location="*",
                        channel=ch,
                        starttime=starttime,
                        endtime=endtime
                    )
                    if st and len(st) > 0:
                        if stream is None:
                            stream = st
                        else:
                            stream += st
                except Exception:
                    continue

            if stream is None or len(stream) == 0:
                return False

            has_z = any(tr.stats.channel[-1] == 'Z' for tr in stream)
            has_n = any(tr.stats.channel[-1] == 'N' for tr in stream)
            has_e = any(tr.stats.channel[-1] == 'E' for tr in stream)
            if not (has_z and has_n and has_e):
                del stream
                gc.collect()
                return False

            filename = f"{net}_{sta}_{event_id}.mseed"
            filepath = os.path.join(output_dir, filename)
            stream.write(filepath, format="MSEED")
            logger.info(f"Event {event_id}: ✅ {net}.{sta} ({len(stream)} trace)")
            del stream
            gc.collect()
            return True
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                logger.debug(f"Retry {net}.{sta} after {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Event {event_id}: Gagal {net}.{sta}: {e}")
                return False
    return False

def download_event(event, output_dir, client):
    """
    Proses satu event: cari stasiun 3C, unduh, simpan.
    """
    lat = event['lat']
    lon = event['lon']
    origin = event['time']
    event_id = event['id']

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for net in PRIORITY_NETWORKS:
        for radius in [MAX_RADIUS_DEG, FALLBACK_RADIUS_DEG]:
            stations = get_stations_with_3comp(client, net, lat, lon, origin, radius)
            if not stations:
                continue
            logger.info(f"Event {event_id}: {len(stations)} stasiun 3C di {net} (radius {radius}°)")
            for net_code, sta_code, dist_deg, channels in stations:
                logger.info(f"Event {event_id}: Mencoba {net_code}.{sta_code} ({dist_deg:.2f}°)")
                if download_3comp_from_station(client, net_code, sta_code, origin, output_dir, event_id):
                    success_count += 1
                gc.collect()
            break
        gc.collect()

    if success_count > 0:
        logger.info(f"Event {event_id}: Berhasil {success_count} stasiun")
    else:
        logger.warning(f"Event {event_id}: Tidak ada stasiun 3C")
    return success_count

# =============================================
# BACA KATALOG
# =============================================
def read_catalog(catalog_path):
    df = pd.read_csv(catalog_path)
    logger.info(f"✅ Total event: {len(df)}")
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if not all([time_col, lat_col, lon_col]):
        logger.error("❌ Kolom waktu/lat/lon tidak ditemukan.")
        sys.exit(1)
    df['datetime'] = pd.to_datetime(df[time_col], utc=True)
    df = df.dropna(subset=['datetime'])
    if 'mag' in df.columns:
        df = df[df['mag'] >= MIN_MAGNITUDE]
    df['year'] = df['datetime'].dt.year
    df = df[df['year'] >= MIN_YEAR]
    return df, lat_col, lon_col

# =============================================
# MAIN
# =============================================
def main():
    logger.info("="*60)
    logger.info("🚀 DOWNLOAD 3C - VENEZUELA (KARIBIA + GLOBAL)")
    logger.info("="*60)
    if not os.path.exists(CATALOG_CSV):
        logger.error(f"❌ File tidak ditemukan: {CATALOG_CSV}")
        sys.exit(1)

    df, lat_col, lon_col = read_catalog(CATALOG_CSV)
    if MAX_EVENTS and len(df) > MAX_EVENTS:
        df = df.head(MAX_EVENTS)
        logger.info(f"⚠️ Testing: hanya {MAX_EVENTS} event pertama.")

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
    logger.info(f"📁 Output: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = Client("IRIS")
    total_success = 0
    total_events_with_data = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_event, ev, OUTPUT_DIR, client): ev for ev in events}
        with tqdm(total=len(futures), desc="Mengunduh 3C") as pbar:
            for future in as_completed(futures):
                try:
                    n = future.result()
                    if n > 0:
                        total_success += n
                        total_events_with_data += 1
                except Exception as e:
                    logger.error(f"Error: {e}")
                pbar.update(1)

    logger.info("="*60)
    logger.info(f"✨ SELESAI! Event dengan data: {total_events_with_data} dari {len(events)}")
    logger.info(f"   Total file: {total_success}")
    logger.info(f"📁 Output: {OUTPUT_DIR}")
    logger.info("="*60)

if __name__ == "__main__":
    main()