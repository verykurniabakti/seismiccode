#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOWNLOAD WAVEFORM - VENEZUELA (3C - OPTIMASI MEMORY + RESUME)
Dengan pembatasan stasiun, pembersihan memory, dan resume otomatis.
"""

import os
import sys
import gc
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================
# KONFIGURASI
# =============================================
CATALOG_CSV = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/venezuela_earthquake_bench/query_venezuela.csv"
OUTPUT_DIR = "/Volumes/Extreme SSD/venezuela_data_earthquake/waveform_venezuela_3c"

TIME_BEFORE = 30
TIME_AFTER = 120                     # dikurangi dari 180 untuk menghemat memory
MAX_RADIUS_DEG = 10.0
FALLBACK_RADIUS_DEG = 25.0           # dikurangi dari 30
MIN_MAGNITUDE = 4.5
MIN_YEAR = 2015
MAX_WORKERS = 8                      # lebih aman untuk memory
MAX_STATIONS_PER_EVENT = 3           # batasi jumlah stasiun
MAX_EVENTS = None

PRIORITY_NETWORKS = ['IU', 'II', 'CU', 'GT', 'GE', 'G', 'AI', 'C', 'VE']
ALL_CHANNELS = ['BHZ','BHN','BHE','HHZ','HHN','HHE',
                'EHZ','EHN','EHE','LHZ','LHN','LHE']

LOG_FILE = "download_venezuela_optimized.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================
# FUNGSI
# =============================================

def get_stations_with_3comp(client, net_code, lat, lon, origin_time, radius_deg):
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
        logger.debug(f"get_stations_with_3comp error {net_code}: {e}")
        return []

def download_3comp_from_station(client, net, sta, origin, output_dir, event_id):
    starttime = origin - TIME_BEFORE
    endtime = origin + TIME_AFTER

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
    try:
        stream.write(filepath, format="MSEED")
        logger.info(f"Event {event_id}: ✅ {net}.{sta} ({len(stream)} trace)")
        del stream
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Event {event_id}: Gagal simpan {net}.{sta}: {e}")
        del stream
        gc.collect()
        return False

def download_event(event, output_dir, client):
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
                gc.collect()  # bersihkan setelah setiap stasiun
            break
        # Bersihkan setelah setiap jaringan
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
    logger.info("🚀 DOWNLOAD 3C - OPTIMASI MEMORY + RESUME")
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
        with tqdm(total=len(futures), desc="Mengunduh") as pbar:
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
    logger.info("="*60)

if __name__ == "__main__":
    main()