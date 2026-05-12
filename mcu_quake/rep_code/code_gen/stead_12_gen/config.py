import tensorflow as tf
import os
import sys
from datetime import datetime

# ===================================================================
#                          PATH CONFIGURATIONS
# ===================================================================

# Path folder 'code_gen' (lokasi skrip ini)
GEN_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/code_gen"

# Path folder induk (Root) agar folder 'Library' bisa terdeteksi
ROOT_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
# Folder khusus data demo
DEMO_DATA_PATH = os.path.join(ROOT_PATH, "Data train-demo")
# Tambahkan ROOT_PATH ke sys.path agar 'import Library' bekerja
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# Path Dataset .pkl
TRAIN_DATA_PATH = os.path.join(DEMO_DATA_PATH, "UUSS Train demo dataset.pkl")
VAL_DATA_PATH = os.path.join(DEMO_DATA_PATH, "UUSS Validation demo dataset.pkl")

# Path Output di Extreme SSD
BASE_OUTPUT_PATH = "/Volumes/Extreme SSD/mcu_quake_output"
now = datetime.now()
time_str = now.strftime("%d%H%M")
OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, "output_{}".format(time_str))

# Buat folder output jika belum ada
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# ===================================================================
#                        TRAINING PARAMETERS
# ===================================================================

SEED = 2023
SAMPLING_RATE = 100

# Jendela input 7 detik @ 100Hz = 700 sampel [cite: 716, 721]
INPUT_SIZE = 700 
NUM_CHANNELS = 1 # Fokus pada komponen vertikal (Z) [cite: 699, 700]

# Network settings
MARGIN = 0.5
ALPHA = 1
ACTIVITION = "relu"

# Hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 2
AUTO = tf.data.AUTOTUNE
LEARNING_RATE = 0.001

# Pengaturan untuk demonstrasi awal di PC
EPOCHS = 15 
STEPS_PER_EPOCH = 30 
VALIDATION_STEPS = 5 

# Metrics & Test
FEATURE_DISTANCE = "Wasserstein" # Metrik utama untuk Contrastive Learning [cite: 735, 736]
TEST_BATCH_SIZE = 128
CROP_CONFIDENCE = 0.5