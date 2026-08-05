# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Impor dari Library MCU-Quake bawaan (Pastikan utils.py sudah diedit sesuai instruksi sebelumnya)
from Library.model import Embedding_Network, Contrastive_Network, Contrastive_Model
from Library.metrics import wasserstein_1D

# ==========================================
# 1. KONFIGURASI PATH & HYPERPARAMETER
# ==========================================
JSON_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia/extracted_data_3c_sesi_clean.json"
OUTPUT_DIR = "output_models"

INPUT_SIZE = 700
BATCH_SIZE = 64
EPOCHS = 100
CHANNEL = "Z" # Kita mulai retraining dari komponen Vertikal (Z) terlebih dahulu

# Kromosom standar hasil Neural Architecture Search (NAS) MCU-Quake untuk target Mikrokontroler
# (conv_stride, k1, k3, k5, k7, k9, pool_size, pool_stride, mlp1, mlp2, mlp3)
CHROMOSOME = [4, 1, 1, 1, 1, 1, 4, 4, 16, 16, 8] 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. CUSTOM JSON DATA GENERATOR (KERAS SEQUENCE)
# ==========================================
class JSONTripletGenerator(keras.utils.Sequence):
    def __init__(self, json_data, keys, batch_size=32, channel="Z", input_size=700):
        self.json_data = json_data
        self.keys = keys
        self.batch_size = batch_size
        self.channel = channel
        self.input_size = input_size
        self.noise_channel = f"{channel}_noise"

    def __len__(self):
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        # Generate satu batch (Refer, Positive, Negative, Silence)
        batch_refer = np.empty((self.batch_size, self.input_size, 1))
        batch_pos = np.empty((self.batch_size, self.input_size, 1))
        batch_neg = np.empty((self.batch_size, self.input_size, 1))
        batch_sil = np.zeros((self.batch_size, self.input_size, 1)) # Hening (Nol)

        for i in range(self.batch_size):
            # Acak ID untuk triplet
            key_refer = np.random.choice(self.keys)
            key_pos = np.random.choice(self.keys)
            key_neg = np.random.choice(self.keys) # Untuk mengekstrak noise maritim

            # Ekstraksi sinyal 700 sampel
            sig_refer = np.array(self.json_data[key_refer][self.channel][:self.input_size])
            sig_pos = np.array(self.json_data[key_pos][self.channel][:self.input_size])
            # Ambil derau dari ujung belakang window
            sig_neg = np.array(self.json_data[key_neg][self.noise_channel][-self.input_size:])

            batch_refer[i,] = sig_refer.reshape(self.input_size, 1)
            batch_pos[i,] = sig_pos.reshape(self.input_size, 1)
            batch_neg[i,] = sig_neg.reshape(self.input_size, 1)

        # Keras mengharapkan format: X, y (Karena custom training step, y diabaikan/dummy)
        dummy_y = np.zeros(self.batch_size)
        
        # Sesuai input Contrastive_Network: [input_refer, input_pos, input_neg, input_sil]
        return (batch_refer, batch_pos, batch_neg, batch_sil), dummy_y

# ==========================================
# 3. PIPELINE EKSEKUSI UTAMA
# ==========================================
def main():
    print("Membaca file JSON Indonesia...")
    with open(JSON_PATH, "r") as f:
        data_indonesia = json.load(f)
    
    all_keys = list(data_indonesia.keys())
    np.random.shuffle(all_keys)
    
    # Split 80% Training, 20% Validation
    split_idx = int(0.8 * len(all_keys))
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]
    
    print(f"Total Event: {len(all_keys)} | Train: {len(train_keys)} | Val: {len(val_keys)}")

    # Inisialisasi Generator
    train_gen = JSONTripletGenerator(data_indonesia, train_keys, BATCH_SIZE, CHANNEL, INPUT_SIZE)
    val_gen = JSONTripletGenerator(data_indonesia, val_keys, BATCH_SIZE, CHANNEL, INPUT_SIZE)

    print("Membangun Arsitektur Model MCU-Quake...")
    # 1. Bangun Feature Extractor
    embedding_model = Embedding_Network(input_size=INPUT_SIZE, chromosome=CHROMOSOME)
    
    # 2. Bangun Contrastive Network (Pembungkus)
    contrastive_net = Contrastive_Network(input_size=INPUT_SIZE, embedding_model=embedding_model)
    
    # 3. Compile ke Custom Training Model (Menggunakan Wasserstein Distance)
    # Tracker loss
    loss_tracker = keras.metrics.Mean(name="loss")
    # Tracker metrik (Pseudo-Accuracy)
    metric_acc = keras.metrics.BinaryAccuracy(name="acc")
    
    model = Contrastive_Model(
        Contrastive_network=contrastive_net,
        batch_size=BATCH_SIZE,
        loss_tracker=loss_tracker,
        feature_distance="Wasserstein", # Sangat disarankan untuk memisahkan domain noise maritim
        metric_acc=metric_acc,
        margin=1.0,
        alpha=1.0
    )

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)

    # Callbacks
    checkpoint_path = os.path.join(OUTPUT_DIR, f"mcu_quake_indonesia_{CHANNEL}_best.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    print("Memulai Proses Retraining...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Ekstraksi dan simpan HANYA arsitektur Embedding Network (Ini yang akan masuk ke Mikrokontroler)
    final_model_path = os.path.join(OUTPUT_DIR, f"frozen_extractor_indonesia_{CHANNEL}.keras")
    embedding_model.save(final_model_path)
    print(f"\n✅ Retraining Selesai! Model Frozen Extractor tersimpan di: {final_model_path}")

if __name__ == "__main__":
    main()