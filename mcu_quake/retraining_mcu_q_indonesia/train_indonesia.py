# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Impor dari Library MCU-Quake bawaan
from Library.model import Embedding_Network, Contrastive_Network, Contrastive_Model
from Library.metrics import wasserstein_1D

# ==========================================
# 1. KONFIGURASI PATH & HYPERPARAMETER
# ==========================================
BASE_DATA_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia"
MASTER_JSON_PATH = os.path.join(BASE_DATA_DIR, "extracted_data_3c_sesi_clean.json")

# Dua file fisik baru hasil belahan
TRAIN_JSON_PATH = os.path.join(BASE_DATA_DIR, "indonesia_train_data.json")
TEST_JSON_PATH = os.path.join(BASE_DATA_DIR, "indonesia_test_data.json")

OUTPUT_DIR = "output_models"

INPUT_SIZE = 700
BATCH_SIZE = 64
EPOCHS = 100
CHANNEL = "Z" 

CHROMOSOME = [4, 1, 1, 1, 1, 1, 4, 4, 16, 16, 8] 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. CUSTOM JSON DATA GENERATOR (DIKOREKSI)
# ==========================================
class JSONTripletGenerator(keras.utils.Sequence):
    def __init__(self, json_data, keys, batch_size=32, channel="Z", input_size=700):
        self.json_data = json_data
        self.keys = np.array(keys) # Pastikan format array untuk slicing
        self.batch_size = batch_size
        self.channel = channel
        self.input_size = input_size
        self.noise_channel = f"{channel}_noise"
        self.on_epoch_end() # Inisialisasi pengacakan pertama

    def __len__(self):
        return int(np.floor(len(self.keys) / self.batch_size))

    def on_epoch_end(self):
        # Mengacak urutan data SETIAP KALI epoch selesai
        np.random.shuffle(self.keys)

    def __getitem__(self, index):
        batch_refer = np.empty((self.batch_size, self.input_size, 1))
        batch_pos = np.empty((self.batch_size, self.input_size, 1))
        batch_neg = np.empty((self.batch_size, self.input_size, 1))
        batch_sil = np.zeros((self.batch_size, self.input_size, 1)) 

        # Mengambil subset kunci berdasarkan index batch
        batch_keys = self.keys[index * self.batch_size : (index + 1) * self.batch_size]

        for i, key_refer in enumerate(batch_keys):
            # Memastikan key_pos BERBEDA dengan key_refer
            available_pos_keys = self.keys[self.keys != key_refer]
            key_pos = np.random.choice(available_pos_keys)
            
            # Negatif bisa diambil dari mana saja
            key_neg = np.random.choice(self.keys)

            sig_refer = np.array(self.json_data[key_refer][self.channel][:self.input_size])
            sig_pos = np.array(self.json_data[key_pos][self.channel][:self.input_size])
            sig_neg = np.array(self.json_data[key_neg][self.noise_channel][-self.input_size:])

            batch_refer[i,] = sig_refer.reshape(self.input_size, 1)
            batch_pos[i,] = sig_pos.reshape(self.input_size, 1)
            batch_neg[i,] = sig_neg.reshape(self.input_size, 1)

        dummy_y = np.zeros(self.batch_size)
        return (batch_refer, batch_pos, batch_neg, batch_sil), dummy_y

# ==========================================
# 3. PIPELINE EKSEKUSI UTAMA
# ==========================================
def main():
    
    # --- BLOK PEMISAHAN FISIK (STRICT DATA SPLIT) ---
    if not os.path.exists(TRAIN_JSON_PATH) or not os.path.exists(TEST_JSON_PATH):
        print("Membelah dataset master menjadi Train dan Test fisik (meniru metodologi Zhi Geng)...")
        with open(MASTER_JSON_PATH, "r") as f:
            master_data = json.load(f)
            
        all_keys = list(master_data.keys())
        all_keys.sort() 
        np.random.seed(42) # Kunci seed agar split selalu deterministik
        np.random.shuffle(all_keys)
        
        # Partisi: 80% Train/KDE, 20% Blind Test
        split_point = int(0.8 * len(all_keys))
        train_keys_master = all_keys[:split_point]
        test_keys_master = all_keys[split_point:]
        
        train_data = {k: master_data[k] for k in train_keys_master}
        test_data = {k: master_data[k] for k in test_keys_master}
        
        with open(TRAIN_JSON_PATH, "w") as f:
            json.dump(train_data, f)
        with open(TEST_JSON_PATH, "w") as f:
            json.dump(test_data, f)
            
        print(f"✅ Split Selesai! Train: {len(train_keys_master)} event | Test: {len(test_keys_master)} event")
        # Bebaskan memori dari master data
        del master_data, train_data, test_data
    # ------------------------------------------------

    # Muat data_train_only dari TRAIN_JSON_PATH (dipakai generator train & val di bawah).
    # File ini sudah pasti ada di titik ini, baik dari blok split di atas maupun dari run sebelumnya.
    with open(TRAIN_JSON_PATH, "r") as f:
        data_train_only = json.load(f)
    train_keys = list(data_train_only.keys())

    # Split internal (dari file Train) untuk Validasi Keras saat Epoch berjalan
    val_split_idx = int(0.8 * len(train_keys))
    final_train_keys = train_keys[:val_split_idx]
    final_val_keys = train_keys[val_split_idx:]
    
    print(f"Total Data Latih Utama: {len(train_keys)} | Eksekusi Train: {len(final_train_keys)} | Eksekusi Val: {len(final_val_keys)}")

    # Inisialisasi Generator HANYA menggunakan data_train_only
    train_gen = JSONTripletGenerator(data_train_only, final_train_keys, BATCH_SIZE, CHANNEL, INPUT_SIZE)
    val_gen = JSONTripletGenerator(data_train_only, final_val_keys, BATCH_SIZE, CHANNEL, INPUT_SIZE)

    print("Membangun Arsitektur Model MCU-Quake...")
    embedding_model = Embedding_Network(input_size=INPUT_SIZE, chromosome=CHROMOSOME)
    contrastive_net = Contrastive_Network(input_size=INPUT_SIZE, embedding_model=embedding_model)
    
    loss_tracker = keras.metrics.Mean(name="loss")
    metric_acc = keras.metrics.BinaryAccuracy(name="acc")
    
    model = Contrastive_Model(
        Contrastive_network=contrastive_net,
        batch_size=BATCH_SIZE,
        loss_tracker=loss_tracker,
        feature_distance="Wasserstein", 
        metric_acc=metric_acc,
        margin=1.0,
        alpha=1.0
    )

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)

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

    final_model_path = os.path.join(OUTPUT_DIR, f"frozen_extractor_indonesia_{CHANNEL}.keras")
    embedding_model.save(final_model_path)
    print(f"\n✅ Retraining Selesai! Model Frozen Extractor tersimpan di: {final_model_path}")

if __name__ == "__main__":
    main()