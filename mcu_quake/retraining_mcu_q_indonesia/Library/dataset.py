# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import json

class Data_Generator:
    def __init__(self, json_data, channel="Z", input_size=700, seed=None):
        """
        json_data: Dictionary hasil dari load_json_data()
        channel: Saluran yang mau di-training ("Z", "N", atau "E")
        input_size: Ukuran input 1D-CNN (default 700)
        """
        self.json_data = json_data
        self.keys_list = list(self.json_data.keys())
        self.channel = channel
        self.input_size = input_size
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def get_next_record(self):
        while True:
            # 1. Pilih ID Gempa Acak untuk Anchor (Refer)
            idx_refer = np.random.choice(len(self.keys_list))
            key_refer = self.keys_list[idx_refer]
            
            # 2. Pilih ID Gempa Acak lain untuk Positive (Harus gempa juga)
            idx_pos = np.random.choice(len(self.keys_list))
            key_pos = self.keys_list[idx_pos]

            # 3. Pilih ID Acak untuk mengambil Noise Lokal (Negative)
            idx_neg = np.random.choice(len(self.keys_list))
            key_neg = self.keys_list[idx_neg]

            # --- EKSTRAKSI & PEMOTONGAN DATA (700 Sampel) ---
            
            # Anchor: Gempa (Diambil dari depan)
            refer_signal = np.array(self.json_data[key_refer][self.channel][:self.input_size])
            
            # Positive: Gempa Lain
            positive_signal = np.array(self.json_data[key_pos][self.channel][:self.input_size])
            
            # Negative: Derau Maritim Indonesia! (Sengaja diambil dari kunci '_noise' dan dari belakang)
            noise_key = f"{self.channel}_noise"
            negative_signal = np.array(self.json_data[key_neg][noise_key][-self.input_size:])
            
            # Silence: Array nol
            silence_signal = np.zeros(self.input_size)

            # --- KONVERSI KE TENSOR KERAS ---
            # Reshape menjadi (700, 1) agar kompatibel dengan layer Conv1D
            refer = tf.convert_to_tensor(refer_signal.reshape(-1, 1), dtype=tf.float32)
            positive = tf.convert_to_tensor(positive_signal.reshape(-1, 1), dtype=tf.float32)
            negative = tf.convert_to_tensor(negative_signal.reshape(-1, 1), dtype=tf.float32)
            refer_silence = tf.convert_to_tensor(silence_signal.reshape(-1, 1), dtype=tf.float32)

            yield (refer, positive, negative, refer_silence)

def load_json_data(file_path, id=None):
    with open(file_path, "r") as read_file:
        dataset = json.load(read_file)
    if id is not None:
        return dataset[str(id)]
    return dataset

def save_json_data(file_path, dataset, indent="\t"):
    with open(file_path, "w") as outfile: 
        json.dump(dataset, outfile, indent=indent)

def load_embedding_data(data_dir, name):
    return load_json_data(os.path.join(data_dir, name))