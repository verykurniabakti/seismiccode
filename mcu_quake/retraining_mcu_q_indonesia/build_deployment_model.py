# -*- coding: utf-8 -*-
"""
Satu file, jalankan dari atas ke bawah, tidak ada sel notebook yang perlu diurutkan.

Yang dilakukan skrip ini, berurutan:
  1. Memuat SavedModel ASLI MCU-Quake (Zhi Geng) dan memotongnya di layer
     32-dimensi (logika identik evaluate_kde_re_embedding_all.py), supaya
     embedding yang dihasilkan cocok dengan vektor KDE 32-dimensi yang sudah ada.
  2. Konversi model yang sudah dipotong itu ke TFLite INT8 penuh.
  3. Ekspor model TFLite tsb ke mcu_quake_model.h (C-array untuk ESP32-S3).
  4. Ekspor vektor referensi KDE (Embedding data, Z.json) ke kde_z_vectors.h,
     dikuantisasi INT8 + skala (bukan float, supaya hemat 4x flash/RAM).

Jalankan:
  python build_deployment_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf

# ============================================================
# KONFIGURASI PATH -- sesuaikan di sini kalau lokasi file berubah
# ============================================================

# SavedModel asli Zhi Geng (folder ini juga berisi lite_model.tflite asli)
MODEL_PRETRAINED_PATH = (
    "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/"
    "indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/"
    "Code & Figure demo/Pre-trained model/MCU-Quake 5-20"
)

# Data mentah untuk kalibrasi kuantisasi INT8
REPRESENTATIVE_DATA_PATH = (
    "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/"
    "data_indonesia/indonesia_test_data.json"
)
NUM_CALIBRATION_SAMPLES = 100

# Vektor referensi KDE 32-dimensi yang sudah ada (hasil skenario "kde_reembed")
KDE_EMBEDDING_JSON = (
    "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/"
    "output_eval_03/Embedding data, Z.json"
)

# Semua output ditulis ke sini
OUTPUT_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/output_models"
TFLITE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mcu_quake_original_32d.tflite")
MODEL_HEADER_PATH = os.path.join(OUTPUT_DIR, "mcu_quake_model.h")
KDE_HEADER_PATH = os.path.join(OUTPUT_DIR, "kde_z_vectors.h")


# ============================================================
# LANGKAH 1+2: Potong model asli di layer 32-dimensi, konversi ke TFLite INT8
# ============================================================

def truncate_at_32d(full_model):
    """Potong model di layer pertama yang output-nya persis 32 dimensi."""
    for layer in full_model.layers:
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and shape[-1] == 32:
            return tf.keras.Model(inputs=full_model.inputs, outputs=layer.output)
    raise ValueError("Tidak ditemukan layer dengan output 32 dimensi di model ini.")


def make_representative_dataset_gen(input_shape):
    with open(REPRESENTATIVE_DATA_PATH, "r") as f:
        raw_data = json.load(f)

    keys = list(raw_data.keys())
    sample_count = min(NUM_CALIBRATION_SAMPLES, len(keys))

    def representative_dataset_gen():
        for k in keys[:sample_count]:
            wave = np.array(raw_data[k]["Z"][:700], dtype=np.float32)
            wave = wave.reshape([1] + list(input_shape[1:]))
            yield [wave]

    return representative_dataset_gen


def build_tflite_model():
    print("=" * 60)
    print("[1/4] Memuat & memotong SavedModel asli di layer 32-dimensi")
    print("=" * 60)
    print(f"    Sumber: {MODEL_PRETRAINED_PATH}")

    full_model = tf.keras.models.load_model(MODEL_PRETRAINED_PATH, compile=False)
    truncated_model = truncate_at_32d(full_model)
    print(f"    Output model setelah dipotong: {truncated_model.output_shape}")

    print("\n" + "=" * 60)
    print("[2/4] Konversi ke TFLite INT8 penuh")
    print("=" * 60)

    converter = tf.lite.TFLiteConverter.from_keras_model(truncated_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset_gen(truncated_model.input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"    ✅ Tersimpan: {TFLITE_OUTPUT_PATH} ({len(tflite_model) / 1024:.2f} KB)")
    return tflite_model


# ============================================================
# LANGKAH 3: Ekspor TFLite -> mcu_quake_model.h (C-array)
# ============================================================

def export_model_header(tflite_data):
    print("\n" + "=" * 60)
    print("[3/4] Ekspor model ke mcu_quake_model.h")
    print("=" * 60)

    hex_array = [f"0x{b:02x}" for b in tflite_data]
    c_code = "#ifndef MCU_QUAKE_MODEL_H\n#define MCU_QUAKE_MODEL_H\n\n"
    c_code += "// File TFLite Feature Extractor MCU-Quake (model asli, dipotong 32D)\n"
    c_code += f"const unsigned int mcu_quake_model_len = {len(tflite_data)};\n"
    c_code += "const unsigned char mcu_quake_model[] = {\n    "

    for i in range(0, len(hex_array), 12):
        c_code += ", ".join(hex_array[i:i + 12]) + ",\n    "
    c_code = c_code.rstrip(",\n    ") + "\n};\n\n#endif // MCU_QUAKE_MODEL_H"

    with open(MODEL_HEADER_PATH, "w") as f:
        f.write(c_code)
    print(f"    ✅ Tersimpan: {MODEL_HEADER_PATH} ({len(tflite_data) / 1024:.2f} KB)")


# ============================================================
# LANGKAH 4: Ekspor vektor referensi KDE -> kde_z_vectors.h (INT8 + skala)
# ============================================================

def quantize_int8(vectors):
    """Kuantisasi simetris INT8 (per-array, skala tunggal)."""
    arr = np.array(vectors, dtype=np.float32)
    max_abs = np.max(np.abs(arr)) if arr.size > 0 else 1.0
    scale = (max_abs / 127.0) if max_abs > 0 else 1.0
    quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return quantized, scale


def format_kde_array(name, vectors):
    quantized, scale = quantize_int8(vectors)
    dim = quantized.shape[1]

    code = f"const int NUM_{name.upper()} = {len(vectors)};\n"
    code += f"const float KDE_{name.upper()}_SCALE = {scale:.10f}f;\n"
    code += f"const int8_t kde_{name.lower()}_vectors[][{dim}] = {{\n"
    for row in quantized:
        formatted_vec = ", ".join(str(int(v)) for v in row)
        code += f"    {{{formatted_vec}}},\n"
    code = code.rstrip(",\n") + "\n};\n\n"
    return code


def export_kde_header():
    print("\n" + "=" * 60)
    print("[4/4] Ekspor vektor KDE ke kde_z_vectors.h (INT8)")
    print("=" * 60)

    if not os.path.exists(KDE_EMBEDDING_JSON):
        print(f"    ❌ File tidak ditemukan: {KDE_EMBEDDING_JSON}")
        return

    with open(KDE_EMBEDDING_JSON, "r") as f:
        data = json.load(f)

    noise_vecs = data.get("noise", [])
    le_vecs = data.get("le", [])

    if len(noise_vecs) == 0:
        print("    ❌ Data vektor kosong.")
        return

    c_code = "#ifndef KDE_Z_VECTORS_H\n#define KDE_Z_VECTORS_H\n\n"
    c_code += "#include <stdint.h>\n\n"
    c_code += "// Ruang Probabilitas Lokal Indonesia (Komponen Z)\n"
    c_code += "// Disimpan sebagai INT8 (bukan float) -- 4x lebih hemat flash/RAM.\n"
    c_code += "// Nilai asli didapat lewat: nilai_float = kode_int8 * SCALE\n\n"
    c_code += format_kde_array("NOISE", noise_vecs)
    c_code += format_kde_array("LE", le_vecs)
    c_code += "#endif // KDE_Z_VECTORS_H"

    with open(KDE_HEADER_PATH, "w") as f:
        f.write(c_code)

    print(f"    ✅ Tersimpan: {KDE_HEADER_PATH} (INT8, {len(noise_vecs) + len(le_vecs)} vektor)")


# ============================================================
# EKSEKUSI
# ============================================================

def main():
    tflite_data = build_tflite_model()
    export_model_header(tflite_data)
    export_kde_header()

    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)
    print(f"Pindahkan kedua file berikut ke folder src/ project ESP32-S3 Anda:")
    print(f"  - {MODEL_HEADER_PATH}")
    print(f"  - {KDE_HEADER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
