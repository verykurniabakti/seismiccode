# -*- coding: utf-8 -*-
from Library import utils, dataset
import os
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging

if __name__ == "__main__":
                    
    #===================================================================
    #                    1. KONFIGURASI PATH STEAD 1C
    #===================================================================
    KEY_DATA_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/Benchmark_ STEAD 3C_ test n15275 r100"
    KEY_TEST_FILE = "STEAD data, test n15275 r100.json" 
    
    DATA_TAG = "STEAD_Global_1C"
    MODEL_TAG = "MCU_5-20"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    
    true_labels = ["NO", "LE"]
    source_to_code = {"noise": 0, "le": 1}

    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    
    # TETAP gunakan embedding STEAD, namun kita hanya akan memanggil data Z
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538")

    #===================================================================
    #                           2. PREPARE FILES
    #===================================================================
    test_data = dataset.load_json_data(os.path.join(KEY_DATA_DIR, KEY_TEST_FILE))
    
    SAVE_BASE = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo"
    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"{MODEL_TAG}_{DATA_TAG}_{time_str}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Logger
    log_file_path = os.path.join(save_dir, "task_log_stead_1c.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # Load Model & Embedding Z STEAD
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")

    logger.info(f"Estimating STEAD 1C embeddings statistics (KDE)...")
    embeddings_Z_PDFs = utils.embedding_PDFs_1D(embedding_Z)

    #===================================================================
    #                            3. EVALUATION (1C ONLY)
    #===================================================================
    total_true_1C_KDE, total_pred_1C_KDE = [], []

    num_points = int(INPUT_WIN * SAMPLING_RATE)
    keys_list = list(test_data.keys())

    logger.info(f"Processing {len(keys_list)} STEAD records for 1C Evaluation...")

    for i in tqdm(range(len(keys_list)), desc="Inference STEAD 1C"):
        record_key = keys_list[i]
        record = test_data[record_key]
        
        try:
            # Hanya mengambil komponen Z (Vertikal)
            Z_n = record["Z_noise"][-num_points:]
            Z_s = record["Z"][:num_points]

            # Latent Embeddings (Z Only)
            _in_Zn = utils.latent_codes_1D(Z_n, embedding_model)
            _in_Zs = utils.latent_codes_1D(Z_s, embedding_model)

            # 1C KDE Inference
            p_n_1c, _, _ = utils.infer_1C_PDFs(_in_Zn, embeddings_Z_PDFs, "Kernel")
            p_s_1c, _, _ = utils.infer_1C_PDFs(_in_Zs, embeddings_Z_PDFs, "Kernel")

            # Mapping ke 2 kelas (Noise vs Earthquake)
            # Jika hasil inferensi >= 1 (QB atau LE), dianggap Earthquake
            total_true_1C_KDE.extend([0, 1])
            total_pred_1C_KDE.extend([1 if p_n_1c >= 1 else 0, 1 if p_s_1c >= 1 else 0])
            
        except KeyError:
            continue

    #===================================================================
    #                        4. METRICS & PLOT
    #===================================================================
    matrix_1C, metrics_1C = utils.calc_confusion_metrics(total_true_1C_KDE, total_pred_1C_KDE)
    
    # Plotting
    fig_1C = utils.plot_confusion(f"{MODEL_TAG} {DATA_TAG} 1C KDE", true_labels, matrix_1C, metrics_1C)
    fig_1C.savefig(os.path.join(save_dir, f"STEAD_Global_1C_KDE.jpg"), dpi=300)
    
    dataset.save_json_data(os.path.join(save_dir, "stead_metrics_1c.json"), metrics_1C)
    
    logger.info("\n" + "="*40)
    logger.info(f"STEAD 1C ACCURACY: {metrics_1C.get('accuracy (avg.)')}")
    logger.info(f"STEAD 1C F1-SCORE: {metrics_1C.get('f1-score (avg.)')}")
    logger.info("="*40)