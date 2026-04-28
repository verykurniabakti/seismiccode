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
    #                    1. KONFIGURASI PATH
    #===================================================================
    # Path dataset UUSS n2222 Bapak
    KEY_DATA_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/Benchmark_ UUSS 3C_ test n2222 r100"
    KEY_TEST_FILE = "UUSS 3C data, test n2222 r100.json" 
    
    DATA_TAG = "UUSS_n2222_1C"
    MODEL_TAG = "MCU_5-20"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    
    true_labels = ["NO", "QB", "LE"]
    source_to_code = {"noise": 0, "qb": 1, "le": 2}

    # Path Model dan Embedding
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    # Untuk 1C, kita hanya butuh embedding Z
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")

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
    log_file_path = os.path.join(save_dir, "task_log_1c.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # Load Model & Embedding Z Only
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")

    logger.info(f"Estimating 1C (Z-Only) embeddings statistics (KDE)...")
    embeddings_Z_PDFs = utils.embedding_PDFs_1D(embedding_Z)

    #===================================================================
    #                            3. EVALUATION (1C ONLY)
    #===================================================================
    total_true_1C_KDE, total_pred_1C_KDE = [], []
    mismatch_noise_1C_KDE, mismatch_seismic_1C_KDE = {}, {}

    num_points = int(INPUT_WIN * SAMPLING_RATE)
    keys_list = list(test_data.keys())

    logger.info(f"Processing {len(keys_list)} records for 1C Evaluation...")

    for i in tqdm(range(len(keys_list)), desc="Inference 1C"):
        record_key = keys_list[i]
        record = test_data[record_key]
        quake_label = record["type"]
    
        try:
            # Hanya mengambil komponen Z
            Z_noise = record["Z_noise"][-num_points:]
            Z_sig = record["Z"][:num_points]

            # Latent Embeddings (Z Only)
            _in_Zn = utils.latent_codes_1D(Z_noise, embedding_model)
            _in_Zs = utils.latent_codes_1D(Z_sig, embedding_model)

            # 1C KDE Inference - Noise
            p_n_1c, _, _ = utils.infer_1C_PDFs(_in_Zn, embeddings_Z_PDFs, "Kernel")
            
            # 1C KDE Inference - Seismic
            p_s_1c, _, _ = utils.infer_1C_PDFs(_in_Zs, embeddings_Z_PDFs, "Kernel")

            total_true_1C_KDE.extend([0, source_to_code[quake_label]])
            total_pred_1C_KDE.extend([p_n_1c, p_s_1c])
            
        except KeyError as e:
            continue

    #===================================================================
    #                        4. METRICS & PLOT
    #===================================================================
    matrix_1C, metrics_1C = utils.calc_confusion_metrics(total_true_1C_KDE, total_pred_1C_KDE)
    
    # Plotting
    fig_1C = utils.plot_confusion(f"{MODEL_TAG} {DATA_TAG} 1C KDE (Z-Only)", true_labels, matrix_1C, metrics_1C)
    fig_1C.savefig(os.path.join(save_dir, f"UUSS_n2222_1C_KDE.jpg"), dpi=300)
    
    # Save Metrics JSON
    dataset.save_json_data(os.path.join(save_dir, "uuss_metrics_1C.json"), metrics_1C)
    
    logger.info("\n" + "="*30)
    logger.info(f"1C ACCURACY AVG: {metrics_1C.get('accuracy (avg.)')}")
    logger.info(f"1C F1-SCORE AVG: {metrics_1C.get('f1-score (avg.)')}")
    logger.info("="*30)
    logger.info("Task UUSS n2222 1C Completed.")