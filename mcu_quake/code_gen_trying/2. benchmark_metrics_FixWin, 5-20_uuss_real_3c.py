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
    # Path Input sesuai yang Bapak berikan
    KEY_DATA_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/Benchmark_ UUSS 3C_ test n2222 r100"
    KEY_TEST_FILE = "UUSS 3C data, test n2222 r100.json" 
    
    DATA_TAG = "UUSS_n2222"
    MODEL_TAG = "MCU_5-20"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    
    true_labels = ["NO", "QB", "LE"]
    source_to_code = {"noise": 0, "qb": 1, "le": 2}

    # Path Model dan Embedding (Path Absolut Mac)
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")

    #===================================================================
    #                           2. PREPARE FILES
    #===================================================================
    print(f"[INFO] Loading UUSS Dataset: {KEY_TEST_FILE}")
    test_data = dataset.load_json_data(os.path.join(KEY_DATA_DIR, KEY_TEST_FILE))
    
    # Simpan Hasil ke SSD Extreme
    SAVE_BASE = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo"
    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"{MODEL_TAG}_{DATA_TAG}_{time_str}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Logger
    log_file_path = os.path.join(save_dir, "task_log_uuss.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # Load Model & Embeddings
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    embedding_N = dataset.load_embedding_data(EMB_DIR, "Embedding data, N.json")
    embedding_E = dataset.load_embedding_data(EMB_DIR, "Embedding data, E.json")

    logger.info(f"Estimating embeddings statistics (KDE)...")
    embeddings_Z_PDFs = utils.embedding_PDFs_1D(embedding_Z)
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E)

    #===================================================================
    #                            3. EVALUATION
    #===================================================================
    total_true_3C_KDE, total_pred_3C_KDE = [], []
    mismatch_noise_3C_KDE, mismatch_seismic_3C_KDE = {}, {}

    num_points = int(INPUT_WIN * SAMPLING_RATE)
    keys_list = list(test_data.keys())

    logger.info(f"Processing {len(keys_list)} records...")

    for i in tqdm(range(len(keys_list)), desc="Inference UUSS"):
        record_key = keys_list[i]
        record = test_data[record_key]
        quake_label = record["type"]
    
        # Parse 3-Component Data
        # Sesuaikan Key ['Z'], ['N'], ['E'] jika di JSON Bapak menggunakan huruf kecil
        try:
            Z_noise, N_noise, E_noise = record["Z_noise"][-num_points:], record["N_noise"][-num_points:], record["E_noise"][-num_points:]
            Z_sig, N_sig, E_sig = record["Z"][:num_points], record["N"][:num_points], record["E"][:num_points]

            # Latent Embeddings
            _in_Z_n = utils.latent_codes_1D(Z_noise, embedding_model)
            _in_N_n = utils.latent_codes_1D(N_noise, embedding_model)
            _in_E_n = utils.latent_codes_1D(E_noise, embedding_model)
            
            _in_Z_s = utils.latent_codes_1D(Z_sig, embedding_model)
            _in_N_s = utils.latent_codes_1D(N_sig, embedding_model)
            _in_E_s = utils.latent_codes_1D(E_sig, embedding_model)

            # 3C KDE Inference - Noise
            emb_n_3c = np.array([_in_E_n, _in_N_n, _in_Z_n]).reshape(1,-1)
            p_n_3c, _, _ = utils.infer_3C_PDFs(emb_n_3c, embeddings_3C_PDFs, "Kernel")
            
            # 3C KDE Inference - Seismic
            emb_s_3c = np.array([_in_E_s, _in_N_s, _in_Z_s]).reshape(1,-1)
            p_s_3c, _, _ = utils.infer_3C_PDFs(emb_s_3c, embeddings_3C_PDFs, "Kernel")

            total_true_3C_KDE.extend([0, source_to_code[quake_label]])
            total_pred_3C_KDE.extend([p_n_3c, p_s_3c])
            
        except KeyError as e:
            logger.warning(f"Key {e} not found in record {record_key}")
            continue

    #===================================================================
    #                        4. METRICS & PLOT
    #===================================================================
    matrix_3C, metrics_3C = utils.calc_confusion_metrics(total_true_3C_KDE, total_pred_3C_KDE)
    
    # Plotting
    fig_3C = utils.plot_confusion(f"{MODEL_TAG} {DATA_TAG} 3C KDE", true_labels, matrix_3C, metrics_3C)
    fig_3C.savefig(os.path.join(save_dir, f"UUSS_n2222_3C_KDE.jpg"), dpi=300)
    
    # Save Metrics
    dataset.save_json_data(os.path.join(save_dir, "uuss_metrics_3C.json"), metrics_3C)
    
    logger.info("\n" + "="*30)
    logger.info(f"ACCURACY AVG: {metrics_3C.get('accuracy (avg.)', 'N/A')}")
    logger.info(f"F1-SCORE AVG: {metrics_3C.get('f1-score (avg.)', 'N/A')}")
    logger.info("="*30)
    logger.info("Task UUSS n2222 Completed.")