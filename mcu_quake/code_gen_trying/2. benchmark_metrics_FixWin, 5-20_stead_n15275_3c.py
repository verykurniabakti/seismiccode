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
    #                    1. KONFIGURASI PATH STEAD
    #===================================================================
    # Path dataset STEAD yang Bapak berikan
    KEY_DATA_DIR = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/Benchmark_ STEAD 3C_ test n15275 r100"
    KEY_TEST_FILE = "STEAD data, test n15275 r100.json" 
    
    DATA_TAG = "STEAD_Global"
    MODEL_TAG = "MCU_5-20"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    
    # STEAD fokus pada Noise (NO) dan Earthquake (LE)
    true_labels = ["NO", "LE"]
    source_to_code = {"noise": 0, "le": 1}

    # Path Model dan Embedding (PASTIKAN MENGGUNAKAN EMBEDDING STEAD)
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    
    # Path Embedding Khusus STEAD (n=61099)
    EMB_DIR = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538")

    #===================================================================
    #                           2. PREPARE FILES
    #===================================================================
    print(f"[INFO] Memuat Dataset STEAD: {KEY_TEST_FILE}")
    test_data = dataset.load_json_data(os.path.join(KEY_DATA_DIR, KEY_TEST_FILE))
    
    SAVE_BASE = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo"
    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"{MODEL_TAG}_{DATA_TAG}_{time_str}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Logger
    log_file_path = os.path.join(save_dir, "task_log_stead_global.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # Load Model & Embeddings STEAD
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)
    embedding_Z = dataset.load_embedding_data(EMB_DIR, "Embedding data, Z.json")
    embedding_N = dataset.load_embedding_data(EMB_DIR, "Embedding data, N.json")
    embedding_E = dataset.load_embedding_data(EMB_DIR, "Embedding data, E.json")

    logger.info(f"Estimating STEAD Global embeddings statistics (KDE)...")
    # Menggunakan statistik ruang laten STEAD
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E)

    #===================================================================
    #                            3. EVALUATION
    #===================================================================
    total_true_3C, total_pred_3C = [], []
    num_points = int(INPUT_WIN * SAMPLING_RATE)
    keys_list = list(test_data.keys())

    logger.info(f"Memproses {len(keys_list)} data STEAD...")

    for i in tqdm(range(len(keys_list)), desc="Inference STEAD"):
        record_key = keys_list[i]
        record = test_data[record_key]
        quake_label = record["type"] # label asli: 'le'
    
        try:
            # Sinyal & Noise 3C
            Z_n, N_n, E_n = record["Z_noise"][-num_points:], record["N_noise"][-num_points:], record["E_noise"][-num_points:]
            Z_s, N_s, E_s = record["Z"][:num_points], record["N"][:num_points], record["E"][:num_points]

            # Embeddings
            _in_Zn, _in_Nn, _in_En = utils.latent_codes_1D(Z_n, embedding_model), utils.latent_codes_1D(N_n, embedding_model), utils.latent_codes_1D(E_n, embedding_model)
            _in_Zs, _in_Ns, _in_Es = utils.latent_codes_1D(Z_s, embedding_model), utils.latent_codes_1D(N_s, embedding_model), utils.latent_codes_1D(E_s, embedding_model)

            # Inferensi 3C KDE - Noise
            emb_n_3c = np.array([_in_En, _in_Nn, _in_Zn]).reshape(1,-1)
            p_n_3c, _, _ = utils.infer_3C_PDFs(emb_n_3c, embeddings_3C_PDFs, "Kernel")
            
            # Inferensi 3C KDE - Earthquake
            emb_s_3c = np.array([_in_Es, _in_Ns, _in_Zs]).reshape(1,-1)
            p_s_3c, _, _ = utils.infer_3C_PDFs(emb_s_3c, embeddings_3C_PDFs, "Kernel")

            # Mapping Label STEAD (Biner: NO vs LE)
            # Di STEAD, output model index 1 (QB) atau 2 (LE) dianggap sebagai Earthquake (1)
            total_true_3C.extend([0, 1]) 
            total_pred_3C.extend([1 if p_n_3c >= 1 else 0, 1 if p_s_3c >= 1 else 0])
            
        except KeyError:
            continue

    #===================================================================
    #                        4. METRICS & SAVE
    #===================================================================
    matrix_3C, metrics_3C = utils.calc_confusion_metrics(total_true_3C, total_pred_3C)
    
    # Visualisasi
    fig_3C = utils.plot_confusion(f"{MODEL_TAG} {DATA_TAG} 3C KDE", true_labels, matrix_3C, metrics_3C)
    fig_3C.savefig(os.path.join(save_dir, "STEAD_Global_3C_KDE.jpg"), dpi=300)
    
    dataset.save_json_data(os.path.join(save_dir, "stead_global_metrics.json"), metrics_3C)
    
    logger.info("\n" + "="*40)
    logger.info(f"STEAD GLOBAL ACCURACY: {metrics_3C.get('accuracy (avg.)')}")
    logger.info(f"STEAD GLOBAL F1-SCORE: {metrics_3C.get('f1-score (avg.)')}")
    logger.info("="*40)
    logger.info("Pengujian STEAD Selesai.")