# -*- coding: utf-8 -*- sesi 2
from Library import utils, dataset
import os
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt

# ===================================================================
# FUNGSI VISUALISASI
# ===================================================================
def plot_comprehensive_metrics(metrics, save_path, domain_title):
    labels = ['Accuracy', 'Recall', 'Precision', 'FPR', 'F1-Score']
    keys = [
        'Accuracy (avg.)', 
        'True positive rate (avg.)', 
        'Positive predictive value (avg.)', 
        'False positive rate (avg.)', 
        'F1-score (avg.)'
    ]
    
    values = [metrics.get(k, 0) for k in keys]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=['#2c3e50', '#3498db', '#e67e22', '#e74c3c', '#27ae60'])
    
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title(f'Performance Metrics Evaluation - {domain_title}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":

    # 1. KONFIGURASI PATH DATA TEST & MODEL
    KEY_DATA_DIR = "/Volumes/Extreme SSD/json_indonesia_juli_sesi_5"
    KEY_TEST_FILE = "extracted_data_3c_sesi_clean.json"
    
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/Code & Figure demo"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")

    # 2. PATH KE-3 EMBEDDING (KDE)
    EMB_DIR_STEAD = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538")
    EMB_DIR_UUSS = os.path.join(BASE_REP, "Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909")
    EMB_DIR_INDO = "/Volumes/Extreme SSD/unduhan_waveform_geofon/output/indonesia_domain_embeddings_3c_le"

    DATA_TAG = "INDONESIA_TEST_DATA"
    MODEL_TAG = "MCU_Quake_3C"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    num_points = int(INPUT_WIN * SAMPLING_RATE)
    true_labels_name = ["NO", "LE"]

    # Persiapan Direktori Output
    SAVE_BASE = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/output_3_embedding'
    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"Comparison_3KDE_{time_str}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Logger
    log_file_path = os.path.join(save_dir, "task_log_comparison.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # ===================================================================
    # 3. MEMUAT DATA & MODEL
    # ===================================================================
    logger.info(f"Memuat Test Dataset Indonesia: {KEY_TEST_FILE}")
    test_data = dataset.load_json_data(os.path.join(KEY_DATA_DIR, KEY_TEST_FILE))
    
    logger.info(f"Memuat Model 1D-CNN MCU-Quake...")
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)

    # ===================================================================
    # 4. MEMBANGUN 3 KDE SEKALIGUS
    # ===================================================================
    logger.info("Membangun KDE STEAD...")
    emb_Z_stead = dataset.load_embedding_data(EMB_DIR_STEAD, "Embedding data, Z.json")
    emb_N_stead = dataset.load_embedding_data(EMB_DIR_STEAD, "Embedding data, N.json")
    emb_E_stead = dataset.load_embedding_data(EMB_DIR_STEAD, "Embedding data, E.json")
    kde_stead = utils.embedding_PDFs_3D(emb_Z_stead, emb_N_stead, emb_E_stead, source_list=['noise', 'le'])

    logger.info("Membangun KDE UUSS...")
    emb_Z_uuss = dataset.load_embedding_data(EMB_DIR_UUSS, "Embedding data, Z.json")
    emb_N_uuss = dataset.load_embedding_data(EMB_DIR_UUSS, "Embedding data, N.json")
    emb_E_uuss = dataset.load_embedding_data(EMB_DIR_UUSS, "Embedding data, E.json")
    kde_uuss = utils.embedding_PDFs_3D(emb_Z_uuss, emb_N_uuss, emb_E_uuss, source_list=['noise', 'le'])

    logger.info("Membangun KDE INDONESIA...")
    emb_Z_indo = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, Z.json")
    emb_N_indo = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, N.json")
    emb_E_indo = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, E.json")
    kde_indo = utils.embedding_PDFs_3D(emb_Z_indo, emb_N_indo, emb_E_indo, source_list=['noise', 'le'])

    # ===================================================================
    # 5. INFERENSI PARALEL
    # ===================================================================
    true_labels = []
    pred_stead = []
    pred_uuss = []
    pred_indo = []
    
    keys_list = list(test_data.keys())
    logger.info(f"Memulai Inferensi Paralel untuk {len(keys_list)} rekaman...")

    for i in tqdm(range(len(keys_list)), desc="Inference Laten & 3 KDE"):
        record_key = keys_list[i]
        record = test_data[record_key]
    
        try:
            # 5a. Potong Sinyal
            Z_n, N_n, E_n = record["Z_noise"][-num_points:], record["N_noise"][-num_points:], record["E_noise"][-num_points:]
            Z_s, N_s, E_s = record["Z"][:num_points], record["N"][:num_points], record["E"][:num_points]

            # 5b. Ekstraksi Fitur dari Model (HANYA DILAKUKAN 1 KALI)
            _in_Zn, _in_Nn, _in_En = utils.latent_codes_1D(Z_n, embedding_model), utils.latent_codes_1D(N_n, embedding_model), utils.latent_codes_1D(E_n, embedding_model)
            _in_Zs, _in_Ns, _in_Es = utils.latent_codes_1D(Z_s, embedding_model), utils.latent_codes_1D(N_s, embedding_model), utils.latent_codes_1D(E_s, embedding_model)

            emb_n_3c = np.array([_in_En, _in_Nn, _in_Zn]).reshape(1,-1)
            emb_s_3c = np.array([_in_Es, _in_Ns, _in_Zs]).reshape(1,-1)

            # 5c. Uji Jarak ke KDE STEAD
            p_n_stead, _, _ = utils.infer_3C_PDFs(emb_n_3c, kde_stead, "Kernel")
            p_s_stead, _, _ = utils.infer_3C_PDFs(emb_s_3c, kde_stead, "Kernel")
            
            # 5d. Uji Jarak ke KDE UUSS
            p_n_uuss, _, _ = utils.infer_3C_PDFs(emb_n_3c, kde_uuss, "Kernel")
            p_s_uuss, _, _ = utils.infer_3C_PDFs(emb_s_3c, kde_uuss, "Kernel")

            # 5e. Uji Jarak ke KDE INDO
            p_n_indo, _, _ = utils.infer_3C_PDFs(emb_n_3c, kde_indo, "Kernel")
            p_s_indo, _, _ = utils.infer_3C_PDFs(emb_s_3c, kde_indo, "Kernel")

            # Simpan hasil perbandingan
            true_labels.extend([0, 1]) 
            
            pred_stead.extend([1 if p_n_stead >= 1 else 0, 1 if p_s_stead >= 1 else 0])
            pred_uuss.extend([1 if p_n_uuss >= 1 else 0, 1 if p_s_uuss >= 1 else 0])
            pred_indo.extend([1 if p_n_indo >= 1 else 0, 1 if p_s_indo >= 1 else 0])
            
        except Exception as e:
            logger.error(f"Error pada {record_key}: {e}")
            continue

    # ===================================================================
    # 6. HITUNG METRIK DAN CETAK HASIL
    # ===================================================================
    logger.info("Menghitung metrik dan membuat visualisasi...")

    domains = [
        ("STEAD", pred_stead),
        ("UUSS", pred_uuss),
        ("INDO", pred_indo)
    ]

    for domain_name, predictions in domains:
        matrix, metrics = utils.calc_confusion_metrics(true_labels, predictions)
        
        # Plot Confusion Matrix
        fig = utils.plot_confusion(f"MCU-Quake,{domain_name},Indonesia(3C)", true_labels_name, matrix, metrics)
        fig.savefig(os.path.join(save_dir, f"CM_{domain_name}_KDE.jpg"), dpi=300)
        plt.close(fig)
       
        # Plot Bar Chart
        plot_comprehensive_metrics(metrics, os.path.join(save_dir, f"BarMetrics_{domain_name}_KDE.jpg"), domain_name)
        
        # Simpan JSON
        dataset.save_json_data(os.path.join(save_dir, f"Metrics_{domain_name}.json"), metrics)
        
        # Catat di Log
        logger.info(f"\n--- HASIL MENGGUNAKAN KDE {domain_name} ---")
        logger.info(f"Accuracy : {metrics.get('Accuracy (avg.)')}")
        logger.info(f"Precision: {metrics.get('Positive predictive value (avg.)')}")
        logger.info(f"Recall   : {metrics.get('True positive rate (avg.)')}")
        logger.info(f"F1-Score : {metrics.get('F1-score (avg.)')}")

    logger.info("\n" + "="*50)
    logger.info("✅ Pengujian Komparasi 3 KDE Selesai.")
    logger.info(f"Seluruh output tersimpan di: {save_dir}")