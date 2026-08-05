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
def plot_comprehensive_metrics(metrics, save_path, domain_name):
    # Mapping nama label ke key yang tepat sesuai output
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
    plt.title(f'MCU-Quake, {domain_name}, KDE-Indonesia')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tambahkan angka di atas batang
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # Mencegah gambar bertumpuk di memori
    print(f"[INFO] Grafik metrik {domain_name} tersimpan di: {save_path}")

# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":

    # 1. KONFIGURASI GLOBAL (MODEL & KDE INDONESIA)
    BASE_REP = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/Code & Figure demo"
    MODEL_PATH = os.path.join(BASE_REP, "Pre-trained model/MCU-Quake 5-20")
    
    # SATU KDE UTAMA (INDONESIA)
    EMB_DIR_INDO = "/Volumes/Extreme SSD/unduhan_waveform_geofon/output/indonesia_domain_embeddings_3c_le"

    # 2. KONFIGURASI DATASET PENGUJIAN (3 DATASET)
    # Catatan: Pastikan path ini sesuai dengan struktur direktori di Mac Anda
    TEST_DATASETS = {
        "INDONESIA_TEST": "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/data_indonesia/indonesia_test_data.json",
        "UUSS": "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ UUSS 3C_ test n2222 r100/UUSS 3C data, test n2222 r100.json",
        "STEAD": "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/Benchmark_ STEAD 3C_ test n15275 r100/STEAD data, test n15275 r100.json"
    }

    MODEL_TAG = "MCU_Quake_3C"
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    num_points = int(INPUT_WIN * SAMPLING_RATE)
    
    true_labels_name = ["NO", "LE"]

    # 3. PERSIAPAN OUTPUT DIREKTORI & LOGGER
    SAVE_BASE = '/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/retraining_mcu_q_indonesia/output_indo_embedding_for_all_test'
    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"Benchmarking_vs_INDO_KDE_{time_str}")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    log_file_path = os.path.join(save_dir, "task_log_benchmarking.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - [%(levelname)s]: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    # ===================================================================
    # 4. MEMUAT MODEL & MEMBANGUN KDE INDONESIA (DILAKUKAN SEKALI SAJA)
    # ===================================================================
    logger.info("Memuat Model 1D-CNN MCU-Quake...")
    embedding_model = keras.models.load_model(filepath=MODEL_PATH)

    logger.info("Membangun 3D KDE INDONESIA...")
    embedding_Z = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, Z.json")
    embedding_N = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, N.json")
    embedding_E = dataset.load_embedding_data(EMB_DIR_INDO, "Embedding data, E.json")
    
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E, source_list=['noise', 'le'])

    # ===================================================================
    # 5. LOOPING EVALUASI KE-3 DATASET PENGUJIAN
    # ===================================================================
    for dataset_name, dataset_path in TEST_DATASETS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 MENGUJI DATASET: {dataset_name}")
        logger.info(f"{'='*50}")

        try:
            logger.info(f"Memuat {dataset_name} ({dataset_path})...")
            test_data = dataset.load_json_data(dataset_path)
            keys_list = list(test_data.keys())
            
            total_true_3C, total_pred_3C = [], []
            
            for i in tqdm(range(len(keys_list)), desc=f"Inferensi {dataset_name}"):
                record_key = keys_list[i]
                record = test_data[record_key]
                
                try:
                    # Sinyal & Noise 3C
                    Z_n, N_n, E_n = record["Z_noise"][-num_points:], record["N_noise"][-num_points:], record["E_noise"][-num_points:]
                    Z_s, N_s, E_s = record["Z"][:num_points], record["N"][:num_points], record["E"][:num_points]

                    # Embeddings (Ekstraksi)
                    _in_Zn, _in_Nn, _in_En = utils.latent_codes_1D(Z_n, embedding_model), utils.latent_codes_1D(N_n, embedding_model), utils.latent_codes_1D(E_n, embedding_model)
                    _in_Zs, _in_Ns, _in_Es = utils.latent_codes_1D(Z_s, embedding_model), utils.latent_codes_1D(N_s, embedding_model), utils.latent_codes_1D(E_s, embedding_model)

                    # Inferensi 3C KDE - Noise (Terhadap KDE Indonesia)
                    emb_n_3c = np.array([_in_En, _in_Nn, _in_Zn]).reshape(1,-1)
                    p_n_3c, _, _ = utils.infer_3C_PDFs(emb_n_3c, embeddings_3C_PDFs, "Kernel")
                    
                    # Inferensi 3C KDE - Earthquake (Terhadap KDE Indonesia)
                    emb_s_3c = np.array([_in_Es, _in_Ns, _in_Zs]).reshape(1,-1)
                    p_s_3c, _, _ = utils.infer_3C_PDFs(emb_s_3c, embeddings_3C_PDFs, "Kernel")

                    # Labeling (0 = NO, 1 = LE)
                    total_true_3C.extend([0, 1]) 
                    total_pred_3C.extend([1 if p_n_3c >= 1 else 0, 1 if p_s_3c >= 1 else 0])
                    
                except Exception as e:
                    continue # Lewati jika ada record yang corrupt/kosong
                    
            # 6. HITUNG METRIK & SIMPAN UNTUK DATASET INI
            logger.info(f"Menyimpan hasil pengujian {dataset_name}...")
            matrix_3C, metrics_3C = utils.calc_confusion_metrics(total_true_3C, total_pred_3C)
            
            # Confusion Matrix
            fig_3C = utils.plot_confusion(f"MCU-Quake, {dataset_name}, KDE-Indonesia", true_labels_name, matrix_3C, metrics_3C)
            fig_3C.savefig(os.path.join(save_dir, f"CM_{dataset_name}_vs_INDO_KDE.jpg"), dpi=300)
            plt.close(fig_3C)
            
            # Bar Chart
            plot_comprehensive_metrics(metrics_3C, os.path.join(save_dir, f"BarMetrics_{dataset_name}.jpg"), dataset_name)
            
            # JSON Data
            dataset.save_json_data(os.path.join(save_dir, f"Metrics_{dataset_name}.json"), metrics_3C)

            logger.info(f"--- HASIL {dataset_name} ---")
            logger.info(f"Accuracy : {metrics_3C.get('Accuracy (avg.)')}")
            logger.info(f"F1-Score : {metrics_3C.get('F1-score (avg.)')}")
            
        except Exception as e:
            logger.error(f"❌ Gagal mengevaluasi dataset {dataset_name}. Error: {e}")

    logger.info("\n" + "="*50)
    logger.info("✅ Pengujian Benchmarking 3 Dataset terhadap KDE Indonesia Selesai.")
    logger.info(f"Seluruh output tersimpan di folder: {save_dir}")