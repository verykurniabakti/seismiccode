# -*- coding: utf-8 -*-
"""
A demonstration benchmarks the MCU-Quake (5-20) using 100 samples from the UUSS dataset. The full data are provided in the Data Availability section of the manuscript. 

Note that the performance of MCU-Quake can be improved by employing a more sophisticated interpretation method for the model's output embeddings.
"""


from Library import utils, dataset
import pandas as pd
import os
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging
import config

def plot_save_mismatches(plot_func, mismatch_pred, name_mismatch_pred,
                         source_data, source_meta,
                         plot_num, save_dir,
                         input_win=None):
        # save fig dir
    if len(mismatch_pred)>1:
        _plot_num = min(plot_num, len(mismatch_pred))
        _fig_dir = os.path.join(save_dir, f"Figures, {name_mismatch_pred.split('.')[0]}")
        if not os.path.exists(_fig_dir): os.makedirs(_fig_dir)
        _id_list = np.random.choice(list(mismatch_pred.keys()), size=_plot_num, replace=False)
        plot_func(_id_list, source_data, source_meta,
                  pred_info=mismatch_pred, save_dir=_fig_dir,
                  input_win=input_win)




if __name__ == "__main__":
                    
    #===================================================================
    #                    3-component dataset and embeddings
    #===================================================================

    source_to_code = {"noise": 0, "qb": 1} # Buang 'le' sementara
    code_to_source = {0: "noise", 1: "qb"}

    # --- Karena menggunakan mac GANTI BAGIAN INI ---
    # key_data_dir = r"Data benchmark-demo"
    # model_path = r"Pre-trained model\MCU-Quake 5-20"
    # embedding_3C_dir = r"Typical embedding\Embedding_data train 3C, UUSS n11275 std15, 30120909"

    # --- BAGIAN KONFIGURASI PATH ---
    base_path = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"

    # 1. Tentukan folder data
    key_data_dir = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo"

    # 2. DEFINISIKAN NAMA FILE (Ini yang tadi hilang/error)
    key_test_file = "Ukraine_1C_golden_standard.json" 

    # 3. Selebihnya...
    DATA_TAG = "Ukraine_1C"
    true_labels = ["NO", "QB", "LE"]
    model_path = os.path.join(base_path, "Pre-trained model", "MCU-Quake 5-20")
    embedding_3C_dir = os.path.join(base_path, "Typical embedding", "Embedding_data train 3C, UUSS n11275 std15, 30120909")


    """ STEAD dataset """
    # DATA_TAG = "STEAD" 

    # key_test_file = "STEAD MCU, test n15275 r100.json"
    # true_labels = ["NO", "LE"]


    # ''' embeddings, STEAD dataset '''
    # embedding_3C_dir = r"Typical embedding\Embedding_data train 3C, STEAD norm7 mag3 L n61099, 30172538"


    #===================================================================
    #                        MCU-Quake: 5-20, 7-second
    #===================================================================

    """ model config """

    # 7s model
    INPUT_WIN = 7  # seconds
    SAMPLING_RATE = 100

    MODEL_TAG = "MCU_5-20"

    # model
    model_path = r"Pre-trained model\MCU-Quake 5-20"



    #===================================================================
    #                           Prepare files
    #===================================================================

    """ Load dataset """
    """ Load dataset """
    print(f"[INFO] load test dataset ...")
    test_data = dataset.load_json_data(os.path.join(key_data_dir, key_test_file)) 

    NUM_RECORD = len(test_data)   # all the dataset

    """ create log file """
    # KOREKSI: Gunakan path SSD Extreme secara langsung
    SAVE_BASE = "/Volumes/Extreme SSD/mcu_quake_output_replikasi_demo"

    # Pastikan folder utama dibuat jika belum ada
    if not os.path.exists(SAVE_BASE):
        os.makedirs(SAVE_BASE)

    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    
    # Gunakan underscore (_) sebagai pengganti koma atau spasi agar aman di sistem file
    folder_name = f"{MODEL_TAG}_{DATA_TAG}_{time_str}"
    save_dir = os.path.join(SAVE_BASE, folder_name)
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # Konfigurasi Logger (Simpan Log ke SSD)
    log_file_path = os.path.join(save_dir, "task_log.txt")
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s]: %(message)s',
                        datefmt = "%Y-%m-%d %H:%M:%S",
                        filemode='w')
    
    logger = logging.getLogger()
    # Hapus handler lama jika ada agar tidak double logging
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Task summary:\n"
                f"Dataset: {os.path.join(key_data_dir, key_test_file)}\n"
                f"Model: {MODEL_TAG}, key embeddings dir: {embedding_3C_dir}\n"
                f"True labels: {true_labels}\n"
                f"Source to code: {source_to_code}\n"
                f"Code to source: {code_to_source}\n"
                f"Output Directory: {save_dir}\n"
                )

    logger.info(f"Load data files ...")

    # embedding model
    # --- 1. Load Embedding Model ---
    # Definisikan path model secara absolut
    final_model_path = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Pre-trained model/MCU-Quake 5-20"
    
    # Load model menggunakan path tersebut
    embedding_model = keras.models.load_model(filepath=final_model_path)

    # --- 2. Load Embedding JSON Data ---
    # Definisikan path folder embedding secara absolut
    final_emb_dir = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code/Typical embedding/Embedding_data train 3C, UUSS n11275 std15, 30120909"

    train_embedding_Z_file = "Embedding data, Z.json"
    train_embedding_N_file = "Embedding data, N.json"
    train_embedding_E_file = "Embedding data, E.json"

    # Gunakan final_emb_dir yang sudah kita buat
    embedding_Z = dataset.load_embedding_data(final_emb_dir, train_embedding_Z_file)
    embedding_N = dataset.load_embedding_data(final_emb_dir, train_embedding_N_file)
    embedding_E = dataset.load_embedding_data(final_emb_dir, train_embedding_E_file)

    # --- 3. Hitung Statistik ---
    logger.info(f"Estimate embeddings statistics ...")
    embeddings_Z_PDFs = utils.embedding_PDFs_1D(embedding_Z)
    embeddings_3C_PDFs = utils.embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E)


    #===================================================================
    #                            Evaluation
    #===================================================================

    # 1C: noise qb le
    total_true_1C_KDE = []  # fill source_code, 
    total_pred_1C_KDE = []

    total_true_1C_Norm = []  # fill source_code, 
    total_pred_1C_Norm = []

    # 3C: noise qb le
    total_true_3C_KDE = []  # fill source_code, 
    total_pred_3C_KDE = []

    total_true_3C_Norm = []  # fill source_code, 
    total_pred_3C_Norm = []

    # mismatch result
    # {"ID":
    #    {"true": int,
    #     "pred": int,
    #     "softmax[NO,QB,LE]": list,
    #    } 
    # }
    mismatch_noise_1C_KDE = {}
    mismatch_seismic_1C_KDE = {}
    mismatch_noise_3C_KDE = {}
    mismatch_seismic_3C_KDE = {}

    """ extract embedding then make inference """

    logger.info(f"Evaluate metrics on test dataset ...")

    num_points = int(INPUT_WIN*SAMPLING_RATE)

    keys_list = list(test_data.keys())[:NUM_RECORD]


    for i in tqdm(range(len(keys_list)), desc="Progress", position=0):

        record_key = keys_list[i]

        """ parse data record """
        record = test_data[record_key]
        quake_label = record["type"].lower() # Paksa ke huruf kecil agar cocok dengan 'qb'
    
        # Ukraine hanya punya Z, kita ambil datanya
        Z_noise_data = record["Z_noise"][-num_points:]
        Z_data = record["Z"][:num_points]

        """ embed data (Z Only) """ 
        # Kita hanya melakukan embedding pada komponen Vertical (Z)
        _input_Z_noise = utils.latent_codes_1D(Z_noise_data, embedding_model)
        _input_Z = utils.latent_codes_1D(Z_data, embedding_model)

        """ infer on Z only, KDE distribution """
        # infer noise
        _noise_or_quake, _noise_likelihood, _noise_softmax = \
            utils.infer_1C_PDFs(_input_Z_noise, embeddings_Z_PDFs, choose_pdf="Kernel")
        
        # infer seismic (Ukraine LE/QB)
        _sources_code, _source_likelihood, _source_softmax = \
            utils.infer_1C_PDFs(_input_Z, embeddings_Z_PDFs, choose_pdf="Kernel") 
        
        # Simpan hasil Noise (KDE)
        total_true_1C_KDE.append(source_to_code["noise"])       
        total_pred_1C_KDE.append(_noise_or_quake)
        if source_to_code["noise"] != _noise_or_quake:
            mismatch_noise_1C_KDE[record_key] = {"true": source_to_code["noise"],
                                                "pred": int(_noise_or_quake),
                                                "softmax[NO,QB,LE]": _noise_softmax}

        # Simpan hasil Seismic (KDE)
        total_true_1C_KDE.append(source_to_code[quake_label])  
        total_pred_1C_KDE.append(_sources_code)
        if source_to_code[quake_label] != _sources_code:
            mismatch_seismic_1C_KDE[record_key] = {"true": source_to_code[quake_label],
                                                "pred": int(_sources_code),
                                                "softmax[NO,QB,LE]": _source_softmax}


        """ infer on Z only, Norm distribution """
        # infer noise
        _noise_or_quake, _, _ = \
            utils.infer_1C_PDFs(_input_Z_noise, embeddings_Z_PDFs, choose_pdf="Norm")
        # infer seismic
        _sources_code, _, _ = \
            utils.infer_1C_PDFs(_input_Z, embeddings_Z_PDFs, choose_pdf="Norm") 
        
        # Simpan hasil Noise (Norm)
        total_true_1C_Norm.append(source_to_code["noise"])       
        total_pred_1C_Norm.append(_noise_or_quake)

        # Simpan hasil Seismic (Norm)
        total_true_1C_Norm.append(source_to_code[quake_label])  
        total_pred_1C_Norm.append(_sources_code)

        

    """ calculate metrics and plot """

    ''' 1C confusion metrics '''

    #===================================================================
    #                        Calculate Metrics and Plot
    #===================================================================

    ''' 1C confusion metrics (UKRAINE - Sinkronisasi 2 Kelas) '''

    # --- 1. FILTERING DATA (Penting!) ---
    # Kita buang semua data yang berlabel 'LE' (2) agar matrix jadi 2x2
    filtered_true_KDE = []
    filtered_pred_KDE = []
    for t, p in zip(total_true_1C_KDE, total_pred_1C_KDE):
        if t < 2: # Ambil hanya NO (0) dan QB (1)
            filtered_true_KDE.append(t)
            # Jika prediksi model adalah LE (2), kita arahkan ke QB (1) atau biarkan tetap 2 
            # agar terhitung sebagai 'salah' di matrix 2x2. 
            # Namun, utils.calc_confusion_metrics biasanya butuh range yang sesuai.
            filtered_pred_KDE.append(p if p < 2 else 1) 

    # Ulangi filtering untuk Norm
    filtered_true_Norm = []
    filtered_pred_Norm = []
    for t, p in zip(total_true_1C_Norm, total_pred_1C_Norm):
        if t < 2:
            filtered_true_Norm.append(t)
            filtered_pred_Norm.append(p if p < 2 else 1)

    # --- 2. HITUNG MATRIX (Sekarang pasti 2x2) ---
    target_labels_ukr = ["NO", "QB"]

    # 1C confusion matrix, KDE
    matrix_1C_KDE, metrics_1C_KDE = utils.calc_confusion_metrics(filtered_true_KDE, filtered_pred_KDE)
    title_1C_KDE = f"{MODEL_TAG} {DATA_TAG}, 1C, KDE"
    fig_1C_KDE = utils.plot_confusion(title=title_1C_KDE,
                                    true_labels=target_labels_ukr,
                                    matrix=matrix_1C_KDE,
                                    metrics=metrics_1C_KDE,
                                    fig_size=[6, 5])

    # 1C confusion matrix, Norm
    matrix_1C_Norm, metrics_1C_Norm = utils.calc_confusion_metrics(filtered_true_Norm, filtered_pred_Norm)
    title_1C_Norm = f"{MODEL_TAG} {DATA_TAG}, 1C, Norm"
    fig_1C_Norm = utils.plot_confusion(title=title_1C_Norm,
                                    true_labels=target_labels_ukr,
                                    matrix=matrix_1C_Norm,
                                    metrics=metrics_1C_Norm,
                                    fig_size=[6, 5])

    # --- 3. TWO-CLASS (NOISE VS SEISMIC) ---
    # Bagian ini tetap bisa berjalan karena utils.two_class_convert menggabungkan QB & LE
    two_class_true_1C_KDE, two_class_pred_1C_KDE = utils.two_class_convert(filtered_true_KDE, filtered_pred_KDE)
    two_class_matrix_1C_KDE, two_class_metrics_1C_KDE = utils.calc_confusion_metrics(two_class_true_1C_KDE, two_class_pred_1C_KDE)
    title_2Class_1C_KDE = f"{MODEL_TAG} {DATA_TAG}, 2-class 1C, KDE"
    fig_2Class_1C_KDE = utils.plot_confusion(title=title_2Class_1C_KDE,
                                        true_labels=["NO", "SE"],
                                        matrix=two_class_matrix_1C_KDE,
                                        metrics=two_class_metrics_1C_KDE,
                                        fig_size=[6, 5],
                                        subAdjust=(0.34, 0.66, 0.2, 0.8))

    # --- BAGIAN 3C DIHAPUS/DIMATIKAN KARENA DATA UKRAINE HANYA 1C ---


    #===================================================================
    #                            Save Results
    #===================================================================
    # Blok ini khusus untuk output 1-Component (Z Only)
    
    logger.info(f"Save results ...")

    ''' 1. Save Figures '''
    # figure 1C, KDE
    fig_1C_KDE_name = f"{title_1C_KDE}, {time_str}.jpg"
    fig_1C_KDE.savefig(os.path.join(save_dir, fig_1C_KDE_name), dpi=300)

    # figure 2-class 1C, KDE
    fig_2Class_1C_KDE_name = f"{title_2Class_1C_KDE}, {time_str}.jpg"
    fig_2Class_1C_KDE.savefig(os.path.join(save_dir, fig_2Class_1C_KDE_name), dpi=300)

    # figure 1C, Norm
    fig_1C_Norm_name = f"{title_1C_Norm}, {time_str}.jpg"
    fig_1C_Norm.savefig(os.path.join(save_dir, fig_1C_Norm_name), dpi=300)


    ''' 2. Save Metrics (JSON) '''
    # metrics 1C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_1C_KDE_name.split('.')[0]}.json"), metrics_1C_KDE)

    # metrics 2-class 1C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_2Class_1C_KDE_name.split('.')[0]}.json"), two_class_metrics_1C_KDE)

    # metrics 1C, Norm
    dataset.save_json_data(os.path.join(save_dir, f"{fig_1C_Norm_name.split('.')[0]}.json"), metrics_1C_Norm)


    ''' 3. Save Mismatches '''
    # Membantu Bapak menganalisis record mana yang gagal dideteksi
    name_mismatch_noise_1C_KDE = f"Miss_noise_n{len(mismatch_noise_1C_KDE)}_{time_str}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_noise_1C_KDE), mismatch_noise_1C_KDE)

    name_mismatch_seismic_1C_KDE = f"Miss_seismic_n{len(mismatch_seismic_1C_KDE)}_{time_str}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_seismic_1C_KDE), mismatch_seismic_1C_KDE)

    logger.info(f"Task completed successfully. Results saved in: {save_dir}")