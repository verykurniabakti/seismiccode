# -*- coding: utf-8 -*-
"""
A demonstration benchmarks the MCU-Quake (5-20) using 100 samples from the UUSS dataset. The full data are provided in the Data Availability section of the manuscript. 

Note that the performance of MCU-Quake can be improved by employing a more sophisticated interpretation method for the model's output embeddings.
"""


# -*- coding: utf-8 -*-
import sys
import os

# WAJIB PALING ATAS: Daftarkan path agar folder 'Library' bisa ditemukan
ROOT_PATH = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# Sekarang baru bisa import modul lokal dan library lainnya
from Library import utils, dataset
import config
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging
# ... lanjutkan dengan fungsi plot_save_mismatches dan blok main

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
    source_to_code = {"noise": 0, "qb": 1, "le": 2}
    code_to_source = {0: "noise", 1: "qb", 2: "le"}

    # Path ke folder STEAD
    key_data_dir = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/Benchmark_ STEAD 3C_ test n15275 r100"

    """ Dataset STEAD """ 
    DATA_TAG = "STEAD"
    key_test_file = "STEAD data, test n15275 r100.json"
    
    true_labels = ["NO", "QB", "LE"]

    # Path base untuk model dan embedding
    base_path = "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/rep_code"
    
    embedding_3C_dir = os.path.join(base_path, "Typical embedding", "Embedding_data train 3C, UUSS n11275 std15, 30120909")
   

    #===================================================================
    #                        MCU-Quake: 5-20, 7-second
    #===================================================================

    """ model config """
    INPUT_WIN = 7 
    SAMPLING_RATE = 100
    
    # TAMBAHKAN BARIS INI:
    MODEL_TAG = "MCU_5-20" 

    # Gunakan Path Absolut ke folder Pre-trained model
    model_path = os.path.join(base_path, "Pre-trained model", "MCU-Quake 5-20")

    #===================================================================
    #                           Prepare files
    #===================================================================

    """ Load dataset """
    print(f"[INFO] load test dataset ...")
    test_data = dataset.load_json_data(os.path.join(key_data_dir, key_test_file)) 

    NUM_RECORD = len(test_data)   # all the dataset
    # NUM_RECORD = 256             # debug

    """ create log file """

    SAVE_BASE = f"{config.OUTPUT_PATH} benchmark metrics"

    now = datetime.now()
    time_str = now.strftime("%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE, f"{MODEL_TAG} {DATA_TAG}, {time_str}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # create and configure logger
    logging.basicConfig(filename=os.path.join(save_dir, "task log.txt"),
                        level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s]: %(message)s',
                        datefmt = "%Y-%m-%d %H:%M:%S",
                        filemode='w')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Task summary:\n"
                f"Dataset: {os.path.join(key_data_dir, key_test_file)}\n"
                f"Model: {MODEL_TAG}, key embeddings dir: {embedding_3C_dir}\n"
                f"True labels: {true_labels}\n"
                f"Source to code: {source_to_code}\n"
                f"Code to source: {code_to_source}\n"
                )


    logger.info(f"Load data files ...")

    # embedding model
    embedding_model = keras.models.load_model(filepath=model_path)

    train_embedding_Z_file = "Embedding data, Z.json"
    train_embedding_N_file = "Embedding data, N.json"
    train_embedding_E_file = "Embedding data, E.json"

    embedding_Z = dataset.load_embedding_data(embedding_3C_dir, train_embedding_Z_file)
    embedding_N = dataset.load_embedding_data(embedding_3C_dir, train_embedding_N_file)
    embedding_E = dataset.load_embedding_data(embedding_3C_dir, train_embedding_E_file)

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
        record = test_data[record_key]
        quake_label = record["type"]
    
        # 1. Ambil sinyal utama
        Z_data = record["Z"][:num_points]
        N_data = record["N"][:num_points]
        E_data = record["E"][:num_points]

        # 2. Ambil sinyal noise (KOREKSI: Samakan nama variabel di sini)
        Z_noise_data = record["Z_noise"][-num_points:]
        N_noise_data = record["N_noise"][-num_points:]
        E_noise_data = record["E_noise"][-num_points:]

        """ embed data """ 
        # Sekarang pemanggilan ini tidak akan menyebabkan NameError
        _input_Z_noise = utils.latent_codes_1D(Z_noise_data, embedding_model)
        _input_N_noise = utils.latent_codes_1D(N_noise_data, embedding_model)
        _input_E_noise = utils.latent_codes_1D(E_noise_data, embedding_model)

        _input_Z = utils.latent_codes_1D(Z_data, embedding_model)
        _input_N = utils.latent_codes_1D(N_data, embedding_model)
        _input_E = utils.latent_codes_1D(E_data, embedding_model)    


        """ infer on Z only, KDE distribution """
        # infer noise
        _noise_or_quake, _noise_likelihood, _noise_softmax = \
            utils.infer_1C_PDFs(_input_Z_noise, embeddings_Z_PDFs, choose_pdf="Kernel")
        # infer seismic
        _sources_code, _source_likelihood, _source_softmax = \
            utils.infer_1C_PDFs(_input_Z, embeddings_Z_PDFs, choose_pdf="Kernel") 
        
        # add noise result
        total_true_1C_KDE.append(source_to_code["noise"])       
        total_pred_1C_KDE.append(_noise_or_quake)
        if source_to_code["noise"] != _noise_or_quake:
            mismatch_noise_1C_KDE[record_key] = {"true": source_to_code["noise"],
                                                "pred": int(_noise_or_quake),
                                                "softmax[NO,QB,LE]": _noise_softmax}

        # add seismic source result
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
        
        # add noise result
        total_true_1C_Norm.append(source_to_code["noise"])       
        total_pred_1C_Norm.append(_noise_or_quake)

        # add seismic source result
        total_true_1C_Norm.append(source_to_code[quake_label])  
        total_pred_1C_Norm.append(_sources_code)


        """ infer on 3-component in 3D space, KDE distribution """

        # infer noise
        input_embeddings_3C_noise = np.array([_input_E_noise, _input_N_noise, _input_Z_noise]).reshape(1,-1)
        _noise_or_quake,_noise_likelihood, _noise_softmax = \
            utils.infer_3C_PDFs(input_embeddings_3C_noise, embeddings_3C_PDFs, choose_pdf="Kernel")

        # infer quake sources
        input_embeddings_3C = np.array([_input_E, _input_N, _input_Z]).reshape(1,-1)
        _sources_code, _source_likelihood, _source_softmax = \
            utils.infer_3C_PDFs(input_embeddings_3C, embeddings_3C_PDFs, choose_pdf="Kernel")

        # add noise result
        total_true_3C_KDE.append(source_to_code["noise"])       
        total_pred_3C_KDE.append(_noise_or_quake)
        if source_to_code["noise"] != _noise_or_quake:
            mismatch_noise_3C_KDE[record_key] = {"true": source_to_code["noise"],
                                                "pred": int(_noise_or_quake),
                                                "softmax[NO,QB,LE]": _noise_softmax}

        # add seismic sources result
        total_true_3C_KDE.append(source_to_code[quake_label])  
        total_pred_3C_KDE.append(_sources_code)      
        if source_to_code[quake_label] != _sources_code:
            mismatch_seismic_3C_KDE[record_key] = {"true": source_to_code[quake_label],
                                                "pred": int(_sources_code),
                                                "softmax[NO,QB,LE]": _source_softmax}

        """ infer on 3-component in 3D space, Normal distribution """
        # for noise
        _noise_or_quake, _, _ = \
            utils.infer_3C_PDFs(input_embeddings_3C_noise, embeddings_3C_PDFs, choose_pdf="Norm")
        
        # for quake
        _sources_code, _, _ = \
            utils.infer_3C_PDFs(input_embeddings_3C, embeddings_3C_PDFs, choose_pdf="Norm")

        # add noise result
        total_true_3C_Norm.append(source_to_code["noise"])       
        total_pred_3C_Norm.append(_noise_or_quake)
        # add seismic source result
        total_true_3C_Norm.append(source_to_code[quake_label])  
        total_pred_3C_Norm.append(_sources_code)   



    """ calculate metrics and plot """

    ''' 1C confusion metrics '''

    # 1C confusion matrix, KDE
    matrix_1C_KDE, metrics_1C_KDE = utils.calc_confusion_metrics(total_true_1C_KDE, total_pred_1C_KDE)

    title_1C_KDE = f"{MODEL_TAG} {DATA_TAG}, 1C, KDE"
    fig_1C_KDE = utils.plot_confusion(title=title_1C_KDE,
                                    true_labels=true_labels,
                                    matrix=matrix_1C_KDE,
                                    metrics=metrics_1C_KDE,
                                    fig_size=[6, 5])

    # confusion matrix for 2-class only, KDE: noise and seismic
    two_class_true_1C_KDE, two_class_pred_1C_KDE= utils.two_class_convert(total_true_1C_KDE, total_pred_1C_KDE)
    two_class_matrix_1C_KDE, two_class_metrics_1C_KDE = utils.calc_confusion_metrics(two_class_true_1C_KDE, two_class_pred_1C_KDE)
    title_2Class_1C_KDE = f"{MODEL_TAG} {DATA_TAG}, 2-class 1C, KDE"
    fig_2Class_1C_KDE = utils.plot_confusion(title=title_2Class_1C_KDE,
                                        true_labels=["NO", "SE"],
                                        matrix=two_class_matrix_1C_KDE,
                                        metrics=two_class_metrics_1C_KDE,
                                        fig_size=[6, 5],
                                        subAdjust=(0.34, 0.66, 0.2, 0.8))


    # 1C confusion matrix, Norm
    matrix_1C_Norm, metrics_1C_Norm = utils.calc_confusion_metrics(total_true_1C_Norm, total_pred_1C_Norm)

    title_1C_Norm = f"{MODEL_TAG} {DATA_TAG}, 1C, Norm"
    fig_1C_Norm = utils.plot_confusion(title=title_1C_Norm,
                                    true_labels=true_labels,
                                    matrix=matrix_1C_Norm,
                                    metrics=metrics_1C_Norm,
                                    fig_size=[6, 5])

    ''' 3C confusion metrics '''

    # 3C confusion matrix, KDE
    matrix_3C_KDE, metrics_3C_KDE = utils.calc_confusion_metrics(total_true_3C_KDE, total_pred_3C_KDE)
    title_3C_KDE = f"{MODEL_TAG} {DATA_TAG}, 3C, KDE"
    fig_3C_KDE = utils.plot_confusion(title=title_3C_KDE,
                                true_labels=true_labels,
                                matrix=matrix_3C_KDE,
                                metrics=metrics_3C_KDE,
                                fig_size=[6, 5])

    # confusion matrix for 2-class only: noise and seismic
    two_class_true_3C_KDE, two_class_pred_3C_KDE = utils.two_class_convert(total_true_3C_KDE, total_pred_3C_KDE)
    two_class_matrix_3C_KDE, two_class_metrics_3C_KDE = utils.calc_confusion_metrics(two_class_true_3C_KDE,
                                                                            two_class_pred_3C_KDE)
    title_2Class_3C_KDE = f"{MODEL_TAG} {DATA_TAG}, 2-class 3C, KDE"
    fig_2Class_3C_KDE = utils.plot_confusion(title=title_2Class_3C_KDE,
                                        true_labels=["NO", "SE"],
                                        matrix=two_class_matrix_3C_KDE,
                                        metrics=two_class_metrics_3C_KDE,
                                        fig_size=[6, 5],
                                        subAdjust=(0.34, 0.66, 0.2, 0.8))

    # 3C confusion matrix, Norm
    matrix_3C_Norm, metrics_3C_Norm = utils.calc_confusion_metrics(total_true_3C_Norm, total_pred_3C_Norm)
    title_3C_Norm = f"{MODEL_TAG} {DATA_TAG}, 3C, Norm"
    fig_3C_Norm = utils.plot_confusion(title=title_3C_Norm,
                                true_labels=true_labels,
                                matrix=matrix_3C_Norm,
                                metrics=metrics_3C_Norm,
                                fig_size=[6, 5])


    #===================================================================
    #                            Save results
    #===================================================================

    logger.info(f"Save results ...")

    ''' save figures '''

    # figure 1C, KDE
    fig_1C_KDE_name = f"{title_1C_KDE}, {time_str}.jpg"
    fig_1C_KDE.savefig(os.path.join(save_dir, fig_1C_KDE_name), dpi=300)

    # figure 2-class 1C, KDE
    fig_2Class_1C_KDE_name = f"{title_2Class_1C_KDE}, {time_str}.jpg"
    fig_2Class_1C_KDE.savefig(os.path.join(save_dir, fig_2Class_1C_KDE_name), dpi=300)

    # figure 1C, Norm
    fig_1C_Norm_name = f"{title_1C_Norm}, {time_str}.jpg"
    fig_1C_Norm.savefig(os.path.join(save_dir, fig_1C_Norm_name), dpi=300)


    # figure 3C, KDE
    fig_3C_KDE_name = f"{title_3C_KDE}, {time_str}.jpg"
    fig_3C_KDE.savefig(os.path.join(save_dir, fig_3C_KDE_name), dpi=300)

    # figure 2-class 3C, KDE
    fig_2Class_3C_KDE_name  = f"{title_2Class_3C_KDE}, {time_str}.jpg"
    fig_2Class_3C_KDE.savefig(os.path.join(save_dir, fig_2Class_3C_KDE_name), dpi=300)

    # figure 3C, Norm
    fig_3C_Norm_name = f"{title_3C_Norm}, {time_str}.jpg"
    fig_3C_Norm.savefig(os.path.join(save_dir, fig_3C_Norm_name), dpi=300)


    ''' save metrics '''

    # metrics 1C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_1C_KDE_name.split('.')[0]}.json"), metrics_1C_KDE)

    # metrics 2-class 1C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_2Class_1C_KDE_name.split('.')[0]}.json"), two_class_metrics_1C_KDE)

    # metrics 1C, Norm
    dataset.save_json_data(os.path.join(save_dir, f"{fig_1C_Norm_name.split('.')[0]}.json"), metrics_1C_Norm)

    # metrics 3C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_3C_KDE_name.split('.')[0]}.json"), metrics_3C_KDE)

    # metrics 2-class 3C, KDE
    dataset.save_json_data(os.path.join(save_dir, f"{fig_2Class_3C_KDE_name.split('.')[0]}.json"), two_class_metrics_3C_KDE)

    # metrics 3C, Norm
    dataset.save_json_data(os.path.join(save_dir, f"{fig_3C_Norm_name.split('.')[0]}.json"), metrics_3C_Norm)


    ''' save mismatches '''
    name_mismatch_noise_1C_KDE = f"Miss noise n{len(mismatch_noise_1C_KDE)}, {title_1C_KDE}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_noise_1C_KDE), mismatch_noise_1C_KDE)

    name_mismatch_seismic_1C_KDE = f"Miss seismic n{len(mismatch_seismic_1C_KDE)}, {title_1C_KDE}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_seismic_1C_KDE), mismatch_seismic_1C_KDE)

    name_mismatch_noise_3C_KDE = f"Miss noise n{len(mismatch_noise_3C_KDE)}, {title_3C_KDE}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_noise_3C_KDE), mismatch_noise_3C_KDE)

    name_mismatch_seismic_3C_KDE = f"Miss seismic n{len(mismatch_seismic_3C_KDE)}, {title_3C_KDE}.json"
    dataset.save_json_data(os.path.join(save_dir, name_mismatch_seismic_3C_KDE), mismatch_seismic_3C_KDE)


    logger.info(f"Task completed.")