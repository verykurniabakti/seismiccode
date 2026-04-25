#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code base is created by: mostafamousavi
modified for LightEQ by: TayyabaZainab0807
""" 
  
from __future__ import print_function 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import gc # Garbage Collector
import tensorflow as tf
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn import utils
import pathlib
import os
import shutil
import csv
import h5py
import time
import datetime
from tensorflow.keras.models import load_model
np.seterr(divide='ignore', invalid='ignore')
from lighteq_utils import DataGenerator_test ,data_generation , DataGenerator, model_lighteq_model0,model_lighteq_model1,model_lighteq_model2, lr_schedule, generate_arrays_from_file, detector, output_writter_test, plotter
import argparse
from tqdm import tqdm
from lighteq_gen import gen
import math
import random
import sys
import statistics


np.set_printoptions(threshold=sys.maxsize)
 

parser = argparse.ArgumentParser(description='Inputs for LightEQ')    
parser.add_argument("--mode", dest='mode', default='test', help="splitdata, prepare, train,quant, test,test_accuracy")
parser.add_argument("--data_dir", dest='data_dir', default="/home/tza/STEAD/tza/merged.hdf5", type=str, help="Input file directory") 
parser.add_argument("--data_list", dest='data_list', default="/home/tza/STEAD/tza/merged.csv", type=str, help="Input csv file")
parser.add_argument("--input_model", dest='input_model', default="./lighteq_original_test_outputs/final_model.h5", type=str, help="The pre-trained model used for the prediction")
parser.add_argument("--input_testdir", dest='input_testdir', default= "/home/tza/STEAD/tza/", type=str, help="List set set directory")
parser.add_argument("--output_dir", dest='output_dir', default='lighteq_original_test', type=str, help="Output directory")
parser.add_argument("--batch_size", dest='batch_size', default= 100, type=int, help="batch size")  
parser.add_argument("--epochs", dest='epochs', default= 20, type=int, help="number of epochs (default: 100)")
parser.add_argument('--gpuid', dest='gpuid', type=int, default=3, help='specifyin the GPU')
parser.add_argument('--gpu_limit', dest='gpu_limit', type=float, default=0.8, help='limiting the GPU memory')
parser.add_argument("--input_dimention", dest='input_dimention', default=(151, 41), type=int, help="a tuple including the time series lenght and number of channels.")  
parser.add_argument("--shuffle", dest='shuffle', default= True, type=bool, help="shuffling the list during the preprocessing and training")
parser.add_argument("--label_type",dest='label_type',  default='triangle', type=str, help="label type for picks: 'gaussian', 'triangle', 'box' ") 
parser.add_argument("--normalization_mode", dest='normalization_mode', default='max', type=str, help="normalization mode for preprocessing: 'std' or 'max' ") 
parser.add_argument("--augmentation", dest='augmentation', default= False, type=bool, help="if True, half of each batch will be augmented")  
parser.add_argument("--add_event_r", dest='add_event_r', default= 0.5, type=float, help=" chance for randomly adding a second event into the waveform") 
parser.add_argument("--shift_event_r", dest='shift_event_r', default= 0.9, type=float, help=" shift the event") 
parser.add_argument("--add_noise_r", dest='add_noise_r', default= 0.4, type=float, help=" chance for randomly adding Gaussian noise into the waveform")  
parser.add_argument("--scale_amplitude_r", dest='scale_amplitude_r', default= None, type=float, help=" chance for randomly amplifying the waveform amplitude ") 
parser.add_argument("--pre_emphasis", dest='pre_emphasis', default= False, type= bool, help=" if raw waveform needs to be pre emphesis ")
parser.add_argument("--train_valid_test_split", dest='train_valid_test_split', default=[0.85, 0.05, 0.10], type= float, help=" precentage for spliting the data into training, validation, and test sets")  
parser.add_argument("--patience", dest='patience', default= 5, type= int, help=" patience for early stop monitoring ") 
parser.add_argument("--detection_threshold",dest='detection_threshold',  default=0.005, type=float, help="Probability threshold for P pick")
parser.add_argument("--report", dest='report', default=False, type=bool, help="summary of the training settings and results.")
parser.add_argument("--plot_num", dest='plot_num', default= 50, type=int, help="number of plots for the test or predition results")
parser.add_argument("--tflite_modes", dest='tflite_modes', default= 0, type=int, help="flavor of tensorflow lite model")
parser.add_argument("--quant", dest='quant', default= 0, type=int, help="do you want to test the quantized model?")
parser.add_argument("--output", dest='output', default= 76, type=int, help="number of predictions")




args = parser.parse_args()


tflite=args.quant
tflite_modes=args.tflite_modes  

print("mode is: ",tflite_modes)
print('Mode is: {}'.format(args.mode))  

#Using GPUs
if (args.gpuid!=None ) :           
    os.environ['CUDA_VISIBLE_DEVICES'] ='{}'.format(args.gpuid)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

        
#Finding the output directory
save_dir = os.path.join(os.getcwd(), str(args.output_dir)+'_outputs')       


def subset_data():
    df = pd.read_csv(args.data_list)
    ev_list = df.trace_name.tolist()
    
    np.random.shuffle(ev_list)  
    
    # Rasio pembagian sesuai riset LightEQ: 85% train, 5% valid, 10% test
    # Kita bagi ev_list secara berurutan
    train_end = int(args.train_valid_test_split[0] * len(ev_list))
    val_end = int((args.train_valid_test_split[0] + args.train_valid_test_split[1]) * len(ev_list))
    
    training = ev_list[:train_end]
    validation = ev_list[train_end:val_end]
    test = ev_list[val_end:]
    
    # Path output ke SSD kamu
    ssd_output_path = '/Volumes/Extreme SSD/stream_stead/data_stead/'

    # Pastikan folder tujuan ada (opsional tapi aman)
    if not os.path.exists(ssd_output_path):
        os.makedirs(ssd_output_path)

    # Simpan ke SSD
    np.save(ssd_output_path + 'training.npy', training)
    np.save(ssd_output_path + 'validation.npy', validation)
    np.save(ssd_output_path + 'test.npy', test)
    
    print(f"Berhasil menyimpan indeks ke {ssd_output_path}")
    print(f"Training: {len(training)}, Validation: {len(validation)}, Test: {len(test)}")


    
### Convert to TFLite Model
def convert_model(model):
    
    run_model = tf.function(lambda x: model(x))
    # Set the fixed input to the model as a concrete function
    # NEW HERE: I fix the bach size to 1, but keep the sequence size to None (dynamic)
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,151,41,3], model.inputs[0].dtype))
    # save the Keras model with fixed input
    MODEL_DIR = "keras_lstm"
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    # Create converter from saved model
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    Choice="_float32"
    #tflite_modes==0 is for 32-bit float model..
    
    #code for compressed model..
    if(tflite_modes==1):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    #code for 8-bit quantized model..
    if(tflite_modes==2):
        Choice="_int8"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
        def generate_representative_dataset():
            X_train=np.load('/home/tza/STEAD/tza/X_train'+str(args.output)+'.npy')
            for i in range(int(X_train.shape[0]/100)):
                print(i,end="\r")
                yield [tf.expand_dims(X_train[i], axis=0)]
                #yield [X_train[i]]
        # Converter will use the above function to optimize quantization
        converter.representative_dataset = generate_representative_dataset
     
    #code for 8-bit quantized weights and 16-bit activation model..
    if(tflite_modes==3): 
        Choice="_int8w16"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
     
    # Convert model
    tflite_model = converter.convert()
    open("keras_lstm/model"+Choice+".tflite", "wb").write(tflite_model)
    return tflite_model



def predict_TFLite(model, X):
    x_data = np.copy(X) # the function quantizes the input, so we must make a copy
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    outputs = []
    
    # Quantize input if needed
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        x_data = x_data / input_scale + input_zero_point
        x_data = x_data.astype(input_details["dtype"])
    
    
    for i in range(len(x_data)):
        # We need to resize the input shape to fit the dynamic sequence (batch size must be queal to 1)
        interpreter.resize_tensor_input(input_details['index'], (1,)+x_data[i].shape, strict=True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details["index"], [x_data[i]])
        interpreter.invoke()
        outputs.append(np.copy(interpreter.get_tensor(output_details["index"])))
    
    
    # Dequantize output
    outputs = np.array(outputs)
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        outputs = outputs.astype(np.float32)
        outputs = (outputs - output_zero_point) * output_scale
    # todo reshape output into array for each exit
    return outputs


##################################### Preparing Data ##########################################
###############################################################################################


if args.mode == 'splitdata':
    subset_data() # splits the data into train_valid_test_split= [0.85, 0.05, 0.10]


    
if args.mode == 'prepare':
    # Path ke SSD kamu
    ssd_path = '/Volumes/Extreme SSD/stream_stead/data_stead/'
    
    # Memuat file indeks yang dibuat di tahap splitdata
    training = np.load(ssd_path + 'training.npy')
    test = np.load(ssd_path + 'test.npy')
    
    params = {
        'file_name': args.data_dir, # Ini akan mengambil dari --data_dir di terminal
        'dim': args.input_dimention,
        'batch_size': args.batch_size,
        'n_channels': 3,
        'target_length': args.output,
        'shuffle': True,
        'norm_mode': args.normalization_mode,
        'augmentation': args.augmentation,
        'add_event_r': args.add_event_r,  
        'shift_event_r': args.shift_event_r,  
        'add_noise_r': args.add_noise_r, 
        'scale_amplitude_r': args.scale_amplitude_r,
        'pre_emphasis': args.pre_emphasis
    }  
    
    # Menjalankan generator untuk membuat fitur STFT
    #gen(params, 'train', training)
    gen(params, 'test', test)

    
##################################### Training Model ##########################################
###############################################################################################

if args.mode == 'train':
    print("tflite_modes", args.tflite_modes)
    ssd_path = '/Volumes/Extreme SSD/stream_stead/data_stead/stead_lighteq_sfft/'
    
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir) 
    save_models = os.path.join(save_dir, 'models')
    os.makedirs(save_models)

    # 1. Inisialisasi Arsitektur Model
    # Kita gunakan model_lighteq_model2 sesuai pilihan riset Anda
    model = model_lighteq_model2(8,16,32,32,16,1,1,2,1,1,9,7,5,3,3,0.4,0.73,0.82,0.71,0.66,32,32,64,1,3)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['binary_accuracy'])
    model.summary()

    # 2. Loop Pelatihan Bertahap (Incremental Training)
    # Total ada 22 partisi (0-21). Kita muat 3 partisi per sesi (~11GB RAM)
    total_parts = 22
    step_size = 1 
    
    start_time = time.time()

    for epoch_cycle in range(args.epochs):
        print(f"\n========== SIKLUS EPOCH {epoch_cycle + 1} / {args.epochs} ==========")
        
        for start_p in range(0, total_parts, step_size):
            end_p = min(start_p + step_size, total_parts)
            print(f"\n--- Memuat data Part {start_p} sampai {end_p-1} ---")
            
            X_block_list = []
            y_block_list = []
            
            # Memuat file dari folder stead_lighteq_sfft
            for p in range(start_p, end_p):
                X_block_list.append(np.load(f"{ssd_path}X_train_{args.output}_part{p}.npy"))
                y_block_list.append(np.load(f"{ssd_path}y_train_{args.output}_part{p}.npy"))
            
            # Menggabungkan partisi yang dimuat menjadi satu blok training
            X_train_block = np.concatenate(X_block_list, axis=0)
            y_train_block = np.concatenate(y_block_list, axis=0)
            
            # Acak data di dalam blok agar tidak bias
            X_train_block, y_train_block = utils.shuffle(X_train_block, y_train_block)

            # Training pada blok ini (kita gunakan 10% data blok sebagai validasi internal)
            model.fit(X_train_block, y_train_block, 
                      batch_size=args.batch_size, 
                      epochs=1, 
                      validation_split=0.1,
                      verbose=1)
            
            # PENTING: Bersihkan RAM segera setelah blok ini selesai dilatih
            # Hapus variabel dari referensi Python
            del X_train_block, y_train_block, X_block_list, y_block_list

            # PAKSA pembersihan memori fisik
            gc.collect() 

            # Bersihkan graph dan memori pada GPU Metal
            tf.keras.backend.clear_session() 

            print(f"--- Selesai Blok {start_p}-{end_p-1}. RAM & GPU dibersihkan. ---")

        # Simpan checkpoint setiap akhir satu siklus penuh (22 partisi)
        checkpoint_path = os.path.join(save_models, f'model_epoch_{epoch_cycle+1}.h5')
        model.save(checkpoint_path)
        print(f"Checkpoint disimpan: {checkpoint_path}")

    # 3. Simpan Model Akhir
    model.save(save_dir + '/final_model_lighteq_m3pro.h5')
    print(f"\n[SELESAI] Total waktu: {time.time() - start_time:.2f} detik")


    # we train the model
    model.summary()
    
    patiences= 5
    early_stopping_monitor = EarlyStopping(patience=patiences)
    
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=patiences-2,
                                   min_lr=0.5e-6)

    history=model.fit(training_generator, validation_data=validation_generator,epochs=args.epochs,callbacks=[lr_reducer, lr_scheduler, early_stopping_monitor])

        
    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'], '--')
    ax.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['binary_accuracy'])
    ax.plot(history.history['val_binary_accuracy'], '--')
    ax.legend(['accuracy', 'val_accuracy'], loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_accuracy.png')))
    
##################################### Quantizing ##############################################
###############################################################################################
    
    
if args.mode == 'quant':    
    model1 = load_model(args.input_model)
    tflite_model = convert_model(model1)


##################################### Testing Model ###########################################
###############################################################################################
if args.mode == 'test':
    # Function for testing model withoput quantization
    if(tflite==0): 
        save_figs = os.path.join(save_dir, 'figures')
        if os.path.isdir(save_figs):shutil.rmtree(save_figs) 
        os.makedirs(save_figs) 

        csvTst = open(os.path.join(save_dir,'X_test_results_normal.csv'), 'w')          
        test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['network_code', 
                              'ID', 
                              'earthquake_distance_km', 
                              'snr_db', 
                              'trace_name', 
                              'trace_category', 
                              'trace_start_time', 
                              'source_magnitude', 
                              'p_arrival_sample',
                              'p_status', 
                              'p_weight',
                              's_arrival_sample', 
                              's_status', 
                              's_weight', 
                              'receiver_type',

                              'number_of_detections',
                               'detection_probability'
                               ])  
        csvTst.flush()        
        plt_n = 0
        
        print('Loading the model ...')        
        model = load_model(args.input_model)

        model.compile(loss='binary_crossentropy',
                       optimizer=Adam(lr=lr_schedule(0)),
                       metrics=['binary_accuracy'])
        
        print('Loading is complete!')  
          
        testlist = np.load(args.input_testdir+"test.npy")
        final_test_x= np.load('/home/tza/STEAD/tza/X_test'+str(args.output)+'.npy')
        final_test_y= np.load('/home/tza/STEAD/tza/y_test'+str(args.output)+'.npy')
    
        list_generator = generate_arrays_from_file(testlist, args.batch_size)
        list_generator_series_x = generate_arrays_from_file(final_test_x, args.batch_size) 
        list_generator_series_y = generate_arrays_from_file(final_test_y, args.batch_size) 
        
        length=int(np.floor(len(final_test_x) / args.batch_size))
        pbar_test = tqdm(total=length) 
    
        fl = h5py.File(args.data_dir, 'r')
        my_i = 0
       
        loss = [0] * length
        accuracy = [0] * length
        
        params_test = {'file_name': args.data_dir,
                       'batch_size': args.batch_size} 
    
        
  
        for i in range(length):
            pbar_test.update()
            new_list = next(list_generator)
            new_series= next(list_generator_series_x)
            predD = model.predict(new_series,batch_size=args.batch_size)
         
            test_set={}
            
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/'+str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/'+str(ID))
                test_set.update( {str(ID) : dataset})                 

            if len(predD) > 0:
                for ts in range(predD.shape[0]):
                    evi =  new_list[ts] 
                    dataset = test_set[evi]  
                    try:
                        spt = int(dataset.attrs['p_arrival_sample']);
                    except Exception:     
                        spt = None
                    try:
                        sst = int(dataset.attrs['s_arrival_sample']);
                    except Exception:     
                        sst = None   
                    if(my_i<=5):
                        print(evi,predD[ts])
                        my_i+=1
                    matches = detector(args, predD[ts])
                    output_writter_test(args, 
                                       dataset, 
                                       evi, 
                                       test_writer,
                                       csvTst,
                                       matches)

                    if plt_n < args.plot_num:
                        plotter(ts, 
                               dataset,
                               evi,
                               args, 
                               save_figs, 
                               predD[ts], 
                                matches)
                        plt_n += 1;  
       
    # Function for testing model with quantization
    if(tflite==1):
        
        interpreter = tf.lite.Interpreter(model_path='keras_lstm/model.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(input_details)
        print(output_details)


        save_figs = os.path.join(save_dir, 'figures')
        if os.path.isdir(save_figs):
            shutil.rmtree(save_figs) 
        os.makedirs(save_figs) 

        csvTst = open(os.path.join(save_dir,'X_test_results_normal.csv'), 'w')          
        test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['network_code', 
                              'ID', 
                              'earthquake_distance_km', 
                              'snr_db', 
                              'trace_name', 
                              'trace_category', 
                              'trace_start_time', 
                              'source_magnitude', 
                              'p_arrival_sample',
                              'p_status', 
                              'p_weight',
                              's_arrival_sample', 
                              's_status', 
                              's_weight', 
                              'receiver_type',

                              'number_of_detections',
                              'detection_probability'
                              ])  
        csvTst.flush()        

        plt_n = 0
        testlist = np.load(args.input_testdir+"test.npy")
        testTimeSeries = np.load('/home/tza/STEAD/tza/X_test'+str(args.output)+'.npy')

        start_time = time.time() 
        list_generator = generate_arrays_from_file(testlist, args.batch_size)
        list_generator_series = generate_arrays_from_file(testTimeSeries, args.batch_size)  
        i = 0
        my_i=0
        pbar_test = tqdm(total=int(np.floor(len(testlist)/args.batch_size)))
        fl = h5py.File(args.data_dir, 'r')
        for i in range(int(np.floor(len(testlist) / args.batch_size))):
            pbar_test.update()
            new_list = next(list_generator)
            new_series = next(list_generator_series)
            
            predD = np.zeros((0,args.output,1))  
            for x in range(int(args.batch_size)):
                test_sample = np.expand_dims(new_series[x,:,:,:], axis=0).astype(input_details[0]["dtype"])
                interpreter.set_tensor(input_details[0]['index'], test_sample)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                predD=np.concatenate((predD, pred), axis=0)
                

            test_set={}
            
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/'+str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/'+str(ID))
                test_set.update( {str(ID) : dataset})                 

            if len(predD) > 0:
                for ts in range(predD.shape[0]):
                    evi =  new_list[ts] 
                    dataset = test_set[evi]  
                    try:
                        spt = int(dataset.attrs['p_arrival_sample']);
                    except Exception:     
                        spt = None
                    try:
                        sst = int(dataset.attrs['s_arrival_sample']);
                    except Exception:     
                        sst = None 
                    if(my_i<=5):
                        print(evi,predD[ts])
                        my_i+=1
                    matches = detector(args, predD[ts]) 
                    output_writter_test(args, 
                                       dataset, 
                                       evi, 
                                       test_writer,
                                       csvTst,
                                       matches)


                    if plt_n < args.plot_num:
                        plotter(ts, 
                               dataset,
                               evi,
                               args, 
                               save_figs, 
                               predD[ts], 
                               matches)
                        
                        plt_n += 1;  
        
        end_time = time.time()
    

    

   
    
    
    
