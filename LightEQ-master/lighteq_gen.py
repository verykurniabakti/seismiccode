"""
Code base is created by: mostafamousavi
modified for LightEQ by: TayyabaZainab0807
""" 

import tensorflow as tf
import numpy as np
from scipy import signal
import math
import h5py




def normalize(data, mode = 'max'):
    #print('*************normalize')
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        data /= max_data    
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data


def shift_event(data, addp, adds, coda_end, snr, rate):
    org_len = len(data) 
    data2 = None
    addp2 = None
    adds2 = None
    coda_end2 = None        
    if np.random.uniform(0, 1) < rate and all(snr >= 5.0):

        space = int(org_len - coda_end)
        preNoise = int(addp)-100 

        noise0 = data[:preNoise, :]
        noise1 = noise0
        if preNoise > 0:
            repN = int(np.floor(space/preNoise))-1            

            if repN >= 5:
                for _ in range(np.random.randint(1, repN)):        
                    noise1 = np.concatenate([noise1, noise0], axis=0)
            else: 
                for _ in range(repN):        
                    noise1 = np.concatenate([noise1, noise0], axis=0)                

            data2 = np.concatenate([noise1, data], axis=0)
            data2 = data2[:org_len, :]
            if addp+len(noise1) >= 0 and addp+len(noise1) < org_len:
                addp2 = addp+len(noise1)
            else:
                addp2 = None

            if adds+len(noise1) >= 0 and adds+len(noise1) < org_len:               
                adds2 = adds+len(noise1)
            else:
                adds2 = None

            if coda_end+len(noise1) < org_len:                              
                coda_end2 = coda_end+len(noise1) 
            else:
                coda_end2 = org_len 

            if addp2 and adds2:
                data = data2
                addp = addp2
                adds = adds2
                coda_end= coda_end2 

    return data, addp, adds, coda_end     


def scale_amplitude(self, data, rate):
    #print('*************scale amplitude')
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2*rate:
        data /= np.random.uniform(1, 3)
    return data

def add_noise(data, snr, rate):
    #print('*************add noise')
    data_noisy = np.empty((data.shape))
    if np.random.uniform(0, 1) < rate and all(snr >= 5.0): 
        data_noisy = np.empty((data.shape))
        noise = np.random.normal(0,1,data.shape[0])
        data_noisy[:, 0] = data[:,0] + 0.5*(noise*(10**(snr[0]/10)))* np.random.random()
        data_noisy[:, 1] = data[:,1] + 0.5*(noise*(10**(snr[1]/10)))* np.random.random()
        data_noisy[:, 2] = data[:,2] + 0.5*(noise*(10**(snr[2]/10)))* np.random.random()    
    else:
        data_noisy = data
    return data_noisy  

def add_event(data, addp, adds, coda_end, snr, rate): 
    #print('*************add event')
    added = np.copy(data)
    additions = None
    spt_secondEV = None
    sst_secondEV = None
    if addp and adds:
        s_p = adds - addp
        if np.random.uniform(0, 1) < rate and all(snr >= 5.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
            secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
            space = data.shape[0]-secondEV_strt  
            added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*np.random.uniform(0, 1)
            added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*np.random.uniform(0, 1) 
            added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*np.random.uniform(0, 1)          
            spt_secondEV = secondEV_strt   
            if  spt_secondEV + s_p + 21 <= data.shape[0]:
                sst_secondEV = spt_secondEV + s_p
            if spt_secondEV and sst_secondEV:                                                                     
                additions = [spt_secondEV, sst_secondEV] 
                data = added

    return data, additions    
    

def pre_emphasis(n_channels, data, pre_emphasis = 0.97):
    #print('*************pre emphasis')
    for ch in range(n_channels): 
        bpf = data[:, ch]  
        data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
    return data

    
##############################helper functions################################

def gen(param, opt, list_IDs_temp):
    lengthTotal = len(list_IDs_temp)
    # Kita proses per 50.000 data agar beban RAM hanya ~3.7 GB
    chunk_size = 50000  
    ssd_output_path = '/Volumes/Extreme SSD/stream_stead/data_stead/'
    fl = h5py.File(param['file_name'], 'r')

    num_chunks = math.ceil(lengthTotal / chunk_size)
    print(f"Memulai pemrosesan bertahap: Total {lengthTotal} data dibagi menjadi {num_chunks} bagian.")

    for c in range(num_chunks):
        start_idx = c * chunk_size
        end_idx = min((c + 1) * chunk_size, lengthTotal)
        current_list = list_IDs_temp[start_idx:end_idx]
        current_len = len(current_list)

        # Alokasi memori hanya untuk bagian (chunk) ini saja
        X_chunk = np.empty((current_len, param['dim'][0], param['dim'][1], param['n_channels']), dtype='float32')
        y_chunk = np.zeros((current_len, param['target_length'], 1), dtype='float32')

        for i, ID in enumerate(current_list):
            try:
                # Logika pembacaan data
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/'+str(ID))
                    data = np.array(dataset)                   
                    snr = dataset.attrs['snr_db']
                    coda_end = int(dataset.attrs['coda_end_sample'])
                    spt = int(dataset.attrs['p_arrival_sample'])
                    sst = int(dataset.attrs['s_arrival_sample'])
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/'+str(ID))
                    data = np.array(dataset)
                
                # Logika Augmentasi (Sama seperti kode asli)
                if param['augmentation'] == True:                 
                    if i <= param['batch_size']//2:   
                        if param['shift_event_r'] and dataset.attrs['trace_category'] == 'earthquake_local' and all(snr):
                            data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, param['shift_event_r']/2)
                        if param['norm_mode']:                    
                            data = normalize(data, param['norm_mode'])  
                    else:                  
                        if dataset.attrs['trace_category'] == 'earthquake_local':                   
                            if param['shift_event_r'] and all(snr):
                                data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, param['shift_event_r'])
                            if param['add_event_r'] and all(snr):
                                data, additions = add_event(data, spt, sst, coda_end, snr, param['add_event_r'])
                            if param['add_noise_r'] and all(snr):
                                data = add_noise(data, snr, param['add_noise_r'])
                            if param['scale_amplitude_r']:
                                data = scale_amplitude(data, param['scale_amplitude_r'])
                            if param['pre_emphasis']:  
                                data = pre_emphasis(param['n_channels'], data) 
                            if param['norm_mode']:    
                                data = normalize(data, param['norm_mode'])                            
                        elif dataset.attrs['trace_category'] == 'noise':
                            if param['norm_mode']:                    
                                data = normalize(data, param['norm_mode']) 
                
                elif param['augmentation'] == False:
                    if param['shift_event_r'] and dataset.attrs['trace_category'] == 'earthquake_local' and all(snr):
                        data, spt, sst, coda_end = shift_event(data, spt, sst, coda_end, snr, param['shift_event_r']/2) 
                    if param['norm_mode']:                    
                        data = normalize(data, param['norm_mode']) 
                
                # Proses STFT
                if not np.any(np.isnan(data).any()):
                    for ch in range(param['n_channels']): 
                        bpf = data[:, ch]                        
                        f, t, Pxx = signal.stft(bpf, fs=100, nperseg=80)
                        Pxx = np.abs(Pxx)
                        X_chunk[i, :, :, ch] = Pxx.T 

                    # Labeling
                    if ID.split('_')[-1] == 'EV':
                        sptS = int(spt*param['target_length']/6000)
                        sstS = int(sst*param['target_length']/6000)                
                        delta = sstS - sptS                
                        y_chunk[i, sptS:int(sstS+(1.2*delta)), 0] = 1

                if (i + start_idx) % 10000 == 0:
                    print(f"Total Progres: {i + start_idx} / {lengthTotal}")

            except Exception:
                pass

        # Simpan chunk ini langsung ke SSD Eksternal
        print(f"--- Menyimpan Bagian {c} ke SSD Eksternal... ---")
        np.save(f"{ssd_output_path}X_{opt}_{param['target_length']}_part{c}.npy", X_chunk)
        np.save(f"{ssd_output_path}y_{opt}_{param['target_length']}_part{c}.npy", y_chunk)
        
        # Paksa hapus variabel besar dari RAM agar Swap SSD Internal lega kembali
        del X_chunk
        del y_chunk
            
 
