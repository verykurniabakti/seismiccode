import json
import numpy as np

# Path file demo Bapak
file_path = "Data benchmark-demo/UUSS MCU, test n100 r100.json"
with open(file_path, 'r') as f:
    data = json.load(f)

first_key = list(data.keys())[0]
sample_z = np.array(data[first_key]['Z'])

print(f"--- AUDIT DATA DEMO ({first_key}) ---")
print(f"Amplitudo Maksimum : {np.max(np.abs(sample_z)):.4f}")
print(f"Standar Deviasi    : {np.std(sample_z):.4f}")
print(f"Rata-rata (Mean)   : {np.mean(sample_z):.4f}")