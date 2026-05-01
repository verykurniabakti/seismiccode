import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load hasil akhir
CSV_PATH = '/Volumes/Extreme SSD/mcu_quake_bis_stead_output_normalisasi_final/final_calibrated_results.csv'
df = pd.read_csv(CSV_PATH)

# Bersihkan jika ada nilai NaN
df = df.dropna(subset=['y_true', 'y_pred'])

print("==================================================")
print("       LAPORAN PERFORMA FINAL (CALIBRATED)")
print("==================================================")

# 1. Classification Report
# Label 0: Noise, Label 2: Earthquake
target_names = ['Noise (0)', 'Earthquake (2)']
report = classification_report(df['y_true'], df['y_pred'], target_names=target_names)
print(report)

# 2. Confusion Matrix
cm = confusion_matrix(df['y_true'], df['y_pred'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix Final - Calibrated MCU-Quake')
plt.xlabel('Predicted')
plt.ylabel('Actual')

output_img = '/Volumes/Extreme SSD/mcu_quake_bis_stead_output_normalisasi_final/confusion_matrix_FINAL_FULL.png'
plt.savefig(output_img)
print(f"\n[INFO] Confusion Matrix disimpan di: {output_img}")
print("==================================================")