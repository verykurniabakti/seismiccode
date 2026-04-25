import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH ---
CSV_RESULT_PATH = '/Volumes/Extreme SSD/mcu_quake_big_stead_output_normalisasi/checkpoint_results_stead_full.csv'
OUTPUT_PLOT = '/Volumes/Extreme SSD/mcu_quake_big_stead_output_normalisasi/confusion_matrix_final.png'

def evaluate_final():
    print("[INFO] Memuat hasil benchmark dengan normalisasi...")
    df = pd.read_csv(CSV_RESULT_PATH)
    
    # Pastikan data bersih
    df = df.dropna(subset=['y_true', 'y_pred'])
    
    y_true = df['y_true'].astype(int)
    y_pred = df['y_pred'].astype(int)

    print("\n" + "="*50)
    print("      HASIL EVALUASI AKHIR (DENGAN NORMALISASI)")
    print("="*50)
    print(f"Total Data : {len(df):,}")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=['Noise (0)', 'Earthquake (2)']))
    print("="*50)

    # Plot Confusion Matrix untuk Lampiran Disertasi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Noise (0)', 'Earthquake (2)'], 
                yticklabels=['Noise (0)', 'Earthquake (2)'])
    plt.title('Confusion Matrix: MCU-Quake with Z-Score Normalization')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"[INFO] Gambar Confusion Matrix disimpan di: {OUTPUT_PLOT}")

if __name__ == "__main__":
    evaluate_final()