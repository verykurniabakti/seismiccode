import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- KONFIGURASI PATH ---
CSV_RESULT_PATH = '/Volumes/Extreme SSD/mcu_quake_big_stead_output/checkpoint_results_stead_full.csv'
OUTPUT_PLOT_PATH = '/Volumes/Extreme SSD/mcu_quake_big_stead_output/confusion_matrix_stead_full.png'

def evaluate_results():
    print(f"[INFO] Memuat data hasil evaluasi ({CSV_RESULT_PATH})...")
    
    # Memuat data
    df = pd.read_csv(CSV_RESULT_PATH)
    
    # Menghapus baris jika ada nilai NaN (proteksi)
    df = df.dropna(subset=['y_true', 'y_pred'])
    
    y_true = df['y_true'].astype(int)
    y_pred = df['y_pred'].astype(int)

    # 1. Menghitung Metrik Utama
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Noise (0)', 'Earthquake (2)'])
    
    print("\n" + "="*50)
    print("           HASIL EVALUASI MCU-QUAKE")
    print("="*50)
    print(f"Total Data Terproses : {len(df):,}")
    print(f"Overall Accuracy      : {accuracy:.4f}")
    print("-" * 50)
    print("Classification Report:")
    print(report)
    print("="*50)

    # 2. Membuat Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalisasi untuk melihat persentase
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 3. Plotting Visual
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Noise (0)', 'Earthquake (2)'], 
                yticklabels=['Noise (0)', 'Earthquake (2)'])
    
    plt.title(f'Confusion Matrix - MCU-Quake on STEAD\n(Total: {len(df):,} Samples)', fontsize=14)
    plt.ylabel('Actual Label (y_true)')
    plt.xlabel('Predicted Label (y_pred)')
    
    # Simpan Gambar
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot Confusion Matrix disimpan di: {OUTPUT_PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    evaluate_results()