import os
import shutil
from pathlib import Path

print("🧹 Membersihkan semua cache Python, VS Code, TensorFlow, dan Keras...\n")

def remove_path(path):
    p = Path(path).expanduser()
    if p.exists():
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print(f"🗑  Deleted: {p}")
        except Exception as e:
            print(f"⚠️  Cannot delete {p}: {e}")
    else:
        print(f"✔️  Skip (not found): {p}")

# 1. Hapus __pycache__ di project saat ini
print("🔍 Menghapus __pycache__ di project...")
for pycache in Path(".").rglob("__pycache__"):
    remove_path(pycache)

# 2. Hapus pip cache
print("\n🔍 Menghapus pip cache...")
remove_path("~/.cache/pip")
remove_path("~/Library/Caches/pip")

# 3. Hapus Python cache global
print("\n🔍 Menghapus Python cache global...")
remove_path("~/.cache/python")
remove_path("~/.cache/pypoetry")
remove_path("~/.cache/pipenv")

# 4. Hapus cache VS Code
print("\n🔍 Menghapus cache VS Code...")
remove_path("~/Library/Application Support/Code/Cache")
remove_path("~/Library/Application Support/Code/CachedData")
remove_path("~/Library/Application Support/Code/User/workspaceStorage")
remove_path("~/Library/Application Support/Code/Service Worker/CacheStorage")

# 5. Hapus cache Pylance
print("\n🔍 Menghapus cache Pylance...")
remove_path("~/Library/Application Support/Code/User/globalStorage/ms-python.vscode-pylance")

# 6. Hapus cache TensorFlow
print("\n🔍 Menghapus cache TensorFlow...")
remove_path("~/.cache/tensorflow")
remove_path("~/.cache/tfhub_modules")
remove_path("~/.cache/tfds")
remove_path("~/tensorflow_datasets")
remove_path("~/Library/Application Support/tensorflow")

# 7. Hapus cache Keras
print("\n🔍 Menghapus cache Keras...")
remove_path("~/.keras")
remove_path("~/Library/Application Support/keras")

# 8. Hapus cache ML umum
print("\n🔍 Menghapus cache ML umum (HuggingFace, PyTorch)...")
remove_path("~/.cache/huggingface")
remove_path("~/.cache/torch")

print("\n✨ Selesai! Semua cache berhasil dibersihkan.")
