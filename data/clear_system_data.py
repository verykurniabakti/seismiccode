import os
import shutil
from pathlib import Path
import subprocess

print("🧹 Membersihkan cache & log untuk mengurangi System Data...\n")

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

# 1. Cache user-level
print("🔍 Menghapus cache user-level...")
user_cache = ["~/Library/Caches", "~/.cache"]
for base in user_cache:
    base_path = Path(base).expanduser()
    if base_path.exists():
        for item in base_path.iterdir():
            if item.name.startswith("com.apple.Safari") or item.name.startswith("CloudKit"):
                print(f"✔️  Skip (Apple core cache): {item}")
                continue
            remove_path(item)

# 2. Log user
print("\n🔍 Menghapus log user...")
remove_path("~/Library/Logs")

# 3. Log sistem non-kritis
print("\n🔍 Menghapus log sistem (butuh sudo saat menjalankan script)...")
try:
    subprocess.run(["sudo", "rm", "-rf", "/private/var/log/*"], check=False)
    print("🗑  Log sistem dibersihkan.")
except Exception as e:
    print(f"⚠️  Tidak bisa menghapus log sistem: {e}")

# 4. Cache aplikasi besar
print("\n🔍 Menghapus cache aplikasi besar...")
big_app_cache = [
    # Chrome
    "~/Library/Application Support/Google/Chrome/Default/Cache",
    "~/Library/Application Support/Google/Chrome/ShaderCache",
    "~/Library/Application Support/Google/Chrome/GrShaderCache",
    "~/Library/Application Support/Google/Chrome/Crashpad",

    # Edge
    "~/Library/Application Support/Microsoft Edge/Default/Cache",
    "~/Library/Application Support/Microsoft Edge/ShaderCache",
    "~/Library/Application Support/Microsoft Edge/GrShaderCache",
    "~/Library/Application Support/Microsoft Edge/Crashpad",

    # VS Code
    "~/Library/Application Support/Code/Cache",
    "~/Library/Application Support/Code/CachedData",
    "~/Library/Application Support/Code/User/workspaceStorage",
    "~/Library/Application Support/Code/Service Worker/CacheStorage",

    # Zoom
    "~/Library/Application Support/zoom.us/data",
    "~/Library/Application Support/zoom.us/logs",

    # Spotify
    "~/Library/Application Support/Spotify/PersistentCache",
]

for p in big_app_cache:
    remove_path(p)

# 5. Cache Python
print("\n🔍 Menghapus cache Python...")
remove_path("~/.cache/pip")
remove_path("~/.cache/python")

# 6. Cache TensorFlow & Keras
print("\n🔍 Menghapus cache TensorFlow & Keras...")
remove_path("~/.cache/tensorflow")
remove_path("~/.cache/tfds")
remove_path("~/tensorflow_datasets")
remove_path("~/.keras")

# 7. Cache ML umum
print("\n🔍 Menghapus cache ML umum...")
remove_path("~/.cache/huggingface")
remove_path("~/.cache/torch")

# 8. __pycache__ di project
print("\n🔍 Menghapus __pycache__ di project saat ini...")
for pycache in Path(".").rglob("__pycache__"):
    remove_path(pycache)

# 9. Tampilkan APFS snapshots
print("\n📸 APFS Local Snapshots (penyebab terbesar System Data):")
try:
    subprocess.run(["tmutil", "listlocalsnapshots", "/"])
    print("\nℹ️ Hapus snapshot dengan:")
    print("   sudo tmutil deletelocalsnapshots <nama-snapshot>")
except Exception as e:
    print(f"⚠️ Tidak bisa menampilkan snapshot: {e}")

print("\n✨ Selesai! Cache & log aman sudah dibersihkan.")
