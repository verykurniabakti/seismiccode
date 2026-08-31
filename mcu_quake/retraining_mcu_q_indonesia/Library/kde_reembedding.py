# -*- coding: utf-8 -*-
"""
KDE Re-Embedding — pipeline lengkap sebagai fungsi/kelas yang bisa dipanggil
langsung, tanpa perlu menulis ulang alur ekstraksi embedding -> bangun KDE ->
inferensi -> metrik setiap kali dibutuhkan.

Sebelum modul ini ada, logika berikut terduplikasi (dengan variasi kecil) di
tiga skrip: evaluate_kde_re_embedding.py, evaluate_kde_re_embedding_all.py,
dan evaluate_kde_reembed_noise_only.py:
  - NpEncoder (JSON encoder untuk tipe numpy)
  - build_embeddings (baca JSON event -> ekstrak embedding per kelas,
    sadar-tipe: record berlabel type="no" adalah noise MURNI (hanya kanal
    utama valid); selain itu event standar dengan kanal utama=le dan
    kanal_noise=noise)
  - assert_disjoint (cek tidak ada event_id bocor antara referensi & uji)
  - load_frozen_extractor (potong model penuh ke layer vektor laten)
  - suntikan jitter epsilon yang reproducible sebelum membangun KDE, untuk
    mencegah scipy.stats.gaussian_kde melempar LinAlgError saat matriks
    kovariansi referensi mendekati singular
  - fix numpy 2.4.x pada embedding_likelihood/infer_1C_PDFs (pdf() KDE
    mengembalikan array shape (1,), bukan skalar 0-d, sehingga float()
    langsung gagal di numpy >=2.4) -- diberi versi aman di sini tanpa
    mengubah Library/utils.py, supaya hasil evaluasi yang sudah
    dipublikasikan dengan versi asli tidak ikut berubah.

Modul ini menggunakan ulang fungsi inti yang sudah ada di Library/utils.py
(embedding_PDFs_1D, embedding_PDFs_3D, calc_confusion_metrics,
latent_codes_1D) -- bukan menulis ulang matematikanya, hanya menyatukan
bagian "lem" (boilerplate) yang sebelumnya disalin-tempel.

Contoh pemakaian paling sederhana (1 kanal, embedding sudah diekstrak):

    from Library.kde_reembedding import KDEReEmbedder

    kde = KDEReEmbedder()
    kde.fit({"noise": ref_noise_embeddings, "le": ref_le_embeddings})
    label, proba = kde.predict(satu_embedding_uji)

Atau langsung dari sinyal mentah + model, sampai metrik jadi:

    from Library.kde_reembedding import kde_reembedding_from_signals

    hasil = kde_reembedding_from_signals(
        model=model,
        train_signals={"noise": [...], "le": [...]},
        test_signals={"noise": [...], "le": [...]},
    )
    print(hasil["metrics"])

Modul ini generik (tidak terikat data seismik) -- selama input berupa
embedding/vektor laten per kelas, bisa dipakai untuk kandidat generalisasi
domain lain (getaran mesin, HAR, drift sensor gas, dst., lihat
Bab 5 Butir 4 proposal disertasi) tanpa mengubah kode ini sama sekali.
"""

import json
import os

import numpy as np
from scipy.special import softmax

from Library.utils import (
    embedding_PDFs_1D,
    embedding_PDFs_3D,
    calc_confusion_metrics,
    latent_codes_1D,
)


# ==========================================================================
# Utilitas JSON aman untuk tipe numpy (dipakai saat menyimpan metrik/embedding)
# ==========================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ==========================================================================
# Ekstraksi model: potong model penuh sampai layer vektor laten
# ==========================================================================
def load_frozen_extractor(model_path, latent_dim=32):
    """Muat model Keras penuh lalu potong sampai layer berdimensi
    `latent_dim` (default 32, ukuran vektor laten ekstraktor 1D-CNN
    MCU-Quake) untuk dipakai sebagai ekstraktor beku (frozen extractor).

    Jika tidak ada layer dengan dimensi output persis `latent_dim`, jatuh
    kembali ke layer kedua-dari-belakang (asumsi layer terakhir adalah
    kepala klasifikasi/softmax).
    """
    import tensorflow as tf

    full_model = tf.keras.models.load_model(model_path, compile=False)

    latent_layer = None
    for layer in full_model.layers:
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, tuple) and output_shape[-1] == latent_dim:
            latent_layer = layer.output
            break
    if latent_layer is None:
        latent_layer = full_model.layers[-2].output

    return tf.keras.Model(inputs=full_model.inputs, outputs=latent_layer)


# ==========================================================================
# Ekstraksi embedding dari berkas JSON event (sadar-tipe)
# ==========================================================================
def build_embeddings(json_path, model, channel="Z", input_size=700, label=""):
    """Kembalikan ({'keys', 'noise', 'le'}, skipped) dari satu berkas JSON.

    Sadar-tipe: dataset satu-event-per-record (Indonesia/UUSS/STEAD, tanpa
    field 'type' atau type='se'/'ev') selalu {channel}=le & {channel}_noise
    =noise. Record dengan type="no" adalah noise MURNI -- hanya kanal
    `channel` yang valid sebagai sampel noise (kanal `channel`_noise
    diabaikan karena bukan kelas terpisah pada record semacam ini).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Berkas tidak ditemukan: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    noise_channel = f"{channel}_noise"
    out = {"keys": [], "noise": [], "le": []}
    skipped = []

    for key, rec in data.items():
        try:
            rec_type = rec.get("type", "se")
            sig_main = np.array(rec[channel][:input_size])
            if len(sig_main) != input_size:
                raise ValueError(f"panjang jendela utama {len(sig_main)}")

            if rec_type == "no":
                emb_no = np.array(latent_codes_1D(sig_main, model)).flatten()
                out["noise"].append(emb_no)
                out["keys"].append(key)
                continue

            sig_noise = np.array(rec[noise_channel][-input_size:])
            if len(sig_noise) != input_size:
                raise ValueError(f"panjang jendela noise {len(sig_noise)}")

            out["le"].append(np.array(latent_codes_1D(sig_main, model)).flatten())
            out["noise"].append(np.array(latent_codes_1D(sig_noise, model)).flatten())
            out["keys"].append(key)
        except Exception as e:
            skipped.append((key, str(e)))

    print(f"[{label}] {json_path}")
    print(f"[{label}] terpakai: {len(out['keys'])} kejadian | dilewati: {len(skipped)}")
    if skipped:
        for k, e in skipped[:5]:
            print(f"[{label}]   lewat: {k} -> {e}")
    return out, skipped


def assert_disjoint(ref, test):
    """Pastikan tidak ada satu pun event_id yang bocor antara referensi & uji."""
    overlap = set(ref["keys"]) & set(test["keys"])
    if overlap:
        raise RuntimeError(
            f"KEBOCORAN DATA: {len(overlap)} event_id muncul di referensi dan uji. "
            f"Contoh: {sorted(overlap)[:5]}"
        )
    print(f"[CEK] referensi {len(ref['keys'])} | uji {len(test['keys'])} | irisan 0 ✓")


# ==========================================================================
# Likelihood/inferensi aman-numpy (fix untuk numpy >=2.4: gaussian_kde.pdf()
# mengembalikan array shape (1,), bukan skalar 0-d, sehingga float() polos
# gagal). Tidak mengubah Library/utils.py -- lihat catatan modul di atas.
# ==========================================================================
def _embedding_likelihood_safe(input_embedding, embeddings_pdf):
    likelihood_noise = embeddings_pdf["noise"].pdf(input_embedding)
    likelihood_le = embeddings_pdf["le"].pdf(input_embedding)
    return {
        "noise": float(np.asarray(likelihood_noise).ravel()[0]),
        "le": float(np.asarray(likelihood_le).ravel()[0]),
    }


def infer_1C_PDFs_safe(input_embedding, embeddings_pdfs, choose_pdf="Kernel"):
    """0: noise, 1: le. Setara Library.utils.infer_1C_PDFs, aman untuk numpy >=2.4."""
    likelihood_dict = _embedding_likelihood_safe(input_embedding, embeddings_pdfs[choose_pdf])
    likelihood_softmax = softmax([likelihood_dict["noise"], likelihood_dict["le"]])
    infer_type = int(np.argmax(likelihood_softmax))
    return infer_type, likelihood_dict, list(np.round(likelihood_softmax, 4))


# ==========================================================================
# Fit KDE dengan jitter epsilon (reproducible) -- cegah LinAlgError akibat
# matriks kovariansi referensi yang mendekati singular.
# ==========================================================================
def fit_kde_with_jitter(reference_embeddings, source_list=("noise", "le"), seed=42, eps=1e-6):
    """Bangun KDE dari embedding referensi, dengan derau Gaussian kecil
    (eps, di-seed ulang setiap panggilan agar hasilnya reproducible
    terlepas dari skenario apa saja yang dijalankan sebelumnya)."""
    rng = np.random.RandomState(seed)
    jittered = {}
    for cls in source_list:
        values = np.array(reference_embeddings[cls])
        jittered[cls] = (values + rng.normal(0, eps, values.shape)).tolist()
    return embedding_PDFs_1D(jittered, source_list=list(source_list))


# ==========================================================================
# Kelas utama: antarmuka fit/predict/evaluate seperti pengklasifikasi
# ==========================================================================
class KDEReEmbedder:
    """Estimasi kerapatan non-parametrik (KDE) pada ruang laten beku, dengan
    antarmuka fit/predict/evaluate seperti pengklasifikasi pada umumnya.

    Parameters
    ----------
    choose_pdf : str
        "Kernel" (KDE, default -- sesuai H1 di proposal) atau "Norm"
        (distribusi normal/Gaussian parametrik, untuk pembanding).
    jitter : bool
        Jika True (default), suntikkan jitter epsilon reproducible saat fit()
        untuk mencegah LinAlgError pada referensi yang hampir singular
        (lihat fit_kde_with_jitter). Nonaktifkan hanya untuk pembanding
        1:1 dengan hasil skrip lama yang tidak memakai jitter.
    seed : int
        Seed jitter, dipakai ulang di setiap fit() supaya hasilnya
        reproducible.
    """

    def __init__(self, choose_pdf="Kernel", jitter=True, seed=42):
        self.choose_pdf = choose_pdf
        self.jitter = jitter
        self.seed = seed
        self.pdfs_ = None
        self._mode = None

    # ------------------------------------------------------------------
    # 1 dimensi / 1 kanal
    # ------------------------------------------------------------------
    def fit(self, reference_embeddings, source_list=("noise", "le")):
        """Bangun KDE dari embedding referensi (HANYA dari himpunan latih).

        reference_embeddings: {"noise": [...], "le": [...]}
        """
        if self.jitter:
            self.pdfs_ = fit_kde_with_jitter(
                reference_embeddings, source_list=source_list, seed=self.seed
            )
        else:
            self.pdfs_ = embedding_PDFs_1D(reference_embeddings, source_list=list(source_list))
        self._mode = "1D"
        return self

    def predict(self, embedding):
        """Klasifikasi satu embedding. Kembalikan (label, likelihood_dict, proba)."""
        self._check_fitted("1D")
        return infer_1C_PDFs_safe(embedding, self.pdfs_, choose_pdf=self.choose_pdf)

    def predict_batch(self, embeddings):
        """Klasifikasi banyak embedding sekaligus. Kembalikan list label int."""
        return [self.predict(e)[0] for e in embeddings]

    # ------------------------------------------------------------------
    # 3 dimensi / 3 kanal (Z, N, E digabung)
    # ------------------------------------------------------------------
    def fit_3d(self, reference_embeddings_Z, reference_embeddings_N, reference_embeddings_E,
               source_list=("noise", "le")):
        self.pdfs_ = embedding_PDFs_3D(
            reference_embeddings_Z, reference_embeddings_N, reference_embeddings_E,
            source_list=list(source_list),
        )
        self._mode = "3D"
        return self

    def predict_3d(self, embedding_3c):
        from Library.utils import infer_3C_PDFs

        self._check_fitted("3D")
        return infer_3C_PDFs(embedding_3c, self.pdfs_, choose_pdf=self.choose_pdf)

    # ------------------------------------------------------------------
    def evaluate(self, test_embeddings, class_to_label=None):
        """Evaluasi menyeluruh: prediksi seluruh himpunan uji lalu hitung
        confusion matrix + metrik (Accuracy, TPR, TNR, FPR, F1, dst.).

        test_embeddings: {"noise": [...], "le": [...]} (atau kelas lain,
            selama namanya sama dengan yang dipakai saat fit()).
        class_to_label: pemetaan nama kelas -> label ground-truth integer.
            Default {"noise": 0, "le": 1} mengikuti konvensi kode ini
            (0 = noise, 1 = local earthquake/kejadian positif).
        """
        self._check_fitted()
        if class_to_label is None:
            class_to_label = {"noise": 0, "le": 1}

        true_labels, pred_labels = [], []
        predict_fn = self.predict if self._mode == "1D" else self.predict_3d
        for cls, gt in class_to_label.items():
            for emb in test_embeddings.get(cls, []):
                infer_type, _, _ = predict_fn(np.array(emb).flatten() if self._mode == "1D" else emb)
                true_labels.append(gt)
                pred_labels.append(infer_type)

        cnf, metrics = calc_confusion_metrics(true_labels, pred_labels)
        return {
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "confusion_matrix": cnf,
            "metrics": metrics,
        }

    def _check_fitted(self, expected_mode=None):
        if self.pdfs_ is None:
            raise RuntimeError("Panggil .fit() (atau .fit_3d()) dahulu sebelum predict/evaluate.")
        if expected_mode is not None and self._mode != expected_mode:
            raise RuntimeError(
                f"Model dilatih dalam mode {self._mode}, tapi predict yang dipanggil "
                f"untuk mode {expected_mode}."
            )


# ==========================================================================
# Fungsi tingkat-tinggi: dari sinyal mentah + model sampai metrik jadi,
# dalam satu panggilan. Menggantikan pola build_embeddings()+run_scenario()
# yang sebelumnya ditulis ulang di tiap skrip evaluate_kde_*.py.
# ==========================================================================
def kde_reembedding_from_signals(model, train_signals, test_signals,
                                  choose_pdf="Kernel", class_to_label=None, jitter=True):
    """Jalankan KDE Re-Embedding lengkap dari sinyal 1 kanal mentah.

    Parameters
    ----------
    model : tf.keras.Model
        Ekstraktor beku (frozen 1D-CNN); dipanggil lewat latent_codes_1D.
        Untuk model penuh (belum dipotong ke layer laten), potong dulu
        dengan load_frozen_extractor().
    train_signals, test_signals : dict
        {"noise": [sinyal_1, sinyal_2, ...], "le": [...]} -- setiap sinyal
        adalah array 1D mentah (belum di-embedding). train_signals dipakai
        HANYA untuk membangun KDE referensi; test_signals HANYA untuk
        inferensi (jaga agar tidak ada kebocoran data antara keduanya).
    choose_pdf : str
        "Kernel" (default) atau "Norm".
    class_to_label : dict, optional
        Lihat KDEReEmbedder.evaluate().
    jitter : bool
        Lihat KDEReEmbedder.

    Returns
    -------
    dict: {"kde": KDEReEmbedder, "confusion_matrix", "metrics",
           "true_labels", "pred_labels"}
    """
    def to_embeddings(signals_by_class):
        return {
            cls: [latent_codes_1D(sig, model) for sig in sigs]
            for cls, sigs in signals_by_class.items()
        }

    train_embeddings = to_embeddings(train_signals)
    test_embeddings = to_embeddings(test_signals)

    kde = KDEReEmbedder(choose_pdf=choose_pdf, jitter=jitter).fit(train_embeddings)
    hasil = kde.evaluate(test_embeddings, class_to_label=class_to_label)
    hasil["kde"] = kde
    return hasil
