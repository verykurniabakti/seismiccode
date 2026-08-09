# -*- coding: utf-8 -*-
"""Visualize latent-domain shift before and after target adaptation.

The "after" representation is extracted with the retrained/frozen extractor.
KDE is used by the project as a classifier; it is not itself a coordinate
transform, so this script does not invent a KDE-based displacement formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from Library.utils import latent_codes_1D


BASE_DIR = Path(__file__).resolve().parent
BASE_DATA = BASE_DIR / "data_indonesia"
DEFAULT_SOURCE = {
    "UUSS": BASE_DIR / "Benchmark_ UUSS 3C_ test n2222 r100" / "UUSS 3C data, test n2222 r100.json",
    "STEAD": BASE_DIR / "Benchmark_ STEAD 3C_ test n15275 r100" / "STEAD data, test n15275 r100.json",
}
DEFAULT_TARGET = BASE_DATA / "indonesia_test_data.json"
DEFAULT_PRETRAINED = Path(
    "/Volumes/Local Disk/Code_Git/S3_code/seismic/mcu_quake/code_gen_trying/"
    "indonesia_jaya_benchmarking_data/mulai_juli/mcquake_ori_file/"
    "Code & Figure demo/Pre-trained model/MCU-Quake 5-20"
)
DEFAULT_RETRAINED = BASE_DIR / "output_models" / "frozen_extractor_indonesia_Z.keras"

INPUT_SIZE = 700
CHANNEL = "Z"
NOISE_CHANNEL = "Z_noise"
CLASS_LABELS = {"le": "Earthquake", "noise": "Noise"}
CLASS_COLORS = {"le": "#0072B2", "noise": "#D55E00"}
DOMAIN_MARKERS = {"source": "o", "target_before": "^", "target_after": "s"}


def _check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


def _load_model(path: Path):
    _check_file(path, "Model")
    print(f"[INFO] Memuat model: {path}")
    return tf.keras.models.load_model(path, compile=False)


def _sample_indices(size: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or size <= max_samples:
        return np.arange(size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=max_samples, replace=False))


def extract_dataset(
    json_path: Path,
    model: Any,
    domain: str,
    stage: str,
    max_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Extract earthquake/noise latent vectors from one waveform JSON."""
    _check_file(json_path, f"Dataset {domain}")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    keys = list(data)
    selected = _sample_indices(len(keys), max_samples, seed)
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for index in selected:
        key = keys[index]
        record = data[key]
        try:
            signal_le = np.asarray(record[CHANNEL][:INPUT_SIZE], dtype=np.float32)
            signal_noise = np.asarray(record[NOISE_CHANNEL][-INPUT_SIZE:], dtype=np.float32)
            if signal_le.size != INPUT_SIZE or signal_noise.size != INPUT_SIZE:
                raise ValueError(
                    f"window {signal_le.size}/{signal_noise.size}, expected {INPUT_SIZE}"
                )
            embeddings = {
                "le": np.asarray(latent_codes_1D(signal_le, model), dtype=np.float32).ravel(),
                "noise": np.asarray(latent_codes_1D(signal_noise, model), dtype=np.float32).ravel(),
            }
            if any(not np.all(np.isfinite(vector)) for vector in embeddings.values()):
                raise ValueError("embedding mengandung NaN/Inf")
            for class_name, vector in embeddings.items():
                rows.append(
                    {
                        "domain": domain,
                        "stage": stage,
                        "class": class_name,
                        "class_label": CLASS_LABELS[class_name],
                        "event_key": str(key),
                        "embedding": vector,
                    }
                )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            skipped.append((str(key), str(error)))

    print(
        f"[{stage}/{domain}] dipakai {len(rows) // 2} event "
        f"({len(rows)} titik), dilewati {len(skipped)}"
    )
    return rows, skipped


def project_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("Tidak ada embedding yang dapat divisualisasikan.")
    dimensions = {row["embedding"].shape for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"Dimensi embedding tidak konsisten: {sorted(dimensions)}")
    matrix = np.vstack([row["embedding"] for row in rows])
    if matrix.shape[0] < 2:
        raise ValueError("Minimal diperlukan dua titik embedding untuk PCA.")
    pca = PCA(n_components=2, random_state=0)
    coordinates = pca.fit_transform(matrix)
    for row, coordinate in zip(rows, coordinates):
        row["pca_1"], row["pca_2"] = map(float, coordinate)
    print(
        f"[INFO] PCA: {matrix.shape[1]}D -> 2D | "
        f"varian dijelaskan: {pca.explained_variance_ratio_.sum():.3f}"
    )
    return pca.explained_variance_ratio_


def _centroids(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["stage"]][row["class"]].append(
            np.array([row["pca_1"], row["pca_2"]], dtype=float)
        )
    return {
        stage: {
            class_name: np.mean(points, axis=0).tolist()
            for class_name, points in classes.items()
        }
        for stage, classes in grouped.items()
    }


def _plot_points(ax, rows: list[dict[str, Any]], stage: str, title: str) -> None:
    subset = [row for row in rows if row["stage"] == stage or row["stage"] == "source"]
    for class_name in ("le", "noise"):
        points = [row for row in subset if row["class"] == class_name]
        for domain in ("source", "target"):
            domain_points = [row for row in points if row["domain"] == domain]
            if not domain_points:
                continue
            ax.scatter(
                [row["pca_1"] for row in domain_points],
                [row["pca_2"] for row in domain_points],
                s=22,
                alpha=0.42 if domain == "source" else 0.72,
                c=CLASS_COLORS[class_name],
                marker=DOMAIN_MARKERS["source" if domain == "source" else stage],
                edgecolors="white",
                linewidths=0.25,
                rasterized=len(domain_points) > 2000,
            )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_shift(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    _plot_points(axes[0], rows, "target_before", "Target sebelum adaptasi")
    _plot_points(axes[1], rows, "target_after", "Target setelah KDE Re-Embedding")

    centroids = _centroids(rows)
    for class_name in ("le", "noise"):
        before = np.asarray(centroids.get("target_before", {}).get(class_name, [np.nan, np.nan]))
        after = np.asarray(centroids.get("target_after", {}).get(class_name, [np.nan, np.nan]))
        if np.all(np.isfinite(np.r_[before, after])):
            axes[1].annotate(
                "",
                xy=after,
                xytext=before,
                arrowprops={"arrowstyle": "->", "color": CLASS_COLORS[class_name], "lw": 1.8},
            )
            axes[1].text(*after, f"  {CLASS_LABELS[class_name]} centroid", color=CLASS_COLORS[class_name], fontsize=8)

    class_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[name],
               markeredgecolor="white", markersize=8, label=CLASS_LABELS[name])
        for name in ("le", "noise")
    ]
    domain_handles = [
        Line2D([0], [0], marker=marker, color="#555555", linestyle="None", markersize=8, label=label)
        for marker, label in (("o", "Source"), ("^", "Target sebelum"), ("s", "Target setelah"))
    ]
    fig.legend(handles=class_handles + domain_handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Pergeseran domain pada ruang laten", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot tersimpan: {output_path}")


def save_outputs(rows: list[dict[str, Any]], explained_variance: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "latent_domain_shift_pca.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "stage", "class", "class_label", "event_key", "pca_1", "pca_2"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    counts = defaultdict(int)
    for row in rows:
        counts[f"{row['stage']}|{row['domain']}|{row['class']}"] += 1
    summary = {
        "representation_after": "MODEL_RETRAINED / frozen extractor",
        "pca_explained_variance_ratio": explained_variance.tolist(),
        "counts": dict(counts),
        "centroids_pca": _centroids(rows),
    }
    with (output_dir / "latent_domain_shift_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[OK] Data PCA tersimpan: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuss", type=Path, default=DEFAULT_SOURCE["UUSS"])
    parser.add_argument("--stead", type=Path, default=DEFAULT_SOURCE["STEAD"])
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--retrained", type=Path, default=DEFAULT_RETRAINED)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output_eval" / "latent_domain_shift")
    parser.add_argument("--max-samples-per-domain", type=int, default=1500, help="0 berarti semua event.")
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples_per_domain < 0:
        raise ValueError("--max-samples-per-domain harus >= 0")

    source_rows: list[dict[str, Any]] = []
    for offset, (domain, path) in enumerate((("UUSS", args.uuss), ("STEAD", args.stead))):
        model = _load_model(args.pretrained) if offset == 0 else model
        rows, _ = extract_dataset(path, model, "source", "source", args.max_samples_per_domain, args.seed + offset)
        source_rows.extend(rows)

    target_before, _ = extract_dataset(args.target, model, "target", "target_before", args.max_samples_per_domain, args.seed + 10)
    adapted_model = _load_model(args.retrained)
    target_after, _ = extract_dataset(args.target, adapted_model, "target", "target_after", args.max_samples_per_domain, args.seed + 11)

    rows = source_rows + target_before + target_after
    explained_variance = project_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(rows, explained_variance, args.output_dir)
    plot_shift(rows, args.output_dir / "latent_domain_shift_pca.png")


if __name__ == "__main__":
    main()
