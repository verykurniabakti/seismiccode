import pandas as pd
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

CSV_FILE = "merge.csv"
HDF5_FILE = "merge.hdf5"
OUTPUT_CSV = "validation_results.csv"

REQUIRED_ATTRS = [
    "p_arrival_sample",
    "s_arrival_sample",
    "coda_end_sample",
    "receiver_latitude",
    "receiver_longitude",
    "source_latitude",
    "source_longitude",
]

# ---------------------------------------------------------
# Helper: Convert attribute to numeric safely
# ---------------------------------------------------------
def to_number(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) > 0:
            x = x[0]

    if isinstance(x, bytes):
        x = x.decode()

    if isinstance(x, str):
        try:
            return float(x)
        except:
            return None

    try:
        return float(x)
    except:
        return None


# ---------------------------------------------------------
# Worker function for parallel validation
# ---------------------------------------------------------
def validate_single_trace(trace):
    errors = []

    try:
        with h5py.File(HDF5_FILE, "r") as h5:
            path = f"data/{trace}"

            if path not in h5:
                return (trace, ["Missing in HDF5"])

            dataset = h5[path]
            data = np.array(dataset)

            # 1. Check waveform shape
            if len(data.shape) != 2 or data.shape[1] != 3:
                errors.append("Waveform shape invalid (should be Nx3)")

            # 2. Check required attributes
            for attr in REQUIRED_ATTRS:
                if attr not in dataset.attrs:
                    errors.append(f"Missing attribute: {attr}")

            # 3. Validate arrival times
            length = len(data)

            p = to_number(dataset.attrs.get("p_arrival_sample"))
            if p is None or p < 0 or p >= length:
                errors.append("P-arrival out of range or invalid")

            s = to_number(dataset.attrs.get("s_arrival_sample"))
            if s is None or s < 0 or s >= length:
                errors.append("S-arrival out of range or invalid")

            c = to_number(dataset.attrs.get("coda_end_sample"))
            if c is None or c < 0 or c >= length:
                errors.append("Coda-end out of range or invalid")

    except Exception as e:
        errors.append(f"Exception: {str(e)}")

    return (trace, errors)


# ---------------------------------------------------------
# MAIN PARALLEL VALIDATOR
# ---------------------------------------------------------
def main():
    df = pd.read_csv(CSV_FILE)
    trace_list = df["trace_name"].tolist()

    print(f"Total traces: {len(trace_list)}")
    print(f"Using {cpu_count()} CPU cores for parallel validation")

    results = []

    # Parallel execution with progress bar
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(validate_single_trace, trace_list),
                           total=len(trace_list),
                           desc="Validating"):
            results.append(result)

    # Convert results to DataFrame
    rows = []
    for trace, errs in results:
        rows.append({
            "trace_name": trace,
            "status": "OK" if len(errs) == 0 else "ERROR",
            "errors": "; ".join(errs) if errs else ""
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Saved detailed results to: {OUTPUT_CSV}")

    total_errors = sum(1 for _, e in results if len(e) > 0)
    missing = sum(1 for _, e in results if "Missing in HDF5" in e)

    print(f"Missing in HDF5: {missing}")
    print(f"Traces with validation errors: {total_errors}")
    print("\nParallel validation complete.")


if __name__ == "__main__":
    main()
