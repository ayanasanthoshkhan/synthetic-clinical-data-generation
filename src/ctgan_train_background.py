"""CTGAN DGX training. Same data, metadata, seed and fit settings as 06_ctgan_training.ipynb.

Run from the DGX project using its existing .venv. Outputs are kept separate from TVAE.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Same GPU selection as the CTGAN notebook.

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

SEED = 42
EPOCHS = 300
BATCH_SIZE = 500
TARGET = "readmitted_30"
NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

# This script is intended to be placed in the project's src/ folder.
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data/processed/train_semantic.csv"
MODEL_PATH = ROOT / "models/ctgan_300ep.pkl"
CSV_PATH = ROOT / "data/synthetic/ctgan_300ep.csv"
METADATA_PATH = ROOT / "results/training/ctgan_300ep_metadata.json"
RUN_PATH = ROOT / "results/training/ctgan_300ep_run.json"

assert TRAIN_PATH.is_file(), f"Training file not found: {TRAIN_PATH}"
for path in (MODEL_PATH, CSV_PATH, METADATA_PATH, RUN_PATH):
    assert not path.exists(), f"Refusing to overwrite existing output: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
assert torch.cuda.is_available(), "CUDA unavailable; check GPU access."
print("Project:", ROOT, flush=True)
print("SDV:", sdv.__version__, "PyTorch:", torch.__version__, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

train = pd.read_csv(TRAIN_PATH, low_memory=False)
assert TARGET in train.columns
assert set(NUMERIC).issubset(train.columns)
assert not ({"patient_nbr", "encounter_id"} & set(train.columns))
assert not train.isna().any().any(), "Missing training values"
for col in train.columns:
    if col in NUMERIC:
        train[col] = pd.to_numeric(train[col], errors="raise")
    else:
        train[col] = train[col].astype(str)
assert set(train[TARGET].unique()) == {"0", "1"}

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train)
for col in train.columns:
    metadata.update_column(
        column_name=col,
        sdtype="numerical" if col in NUMERIC else "categorical",
    )
metadata.validate()
metadata.validate_data(data=train)
print("Training shape:", train.shape, flush=True)
print("Readmission labels:\n", train[TARGET].value_counts().sort_index(), flush=True)

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    enable_gpu=True,
    verbose=True,
)
start = time.perf_counter()
synthesizer.fit(train)
training_seconds = time.perf_counter() - start
print(f"CTGAN training completed in {training_seconds / 60:.2f} minutes", flush=True)

# Save the trained model immediately, before potentially time-consuming sampling.
synthesizer.save(filepath=str(MODEL_PATH))
metadata.save_to_json(filepath=str(METADATA_PATH))
print("Saved model:", MODEL_PATH, flush=True)

synthetic_ctgan = synthesizer.sample(num_rows=len(train))
assert synthetic_ctgan.shape == train.shape, "Unexpected synthetic dataset shape"
assert set(synthetic_ctgan.columns) == set(train.columns), "Synthetic columns mismatch"
synthetic_ctgan.to_csv(CSV_PATH, index=False)

run_info = {
    "model": "CTGAN",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "train_rows": len(train),
    "synthetic_rows": len(synthetic_ctgan),
    "training_seconds": round(training_seconds, 2),
    "sdv_version": sdv.__version__,
    "torch_version": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "training_csv": "data/processed/train_semantic.csv",
    "test_data_used_in_training": False,
}
with RUN_PATH.open("w", encoding="utf-8") as file:
    json.dump(run_info, file, indent=2)

print("Saved synthetic data:", CSV_PATH, flush=True)
print("Saved run summary:", RUN_PATH, flush=True)
print("Real readmission (%):", round((train[TARGET] == "1").mean() * 100, 3), flush=True)
print("CTGAN readmission (%):", round((synthetic_ctgan[TARGET].astype(str) == "1").mean() * 100, 3), flush=True)
visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
print("\nMean prior visits:", flush=True)
print(pd.DataFrame({
    "Real": train[visit_cols].mean(),
    "CTGAN": synthetic_ctgan[visit_cols].apply(pd.to_numeric, errors="coerce").mean(),
}).round(3), flush=True)
print("\nCategory coverage:", flush=True)
for col in ("race", "diag_3"):
    print(f"{col}: real={train[col].nunique()}, CTGAN={synthetic_ctgan[col].nunique()}", flush=True)
print("CTGAN complete.", flush=True)
