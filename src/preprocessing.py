"""Leakage-aware preprocessing for UCI Diabetes 130-US Hospitals (dataset 296).

Only training data determine learned preprocessing choices (rare specialties and
numerical imputation medians). Patient identifiers are used for splitting/audits,
never for model training or synthetic data generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_ID = 296
SEED = 42
TEST_SIZE = 0.20
RARE_SPECIALTY_MIN_COUNT = 100
DEATH_HOSPICE_CODES = (11, 13, 14, 19, 20, 21)
IDENTIFIER_COLS = ("encounter_id", "patient_nbr")
TARGET_SOURCE = "readmitted"
TARGET_BINARY = "readmitted_30"
DROP_COLS = ("weight", "examide", "citoglipton")
DIAG_COLS = ("diag_1", "diag_2", "diag_3")
LAB_COLS = ("A1Cresult", "max_glu_serum")
CODED_CATEGORY_COLS = (
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
)
NUMERIC_FEATURES = (
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
)


def assemble_uci_data(features: pd.DataFrame,
                      targets: pd.DataFrame,
                      identifiers: pd.DataFrame) -> pd.DataFrame:
    """Join UCI-provided aligned data tables by their original row order."""
    for name, frame in (("features", features), ("targets", targets),
                        ("identifiers", identifiers)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
    n = len(features)
    if len(targets) != n or len(identifiers) != n:
        raise ValueError("UCI features, target and identifiers differ in row count")
    for col in IDENTIFIER_COLS:
        if col not in identifiers.columns:
            raise ValueError(f"Required identifier missing: {col}")
    if TARGET_SOURCE not in targets.columns:
        raise ValueError("Target 'readmitted' missing from targets")
    if features.columns.intersection(targets.columns).size:
        raise ValueError("Overlapping features and target column names")
    if set(features.columns) & set(identifiers.columns):
        raise ValueError("Identifiers were unexpectedly included among features")
    combined = pd.concat(
        [identifiers.reset_index(drop=True), features.reset_index(drop=True),
         targets.reset_index(drop=True)], axis=1
    )
    if combined[list(IDENTIFIER_COLS)].isna().any().any():
        raise ValueError("Missing patient or encounter identifiers")
    if combined["encounter_id"].duplicated().any():
        raise ValueError("Encounter identifiers are not unique")
    return combined


def filter_cohort(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply prespecified outcome-relevance and invalid-gender exclusions."""
    required = {TARGET_SOURCE, *IDENTIFIER_COLS, "gender", "discharge_disposition_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    allowed = {"NO", ">30", "<30"}
    values = set(data[TARGET_SOURCE].dropna().astype(str).unique())
    if data[TARGET_SOURCE].isna().any() or not values.issubset(allowed):
        raise ValueError(f"Unexpected target categories: {values}")

    dispositions = pd.to_numeric(data["discharge_disposition_id"], errors="coerce")
    death_hospice = dispositions.isin(DEATH_HOSPICE_CODES)
    invalid_gender = data["gender"].astype("string").str.strip().str.lower().eq(
        "unknown/invalid"
    ).fillna(False)
    eligible = ~(death_hospice | invalid_gender)
    filtered = data.loc[eligible].copy().reset_index(drop=True)
    filtered[TARGET_BINARY] = filtered[TARGET_SOURCE].eq("<30").astype("int8")

    summary: dict[str, Any] = {
        "original_encounters": int(len(data)),
        "excluded_death_or_hospice": int(death_hospice.sum()),
        "excluded_invalid_gender": int(invalid_gender.sum()),
        "excluded_by_both_rules": int((death_hospice & invalid_gender).sum()),
        "excluded_total_unique_rows": int((~eligible).sum()),
        "remaining_encounters": int(len(filtered)),
        "original_unique_patients": int(data["patient_nbr"].nunique()),
        "remaining_unique_patients": int(filtered["patient_nbr"].nunique()),
        "original_readmitted_30_rate": float(data[TARGET_SOURCE].eq("<30").mean()),
        "remaining_readmitted_30_rate": float(filtered[TARGET_BINARY].mean()),
        "candidate_exclusion_readmission_counts": {
            str(k): int(v) for k, v in
            data.loc[death_hospice, TARGET_SOURCE].value_counts().items()
        },
    }
    return filtered, summary


def grouped_train_test_split(data: pd.DataFrame,
                             test_size: float = TEST_SIZE,
                             seed: int = SEED
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "patient_nbr" not in data or TARGET_BINARY not in data:
        raise ValueError("patient_nbr and readmitted_30 are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(data, y=data[TARGET_BINARY], groups=data["patient_nbr"])
    )
    train = data.iloc[train_idx].copy().reset_index(drop=True)
    test = data.iloc[test_idx].copy().reset_index(drop=True)
    assert not set(train["patient_nbr"]) & set(test["patient_nbr"]), (
        "Patient leakage detected"
    )
    assert not set(train["encounter_id"]) & set(test["encounter_id"]), (
        "Encounter leakage detected"
    )
    assert len(train) + len(test) == len(data)
    return train, test


def _as_category(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.mask(result.isin(["", "?"]))
    return result.fillna(missing_label).astype(str)


def group_icd9(code: Any) -> str:
    """Map ICD-9-CM diagnosis strings to broad chapters plus diabetes, V, and E."""
    if pd.isna(code):
        return "Missing"
    text = str(code).strip().upper()
    if text in {"", "?", "NAN", "NONE", "MISSING"}:
        return "Missing"
    if re.fullmatch(r"V\d+(?:\.\d+)?", text):
        return "V_Supplementary_factors"
    if re.fullmatch(r"E\d+(?:\.\d+)?", text):
        return "E_External_causes"
    if not re.fullmatch(r"\d{1,3}(?:\.\d+)?", text):
        return "Other_or_unrecognized_code"
    number = int(text.split(".")[0])
    if number == 250:
        return "Diabetes_250"
    chapters = (
        (1, 139, "Infectious_parasitic"),
        (140, 239, "Neoplasms"),
        (240, 279, "Endocrine_nutritional_metabolic"),
        (280, 289, "Blood"),
        (290, 319, "Mental"),
        (320, 389, "Nervous_sense_organs"),
        (390, 459, "Circulatory"),
        (460, 519, "Respiratory"),
        (520, 579, "Digestive"),
        (580, 629, "Genitourinary"),
        (630, 679, "Pregnancy_childbirth"),
        (680, 709, "Skin"),
        (710, 739, "Musculoskeletal"),
        (740, 759, "Congenital"),
        (760, 779, "Perinatal"),
        (780, 799, "Symptoms_ill_defined"),
        (800, 999, "Injury_poisoning"),
    )
    for lower, upper, name in chapters:
        if lower <= number <= upper:
            return name
    return "Other_or_unrecognized_code"


def _clean_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise cleaning only; no train/test learned statistics."""
    result = frame.drop(columns=[*IDENTIFIER_COLS, TARGET_SOURCE, *DROP_COLS]).copy()
    for col in DIAG_COLS:
        result[col] = result[col].map(group_icd9).astype(str)
    for col in result.columns:
        if col in NUMERIC_FEATURES or col == TARGET_BINARY:
            continue
        fill = "Not_recorded" if col in LAB_COLS else "Missing"
        result[col] = _as_category(result[col], fill)
    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result[TARGET_BINARY] = result[TARGET_BINARY].astype("int8")
    return result


def transform_train_test(train_full: pd.DataFrame, test_full: pd.DataFrame,
                         min_specialty_count: int = RARE_SPECIALTY_MIN_COUNT
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare semantically typed tables with train-only rare mappings/imputation."""
    train = _clean_semantics(train_full)
    test = _clean_semantics(test_full)

    if list(train.columns) != list(test.columns):
        raise AssertionError("Train/test feature columns differ")

    specialty_counts = train["medical_specialty"].value_counts()
    kept_specialties = sorted([
        str(k) for k, count in specialty_counts.items()
        if count >= min_specialty_count and k != "Missing"
    ])
    for frame in (train, test):
        s = frame["medical_specialty"]
        frame["medical_specialty"] = s.where(
            s.isin(kept_specialties) | s.eq("Missing"), "Other"
        ).astype(str)

    numeric_medians: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        median = train[col].median()
        if pd.isna(median):
            raise ValueError(f"No observed training values for numeric feature {col}")
        numeric_medians[col] = float(median)
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    for name, frame in (("training", train), ("test", test)):
        if frame.isna().any().any():
            bad = frame.columns[frame.isna().any()].tolist()
            raise AssertionError(f"{name} still has missing values in {bad}")
        if "patient_nbr" in frame or "encounter_id" in frame or TARGET_SOURCE in frame:
            raise AssertionError("Patient/encounter identifiers or original target leaked")
        if not set(frame[TARGET_BINARY].unique()).issubset({0, 1}):
            raise AssertionError("Invalid binary target")

    categorical_features = [
        c for c in train.columns if c not in NUMERIC_FEATURES and c != TARGET_BINARY
    ]
    schema: dict[str, Any] = {
        "dataset_uci_id": DATASET_ID,
        "seed": SEED,
        "split_method": "GroupShuffleSplit(patient_nbr)",
        "test_size_requested_fraction_of_patients": TEST_SIZE,
        "rare_specialty_min_training_count": min_specialty_count,
        "retained_specialties_from_train": kept_specialties,
        "numerical_imputation_medians_from_train": numeric_medians,
        "categorical_feature_columns": categorical_features,
        "numerical_feature_columns": list(NUMERIC_FEATURES),
        "target_column": TARGET_BINARY,
        "target_semantic_type": "categorical_binary",
        "lab_missing_category": "Not_recorded",
        "other_categorical_missing_category": "Missing",
        "removed_columns": list(DROP_COLS),
        "diagnosis_grouping": "ICD-9-CM broad chapters; 250 separated; V/E separate",
        "excluded_discharge_codes": list(DEATH_HOSPICE_CODES),
        "gender_exclusion": "Unknown/Invalid",
    }
    return train, test, schema


def save_outputs(train: pd.DataFrame, test: pd.DataFrame,
                 train_full: pd.DataFrame, test_full: pd.DataFrame,
                 schema: dict[str, Any], cohort_summary: dict[str, Any],
                 output_dir: str | Path) -> dict[str, Path]:
    """Save model-input CSVs + schema and ID-only audit manifest to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train_semantic.csv"
    test_path = out / "test_semantic.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    manifest = pd.concat([
        train_full[[*IDENTIFIER_COLS]].assign(split="train"),
        test_full[[*IDENTIFIER_COLS]].assign(split="test"),
    ], ignore_index=True)
    manifest_path = out / "split_manifest_IDS_ONLY.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        **schema,
        "cohort_summary": cohort_summary,
        "train_encounters": len(train),
        "test_encounters": len(test),
        "train_unique_patients": int(train_full["patient_nbr"].nunique()),
        "test_unique_patients": int(test_full["patient_nbr"].nunique()),
        "train_positive_rate": float(train[TARGET_BINARY].mean()),
        "test_positive_rate": float(test[TARGET_BINARY].mean()),
        "training_feature_columns": [c for c in train.columns if c != TARGET_BINARY],
    }
    meta_path = out / "preprocessing_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return {"train": train_path, "test": test_path, "manifest": manifest_path,
            "metadata": meta_path}


def load_semantic_csv(path: str | Path, metadata_path: str | Path) -> pd.DataFrame:
    """Reload CSV without losing categorical dtype for integer-coded IDs."""
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    string_columns = metadata["categorical_feature_columns"]
    frame = pd.read_csv(path, dtype={name: "string" for name in string_columns})
    for name in string_columns:
        frame[name] = frame[name].astype(str)
    frame[TARGET_BINARY] = frame[TARGET_BINARY].astype(int)
    return frame
