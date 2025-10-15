#!/usr/bin/env python3
"""
Download and process CDC/WHO growth data into NumPy .npz format.

This script downloads growth reference data from CDC and WHO sources,
parses the CSVs, and saves them as compressed NumPy arrays for use
in the biv package's ZScoreDetector.

WHO data covers ages from birth to <24 months (0-23.99 months).
CDC data covers ages from 24 months and above (24.0+ months).

The Z-score calculations automatically select the appropriate reference
data based on the input age: WHO for ages <24 months, CDC for ages >=24 months.
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Required columns for parsing
REQUIRED_COLS = {
    "cdc_bmi": ["sex", "agemos", "L", "M", "S", "P95", "sigma"],
    "cdc_other": ["Sex", "Agemos", "L", "M", "S"],
}

# Data sources with URLs
DATA_SOURCES: Dict[str, List[Tuple[str, str]]] = {
    "cdc": [
        ("wtage", "https://www.cdc.gov/growthcharts/data/zscore/wtage.csv"),
        ("statage", "https://www.cdc.gov/growthcharts/data/zscore/statage.csv"),
        (
            "bmi_age",
            "https://www.cdc.gov/growthcharts/data/extended-bmi/bmi-age-2022.csv",
        ),
    ],
    "who": [
        (
            "boys_wtage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Boys-Weight-for-age-Percentiles.csv",
        ),
        (
            "boys_lenage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Boys-Length-for-age-Percentiles.csv",
        ),
        (
            "boys_headage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Boys-Head-Circumference-for-age-Percentiles.csv",
        ),
        (
            "boys_wtlen",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Boys-Weight-for-length-Percentiles.csv",
        ),
        (
            "girls_wtage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Girls-Weight-for-age%20Percentiles.csv",
        ),
        (
            "girls_lenage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Girls-Length-for-age-Percentiles.csv",
        ),
        (
            "girls_headage",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Girls-Head-Circumference-for-age-Percentiles.csv",
        ),
        (
            "girls_wtlen",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/growthcharts/WHO-Girls-Weight-for-length-Percentiles.csv",
        ),
    ],
}


def download_csv(url: str, timeout: int = 30) -> str:
    """Download CSV content from URL with retries."""
    try:
        # Create retry configuration
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,  # Exponential backoff
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        with requests.Session() as session:
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            response = session.get(url, timeout=timeout, verify=True)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_array(arr: np.ndarray, array_name: str, source_type: str) -> None:
    """Validate parsed array for common issues."""
    if arr.size == 0:
        logger.warning(f"{array_name}: empty array")
        return

    # Determine age column name
    age_col = "age" if "age" in arr.dtype.names else None

    if age_col:
        ages = arr[age_col]
        # Check finite values but allow NaN
        if not np.all(np.isfinite(ages) | np.isnan(ages)):
            raise ValueError(f"{array_name}: non-finite {age_col} values")
        # Check monotonic increasing for finite values
        finite_ages = ages[np.isfinite(ages)]
        if len(finite_ages) > 1 and not np.all(finite_ages[:-1] <= finite_ages[1:]):
            raise ValueError(f"{array_name}: {age_col} not monotonically increasing")

        # Check for potential unit mismatch (age in years instead of months)
        if len(finite_ages) > 0:
            max_age = np.max(finite_ages)
            if max_age > 241:
                logger.warning(
                    f"{array_name}: Age values up to {max_age:.1f} months detected. "
                    "Values >241 months suggest input may be in years rather than months."
                )

    # Check finite (allow nan) and no negative for M, S (L can be negative)
    for col in ["L", "M", "S"]:
        if col in arr.dtype.names:
            values = arr[col]
            if not np.all(np.isfinite(values) | np.isnan(values)):
                raise ValueError(f"{array_name}: non-finite {col} values")
            if col != "L" and np.any((values < 0) & (~np.isnan(values))):
                raise ValueError(f"{array_name}: negative {col} values")

    # For CDC BMI, check P95 and sigma
    if source_type == "cdc":
        for col in ["P95", "sigma"]:
            if col in arr.dtype.names:
                values = arr[col]
                if not np.all(np.isfinite(values) | np.isnan(values)):
                    raise ValueError(f"{array_name}: non-finite {col} values")
                if np.any((values < 0) & (~np.isnan(values))):
                    raise ValueError(f"{array_name}: negative {col} values")


def parse_cdc_csv(content: str, name: str) -> Dict[str, np.ndarray]:
    """Parse CDC CSV data into minimal structured arrays with only used columns."""
    lines = content.strip().split("\n")
    header = lines[0].split(",")

    # Parse data (skip header)
    data = []
    max_cols = len(header)
    for line in lines[1:]:
        if line.strip():
            raw_values = line.split(",")
            # Normalize to exactly max_cols: truncate extra, pad if fewer
            values = (raw_values + [""] * max_cols)[:max_cols]
            # Convert to float, empty strings to NaN
            row = []
            for val in values:
                val = val.strip()
                try:
                    row.append(float(val) if val else np.nan)
                except ValueError:
                    row.append(val)  # Keep strings for sex
            data.append(row)

    data_array = np.array(data, dtype=object)

    # Split by sex (1=male, 2=female) for CDC data
    if "bmi_age" in name.lower():
        essential_cols = REQUIRED_COLS["cdc_bmi"]
        male_mask = data_array[:, 0] == 1
        female_mask = data_array[:, 0] == 2

        male_data = data_array[male_mask]
        female_data = data_array[female_mask]

        # Create minimal dtype with only essential columns, normalizing age column
        dtype_fields = []
        for col in essential_cols:
            if col == "sex":
                continue
            field_name = "age" if col == "agemos" else col
            dtype_fields.append((field_name, "f8"))
        dt = np.dtype(dtype_fields)

        male_structured = np.zeros(male_data.shape[0], dtype=dt)
        female_structured = np.zeros(female_data.shape[0], dtype=dt)

        for col in essential_cols:
            if col != "sex" and col in header:
                col_idx = header.index(col)
                field_name = "age" if col == "agemos" else col
                male_structured[field_name] = male_data[:, col_idx].astype(float)
                female_structured[field_name] = female_data[:, col_idx].astype(float)

        # Validate arrays
        validate_array(male_structured, "bmi_male", "cdc")
        validate_array(female_structured, "bmi_female", "cdc")

        return {"bmi_male": male_structured, "bmi_female": female_structured}

    else:
        essential_cols = REQUIRED_COLS["cdc_other"]
        sex_col_idx = header.index("Sex")

        male_mask = data_array[:, sex_col_idx] == 1
        female_mask = data_array[:, sex_col_idx] == 2

        male_data = data_array[male_mask]
        female_data = data_array[female_mask]

        # Create minimal dtype with only essential columns, normalizing age column
        dtype_fields = []
        for col in essential_cols:
            if col == "Sex":
                continue
            field_name = "age" if col == "Agemos" else col
            dtype_fields.append((field_name, "f8"))
        dt = np.dtype(dtype_fields)

        male_structured = np.zeros(male_data.shape[0], dtype=dt)
        female_structured = np.zeros(female_data.shape[0], dtype=dt)

        for col in essential_cols:
            if col != "Sex" and col in header:
                col_idx = header.index(col)
                field_name = "age" if col == "Agemos" else col
                male_structured[field_name] = male_data[:, col_idx].astype(float)
                female_structured[field_name] = female_data[:, col_idx].astype(float)

        male_key = f"{name.replace('statage', 'haz').replace('wtage', 'waz')}_male"
        female_key = f"{name.replace('statage', 'haz').replace('wtage', 'waz')}_female"

        # Validate arrays
        validate_array(male_structured, male_key, "cdc")
        validate_array(female_structured, female_key, "cdc")

        return {male_key: male_structured, female_key: female_structured}


def parse_who_csv(content: str, name: str) -> Dict[str, np.ndarray]:
    """Parse WHO CSV data into minimal structured arrays with only used columns."""
    lines = content.strip().split("\n")
    header_raw = lines[0].split(",")

    # Clean header names (remove BOM, special chars)
    header_clean = [
        col.replace("\ufeff", "")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        for col in header_raw
    ]

    # WHO arrays only need: age column (Month or Length), L, M, S columns
    # Use "Length" for weight-for-length files, "Month" for others
    age_col = "Length" if "wtlen" in name else "Month"
    essential_cols = [age_col, "L", "M", "S"]
    present_cols = [col for col in essential_cols if col in header_clean]
    col_indices = [header_clean.index(col) for col in present_cols]

    # Warn for missing essential columns
    for col in essential_cols:
        if col not in present_cols:
            logger.warning(f"Essential column {col} not found in {name} header")

    data = []
    for line in lines[1:]:
        if line.strip():
            values = line.split(",")
            # Extract only present columns
            row = []
            for idx in col_indices:
                if idx < len(values):
                    val = values[idx].strip()
                    try:
                        row.append(float(val) if val else np.nan)
                    except ValueError:
                        row.append(np.nan)
                else:
                    row.append(np.nan)
            data.append(row)

    data_array = np.array(data, dtype=float)

    # Create minimal dtype with unified age column name
    dt = np.dtype([("age", "f8"), ("L", "f8"), ("M", "f8"), ("S", "f8")])

    structured = np.zeros(data_array.shape[0], dtype=dt)

    # Set present fields
    lms_fields = ["L", "M", "S"]
    for col in lms_fields:
        if col in present_cols:
            idx = present_cols.index(col)
            structured[col] = data_array[:, idx]

    # Handle age separately
    if age_col in present_cols:
        idx = present_cols.index(age_col)
        structured["age"] = data_array[:, idx]
    else:
        structured["age"] = np.full(data_array.shape[0], np.nan)

    # Determine measure type and sex from name
    measure_map = {"wtage": "waz", "lenage": "haz", "headage": "headcz", "wtlen": "wlz"}

    for key, value in measure_map.items():
        if key in name:
            measure = value
            break
    else:
        measure = name

    sex = "male" if "boys" in name else "female"

    array_key = f"{measure}_{sex}"

    # For WHO month-based data, filter out ages >= 24.0 months
    # Weight-for-length uses Length in cm, so preserve all data without filtering
    if age_col == "Month" and "wtlen" not in name:
        valid_indices = structured["age"] < 24.0
        structured = structured[valid_indices]

    # Validate array
    validate_array(structured, array_key, "who")

    return {array_key: structured}


def save_npz(data: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save data dictionary as compressed NumPy .npz file."""
    np.savez_compressed(output_path, **data)
    logger.info(f"Saved {len(data)} arrays to {output_path}")


def main(strict_mode=False, source_filter=None, force=False):
    """Main function to download and process all data."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "growth_references.npz"

    all_data = {}

    # Load existing data if available
    if output_path.exists():
        loaded = np.load(output_path)
        # Load all data arrays (non-metadata)
        for key in loaded.files:
            if not key.startswith("metadata_"):
                all_data[key] = loaded[key]
        loaded.close()

    # Track failed sources for strict mode
    failed_sources = []

    # Process each source
    total_sources = sum(len(sources) for sources in DATA_SOURCES.values())
    with tqdm(total=total_sources, desc="Fetching sources") as pbar:
        for source_type, sources in DATA_SOURCES.items():
            if source_filter and source_type != source_filter:
                pbar.update(len(sources))
                continue

            for name, url in sources:
                pbar.set_postfix({"source": f"{source_type.upper()}: {name}"})
                pbar.update(1)

                # Always download since these files are not that big
                try:
                    # Download CSV
                    csv_content = download_csv(url)
                    hash_value = compute_sha256(csv_content)

                    # Parse based on source type
                    if source_type == "cdc":
                        parsed = parse_cdc_csv(csv_content, name)
                    else:  # who
                        parsed = parse_who_csv(csv_content, name)

                    all_data.update(parsed)

                    # Store metadata per source
                    metadata = {
                        "url": url,
                        "hash": hash_value,
                        "timestamp": str(np.datetime64("now")),
                    }

                    # Add metadata as arrays
                    for key, value in metadata.items():
                        all_data[f"metadata_{name}_{key}"] = np.array(
                            [value], dtype="U256"
                        )

                except Exception as e:
                    failed_sources.append(f"{source_type}::{name}")
                    logger.error(f"Failed to process {source_type}::{name}: {e}")
                    if strict_mode:
                        # Raise at end with all failed sources
                        pass
                    continue

    pbar.close()

    # Check for strict mode failures
    if strict_mode and failed_sources:
        raise RuntimeError(
            f"Strict mode failed: Unable to process sources: {', '.join(failed_sources)}"
        )

    # Save combined data
    save_npz(all_data, output_path)

    # Verify saved data
    loaded = np.load(output_path)
    logger.info(f"Verification: {len(loaded.files)} arrays saved")

    for key in loaded.files[:5]:  # Show first 5
        logger.info(f"  {key}: shape {loaded[key].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download growth reference data from CDC and WHO sources."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download all sources, ignoring timestamps",
    )
    parser.add_argument(
        "--source",
        choices=["cdc", "who"],
        help="Download only from specified source type (cdc or who)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: exit on any error instead of continuing with warnings",
    )
    args = parser.parse_args()

    main(strict_mode=args.strict, source_filter=args.source, force=args.force)
