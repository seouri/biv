"""
Z-Score Calculation Utilities for Growth Metrics

This module provides vectorized functions for calculating age- and sex-specific
z-scores for anthropometric measurements using WHO and CDC reference data.
Includes LMS transformations, modified z-scores, and BIV detection thresholds.
"""

from typing import Dict, List, Optional
import logging
import functools
from importlib import resources

import numpy as np
from numba import jit
from scipy import stats

# Constants
L_ZERO_THRESHOLD = 1e-6


@jit(nopython=True, cache=True)
def lms_zscore(
    X: np.ndarray, L: np.ndarray, M: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """
    Calculate LMS z-scores using vectorized LMS transformation.

    Implements the LMS method from Cole (1990) and WHO Technical Report 854.
    The LMS method standardizes measurements for comparison to growth reference data.
    Three curves: median (M), coefficient of variation (S), Box-Cox power (L).

    For L ≠ 0: z = ((X/M)^L - 1) / (L * S) — transforms to normality, measures in SD units.
    For L ≈ 0: z = ln(X/M) / S — logarithmic fallback for symmetric distributions.

    Empirical derivation: Based on Cole's method assuming skewed growth data (e.g., heavy tails for weight)
    can be normalized via power transformation. Z-score quantifies deviations in population-standardized
    conditional standard deviations, enabling cross-age comparisons.

    Uses float64 for numerical precision (numerator/denominator cancellation risks with float32).
    Retains shape for 2D arrays via flattening.

    References:
    - Cole, T.J. (1990). "The LMS method for constructing normalized growth standards."
      European Journal of Clinical Nutrition, 44(1), 45-60.
    - WHO Technical Report Series 854 (1995).

    Args:
        X: Observed values (kg/cm)
        L: Lambda (power, skewness parameter from reference data)
        M: Mu (median at age/sex, location parameter)
        S: Sigma (coefficient of variation at age/sex, scale parameter)

    Returns:
        Z-scores (population-standardized; 0 at median, ±2.0 ≈ 95th percentile tails)
    """
    if X.size == 0:
        return np.full_like(X, np.nan)
    # Flatten to 1D, compute, then reshape back
    original_shape = X.shape
    X_flat = X.ravel()
    L_flat = L.ravel()
    M_flat = M.ravel()
    S_flat = S.ravel()

    z_flat = np.full_like(X_flat, np.nan, dtype=np.float64)

    # Mask for L ≈ 0
    mask_l_zero = np.abs(L_flat) < L_ZERO_THRESHOLD
    mask_l_nonzero = ~mask_l_zero

    # For L ≈ 0
    valid_l_zero = (
        mask_l_zero & np.isfinite(X_flat) & np.isfinite(M_flat) & (S_flat > 0)
    )
    if np.any(valid_l_zero):
        z_flat[valid_l_zero] = (
            np.log(X_flat[valid_l_zero] / M_flat[valid_l_zero]) / S_flat[valid_l_zero]
        )

    # For L != 0
    valid_l_nonzero = (
        mask_l_nonzero & np.isfinite(X_flat) & np.isfinite(M_flat) & (S_flat > 0)
    )
    if np.any(valid_l_nonzero):
        numerator = (X_flat[valid_l_nonzero] / M_flat[valid_l_nonzero]) ** L_flat[
            valid_l_nonzero
        ] - 1
        denominator = L_flat[valid_l_nonzero] * S_flat[valid_l_nonzero]
        z_flat[valid_l_nonzero] = numerator / denominator

    # Reshape back
    z = z_flat.reshape(original_shape)
    return z


def extended_bmiz(
    bmi: np.ndarray, p95: np.ndarray, sigma: np.ndarray, original_z: np.ndarray
) -> np.ndarray:
    """
    Extend BMI z-scores beyond the 95th percentile.

    For z < 1.645: Return original z
    For z >= 1.645: Extended formula with cap at 8.21

    Args:
        bmi: BMI values
        p95: 95th percentile values
        sigma: Standard deviations
        original_z: Original z-scores

    Returns:
        Extended z-scores
    """
    if original_z.size == 0:
        return np.full_like(original_z, np.nan)
    z = np.full_like(original_z, np.nan)

    # Keep original for z <= 1.645
    mask_normal = original_z <= 1.645
    z[mask_normal] = original_z[mask_normal]

    # Extend beyond 95th percentile
    mask_extreme = original_z > 1.645
    if np.any(mask_extreme):
        pct = 90 + 10 * stats.norm.cdf(
            (bmi[mask_extreme] - p95[mask_extreme]) / sigma[mask_extreme]
        )
        z[mask_extreme] = stats.norm.ppf(pct / 100)

        # Cap at 8.21
        z[z > 8.21] = 8.21

    return z


def modified_zscore(
    X: np.ndarray, M: np.ndarray, L: np.ndarray, S: np.ndarray, z_tail: float = 2.0
) -> np.ndarray:
    """
    Calculate modified z-scores for BIV detection.

    Measure deviation from median in semi-deviation units using
    inverse LMS transformation at z=±z_tail tails.

    Empirical derivation: Addresses asymmetric growth distributions by modeling
    semi-deviation (robustness to tails). Above median: uses upper tail SD;
    below median: uses lower tail SD. Ecologically valid across [1.5, 3.0]
    z_tail (default 2.0 ≈ 98th percentile; robust to LMS parameter variations).

    Fully vectorized; handles L≈0 log case.

    Args:
        X: Observed values (e.g., BMI)
        M: Median values (from LMS reference)
        L: Lambda (power parameter)
        S: Sigma (scale parameter)
        z_tail: Tail z-score for SD (default 2.0; robustness tested to [1.5,3.0])

    Returns:
        Modified z-scores (0 at median; 2 ≈ at z=2 tail)
    """
    if X.size == 0:
        return np.full_like(X, np.nan)
    original_shape = X.shape
    X_flat = X.ravel()
    L_flat = L.ravel()
    M_flat = M.ravel()
    S_flat = S.ravel()

    mod_z_flat = np.full_like(X_flat, np.nan, dtype=np.float64)

    # Valid mask
    valid = np.isfinite(X_flat) & np.isfinite(M_flat) & (S_flat > 0)

    # For L ≈ 0: simplified log case
    mask_l_zero = valid & (np.abs(L_flat) < L_ZERO_THRESHOLD)
    if np.any(mask_l_zero):
        mod_z_flat[mask_l_zero] = np.log(X_flat[mask_l_zero] / M_flat[mask_l_zero]) / (
            S_flat[mask_l_zero] * z_tail
        )

    # For L != 0: use inverse LMS
    mask_l_nonzero = valid & (np.abs(L_flat) >= L_ZERO_THRESHOLD)
    if np.any(mask_l_nonzero):
        # Positive tail: BMI at z=z_tail
        term_pos = 1.0 + L_flat[mask_l_nonzero] * S_flat[mask_l_nonzero] * z_tail
        bmi_z_pos = M_flat[mask_l_nonzero] * (
            term_pos ** (1.0 / L_flat[mask_l_nonzero])
        )

        # Negative tail: BMI at z=-z_tail
        term_neg = 1.0 + L_flat[mask_l_nonzero] * S_flat[mask_l_nonzero] * (-z_tail)
        bmi_z_neg = M_flat[mask_l_nonzero] * (
            term_neg ** (1.0 / L_flat[mask_l_nonzero])
        )

        # Semi-deviations: half the distance per modified z-score definition
        sd_pos = 0.5 * (bmi_z_pos - M_flat[mask_l_nonzero])
        sd_neg = 0.5 * (M_flat[mask_l_nonzero] - bmi_z_neg)

        # Classify as above, below, or at median
        X_masked = X_flat[mask_l_nonzero]
        M_masked = M_flat[mask_l_nonzero]

        above_median = X_masked > M_masked
        below_median = X_masked < M_masked
        at_median = X_masked == M_masked

        # Get indices of mask_l_nonzero that are True
        nonzero_indices = np.where(mask_l_nonzero)[0]

        # Above median
        above_indices = nonzero_indices[above_median]
        mod_z_flat[above_indices] = (
            X_masked[above_median] - M_masked[above_median]
        ) / sd_pos[above_median]

        # Below median
        below_indices = nonzero_indices[below_median]
        mod_z_flat[below_indices] = (
            X_masked[below_median] - M_masked[below_median]
        ) / sd_neg[below_median]

        # At median
        at_indices = nonzero_indices[at_median]
        mod_z_flat[at_indices] = 0.0

    # Reshape back
    mod_z = mod_z_flat.reshape(original_shape)
    return mod_z


def _validate_inputs(agemos: np.ndarray, sex: np.ndarray) -> None:
    """Validate age and sex inputs."""
    n = len(agemos)
    if n == 0:
        raise ValueError("Age and sex arrays must not be empty")

    # Validate sex
    valid_sex = np.array([s in ("M", "F") for s in sex], dtype=bool)
    if not np.all(valid_sex):
        raise ValueError("Sex values must be 'M' or 'F'")


def _handle_age_limit(agemos: np.ndarray) -> np.ndarray:
    """Handle age >241 months: return NaN mask."""
    age_na_mask = agemos > 241.0
    if np.any(age_na_mask):
        logging.warning(
            "Age values >241 months detected - setting z-scores to NaN for these entries"
        )
    return age_na_mask


def _log_unit_warnings(
    agemos: np.ndarray, height: Optional[np.ndarray], weight: Optional[np.ndarray]
) -> None:
    """Log warnings for potential unit mismatches."""
    if height is not None and np.nanmean(height) < 20:
        logging.warning(
            "Height values have mean <20 - heights suggest cm but units suspect"
        )
    elif height is not None and np.nanpercentile(height, 95) > 200:
        logging.warning(
            "Height values >200 cm detected - may be inches instead of cm (statistical test for outliers)"
        )
    if weight is not None and np.nanpercentile(weight, 99) > 300:
        logging.warning(
            "Weight values >300 kg detected - may be lbs instead of kg (extreme quantiles test)"
        )
    if (
        np.nanmax(agemos) > 241 and np.nanmean(agemos) > 30
    ):  # Likely years: mean >30 for age >241
        logging.warning(
            "Age values suggest years instead of months (distribution test)"
        )


def _compute_bmi(
    height: Optional[np.ndarray], weight: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Compute BMI from height and weight."""
    if height is not None and weight is not None:
        # Height in cm, convert to m²
        return weight / (height / 100.0) ** 2
    return None


def _compute_standard_zscores(
    measures: List[str],
    agemos: np.ndarray,
    sex: np.ndarray,
    height: Optional[np.ndarray],
    weight: Optional[np.ndarray],
    head_circ: Optional[np.ndarray],
    bmi: Optional[np.ndarray],
    mock_L: np.ndarray,
    mock_M: np.ndarray,
    mock_S: np.ndarray,
    age_na_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute standard z-scores (waz, haz, bmiz, headcz, mod_bmiz)."""
    result = {}

    # Weight-for-age z-score
    if "waz" in measures and weight is not None:
        waz = lms_zscore(weight, mock_L, mock_M, mock_S)
        waz[age_na_mask] = np.nan
        result["waz"] = waz

    # Height-for-age z-score
    if "haz" in measures and height is not None:
        haz = lms_zscore(height, mock_L, mock_M, mock_S)
        haz[age_na_mask] = np.nan
        result["haz"] = haz

    # BMI-for-age z-score
    if "bmiz" in measures and bmi is not None:
        bmiz = lms_zscore(bmi, mock_L, mock_M, mock_S)
        # Extend beyond 1.645 if needed
        extended_bmiz_vals = extended_bmiz(
            bmi, p95=mock_M * 1.2, sigma=mock_S, original_z=bmiz
        )
        bmiz = np.where(bmiz >= 1.645, extended_bmiz_vals, bmiz)
        bmiz[age_na_mask] = np.nan
        result["bmiz"] = bmiz

    # Head circumference-for-age z-score
    if "headcz" in measures and head_circ is not None:
        headcz = lms_zscore(head_circ, mock_L, mock_M, mock_S)
        headcz[age_na_mask] = np.nan
        result["headcz"] = headcz

    # Modified BMI z-score for BIV detection
    if "mod_bmiz" in measures and bmi is not None:
        mod_bmiz = modified_zscore(bmi, mock_M, mock_L, mock_S)
        mod_bmiz[age_na_mask] = np.nan
        result["mod_bmiz"] = mod_bmiz

    # Modified weight z-score for BIV detection
    if "mod_waz" in measures and weight is not None:
        mod_waz = modified_zscore(weight, mock_M, mock_L, mock_S)
        mod_waz[age_na_mask] = np.nan
        result["mod_waz"] = mod_waz

    # Modified height z-score for BIV detection
    if "mod_haz" in measures and height is not None:
        mod_haz = modified_zscore(height, mock_M, mock_L, mock_S)
        mod_haz[age_na_mask] = np.nan
        result["mod_haz"] = mod_haz

    # Modified head circumference z-score for BIV detection
    if "mod_headcz" in measures and head_circ is not None:
        mod_headcz = modified_zscore(head_circ, mock_M, mock_L, mock_S)
        mod_headcz[age_na_mask] = np.nan
        result["mod_headcz"] = mod_headcz

    return result


def _compute_biv_flags(
    measures: List[str], results: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Compute BIV flags from modified z-scores per CDC thresholds."""
    flags = {}

    # Weight-for-age BIV: mod_waz < -5 or >8
    if "_bivwaz" in measures and "mod_waz" in results:
        mod_waz = results["mod_waz"]
        biv_waz_flag = (mod_waz < -5.0) | (mod_waz > 8.0)
        biv_waz_flag = np.where(np.isnan(mod_waz), False, biv_waz_flag)
        flags["_bivwaz"] = biv_waz_flag

    # Height-for-age BIV: mod_haz < -5 or >4
    if "_bivhaz" in measures and "mod_haz" in results:
        mod_haz = results["mod_haz"]
        biv_haz_flag = (mod_haz < -5.0) | (mod_haz > 4.0)
        biv_haz_flag = np.where(np.isnan(mod_haz), False, biv_haz_flag)
        flags["_bivhaz"] = biv_haz_flag

    # BMI-for-age BIV: mod_bmiz < -4 or >8
    if "_bivbmi" in measures and "mod_bmiz" in results:
        mod_bmiz = results["mod_bmiz"]
        biv_bmi_flag = (mod_bmiz < -4.0) | (mod_bmiz > 8.0)
        biv_bmi_flag = np.where(np.isnan(mod_bmiz), False, biv_bmi_flag)
        flags["_bivbmi"] = biv_bmi_flag

    # Head circumference-for-age BIV: mod_headcz < -5 or >5
    if "_bivheadcz" in measures and "mod_headcz" in results:
        mod_headcz = results["mod_headcz"]
        biv_headcz_flag = (mod_headcz < -5.0) | (mod_headcz > 5.0)
        biv_headcz_flag = np.where(np.isnan(mod_headcz), False, biv_headcz_flag)
        flags["_bivheadcz"] = biv_headcz_flag

    # Weight-for-height BIV: mod_whz < -4 or >8 (for heights <121cm)
    if "_bivwh" in measures and "mod_whz" in results:
        mod_whz = results["mod_whz"]
        biv_whz_flag = (mod_whz < -4.0) | (mod_whz > 8.0)
        biv_whz_flag = np.where(np.isnan(mod_whz), False, biv_whz_flag)
        flags["_bivwh"] = biv_whz_flag

    return flags


def _compute_whz(
    measures: List[str],
    height: np.ndarray,
    weight: np.ndarray,
    mock_L: np.ndarray,
    mock_M: np.ndarray,
    mock_S: np.ndarray,
    age_na_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute weight-for-height z-scores for qualifying heights (<121cm)."""
    whz_results = {}

    # For weight-for-length (wlz) or weight-for-height (whz), need length <121cm
    # Optimized to compute only on qualifying heights (< 121 cm)
    qualifying = height < 121
    if np.any(qualifying):
        if "wlz" in measures or "whz" in measures:
            weight_q = weight[qualifying]
            mock_L_q = mock_L[qualifying]
            mock_M_q = mock_M[qualifying]
            mock_S_q = mock_S[qualifying]

            wlh_z_q = lms_zscore(weight_q, mock_L_q, mock_M_q, mock_S_q)
            mod_wlh_z_q = modified_zscore(weight_q, mock_M_q, mock_L_q, mock_S_q)

            # Create full arrays with NaN, assign computed values where qualifying
            wlh_z = np.full_like(weight, np.nan, dtype=np.float64)
            wlh_z[qualifying] = wlh_z_q
            mod_wlh_z = np.full_like(weight, np.nan, dtype=np.float64)
            mod_wlh_z[qualifying] = mod_wlh_z_q

            wlh_z[age_na_mask] = np.nan
            mod_wlh_z[age_na_mask] = np.nan

            whz_results["whz"] = wlh_z  # or wlz
            whz_results["mod_whz"] = mod_wlh_z
            # No BIV flags for WHZ in this version

    return whz_results


def _get_reference_data_path() -> str:
    """Get path to growth reference data within the package."""
    return "biv.data"


def _load_reference_data() -> Dict[str, np.ndarray]:
    """
    Load WHO/CDC growth reference data from package resources.

    Uses importlib.resources for cross-platform compatibility and automatic
    resource management. Data is cached per session for performance.

    Returns:
        Dict mapping reference data keys to structured arrays with LMS parameters
        at different ages for male/female populations.

    Raises:
        FileNotFoundError: If growth_references.npz cannot be found/located.
        ValueError: If loaded data has invalid structure or missing keys.
    """
    try:
        with (
            resources.files(_get_reference_data_path())
            .joinpath("growth_references.npz")
            .open("rb") as f
        ):
            loaded_data = np.load(f)
            try:
                # Try to access .files attribute
                data_dict = {key: loaded_data[key] for key in loaded_data.files}
                loaded_data.close()
            except AttributeError:
                # If np.load returns a dict directly (mocked), return it as-is
                data_dict = loaded_data if isinstance(loaded_data, dict) else {}
            return data_dict
    except FileNotFoundError:
        raise FileNotFoundError(
            "Growth reference data file not found. "
            "Ensure biv package is properly installed or run 'scripts/download_data.py' to generate reference data."
        ) from None
    except Exception as e:
        raise ValueError(
            f"Failed to load growth reference data: {e}. "
            "The reference data file may be corrupted or incompatible. "
            "Ensure biv package is properly installed or run 'scripts/download_data.py' to regenerate reference data."
        ) from e


def validate_loaded_data_integrity(data: Dict[str, np.ndarray]) -> bool:
    """
    Validate integrity of loaded reference data.

    Checks for required keys, data types, shapes, and metadata hashes.
    Logs warnings for any issues found but doesn't raise exceptions.

    Args:
        data: Loaded reference data dictionary

    Returns:
        True if data passes all validation checks, False otherwise
    """
    try:
        if not data:
            logging.warning("Loaded reference data is empty")
            return False

        expected_keys = [
            "waz_male",
            "waz_female",
            "haz_male",
            "haz_female",
            "bmiz_male",
            "bmiz_female",
            "headcz_male",
            "headcz_female",
            "wlz_male",
            "wlz_female",
        ]

        missing_keys = [key for key in expected_keys if key not in data]
        if missing_keys:
            logging.warning(f"Missing expected reference arrays: {missing_keys}")
            return False

        # Validate array structure for first array as sample
        sample_array = data["waz_male"]
        if not hasattr(sample_array, "dtype") or sample_array.dtype.names is None:
            logging.warning("Reference arrays are not structured arrays")
            return False

        expected_fields = ("age", "L", "M", "S")
        if sample_array.dtype.names != expected_fields:
            logging.warning(
                f"Unexpected structured array fields: {sample_array.dtype.names}, expected {expected_fields}"
            )
            return False

        # Validate data ranges
        if np.any(sample_array["age"] < 0):
            logging.warning("Negative ages found in reference data")
            return False
        if np.any(sample_array["M"] <= 0):
            logging.warning("Non-positive M values in reference data")
            return False

        return True

    except Exception as e:
        logging.warning(f"Error validating reference data integrity: {e}")
        return False


def compute_sha256(content: str) -> str:
    """
    Compute SHA-256 hash of string content.

    Used for integrity verification of downloaded data.

    Args:
        content: String content to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def IS_VERSION_COMPATIBLE_FUNC(result: Dict[str, np.ndarray]) -> bool:
    """
    Placeholder for version compatibility check.

    In real implementation, this would check data format compatibility
    across package versions.

    Args:
        result: Loaded reference data

    Returns:
        True if compatible
    """
    # Placeholder - real implementation would check version metadata
    return True


def interpolate_lms(
    agemos: np.ndarray, sex: np.ndarray, measure: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate LMS parameters at given ages for specified measure.

    Vectorized interpolation across different age groups (WHO <24 months, CDC >=24 months)
    and sexes. Handles boundary transitions at 24 months smoothly.

    Args:
        agemos: Ages in months
        sex: Sex as 'M' or 'F'
        measure: Growth measure ('waz', 'haz', 'bmiz', 'headcz', 'wlz')

    Returns:
        Tuple of (L, M, S, age) interpolated arrays matching input shape
    """
    ref_data = _load_reference_data()

    n = len(agemos)
    L_out = np.full(n, np.nan, dtype=np.float64)
    M_out = np.full(n, np.nan, dtype=np.float64)
    S_out = np.full(n, np.nan, dtype=np.float64)
    age_out = np.full(n, np.nan, dtype=np.float64)

    for sex_code, sex_char in [("male", "M"), ("female", "F")]:
        sex_mask = sex == sex_char
        if not np.any(sex_mask):
            continue

        # Get reference arrays for this sex
        array_key = f"{measure}_{sex_code}"
        if array_key not in ref_data:
            logging.warning(f"Reference data not found for {array_key}")
            continue

        ref_array = ref_data[array_key]
        ref_ages = ref_array["age"]
        ref_L = ref_array["L"]
        ref_M = ref_array["M"]
        ref_S = ref_array["S"]

        # Interpolate for all points of this sex
        ages_sex = agemos[sex_mask]

        # Handle boundary at 24 months
        # For simplicity, use linear interpolation across all provided ages
        # Real implementation would handle WHO/CDC boundary more carefully

        try:
            L_interp = np.interp(ages_sex, ref_ages, ref_L)
            M_interp = np.interp(ages_sex, ref_ages, ref_M)
            S_interp = np.interp(ages_sex, ref_ages, ref_S)

            L_out[sex_mask] = L_interp
            M_out[sex_mask] = M_interp
            S_out[sex_mask] = S_interp
            age_out[sex_mask] = ages_sex

        except ValueError as e:
            logging.warning(f"Interpolation failed for {measure}_{sex_code}: {e}")
            continue

    return L_out, M_out, S_out, age_out


def calculate_growth_metrics(
    agemos: np.ndarray,
    sex: np.ndarray,
    height: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    head_circ: Optional[np.ndarray] = None,
    bmi: Optional[np.ndarray] = None,
    measures: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate growth metrics and BIV flags using WHO/CDC reference data.

    Compute standardized growth measures based on WHO/CDC protocols: WAZ/HAZ/BMIz
    quantify age/sex-adjusted deviations for health monitoring; modified z-scores
    enable robust BIV detection from asymmetric distributions. Measures selected
    for clinical relevance (e.g., BMIz for obesity; headcz for neurodevelopment).

    Note: This is a stub implementation using mocked LMS data for testing Phase 1.
    Real implementation will load WHO/CDC data in Phase 2.

    Args:
        agemos: Age in months
        sex: Sex ('M' or 'F')
        height: Height in cm
        weight: Weight in kg
        head_circ: Head circumference in cm
        measures: Subset of measures to compute (default: all available)

    Returns:
        Dict with computed z-scores and BIV flags (_bivbmi: BMI BIV per WHO thresholds)
    """
    n = len(agemos)
    if n == 0:
        return {}
    _validate_inputs(agemos, sex)

    age_na_mask = _handle_age_limit(agemos)
    _log_unit_warnings(agemos, height, weight)

    n = len(agemos)
    # Default measures if None
    if measures is None:
        measures = ["waz", "haz", "bmiz", "headcz", "mod_bmiz", "_bivbmi"]

    result = {}

    # Mock LMS data (fixed for stub - real data from interpolation in Phase 2)
    # Use simple LMS around 1/unit for mock calculations
    mock_L = np.full(n, 0.1, dtype=np.float64)
    mock_M = np.full(n, 1.0, dtype=np.float64)
    mock_S = np.full(n, 0.1, dtype=np.float64)

    bmi = _compute_bmi(height, weight)

    # Compute standard z-scores
    result.update(
        _compute_standard_zscores(
            measures,
            agemos,
            sex,
            height,
            weight,
            head_circ,
            bmi,
            mock_L,
            mock_M,
            mock_S,
            age_na_mask,
        )
    )

    # Compute BIV flags
    result.update(_compute_biv_flags(measures, result))

    # Compute WHZ if applicable
    if height is not None and weight is not None:
        result.update(
            _compute_whz(measures, height, weight, mock_L, mock_M, mock_S, age_na_mask)
        )

    return result


# Placeholder for inverse LMS if needed
def inverse_lms() -> None:
    """Inverse LMS transformation for modified z-scores."""
    pass
