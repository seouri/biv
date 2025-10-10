"""
Z-Score Calculation Utilities for Growth Metrics

This module provides vectorized functions for calculating age- and sex-specific
z-scores for anthropometric measurements using WHO and CDC reference data.
Includes LMS transformations, modified z-scores, and BIV detection thresholds.
"""

from typing import Dict, List, Optional
import logging

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

        # Above median
        mod_z_flat[mask_l_nonzero & above_median] = (
            X_masked[above_median] - M_masked[above_median]
        ) / sd_pos[above_median]

        # Below median
        mod_z_flat[mask_l_nonzero & below_median] = (
            X_masked[below_median] - M_masked[below_median]
        ) / sd_neg[below_median]

        # At median
        mod_z_flat[mask_l_nonzero & at_median] = 0.0

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
    """Handle age >240 months: return NaN mask."""
    age_na_mask = agemos > 240.0
    if np.any(age_na_mask):
        logging.warning(
            "Age values >240 months detected - setting z-scores to NaN for these entries"
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
        np.nanmax(agemos) > 240 and np.nanmean(agemos) > 30
    ):  # Likely years: mean >30 for age >240
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

    return result


def _compute_biv_flags(
    measures: List[str], results: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Compute BIV flags from modified z-scores."""
    flags = {}

    # BIV flags derived from modified z-scores
    # Currently only BMI BIV flag (_bivbmi) as per ZScoreDetector spec
    if "_bivbmi" in measures and "mod_bmiz" in results:
        # BMI: mod_z < -4 or >8
        biv_bmi_flag = (results["mod_bmiz"] < -4.0) | (results["mod_bmiz"] > 8.0)
        biv_bmi_flag = np.where(np.isnan(results["mod_bmiz"]), False, biv_bmi_flag)
        flags["_bivbmi"] = biv_bmi_flag

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


def interpolate_lms() -> None:
    """Placeholder for LMS interpolation function."""
    # TODO: Implement vectorized interpolation using reference data
    pass


def calculate_growth_metrics(
    agemos: np.ndarray,
    sex: np.ndarray,
    height: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    head_circ: Optional[np.ndarray] = None,
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
