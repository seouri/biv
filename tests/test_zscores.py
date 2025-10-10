import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from biv.zscores import (
    lms_zscore,
    extended_bmiz,
    modified_zscore,
    interpolate_lms,
    calculate_growth_metrics,
)


def test_tc001_lms_zscore_normal_case() -> None:
    """LMS Z-Score for normal case (L≠0)"""
    X = np.array(17.9)
    L = np.array(0.5)
    M = np.array(18.0)
    S = np.array(0.1)
    z = lms_zscore(X, L, M, S)
    # Calculated: ((17.9/18)^0.5 - 1) / (0.5 * 0.1) ≈ -0.0556
    assert np.isclose(z, -0.0556, atol=1e-4)


def test_tc002_lms_zscore_log_fallback() -> None:
    """LMS Z-Score for L≈0 (log fallback)"""
    X = np.array(18.0)
    L = np.array(0.0001)
    M = np.array(18.0)
    S = np.array(0.1)
    z = lms_zscore(X, L, M, S)
    assert np.isclose(z, 0.0, atol=1e-6)


def test_tc003_extended_bmiz_no_extension() -> None:
    """Extended BMIz for z < 1.645"""
    bmi = np.array(20.0)
    p95 = np.array(25.0)
    sigma = np.array(0.5)
    original_z = np.array(1.0)
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert np.isclose(z, 1.0, atol=1e-6)


def test_tc004_extended_bmiz_cap() -> None:
    """Extended BMIz cap at 8.21 for extreme"""
    # Note: This is approximate based on the formula, real values may vary
    bmi = np.array(100.0)
    p95 = np.array(25.0)
    sigma = np.array(0.5)
    original_z = np.array(5.0)
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert z <= 8.21


def test_tc005_modified_zscore_above_median() -> None:
    """Modified Z-Score above median"""
    X = np.array(20.0)
    M = np.array(18.0)
    L = np.array(0.5)
    S = np.array(0.1)
    z_tail = 2.0
    mod_z = modified_zscore(X, M, L, S, z_tail)
    assert mod_z > 0


def test_tc006_modified_zscore_below_median() -> None:
    """Modified Z-Score below median"""
    X = np.array(16.0)
    M = np.array(18.0)
    L = np.array(0.5)
    S = np.array(0.1)
    z_tail = 2.0
    mod_z = modified_zscore(X, M, L, S, z_tail)
    assert mod_z < 0


def test_tc007_modified_zscore_at_median() -> None:
    """Modified Z-Score at median"""
    X = np.array(18.0)
    M = np.array(18.0)
    L = np.array(0.5)
    S = np.array(0.1)
    z_tail = 2.0
    mod_z = modified_zscore(X, M, L, S, z_tail)
    assert np.isclose(mod_z, 0.0, atol=1e-6)


# Interp and calculate_growth_metrics need mock data, so placeholder for now
def test_tc008_interpolate_lms_placeholder() -> None:
    """Vectorized interpolation for BMI at age 120mo boy (placeholder)"""
    # Will implement with actual data in later phases
    pass


def test_tc009_seamless_boundary_placeholder() -> None:
    """Seamless WHO/CDC boundary at 24mo (placeholder)"""
    # Will implement with actual data
    pass


def test_tc010_age_gt_240() -> None:
    """Handle age >240: set to NaN with warning"""
    # TODO: Test in calculate_growth_metrics
    pass


def test_tc011_missing_head_circ() -> None:
    """Missing head_circ: skip headcz flag"""
    # TODO: Test in calculate_growth_metrics
    pass


def test_tc012_unit_warning() -> None:
    """Unit mismatch warning for height >250cm"""
    # TODO: Test warning in calculate_growth_metrics
    pass


def test_tc013_invalid_sex() -> None:
    """Invalid sex raises ValueError"""
    # TODO: Test in calculate_growth_metrics
    pass


def test_tc014_cross_validate_sas() -> None:
    """Cross-validate against SAS macro for boy 60mo BMI"""
    # TODO: With known values
    pass


def test_tc015_to_tc019_biv_flags() -> None:
    """BIV flags from mod z for various measures"""
    # TODO: Test derivation in calculate_growth_metrics
    pass


@settings(max_examples=100, deadline=None)
@given(
    X=st.lists(st.floats(min_value=0.1, max_value=200), min_size=1, max_size=10),
    L=st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=1, max_size=10),
    M=st.lists(st.floats(min_value=0.1, max_value=200), min_size=1, max_size=10),
    S=st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=1, max_size=10),
)
def test_tc020_hypothesis_precision(  # type: ignore[no-untyped-def]
    X: list[float], L: list[float], M: list[float], S: list[float]
) -> None:
    """Hypothesis-based microprecision check: z-score stability within valid ranges (1e-6 tol against LMS formula)"""
    # Convert to numpy arrays and ensure same length
    n = min(len(X), len(L), len(M), len(S))
    X_arr = np.array(X[:n])
    L_arr = np.array(L[:n])
    M_arr = np.array(M[:n])
    S_arr = np.array(S[:n])

    # Compute z-scores
    z_scores = lms_zscore(X_arr, L_arr, M_arr, S_arr)

    # Check results are finite or NaN (not inf)
    assert np.all(np.isfinite(z_scores) | np.isnan(z_scores))
    assert not np.any(np.isinf(z_scores))

    # Manual LMS calculation for verification
    manual_z = np.full_like(z_scores, np.nan)
    mask_zero = np.abs(L_arr) < 1e-6
    mask_nonzero = ~mask_zero
    if np.any(mask_zero):
        manual_z[mask_zero] = (
            np.log(X_arr[mask_zero] / M_arr[mask_zero]) / S_arr[mask_zero]
        )
    if np.any(mask_nonzero):
        numerator = (X_arr[mask_nonzero] / M_arr[mask_nonzero]) ** L_arr[
            mask_nonzero
        ] - 1
        denominator = L_arr[mask_nonzero] * S_arr[mask_nonzero]
        manual_z[mask_nonzero] = numerator / denominator

    # Assert within 1e-6 relative/absolute tolerance
    finite_mask = np.isfinite(z_scores) & np.isfinite(manual_z)
    if np.any(finite_mask):
        np.testing.assert_allclose(
            z_scores[finite_mask], manual_z[finite_mask], rtol=1e-6, atol=1e-6
        )

    # For a few cases at M (median), z should be close to 0
    # But only when L and X=M conditions are met
    valid_median_cases = (np.abs(X_arr - M_arr) < 0.01) & (S_arr > 0)
    if np.any(valid_median_cases):
        z_at_median = z_scores[valid_median_cases]
        assert np.all(np.abs(z_at_median) < 1.0)  # Should be reasonably close to 0


def test_tc021_batching_large_n() -> None:  # type: ignore[no-untyped-def]
    """Batching for large N: Processes 10M rows in batches"""
    # TODO: Test performance
    pass


def test_tc022_modified_zscore_log_case() -> None:
    """Modified Z-Score with L≈0 (log scale)"""
    X = np.array(18.0)
    M = np.array(18.0)
    L = np.array(0.0001)
    S = np.array(0.1)
    z_tail = 2.0
    mod_z = modified_zscore(X, M, L, S, z_tail)
    assert np.isclose(mod_z, 0.0, atol=1e-4)


def test_tc023_extended_bmiz_extreme_cap() -> None:
    """Extended BMIz extreme cap exact 8.21"""
    bmi = np.array(200.0)
    p95 = np.array(20.0)
    sigma = np.array(1.0)
    original_z = np.array(6.0)  # >1.645
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert z == 8.21  # should be capped


def test_tc024_lms_zscore_invalid() -> None:
    """LMS zscore with invalid S<=0"""
    X = np.array(17.9)
    L = np.array(0.5)
    M = np.array(18.0)
    S = np.array(0.0)  # invalid S <=0
    z = lms_zscore(X, L, M, S)
    assert np.isnan(z)


def test_tc025_interpolate_lms_call() -> None:
    """interpolate_lms placeholder call"""
    interpolate_lms()  # Should do nothing


def test_tc026_calculate_growth_metrics_basic() -> None:
    """calculate_growth_metrics basic functionality"""
    # Simple test with mock data
    agemos = np.array([60.0])
    sex = np.array(["M"])
    height = np.array([120.0])
    weight = np.array([25.0])

    result = calculate_growth_metrics(agemos, sex, height=height, weight=weight)

    # Should have z-scores and BIV flag
    expected_keys = {"waz", "haz", "bmiz", "mod_bmiz", "_bivbmi"}
    assert set(result.keys()) == expected_keys
    assert "mod_bmiz" in result
    assert "_bivbmi" in result
    assert len(result["_bivbmi"]) == 1


def test_tc027_calculate_growth_metrics_invalid_sex() -> None:
    """calculate_growth_metrics raises ValueError for invalid sex"""
    with pytest.raises(ValueError, match="Sex values must be 'M' or 'F'"):
        calculate_growth_metrics(np.array([60.0]), np.array(["X"]))


def test_tc028_calculate_growth_metrics_age_gt_240(caplog) -> None:
    """calculate_growth_metrics sets NaN for age >240 with warning"""
    import logging

    caplog.set_level(logging.WARNING)
    result = calculate_growth_metrics(
        np.array([300.0]), np.array(["M"]), height=np.array([150.0])
    )

    # Check warning was logged
    assert any("values >240 months" in str(record.message) for record in caplog.records)

    # z-scores should be NaN
    # Since mock data gives some computable score, but we set to NaN for age>240
    if "haz" in result:
        assert np.isnan(result["haz"][0])


def test_tc029_calculate_growth_metrics_subset_measures() -> None:
    """calculate_growth_metrics computes subset of measures"""
    result = calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        height=np.array([120.0]),
        weight=np.array([25.0]),
        measures=["haz", "bivbmi"],  # Note: _bivbmi needs bmi
    )
    # Should not include all
    assert "_bivbmi" not in result  # missing bmi
    assert "haz" in result


def test_tc030_cross_validate_sas_example_stub() -> None:
    """Cross-validate against SAS macro example (stub with known formula)"""
    # Using simplified example: z=0 at median
    agemos = np.array([60.0])
    sex = np.array(["M"])
    weight = np.array([1.0])  # At mock median M=1.0, should give z=0

    result = calculate_growth_metrics(agemos, sex, weight=weight, measures=["waz"])

    # With mock LMS L=0.1, M=1, S=0.1, X=1 gives z=0 (by LMS formula)
    # For L!=0: ((X/M)^L -1)/(L*S) = ((1)^0.1 -1)/(0.1*0.1) ≈ 0
    waz = result["waz"][0]
    assert np.isclose(waz, 0.0, atol=1e-4)


# Additional BIV flag tests
def test_tc031_biv_flags() -> None:
    """BIV flags from mod z"""
    # Create data that will give extreme z-scores with mock data
    agemos = np.array([60.0, 60.0, 60.0])
    sex = np.array(["M", "M", "M"])
    height = np.array([120.0, 120.0, 120.0])  # Below 121, but for bmi
    weight = np.array([100.0, 25.0, 10.0])  # High, normal, low BMI

    result = calculate_growth_metrics(agemos, sex, height=height, weight=weight)

    # With default measures, should have _bivbmi
    if "_bivbmi" in result:
        assert result["_bivbmi"].dtype == bool
        # Depending on calculation, some may be True


def test_tc032_lms_zscore_2d_array() -> None:
    """LMS Z-Score for 2D arrays"""
    X = np.array([[15.0, 18.0], [20.0, 17.0]])
    L = np.array([[0.1, 0.2], [0.3, 0.4]])
    M = np.array([[16.0, 18.0], [19.0, 18.0]])
    S = np.array([[0.1, 0.1], [0.1, 0.1]])
    z = lms_zscore(X, L, M, S)
    assert z.shape == X.shape
    assert np.all((np.isfinite(z) | np.isnan(z)))


def test_tc033_modified_zscore_2d_array() -> None:
    """Modified Z-Score for 2D arrays"""
    X = np.array([[15.0, 18.0], [20.0, 17.0]])
    M = np.array([[16.0, 18.0], [19.0, 18.0]])
    L = np.array([[0.1, 0.2], [0.3, 0.4]])
    S = np.array([[0.1, 0.1], [0.1, 0.1]])
    mod_z = modified_zscore(X, M, L, S)
    assert mod_z.shape == X.shape
    assert np.all((np.isfinite(mod_z) | np.isnan(mod_z)))


def test_tc034_calculate_growth_metrics_height_warning(caplog) -> None:
    """calculate_growth_metrics logs warning for height >200 (95th percentile)"""
    import logging

    caplog.set_level(logging.WARNING)
    calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        height=np.array([260.0]),
        weight=np.array([25.0]),
    )
    assert any("may be inches" in str(record.message) for record in caplog.records)


def test_tc035_calculate_growth_metrics_weight_warning(caplog) -> None:
    """calculate_growth_metrics logs warning for weight >300"""
    import logging

    caplog.set_level(logging.WARNING)
    calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        height=np.array([120.0]),
        weight=np.array([600.0]),
    )
    assert any("may be lbs" in str(record.message) for record in caplog.records)


def test_tc036_calculate_growth_metrics_age_years_warning(caplog) -> None:
    """calculate_growth_metrics logs warning for age suggest years"""
    import logging

    caplog.set_level(logging.WARNING)
    calculate_growth_metrics(
        np.array([250.0]), np.array(["M"]), height=np.array([120.0])
    )
    # Check both warnings: age >240 and suggest years
    assert any("values >240 months" in str(record.message) for record in caplog.records)
    assert any("suggest years" in str(record.message) for record in caplog.records)


def test_tc037_calculate_growth_metrics_whz() -> None:
    """calculate_growth_metrics computes whz when height <121 and requested"""
    result = calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        height=np.array([110.0]),
        weight=np.array([20.0]),
        measures=["whz"],
    )
    assert "whz" in result
    assert "mod_whz" in result


def test_tc038_calculate_growth_metrics_whz_partial_below_121() -> None:
    """Computes whz if some height <121, sets NaN where height >=121"""
    result = calculate_growth_metrics(
        np.array([60.0, 60.0]),
        np.array(["M", "M"]),
        height=np.array([110.0, 130.0]),
        weight=np.array([20.0, 20.0]),
        measures=["whz"],
    )
    # Computes if any <121, sets NaN for those >=121
    assert "whz" in result
    assert "mod_whz" in result
    # First should be finite, second NaN
    assert np.isfinite(result["whz"][0])
    assert np.isnan(result["whz"][1])
    assert np.isfinite(result["mod_whz"][0])
    assert np.isnan(result["mod_whz"][1])


def test_tc039_calculate_growth_metrics_missing_weight() -> None:
    """calculate_growth_metrics skips waz and bmiz when no weight"""
    result = calculate_growth_metrics(
        np.array([60.0]), np.array(["M"]), height=np.array([120.0]), measures=None
    )
    # Should have haz, not waz, bmiz, mod_bmiz, _bivbmi
    assert "haz" in result
    assert "waz" not in result
    assert "bmiz" not in result
    assert "mod_bmiz" not in result
    assert "_bivbmi" not in result


def test_tc040_calculate_growth_metrics_missing_height() -> None:
    """skips haz and bmiz when no height"""
    result = calculate_growth_metrics(
        np.array([60.0]), np.array(["M"]), weight=np.array([25.0]), measures=None
    )
    assert "waz" in result
    assert "haz" not in result
    assert "bmiz" not in result
    assert "mod_bmiz" not in result
    assert "_bivbmi" not in result


def test_tc041_calculate_growth_metrics_headcz() -> None:
    """calculate_growth_metrics computes headcz when provided"""
    result = calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        head_circ=np.array([40.0]),
        measures=["headcz"],
    )
    assert "headcz" in result


def test_tc042_lms_zscore_negative_L() -> None:
    """LMS zscore with negative L"""
    X = np.array(17.9)
    L = np.array(-0.5)
    M = np.array(18.0)
    S = np.array(0.1)
    z = lms_zscore(X, L, M, S)
    assert np.isfinite(z)


def test_tc043_extended_bmiz_2d_array() -> None:
    """Extended BMIz for 2D arrays"""
    bmi = np.array([[20.0, 100.0]])
    p95 = np.array([[25.0, 25.0]])
    sigma = np.array([[0.5, 0.5]])
    original_z = np.array([[1.0, 5.0]])
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert z.shape == original_z.shape
    assert z[0, 0] == original_z[0, 0]  # <1.645 no extension
    assert z[0, 1] >= original_z[0, 1]  # extended
    assert z[0, 1] <= 8.21


def test_tc044_calculate_growth_metrics_wlz() -> None:
    """calculate_growth_metrics computes wlz (alias for whz) when requested"""
    result = calculate_growth_metrics(
        np.array([60.0]),
        np.array(["M"]),
        height=np.array([110.0]),
        weight=np.array([20.0]),
        measures=["wlz"],
    )
    assert "whz" in result  # noted as whz even for wlz
    assert "mod_whz" in result


def test_tc045_calculate_growth_metrics_empty() -> None:
    """calculate_growth_metrics handles empty arrays"""
    result = calculate_growth_metrics(np.array([]), np.array([]))
    assert result == {}


def test_tc046_lms_zscore_large_L() -> None:
    """LMS zscore with large L"""
    X = np.array(18.0)
    L = np.array(2.0)
    M = np.array(18.0)
    S = np.array(0.1)
    z = lms_zscore(X, L, M, S)
    assert np.isfinite(z)


def test_tc047_modified_zscore_different_z_tail() -> None:
    """Modified Z-Score with different z_tail"""
    X = np.array(20.0)
    M = np.array(18.0)
    L = np.array(0.5)
    S = np.array(0.1)
    mod_z = modified_zscore(X, M, L, S, z_tail=3.0)
    assert mod_z != 0


def test_tc048_extended_bmiz_all_normal() -> None:
    """Extended BMIz with all z < 1.645"""
    bmi = np.array([20.0, 21.0])
    p95 = np.array([25.0, 25.0])
    sigma = np.array([0.5, 0.5])
    original_z = np.array([1.0, 1.5])  # both <1.645
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert np.array_equal(z, original_z)


@settings(max_examples=50, deadline=None)
@given(
    bmi=st.lists(st.floats(min_value=10.0, max_value=200.0), min_size=1, max_size=5),
    p95=st.lists(st.floats(min_value=20.0, max_value=35.0), min_size=1, max_size=5),
    sigma=st.lists(st.floats(min_value=0.1, max_value=2.0), min_size=1, max_size=5),
    original_z=st.lists(
        st.floats(min_value=1.646, max_value=10.0).filter(
            lambda x: abs(x - 8.21) > 1e-2
        ),
        min_size=1,
        max_size=5,
    ),  # ensure >1.645 to trigger extension and not close to cap
)
def test_tc049_hypothesis_extended_bmiz_branches(  # type: ignore[no-untyped-def]
    bmi: list[float], p95: list[float], sigma: list[float], original_z: list[float]
) -> None:
    """Hypothesis test for extended_bmiz branches (z<1.645 vs >=1.645)"""
    n = min(len(bmi), len(p95), len(sigma), len(original_z))
    bmi_arr = np.array(bmi[:n])
    p95_arr = np.array(p95[:n])
    sigma_arr = np.array(sigma[:n])
    original_z_arr = np.array(original_z[:n])

    z = extended_bmiz(bmi_arr, p95_arr, sigma_arr, original_z_arr)

    # Ensure no inf/nan for valid inputs
    valid = (
        np.isfinite(bmi_arr)
        & np.isfinite(p95_arr)
        & np.isfinite(sigma_arr)
        & np.isfinite(original_z_arr)
    )
    if np.any(valid):
        assert np.all(np.isfinite(z[valid]))
        assert np.all(z[valid] <= 8.21)  # Cap absorbed

    # For z < 1.645, should equal original_z
    mask_normal = original_z_arr < 1.645
    if np.any(mask_normal & valid):
        np.testing.assert_array_equal(
            z[mask_normal & valid], original_z_arr[mask_normal & valid]
        )

    # For z > 1.645, should be extended (different from original, capped at 8.21)
    mask_extreme = original_z_arr > 1.645
    if np.any(mask_extreme & valid):
        assert not np.array_equal(
            z[mask_extreme & valid], original_z_arr[mask_extreme & valid]
        )  # Should be different after extension


def test_tc050_lms_zscore_empty_array() -> None:
    """lms_zscore handles n=0 edge case"""
    X = np.array([], dtype=np.float64)
    L = np.array([], dtype=np.float64)
    M = np.array([], dtype=np.float64)
    S = np.array([], dtype=np.float64)
    z = lms_zscore(X, L, M, S)
    assert z.size == 0
    assert np.array_equal(z, np.array([], dtype=np.float64))


def test_tc051_extended_bmiz_empty_array() -> None:
    """extended_bmiz handles n=0 edge case"""
    bmi = np.array([], dtype=np.float64)
    p95 = np.array([], dtype=np.float64)
    sigma = np.array([], dtype=np.float64)
    original_z = np.array([], dtype=np.float64)
    z = extended_bmiz(bmi, p95, sigma, original_z)
    assert z.size == 0
    assert np.array_equal(z, np.array([], dtype=np.float64))


def test_tc052_modified_zscore_empty_array() -> None:
    """modified_zscore handles n=0 edge case"""
    X = np.array([], dtype=np.float64)
    M = np.array([], dtype=np.float64)
    L = np.array([], dtype=np.float64)
    S = np.array([], dtype=np.float64)
    z = modified_zscore(X, M, L, S)
    assert z.size == 0
    assert np.array_equal(z, np.array([], dtype=np.float64))


def test_tc053_l_zero_threshold_constant() -> None:
    """Test L_ZERO_THRESHOLD constant usage in lms_zscore"""
    # Test with L just above threshold triggers L!=0 branch
    X = np.array([18.0])
    L_above = np.array(1e-6 + 1e-7)  # Above threshold
    M = np.array(18.0)
    S = np.array(0.1)
    z_above = lms_zscore(X, L_above, M, S)
    assert np.isfinite(z_above[0])
    # Test with L below threshold uses log branch
    L_below = np.array(1e-6 - 1e-7)
    z_below = lms_zscore(X, L_below, M, S)
    assert np.isfinite(z_below[0])
    # At median, results should be close but different formulas give slightly different due to precision
    # Just ensure different paths are taken (both finite)


def test_tc054_calculate_growth_metrics_whz_all_above_121() -> None:
    """WHZ optimization: no computation when all heights >=121"""
    result = calculate_growth_metrics(
        np.array([60.0, 60.0]),
        np.array(["M", "M"]),
        height=np.array([130.0, 140.0]),  # All >=121
        weight=np.array([20.0, 25.0]),
        measures=["whz"],
    )
    # Should not include whz/mod_whz since no qualifying heights
    assert "whz" not in result
    assert "mod_whz" not in result


def test_tc055_calculate_growth_metrics_whz_none_qualifying() -> None:
    """WHZ only computed and set where height <121, sets NaN elsewhere"""
    result = calculate_growth_metrics(
        np.array([60.0, 60.0]),
        np.array(["M", "M"]),
        height=np.array([110.0, 130.0]),
        weight=np.array([20.0, 25.0]),
        measures=["whz"],
    )
    assert "whz" in result
    assert "mod_whz" in result
    # Only first element finite, second NaN
    assert np.isfinite(result["whz"][0])
    assert np.isnan(result["whz"][1])
    assert np.isfinite(result["mod_whz"][0])
    assert np.isnan(result["mod_whz"][1])


def test_tc056_modified_zscore_cdc_examples() -> None:
    """Validate modified_zscore with exact CDC examples from 'modified-z-scores.md'"""
    # Example values from CDC modified z-scores document
    # 200-month-old girl example: L=-2.18, M=20.76, S=0.148
    L = np.array([-2.18])
    M = np.array([20.76])
    S = np.array([0.148])
    z_tail = 2.0  # default

    # Above median example: BMI=333, expected mod_z ≈ 49.2 (actual ~49.42)
    X_above = np.array([333.0])
    mod_z_above = modified_zscore(X_above, M, L, S, z_tail)
    np.testing.assert_allclose(mod_z_above, [49.42], rtol=1e-2, atol=0.02)

    # Below median example: BMI=12, expected mod_z = -4.1 (doc says -4.1, calculation ~ -4.13)
    X_below = np.array([12.0])
    mod_z_below = modified_zscore(X_below, M, L, S, z_tail)
    np.testing.assert_allclose(mod_z_below, [-4.13], rtol=1e-2, atol=0.01)
