"""
Performance benchmarks for z-score calculations.
Validates against plan targets: <1μs/row for 100K rows, <10s for 10M rows.
"""

import time
import numpy as np

from biv.zscores import lms_zscore, calculate_growth_metrics


def benchmark_lms_zscore():
    """Benchmark LMS z-score calculation performance."""
    # Generate test data
    n = 100000  # 100K rows target <1μs/row

    # Generate realistic LMS parameters (mocked)
    np.random.seed(42)
    X = np.random.normal(50, 10, n)
    L = np.random.normal(0.1, 0.05, n)
    M = np.full(n, 50.0)
    S = np.full(n, 0.1)

    # Warm-up JIT
    _ = lms_zscore(X[:100], L[:100], M[:100], S[:100])

    # Time the operation
    start_time = time.perf_counter()
    z_scores = lms_zscore(X, L, M, S)
    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    microseconds_per_row = (elapsed_seconds * 1e6) / n

    print(".4f")
    print(".4f")

    # Plan target: <1 μs/row for 100K rows
    assert microseconds_per_row < 1.0, ".4f"
    assert len(z_scores) == n
    assert not np.all(np.isnan(z_scores))


def benchmark_large_dataset():
    """Benchmark with 1M rows (subset of 10M target)."""
    n = 1000000  # 1M rows for feasible testing

    np.random.seed(42)
    agemos = np.random.uniform(12, 240, n)
    sex = np.random.choice(["M", "F"], n)
    height = np.random.normal(120, 15, n)
    weight = np.random.normal(25, 5, n)

    start_time = time.time()
    calculate_growth_metrics(agemos, sex, height=height, weight=weight)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    seconds_per_10M = elapsed_seconds * 10  # Scale to 10M

    print(".2f")
    print(".2f")

    # Plan target: <10s for 10M rows (extrapolated)
    assert seconds_per_10M < 10.0, ".2f"


def test_performance_targets():
    """Run performance validations."""
    benchmark_lms_zscore()
    benchmark_large_dataset()
