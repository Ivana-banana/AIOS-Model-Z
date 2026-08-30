#!/usr/bin/env python3
"""Проверка одностороннего split-conformal запаса.

CSV columns: scenario,prediction,actual,distance
Residual/score is prediction - actual; positive means surrogate optimism.

The normalized option is deliberately diagnostic: it fits abs(residual) on
1 + distance and uses leave-one-out fitted scales for calibration scores.
It is not a distribution-free conditional-coverage theorem by itself.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def finite_sample_index(n: int, coverage: float) -> int:
    """1-based calibration order-statistic index, with n+1 = infinity."""
    if not (0 < coverage < 1):
        raise ValueError("coverage must be in (0,1)")
    return int(math.ceil((n + 1) * coverage))


def guaranteed_coverage(n: int, coverage: float) -> float:
    """Exact continuous-score marginal coverage of the split-conformal rule."""
    k = finite_sample_index(n, coverage)
    # If k=n+1, the formal threshold is +infinity.
    return 1.0 if k == n + 1 else k / (n + 1)


def order_stat_threshold(scores: np.ndarray, coverage: float) -> tuple[float, int, bool]:
    n = len(scores)
    k = finite_sample_index(n, coverage)
    if k > n:
        return float("inf"), k, True
    return float(np.sort(scores)[k - 1]), k, False


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"scenario", "prediction", "actual", "distance"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    scenarios = [r["scenario"] for r in rows]
    pred = np.array([float(r["prediction"]) for r in rows], dtype=float)
    actual = np.array([float(r["actual"]) for r in rows], dtype=float)
    distance = np.array([float(r["distance"]) for r in rows], dtype=float)
    if not np.all(np.isfinite(np.r_[pred, actual, distance])):
        raise ValueError("prediction, actual and distance must be finite")
    if np.any(distance < 0):
        raise ValueError("distance must be non-negative")
    return scenarios, pred, actual, distance


def scale_fit(x: np.ndarray, y: np.ndarray, leave_out: int | None = None) -> float:
    """Positive fitted scale from abs residual on intercept + distance."""
    mask = np.ones(len(x), dtype=bool)
    if leave_out is not None:
        mask[leave_out] = False
    xx, yy = x[mask], y[mask]
    if len(xx) < 2 or np.ptp(xx) == 0:
        value = float(np.median(yy)) if len(yy) else 1.0
    else:
        fit = stats.linregress(xx, yy)
        value = float(fit.intercept + fit.slope * x[leave_out if leave_out is not None else 0])
        if leave_out is None:
            # caller requests the first value only through this branch; replaced below
            value = float(fit.intercept)
    return max(value, np.finfo(float).eps)


def fitted_scale_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    if len(x) < 2 or np.ptp(x) == 0:
        return max(float(np.median(y)), np.finfo(float).eps)
    fit = stats.linregress(x, y)
    return max(float(fit.intercept + fit.slope * x0), np.finfo(float).eps)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check one-sided conformal margin and a diagnostic distance-normalized variant")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--coverage", type=float, default=0.90, help="target marginal coverage, default 0.90")
    ap.add_argument("--candidate-distance", type=float, default=None, help="optional distance for a new candidate")
    args = ap.parse_args()

    scenarios, pred, actual, distance = read_csv(args.csv)
    n = len(pred)
    coverage = args.coverage
    residual = pred - actual
    raw_q, k, infinity = order_stat_threshold(residual, coverage)

    abs_residual = np.abs(residual)
    norm_scores = np.array([
        residual[i] / fitted_scale_at(
            np.delete(distance, i), np.delete(abs_residual, i), distance[i]
        )
        for i in range(n)
    ])
    norm_q, norm_k, norm_inf = order_stat_threshold(norm_scores, coverage)

    print(f"n={n}; target_coverage={coverage:.6f}")
    print(f"raw: k=ceil((n+1)*coverage)={k}; guaranteed_marginal_coverage={guaranteed_coverage(n, coverage):.8f}; threshold={'inf' if infinity else f'{raw_q:.10g}'}")
    print(f"raw interpretation: q is the {k}-th order statistic of scores prediction-actual; positive q is a downward safety margin in the actual-value direction.")
    print(f"normalized diagnostic: score=(prediction-actual)/scale(distance), scale fitted from abs residual ~ 1 + distance with leave-one-out calibration; k={norm_k}; threshold={'inf' if norm_inf else f'{norm_q:.10g}'}")
    if args.candidate_distance is not None:
        scale0 = fitted_scale_at(distance, abs_residual, args.candidate_distance)
        norm_margin = float("inf") if norm_inf else norm_q * scale0
        print(f"candidate_distance={args.candidate_distance:.10g}; fitted_scale={scale0:.10g}; normalized_margin={'inf' if math.isinf(norm_margin) else f'{norm_margin:.10g}'}")

    if np.ptp(distance) > 0 and n >= 3:
        pearson = stats.pearsonr(distance, residual)
        print(f"diagnostic Pearson(distance,residual): r={pearson.statistic:.8f}; two-sided p={pearson.pvalue:.8g}")
    else:
        print("diagnostic Pearson: unavailable (need n>=3 and non-constant distance)")

    print("ASSUMPTIONS FOR STATISTICAL FORCE:")
    print("1) The calibration pairs and the new candidate pair are exchangeable/i.i.d. from the same target population (or a formally justified weighted-exchangeable alternative).")
    print("2) The surrogate/model version, residual definition, candidate-generation protocol and inclusion rule are fixed independently of these calibration residuals; the candidate is not selected by maximizing the same model in a way that changes its residual distribution.")
    print("3) The reported number is marginal coverage over a random future pair, not a guarantee conditional on this particular optimized candidate, its distance, or the worst candidate among a search.")
    print("4) The normalized number has no conditional-coverage validity from this script alone; it requires a justified scale model, enough data, and a method with a theorem for the intended local/weighted target.")
    print("5) With n near 10, the raw 90% rule is the sample maximum (for n=9, k=9); the result is a finite-sample upper order-statistic bound, not an estimated stable 90th percentile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
