#!/usr/bin/env python3
"""Build comparable uniform and cyclic monthly injection schedules.

Input CSV columns: well,date,rate
Output CSV columns: well,date,rate

The script keeps constant, for every well and calendar year separately:
1) the set of well-month records and their dates;
2) the cumulative injection volume (sum of monthly rates; if rates are
   average daily rates, both schedules have the same annual integral up to
   the common month-length convention used by the input);
3) the well allocation and the year-by-year total injection.

The script does NOT keep constant: the timing of injection, instantaneous
rates, pressure response, oil production, water cut, NPV, or energy use.
It is only a schedule generator, not a reservoir-flow simulator.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Record:
    well: str
    date_text: str
    year: int
    month_index: int
    rate: float


def parse_month(text: str) -> tuple[int, int]:
    """Parse YYYY-MM or an ISO date beginning with YYYY-MM."""
    value = text.strip()
    if len(value) < 7:
        raise ValueError(f"date must begin with YYYY-MM, got {text!r}")
    year = int(value[0:4])
    month = int(value[5:7])
    date(year, month, 1)  # validate
    return year, month


def read_input(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"well", "date", "rate"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain columns well,date,rate")
        for row in reader:
            year, month = parse_month(row["date"])
            rate = float(row["rate"])
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(f"rate must be finite and non-negative: {row}")
            records.append(Record(row["well"].strip(), row["date"].strip(),
                                  year, month, rate))
    if not records:
        raise ValueError("input CSV contains no data rows")
    return records


def cycle_factors(n: int, half_period: int, amplitude: float, phase: int) -> np.ndarray:
    """Return alternating factors 1-amplitude and 1+amplitude."""
    if half_period < 1:
        raise ValueError("half_period must be >= 1 month")
    if not 0 <= amplitude < 1:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1")
    positions = np.arange(n, dtype=int) + int(phase)
    signs = np.where((positions // half_period) % 2 == 0, -1.0, 1.0)
    return 1.0 + amplitude * signs


def build_schedules(records: list[Record], half_period: int,
                    amplitude: float, phase: int) -> tuple[list[Record], list[Record]]:
    groups: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.well, record.year)].append(record)

    uniform: list[Record] = []
    cyclic: list[Record] = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item.date_text)
        base = np.array([item.rate for item in group], dtype=float)
        total = float(base.sum())
        # Uniform means equal rate over the available months in each well-year.
        uniform_rates = np.full(len(group), total / len(group), dtype=float)
        raw_factors = cycle_factors(len(group), half_period, amplitude, phase)
        # Normalize factors so cyclic timing has exactly the same group total.
        weighted = base * raw_factors
        if total == 0:
            cyclic_rates = np.zeros(len(group), dtype=float)
        else:
            cyclic_rates = weighted * (total / float(weighted.sum()))
        for item, u_rate, c_rate in zip(group, uniform_rates, cyclic_rates):
            uniform.append(Record(item.well, item.date_text, item.year,
                                  item.month_index, float(u_rate)))
            cyclic.append(Record(item.well, item.date_text, item.year,
                                 item.month_index, float(c_rate)))
        # Deterministic numerical guard: the normalization error must be tiny.
        if not np.isclose(uniform_rates.sum(), cyclic_rates.sum(), rtol=0, atol=1e-10 * max(1.0, total)):
            raise ArithmeticError(f"annual volume mismatch for {key}")
    return uniform, cyclic


def write_output(path: Path, records: list[Record]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["well", "date", "rate"])
        for item in sorted(records, key=lambda x: (x.well, x.date_text)):
            writer.writerow([item.well, item.date_text, f"{item.rate:.10g}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("uniform_csv", type=Path)
    parser.add_argument("cyclic_csv", type=Path)
    parser.add_argument("--half-period", type=int, default=2,
                        help="cycle half-period in input records/months (default: 2)")
    parser.add_argument("--amplitude", type=float, default=0.50,
                        help="dimensionless modulation amplitude, 0 <= A < 1")
    parser.add_argument("--phase", type=int, default=0,
                        help="integer phase shift in months/records")
    args = parser.parse_args()

    records = read_input(args.input_csv)
    uniform, cyclic = build_schedules(records, args.half_period,
                                      args.amplitude, args.phase)
    write_output(args.uniform_csv, uniform)
    write_output(args.cyclic_csv, cyclic)


if __name__ == "__main__":
    main()
