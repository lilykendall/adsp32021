"""Task 6 — Modified test dataset creation.

Each scenario is a cumulative additive shift applied to the *original* test set,
written to its own CSV. `data/test_original.csv` is never overwritten.

Run standalone:

    python -m src.scenarios
"""

from __future__ import annotations

import json

import pandas as pd

if __package__ in (None, ""):  # allow `python src/scenarios.py` as well as `-m src.scenarios`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config as cfg


def apply_scenario(test_df: pd.DataFrame, shifts: dict[str, float]) -> pd.DataFrame:
    """Return a copy of `test_df` with each named column shifted by its delta."""
    out = test_df.copy()
    for col, delta in shifts.items():
        if col not in out.columns:
            raise KeyError(f"scenario column {col!r} not present in the test set")
        out[col] = out[col] + delta
    return out


def verify(original: pd.DataFrame, modified: pd.DataFrame,
           shifts: dict[str, float]) -> dict:
    """Confirm the intended columns moved by exactly the intended amount, and
    that nothing else changed. This is the validation the rubric asks for."""
    checks = {"shifted": {}, "unchanged_columns_ok": True, "row_count_ok": len(original) == len(modified)}

    for col, delta in shifts.items():
        diff = (modified[col] - original[col]).dropna()
        checks["shifted"][col] = {
            "expected_delta": delta,
            "observed_delta_min": float(diff.min()) if len(diff) else None,
            "observed_delta_max": float(diff.max()) if len(diff) else None,
            "ok": bool(len(diff) and diff.round(6).eq(delta).all()),
            "original_mean": float(original[col].mean()),
            "modified_mean": float(modified[col].mean()),
            # Worth noting in the write-up: the shift pushes values outside their
            # natural range (negative income, poverty > 100%), which is exactly
            # the kind of thing a production data-quality check should catch.
            "modified_min": float(modified[col].min()),
            "modified_max": float(modified[col].max()),
        }

    untouched = [c for c in original.columns if c not in shifts]
    for col in untouched:
        if not original[col].equals(modified[col]):
            checks["unchanged_columns_ok"] = False
            checks.setdefault("unexpectedly_changed", []).append(col)

    return checks


def main() -> None:
    original = pd.read_csv(cfg.DATA_DIR / "test_original.csv")
    all_checks = {}

    for name, shifts in cfg.SCENARIOS.items():
        if name == "original":
            continue
        modified = apply_scenario(original, shifts)
        path = cfg.DATA_DIR / f"test_{name}.csv"
        modified.to_csv(path, index=False)
        all_checks[name] = verify(original, modified, shifts)
        print(f"{name:<14} -> {path.name} "
              f"({', '.join(f'{c}{d:+g}' for c, d in shifts.items())})")

    (cfg.METRIC_DIR / "scenario_verification.json").write_text(
        json.dumps(all_checks, indent=2)
    )
    print(f"\nWrote {(cfg.METRIC_DIR / 'scenario_verification.json').relative_to(cfg.ROOT)}")


if __name__ == "__main__":
    main()
