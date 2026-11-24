#!/usr/bin/env python3
"""Regenerate run-level statistics and convergence data from released CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CODE_DIR / "config" / "analysis_config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def suffix_match(run_id: str, mapping: dict[str, str]) -> str | None:
    matches = [label for label, suffix in mapping.items() if run_id.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous method suffix for run directory: {run_id}: {matches}")
    return matches[0] if matches else None


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_run(csv_path: Path, required_columns: list[str]) -> tuple[list[dict], dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = set(required_columns) - columns
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty result CSV: {csv_path}")

    collisions = [parse_bool(row["collision"]) for row in rows]
    ttc = [float(row["min_TTC"]) for row in rows]
    fit = [float(row["target_fit"]) for row in rows]
    if not all(math.isfinite(value) for value in ttc + fit):
        raise ValueError(f"Non-finite TTC or target_fit value in {csv_path}")

    target_valid = [
        collision and (fit_value - ttc_value) < 0.0
        for collision, fit_value, ttc_value in zip(collisions, fit, ttc)
    ]
    run_metrics = {
        "n_eval": len(rows),
        "collision_rate_percent": 100.0 * sum(collisions) / len(rows),
        "target_valid_collision_rate_percent": 100.0 * sum(target_valid) / len(rows),
        "ttc_lt_1_s_percent": 100.0 * sum(value < 1.0 for value in ttc) / len(rows),
        "ttc_lt_3_s_percent": 100.0 * sum(value < 3.0 for value in ttc) / len(rows),
        "best_target_fit": min(fit),
    }
    return rows, run_metrics


def summarize_runs(runs: list[dict]) -> dict:
    metric_names = [
        "n_eval",
        "collision_rate_percent",
        "target_valid_collision_rate_percent",
        "ttc_lt_1_s_percent",
        "ttc_lt_3_s_percent",
        "best_target_fit",
    ]
    summary = {"runs": len(runs)}
    for metric in metric_names:
        summary[f"mean_{metric}"] = statistics.mean(run[metric] for run in runs)
    return summary


def collect_convergence(rows: list[dict]) -> list[tuple[int, float]]:
    """Return (complete-generation evaluation endpoint, best-so-far fit)."""
    by_step: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_step[float(row["step"])].append(float(row["target_fit"]))

    endpoint = 0
    best = math.inf
    curve = []
    for step in sorted(by_step):
        values = by_step[step]
        endpoint += len(values)
        best = min(best, min(values))
        curve.append((endpoint, best))
    return curve


def curve_at(curve: list[tuple[int, float]], evaluation: int) -> float | None:
    value = None
    for endpoint, best in curve:
        if endpoint > evaluation:
            break
        value = best
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(input_dir: Path, csv_paths: list[Path], scenarios: list[str]) -> list[dict]:
    manifest = []
    for path in csv_paths:
        relative = path.relative_to(input_dir).as_posix()
        scenario = relative.split("/", 1)[0]
        run_id = path.parent.parent.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({
            "relative_path": relative,
            "scenario": scenario if scenario in scenarios else "UNMAPPED",
            "run_id": run_id,
            "bytes": path.stat().st_size,
            "sha256": digest,
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reproduced_outputs")
    parser.add_argument("--plot", action="store_true", help="Also create PDF and PNG convergence plots")
    args = parser.parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    main_methods = config["main_methods"]
    ablation_methods = config["ablation_methods"]
    scenarios = config["scenarios"]
    expected_runs = int(config["expected_runs_per_scenario_method"])
    csv_paths = sorted(input_dir.rglob("sotif_search_results.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No sotif_search_results.csv files found under {input_dir}")

    main_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    ablation_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    run_output_rows = []
    convergence_inputs: dict[tuple[str, str], list[list[tuple[int, float]]]] = defaultdict(list)
    unmapped = []

    for csv_path in csv_paths:
        relative = csv_path.relative_to(input_dir)
        scenario = relative.parts[0]
        if scenario not in scenarios:
            unmapped.append(relative.as_posix())
            continue
        run_id = csv_path.parent.parent.name
        main_method = suffix_match(run_id, main_methods)
        ablation_method = suffix_match(run_id, ablation_methods)
        if main_method is None and ablation_method is None:
            unmapped.append(relative.as_posix())
            continue
        rows, metrics = read_run(csv_path, config["required_csv_columns"])
        run_output_rows.append({
            "scenario": scenario,
            "run_id": run_id,
            "main_method": main_method or "",
            "ablation_method": ablation_method or "",
            **metrics,
        })
        if main_method:
            main_runs[(scenario, main_method)].append(metrics)
            if main_method in config["convergence_methods"]:
                convergence_inputs[(scenario, main_method)].append(collect_convergence(rows))
        if ablation_method:
            ablation_runs[(scenario, ablation_method)].append(metrics)

    if unmapped:
        raise ValueError("Unmapped result files were found: " + ", ".join(unmapped[:10]))
    expected_file_count = len(scenarios) * len(config["all_unique_methods"]) * expected_runs
    if len(csv_paths) != expected_file_count:
        raise ValueError(f"Expected {expected_file_count} result CSVs; found {len(csv_paths)}")

    for group_name, grouped, methods in (
        ("main", main_runs, list(main_methods)),
        ("ablation", ablation_runs, list(ablation_methods)),
    ):
        for scenario in scenarios:
            for method in methods:
                runs = grouped.get((scenario, method), [])
                if len(runs) != expected_runs:
                    raise ValueError(
                        f"{group_name} {scenario}/{method}: expected {expected_runs} runs, found {len(runs)}"
                    )

    run_fields = [
        "scenario", "run_id", "main_method", "ablation_method", "n_eval",
        "collision_rate_percent", "target_valid_collision_rate_percent",
        "ttc_lt_1_s_percent", "ttc_lt_3_s_percent", "best_target_fit",
    ]
    write_csv(output_dir / "run_metrics.csv", run_output_rows, run_fields)

    summary_rows = []
    for group_name, grouped in (("main", main_runs), ("ablation", ablation_runs)):
        rows = []
        for scenario, method in sorted(grouped):
            summary = summarize_runs(grouped[(scenario, method)])
            rows.append({"scenario": scenario, "method": method, **summary})
        summary_rows.extend({"group": group_name, **row} for row in rows)
        write_csv(
            output_dir / f"{group_name}_summary.csv",
            rows,
            ["scenario", "method", "runs", "mean_n_eval", "mean_collision_rate_percent",
             "mean_target_valid_collision_rate_percent", "mean_ttc_lt_1_s_percent",
             "mean_ttc_lt_3_s_percent", "mean_best_target_fit"],
        )

    convergence_rows = []
    max_eval = int(config["convergence_max_valid_evaluations"])
    start_eval = int(config["convergence_start_valid_evaluation"])
    for scenario in scenarios:
        for method in config["convergence_methods"]:
            run_curves = convergence_inputs[(scenario, method)]
            if len(run_curves) != expected_runs:
                raise ValueError(f"Convergence {scenario}/{method}: expected {expected_runs} runs, found {len(run_curves)}")
            for evaluation in range(start_eval, max_eval + 1):
                values = [curve_at(curve, evaluation) for curve in run_curves]
                values = [value for value in values if value is not None]
                if len(values) == expected_runs:
                    convergence_rows.append({
                        "scenario": scenario,
                        "method": method,
                        "valid_evaluations": evaluation,
                        "mean_best_target_fit": statistics.mean(values),
                        "runs_in_mean": len(values),
                    })
    write_csv(
        output_dir / "convergence_means.csv",
        convergence_rows,
        ["scenario", "method", "valid_evaluations", "mean_best_target_fit", "runs_in_mean"],
    )

    manifest = build_manifest(input_dir, csv_paths, scenarios)
    write_csv(output_dir / "source_csv_manifest.csv", manifest,
              ["relative_path", "scenario", "run_id", "bytes", "sha256"])
    report = {
        "input_directory": str(input_dir),
        "result_csv_count": len(csv_paths),
        "result_csv_bytes": sum(item["bytes"] for item in manifest),
        "expected_runs_per_scenario_method": expected_runs,
        "scenario_count": len(scenarios),
        "main_method_count": len(main_methods),
        "ablation_method_count_including_full_method": len(ablation_methods),
        "metrics": config["metric_notes"],
        "outputs": [
            "run_metrics.csv", "main_summary.csv", "ablation_summary.csv",
            "convergence_means.csv", "source_csv_manifest.csv", "analysis_report.json",
        ],
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.plot:
        from plot_convergence import make_plot
        make_plot(output_dir / "convergence_means.csv", output_dir)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

