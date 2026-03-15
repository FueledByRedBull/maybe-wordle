from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import tomllib

import optuna


SCALAR_KEYS = [
    "base_seed_weight",
    "base_history_only_weight",
    "cooldown_days",
    "cooldown_floor",
    "midpoint_days",
    "logistic_k",
    "exact_threshold",
    "exact_exhaustive_threshold",
    "exact_candidate_pool",
    "session_opener_pool",
    "session_reply_pool",
    "session_window_days",
    "lookahead_threshold",
    "medium_state_lookahead_threshold",
    "lookahead_candidate_pool",
    "medium_state_lookahead_candidate_pool",
    "lookahead_reply_pool",
    "medium_state_lookahead_reply_pool",
    "lookahead_root_force_in_two_scan",
    "medium_state_force_in_two_scan",
    "large_state_split_threshold",
    "pool_tight_gap_threshold",
    "pool_medium_gap_threshold",
    "pool_diversity_stride",
    "danger_lookahead_threshold",
    "danger_exact_threshold",
    "danger_reply_pool_bonus",
    "danger_exact_root_pool",
    "danger_exact_survivor_cap",
    "lookahead_trap_penalty",
    "lookahead_large_bucket_penalty",
    "lookahead_dangerous_mass_penalty",
    "lookahead_large_bucket_mass_penalty",
    "trap_size_threshold",
    "trap_mass_threshold",
    "sync_reverify_days",
]

PROXY_KEYS = [
    "entropy_w",
    "bucket_mass_w",
    "bucket_size_w",
    "ambiguous_w",
    "proxy_w",
    "solve_prob_w",
    "posterior_w",
    "smoothness_w",
    "gray_reuse_w",
    "large_bucket_count_w",
    "dangerous_mass_count_w",
    "large_bucket_mass_w",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Maybe Wordle predictive config with Optuna live scoring.")
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--config", default="config/prior.toml")
    parser.add_argument("--output", default="config/prior.optuna.toml")
    parser.add_argument(
        "--storage",
        default="sqlite:///data/derived/predictive/optuna-live-config.sqlite3",
    )
    parser.add_argument("--study-name", default="maybe-wordle-live-config")
    parser.add_argument("--failure-penalty", type=float, default=4.0)
    parser.add_argument("--coverage-gap-penalty", type=float, default=8.0)
    parser.add_argument("--hard-case-failure-penalty", type=float, default=2.0)
    parser.add_argument("--hard-case-average-weight", type=float, default=0.05)
    parser.add_argument("--latency-weight", type=float, default=0.002)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    output_path = root / args.output

    with config_path.open("rb") as handle:
        base_config = tomllib.load(handle)

    if not args.no_build:
        run(
            ["cargo", "build", "--release"],
            cwd=root,
        )

    binary = root / "target" / "release" / ("maybe-wordle.exe" if sys.platform == "win32" else "maybe-wordle")
    if not binary.exists():
        raise SystemExit(f"missing release binary at {binary}")

    storage_path = args.storage
    if storage_path.startswith("sqlite:///"):
        db_path = root / storage_path.removeprefix("sqlite:///")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path = f"sqlite:///{db_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        candidate = suggest_config(trial, base_config)
        rendered = render_toml(candidate)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".toml",
            prefix="maybe-wordle-optuna-",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(rendered)
            temp_path = Path(handle.name)

        try:
            result = run_json(
                [
                    str(binary),
                    "evaluate-live-config",
                    "--config",
                    temp_path.as_posix(),
                    "--from",
                    args.date_from,
                    "--to",
                    args.date_to,
                    "--top",
                    str(args.top),
                    "--json",
                ],
                cwd=root,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        objective_value = (
            result["average_guesses"]
            + (result["failures"] * args.failure_penalty)
            + (result["coverage_gaps"] * args.coverage_gap_penalty)
            + (result["hard_case_failures"] * args.hard_case_failure_penalty)
            + (result["hard_case_average_guesses"] * args.hard_case_average_weight)
            + (result["latency_p95_ms"] * args.latency_weight)
        )
        trial.set_user_attr("average_guesses", result["average_guesses"])
        trial.set_user_attr("failures", result["failures"])
        trial.set_user_attr("coverage_gaps", result["coverage_gaps"])
        trial.set_user_attr("latency_p95_ms", result["latency_p95_ms"])
        trial.set_user_attr("hard_case_average_guesses", result["hard_case_average_guesses"])
        trial.set_user_attr("hard_case_failures", result["hard_case_failures"])
        return objective_value

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)

    best_config = suggest_config_from_params(base_config, study.best_trial.params)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_toml(best_config), encoding="utf-8")

    summary = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_metrics": study.best_trial.user_attrs,
        "output_config": output_path.as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_json(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    return json.loads(completed.stdout)


def suggest_config(trial: optuna.Trial, base: dict) -> dict:
    candidate = copy.deepcopy(base)
    candidate["base_seed_weight"] = trial.suggest_float("base_seed_weight", 0.60, 1.25)
    candidate["base_history_only_weight"] = trial.suggest_float("base_history_only_weight", 0.10, 0.60)
    candidate["cooldown_days"] = trial.suggest_int("cooldown_days", 60, 540, step=30)
    candidate["cooldown_floor"] = trial.suggest_float("cooldown_floor", 0.0, 0.05, step=0.01)
    candidate["midpoint_days"] = trial.suggest_float("midpoint_days", 365.0, 1220.0, step=45.0)
    candidate["logistic_k"] = trial.suggest_float("logistic_k", 0.005, 0.03, log=True)
    candidate["exact_threshold"] = trial.suggest_int("exact_threshold", 48, 96, step=8)
    candidate["exact_exhaustive_threshold"] = trial.suggest_int(
        "exact_exhaustive_threshold",
        8,
        int(candidate["exact_threshold"]),
        step=2,
    )
    candidate["exact_candidate_pool"] = trial.suggest_int("exact_candidate_pool", 64, 160, step=16)
    candidate["lookahead_threshold"] = trial.suggest_int("lookahead_threshold", 96, 224, step=16)
    candidate["medium_state_lookahead_threshold"] = trial.suggest_int(
        "medium_state_lookahead_threshold",
        64,
        min(160, int(candidate["lookahead_threshold"])),
        step=16,
    )
    candidate["lookahead_candidate_pool"] = trial.suggest_int("lookahead_candidate_pool", 16, 48, step=8)
    candidate["medium_state_lookahead_candidate_pool"] = trial.suggest_int(
        "medium_state_lookahead_candidate_pool",
        int(candidate["lookahead_candidate_pool"]),
        80,
        step=8,
    )
    candidate["lookahead_reply_pool"] = trial.suggest_int("lookahead_reply_pool", 8, 24, step=4)
    candidate["medium_state_lookahead_reply_pool"] = trial.suggest_int(
        "medium_state_lookahead_reply_pool",
        int(candidate["lookahead_reply_pool"]),
        32,
        step=4,
    )
    candidate["lookahead_root_force_in_two_scan"] = trial.suggest_int(
        "lookahead_root_force_in_two_scan",
        48,
        128,
        step=16,
    )
    candidate["medium_state_force_in_two_scan"] = trial.suggest_int(
        "medium_state_force_in_two_scan",
        int(candidate["lookahead_root_force_in_two_scan"]),
        288,
        step=32,
    )
    candidate["large_state_split_threshold"] = trial.suggest_int("large_state_split_threshold", 40, 64, step=4)
    candidate["pool_tight_gap_threshold"] = trial.suggest_float("pool_tight_gap_threshold", 0.03, 0.07, step=0.01)
    candidate["pool_medium_gap_threshold"] = trial.suggest_float(
        "pool_medium_gap_threshold",
        candidate["pool_tight_gap_threshold"],
        0.20,
        step=0.01,
    )
    candidate["danger_lookahead_threshold"] = trial.suggest_float("danger_lookahead_threshold", 0.52, 0.70, step=0.02)
    candidate["danger_exact_threshold"] = trial.suggest_float(
        "danger_exact_threshold",
        candidate["danger_lookahead_threshold"],
        0.80,
        step=0.02,
    )
    candidate["danger_reply_pool_bonus"] = trial.suggest_int("danger_reply_pool_bonus", 4, 16, step=4)
    candidate["danger_exact_root_pool"] = trial.suggest_int("danger_exact_root_pool", 16, 40, step=8)
    candidate["danger_exact_survivor_cap"] = trial.suggest_int(
        "danger_exact_survivor_cap",
        int(candidate["exact_threshold"]) + 1,
        224,
    )
    candidate["lookahead_trap_penalty"] = trial.suggest_float("lookahead_trap_penalty", 0.20, 0.60)
    candidate["lookahead_large_bucket_penalty"] = trial.suggest_float("lookahead_large_bucket_penalty", 0.05, 0.20)
    candidate["lookahead_dangerous_mass_penalty"] = trial.suggest_float("lookahead_dangerous_mass_penalty", 0.04, 0.18)
    candidate["lookahead_large_bucket_mass_penalty"] = trial.suggest_float("lookahead_large_bucket_mass_penalty", 0.05, 0.20)
    candidate["trap_size_threshold"] = trial.suggest_int("trap_size_threshold", 5, 8)
    candidate["trap_mass_threshold"] = trial.suggest_float("trap_mass_threshold", 0.10, 0.25, step=0.01)

    proxy = candidate["proxy_weights"]
    proxy["entropy_w"] = trial.suggest_float("proxy_entropy_w", 0.05, 4.0, log=True)
    proxy["bucket_mass_w"] = trial.suggest_float("proxy_bucket_mass_w", 0.05, 4.0, log=True)
    proxy["bucket_size_w"] = trial.suggest_float("proxy_bucket_size_w", 0.02, 1.0, log=True)
    proxy["ambiguous_w"] = trial.suggest_float("proxy_ambiguous_w", 0.05, 2.0, log=True)
    proxy["proxy_w"] = trial.suggest_float("proxy_proxy_w", 0.10, 3.0, log=True)
    proxy["solve_prob_w"] = trial.suggest_float("proxy_solve_prob_w", 0.01, 1.0, log=True)
    proxy["posterior_w"] = trial.suggest_float("proxy_posterior_w", 0.005, 0.5, log=True)
    proxy["smoothness_w"] = trial.suggest_float("proxy_smoothness_w", 0.05, 2.0, log=True)
    proxy["gray_reuse_w"] = trial.suggest_float("proxy_gray_reuse_w", 0.01, 0.5, log=True)
    proxy["large_bucket_count_w"] = trial.suggest_float("proxy_large_bucket_count_w", 0.02, 1.0, log=True)
    proxy["dangerous_mass_count_w"] = trial.suggest_float("proxy_dangerous_mass_count_w", 0.02, 1.0, log=True)
    proxy["large_bucket_mass_w"] = trial.suggest_float("proxy_large_bucket_mass_w", 0.05, 2.0, log=True)

    normalize_config(candidate)
    return candidate


def suggest_config_from_params(base: dict, params: dict) -> dict:
    fixed = optuna.trial.FixedTrial(params)
    return suggest_config(fixed, base)


def normalize_config(config: dict) -> None:
    config["exact_exhaustive_threshold"] = max(2, min(config["exact_exhaustive_threshold"], config["exact_threshold"]))
    config["medium_state_lookahead_threshold"] = max(
        config["medium_state_lookahead_threshold"],
        config["exact_threshold"] + 1,
    )
    config["lookahead_threshold"] = max(
        config["lookahead_threshold"],
        config["medium_state_lookahead_threshold"],
    )
    config["medium_state_lookahead_candidate_pool"] = max(
        config["medium_state_lookahead_candidate_pool"],
        config["lookahead_candidate_pool"],
    )
    config["medium_state_lookahead_reply_pool"] = max(
        config["medium_state_lookahead_reply_pool"],
        config["lookahead_reply_pool"],
    )
    config["medium_state_force_in_two_scan"] = max(
        config["medium_state_force_in_two_scan"],
        config["lookahead_root_force_in_two_scan"],
    )
    config["pool_medium_gap_threshold"] = max(
        config["pool_medium_gap_threshold"],
        config["pool_tight_gap_threshold"],
    )
    config["danger_exact_threshold"] = max(
        config["danger_exact_threshold"],
        config["danger_lookahead_threshold"],
    )
    config["danger_exact_survivor_cap"] = max(
        config["danger_exact_survivor_cap"],
        config["exact_threshold"] + 1,
    )


def render_toml(config: dict) -> str:
    lines: list[str] = []
    for key in SCALAR_KEYS:
        lines.append(f"{key} = {toml_value(config[key])}")
    lines.append("")
    lines.append("[proxy_weights]")
    for key in PROXY_KEYS:
        lines.append(f"{key} = {toml_value(config['proxy_weights'][key])}")
    lines.append("")
    lines.append("[manual_weights]")
    for key, value in sorted(config.get("manual_weights", {}).items()):
        lines.append(f"\"{key}\" = {toml_value(value)}")
    lines.append("")
    return "\n".join(lines)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"unsupported TOML value: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
