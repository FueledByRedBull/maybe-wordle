#!/usr/bin/env python3
"""Archive completed legacy Optuna trials without making them promotable studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_parameter(value: float, distribution_json: str) -> Any:
    distribution = json.loads(distribution_json)
    name = distribution.get("name")
    attributes = distribution.get("attributes", {})
    if name == "CategoricalDistribution":
        choices = attributes.get("choices", [])
        index = int(value)
        if index < 0 or index >= len(choices):
            raise ValueError(f"categorical index {index} is outside {len(choices)} choices")
        return choices[index]
    if name == "IntDistribution":
        return int(value)
    return value


def read_database(path: Path, root: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        studies = []
        for study in connection.execute(
            "select study_id, study_name from studies order by study_id"
        ):
            directions = [
                row["direction"]
                for row in connection.execute(
                    "select direction from study_directions where study_id=? order by objective",
                    (study["study_id"],),
                )
            ]
            trials = []
            ignored_states: dict[str, int] = {}
            for trial in connection.execute(
                "select trial_id, number, state from trials where study_id=? order by number",
                (study["study_id"],),
            ):
                if trial["state"] != "COMPLETE":
                    ignored_states[trial["state"]] = ignored_states.get(trial["state"], 0) + 1
                    continue
                values = [
                    row["value"]
                    for row in connection.execute(
                        "select value from trial_values where trial_id=? order by objective",
                        (trial["trial_id"],),
                    )
                ]
                parameters = {
                    row["param_name"]: external_parameter(
                        row["param_value"], row["distribution_json"]
                    )
                    for row in connection.execute(
                        "select param_name, param_value, distribution_json "
                        "from trial_params where trial_id=? order by param_name",
                        (trial["trial_id"],),
                    )
                }
                metrics = {
                    row["key"]: json.loads(row["value_json"])
                    for row in connection.execute(
                        "select key, value_json from trial_user_attributes "
                        "where trial_id=? order by key",
                        (trial["trial_id"],),
                    )
                }
                trials.append(
                    {
                        "number": trial["number"],
                        "values": values,
                        "parameters": parameters,
                        "reported_metrics": metrics,
                    }
                )
            studies.append(
                {
                    "name": study["study_name"],
                    "directions": directions,
                    "completed_trials": trials,
                    "ignored_noncomplete_states": ignored_states,
                }
            )
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "studies": studies,
        }
    finally:
        connection.close()


def build_archive(root: Path, input_dir: Path) -> dict[str, Any]:
    databases = sorted(input_dir.glob("*optuna*.sqlite3"))
    if not databases:
        raise FileNotFoundError(f"no legacy Optuna databases found under {input_dir}")
    sources = [read_database(path, root) for path in databases]
    completed = sum(
        len(study["completed_trials"])
        for source in sources
        for study in source["studies"]
    )
    ignored = sum(
        sum(study["ignored_noncomplete_states"].values())
        for source in sources
        for study in source["studies"]
    )
    return {
        "schema_version": 1,
        "scope": "legacy Optuna archive; diagnostic only; never resumable or promotable",
        "sealed_test_evaluated": False,
        "completed_trials_archived": completed,
        "noncomplete_trials_ignored": ignored,
        "sources": sources,
        "limitations": [
            "Legacy trials do not bind canonical base config, data snapshot, cutoff, evaluation plan, code identity, or resource budget.",
            "Legacy scalar objectives mix mean guesses, hard-case terms, failures, and latency and are not comparable with the guarded v10 study objective.",
            "Imported parameters are historical suggestions only. Re-evaluate any candidate through the current rolling runner before considering it.",
            "The sealed test remains unopened; this importer reads local SQLite files only.",
        ],
    }


def encoded_archive(root: Path, input_dir: Path) -> bytes:
    return (json.dumps(build_archive(root, input_dir), indent=2, sort_keys=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/derived/predictive")
    parser.add_argument(
        "--output", default="benchmarks/predictive/legacy-optuna-archive.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = (root / args.input_dir).resolve()
    output = (root / args.output).resolve()
    payload = encoded_archive(root, input_dir)
    if args.check:
        if not output.exists() or output.read_bytes() != payload:
            raise SystemExit(f"legacy Optuna archive is stale: {output}")
        print(f"legacy Optuna archive is current: {output.relative_to(root)}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"archived legacy Optuna trials to {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
