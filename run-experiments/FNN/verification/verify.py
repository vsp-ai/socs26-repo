#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROP_INDEX = 1
APPLICABILITY_FILTERING = 1
HEURISTIC_SEARCH = "hamming"
FNN_SAT_SOLVER = "z3"
PLAJA_TIMEOUT_GRACE_SEC = 300
PLAJA_DOCKER_IMAGE = os.environ.get(
    "PLAJA_DOCKER_IMAGE",
    "victorsputrich/plaja_dependencies-chaahat:MRv0.5.1-roundingsat",
)

CSV_FIELDS = (
    "property",
    "Status",
    "HasGoalPath",
    "SpuriousPrefixLength",
    "Iterations",
    "SearchTime",
    "PathLength",
    "ExpandedStates",
    "GeneratedStates",
    "Predicates",
    "Z3Queries",
    "memory_peak",
)
CSV_INT_FIELDS = {
    "Status",
    "HasGoalPath",
    "SpuriousPrefixLength",
    "Iterations",
    "PathLength",
    "ExpandedStates",
    "GeneratedStates",
    "Predicates",
    "Z3Queries",
    "memory_peak",
}
CSV_FLOAT_FIELDS = {"SearchTime"}


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _load_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _required(mapping: dict, keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing {label} keys: {missing}")


def _required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise ValueError(f"Empty {label}: {path}")
    return path


def _load_params(path: Path) -> dict:
    params = _load_json(path)
    if not isinstance(params, dict):
        raise ValueError(f"Expected JSON object in {path}")
    _required(params, ("fnn_root", "plaja_root", "output_root", "services"), "params")
    services = params["services"]
    if not isinstance(services, dict) or "verification" not in services:
        raise KeyError("Missing services.verification in params")
    return params


def _resolve_plaja_bin(plaja_root: Path) -> Path:
    root = plaja_root.expanduser().resolve()
    candidates = [root] if root.is_file() else []
    candidates.extend([root / "build" / "PlaJA", root / "PlaJA"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find PlaJA binary under {root}; expected build/PlaJA")


def _load_evaluation_rows(path: Path) -> list[dict]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected evaluation results list in {path}")

    rows = []
    seen_run_ids = set()
    for row in obj:
        if not isinstance(row, dict):
            raise ValueError(f"Found non-object evaluation result in {path}")
        _required(
            row,
            ("dataset", "run_id", "jani", "iface", "prop", "symbolic_path", "result", "return_code"),
            "evaluation result",
        )
        if row["result"] != "finished" or int(row["return_code"]) != 0:
            raise ValueError(f"Evaluation did not finish for run_id={row.get('run_id')}: result={row.get('result')}")
        run_id = str(row["run_id"])
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id in evaluation results: {run_id}")
        seen_run_ids.add(run_id)
        _required_file(_expand_path(row["symbolic_path"]).resolve(), "symbolic model")
        rows.append(row)

    if not rows:
        raise ValueError(f"No evaluation results found in {path}")
    return rows


def _slug(value: object) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", ".", "@"}:
            keep.append(ch)
        elif ch in {":", "/", " "}:
            keep.append("-")
    slug = "".join(keep).strip("-_")
    return slug or "run"


def _build_cfgs(evaluation_rows: list[dict]) -> list[dict]:
    cfgs = []
    for row in evaluation_rows:
        cfg = {
            "job_id": f"verify::{row['run_id']}",
            "architecture": "FNN",
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "task": row.get("task"),
            "benchmark": row.get("benchmark"),
            "mode": row.get("mode"),
            "loss": row.get("loss"),
            "structure": row.get("structure"),
            "seed": row.get("seed"),
            "lr": row.get("lr"),
            "fidelity": row.get("fidelity"),
            "f1": row.get("f1"),
            "mse": row.get("mse"),
            "rank_acc": row.get("rank_acc"),
            "binarization_size": row.get("binarization_size"),
            "binarization_time_sec": row.get("binarization_time_sec"),
            "train_time_sec": row.get("train_time_sec"),
            "rules_count": row.get("rules_count"),
            "avg_LpR": row.get("avg_LpR"),
            "max_LpR": row.get("max_LpR"),
            "goal": row.get("goal"),
            "episodes": row.get("episodes"),
            "goal_reaching": row.get("goal_reaching"),
            "failure": row.get("failure"),
            "undone": row.get("undone"),
            "deadend": row.get("deadend"),
            "eval_plaja_status": row.get("plaja_status"),
            "eval_search_time": row.get("search_time"),
            "average_score": row.get("average_score"),
            "average_length": row.get("average_length"),
            "model_dir": row.get("model_dir"),
            "model_path": row.get("model_path"),
            "symbolic_path": row["symbolic_path"],
            "symbolic_model_name": Path(row["symbolic_path"]).name,
            "jani": row["jani"],
            "iface": row["iface"],
            "prop": row["prop"],
            "verification_method": FNN_SAT_SOLVER,
        }
        for key, label in (("jani", "JANI model"), ("iface", "JANI2NNET interface"), ("prop", "JANI property")):
            _required_file(_expand_path(cfg[key]).resolve(), label)
        cfgs.append(cfg)
    return cfgs


def _docker_mount_args(paths: list[Path]) -> list[str]:
    args = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        args.extend(["-v", f"{resolved}:{resolved}"])
    return args


def _build_plaja_cmd(*, plaja_bin: Path, cfg: dict, stats_file: Path, timeout_sec: int | None) -> list[str]:
    cmd = [
        str(plaja_bin),
        "--engine",
        "PA_CEGAR",
        "--model-file",
        str(_expand_path(cfg["jani"]).resolve()),
        "--additional-properties",
        str(_expand_path(cfg["prop"]).resolve()),
        "--prop",
        str(PROP_INDEX),
        "--applicability-filtering",
        str(APPLICABILITY_FILTERING),
        "--heuristic-search",
        HEURISTIC_SEARCH,
        "--fnn-sat",
        FNN_SAT_SOLVER,
        "--check-for-pa-terminal-states",
        "true",
        "--set-pa-goal-objective-terminal",
        "true",
        "--split-point",
        "bs",
        "--selection-refinement",
        "all",
        "--entailment-mode",
        "entailment-only",
        "--check-policy-spuriousness",
        "false",
        "--incremental-search",
        "true",
        "--print-stats",
        "--stats-file",
        str(stats_file),
        "--fnn-interface",
        str(_expand_path(cfg["iface"]).resolve()),
        "--fnn",
        str(_expand_path(cfg["symbolic_path"]).resolve()),
    ]
    if timeout_sec is not None and int(timeout_sec) > 0:
        cmd.extend(["--max-time", str(int(timeout_sec))])
    return cmd


def _build_docker_cmd(*, mount_paths: list[Path], plaja_cmd: list[str]) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        *_docker_mount_args(mount_paths),
        "-w",
        str(Path(plaja_cmd[0]).parent),
        PLAJA_DOCKER_IMAGE,
        *plaja_cmd,
    ]


def _process_timeout(timeout_sec: int | None) -> int | None:
    if timeout_sec is None or int(timeout_sec) <= 0:
        return None
    return int(timeout_sec) + PLAJA_TIMEOUT_GRACE_SEC


def _read_last_stats(stats_csv: Path) -> dict:
    if not stats_csv.exists():
        return {}
    with stats_csv.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return {}

    header = rows[0]
    last_row = None
    for row in reversed(rows[1:]):
        if row == header:
            continue
        if row and any(cell.strip() for cell in row):
            last_row = row
            break
    if last_row is None:
        return {}

    first_idx = {}
    for idx, name in enumerate(header):
        if name not in first_idx:
            first_idx[name] = idx

    out = {}
    for key in CSV_FIELDS:
        idx = first_idx.get(key)
        if idx is None or idx >= len(last_row):
            out[key] = None
            continue
        value = str(last_row[idx]).strip()
        out[key] = value if value else None
    return out


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    value = _to_float(value)
    if value is None:
        return None
    return int(value)


def _csv_value(stats: dict, key: str):
    raw = stats.get(key)
    if raw is None:
        return None
    if key in CSV_INT_FIELDS:
        return _to_int(raw)
    if key in CSV_FLOAT_FIELDS:
        return _to_float(raw)
    return raw


def _outcome_from_has_goal_path(has_goal_path: int | None) -> str:
    if has_goal_path == 1:
        return "unsafe"
    if has_goal_path == 0:
        return "safe"
    return "unknown"


def _log_excerpt(path: Path, max_lines: int = 3) -> str:
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("[cmd]")]
    return " | ".join(lines[:max_lines])


def _verify_one(
    cfg: dict,
    *,
    plaja_root: Path,
    plaja_bin: Path,
    repo_root: Path,
    logs_dir: Path,
    stats_dir: Path,
    timeout_sec: int | None,
) -> dict:
    run_id = str(cfg["run_id"])
    stats_file = stats_dir / f"verify_stats__fnn__{_slug(run_id)}__z3.csv"
    log_file = logs_dir / f"verify_log__fnn__{_slug(run_id)}__z3.log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    plaja_cmd = _build_plaja_cmd(
        plaja_bin=plaja_bin,
        cfg=cfg,
        stats_file=stats_file,
        timeout_sec=timeout_sec,
    )
    mount_paths = [
        plaja_root,
        repo_root,
        _expand_path(cfg["jani"]).resolve().parent,
        _expand_path(cfg["symbolic_path"]).resolve().parent,
        logs_dir,
        stats_dir,
    ]
    docker_cmd = _build_docker_cmd(mount_paths=mount_paths, plaja_cmd=plaja_cmd)

    timed_out = False
    with log_file.open("w") as handle:
        handle.write(f"[cmd] {shlex.join(docker_cmd)}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                docker_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=_process_timeout(timeout_sec),
            )
            return_code = int(proc.returncode)
        except FileNotFoundError:
            handle.write("\n[error] docker executable not found\n")
            return_code = 127
        except subprocess.TimeoutExpired:
            handle.write(f"\n[timeout] exceeded process timeout {_process_timeout(timeout_sec)} sec\n")
            return_code = 124
            timed_out = True

    stats = _read_last_stats(stats_file)
    has_goal_path = _csv_value(stats, "HasGoalPath")
    row = {
        **cfg,
        "property": _csv_value(stats, "property"),
        "Status": _csv_value(stats, "Status"),
        "HasGoalPath": has_goal_path,
        "SpuriousPrefixLength": _csv_value(stats, "SpuriousPrefixLength"),
        "Iterations": _csv_value(stats, "Iterations"),
        "SearchTime": _csv_value(stats, "SearchTime"),
        "PathLength": _csv_value(stats, "PathLength"),
        "ExpandedStates": _csv_value(stats, "ExpandedStates"),
        "GeneratedStates": _csv_value(stats, "GeneratedStates"),
        "Predicates": _csv_value(stats, "Predicates"),
        "Z3Queries": _csv_value(stats, "Z3Queries"),
        "memory_peak": _csv_value(stats, "memory_peak"),
        "outcome": _outcome_from_has_goal_path(has_goal_path),
        "verify_stats_exists": bool(stats),
        "timeout": timed_out,
        "return_code": return_code,
        "stats_file": str(stats_file),
        "log_file": str(log_file),
    }
    if timed_out:
        row["result"] = "timeout"
    elif return_code == 0 and stats:
        row["result"] = "finished"
    else:
        error = f"PlaJA exited with code {return_code}"
        if not stats:
            error = f"{error}; verification stats were not produced"
        excerpt = _log_excerpt(log_file)
        if excerpt:
            error = f"{error}: {excerpt}"
        row["result"] = "error"
        row["error"] = error
    return row


def _prepare_outputs(*, output_root: Path, cfgs: list[dict], params_path: Path) -> tuple[Path, Path, Path]:
    verification_root = output_root / "verification"
    cfgs_path = verification_root / "cfgs.json"
    commands_path = verification_root / "commands.txt"
    results_path = verification_root / "results.json"
    _write_json(
        cfgs_path,
        {
            "meta": {
                "params": str(params_path),
                "count": len(cfgs),
                "verification_method": FNN_SAT_SOLVER,
                "heuristic_search": HEURISTIC_SEARCH,
            },
            "cfgs": cfgs,
        },
    )
    commands = [
        shlex.join([sys.executable, str(Path(__file__).resolve()), "--params", str(params_path), "--cfg-index", str(index)])
        for index in range(len(cfgs))
    ]
    commands_path.write_text("\n".join(commands) + ("\n" if commands else ""))
    return cfgs_path, commands_path, results_path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Verify trained FNN policies with PlaJA PA_CEGAR.")
    parser.add_argument(
        "--params",
        default=str((script_dir.parent / "params-subset.json").resolve()),
        help="Path to FNN params JSON.",
    )
    parser.add_argument("--cfg-index", type=int, default=None, help="Run only one generated verification config.")
    args = parser.parse_args()

    params_path = _expand_path(args.params).resolve()
    params = _load_params(params_path)
    fnn_root = _required_file(_expand_path(params["fnn_root"]).resolve(), "FNN root")
    repo_root = fnn_root.parent
    output_root = _expand_path(params["output_root"]).resolve()
    plaja_root = _expand_path(params["plaja_root"]).resolve()
    plaja_bin = _resolve_plaja_bin(plaja_root)

    verify_params = params["services"]["verification"].get("verify_params", {}) or {}
    timeout_sec = verify_params.get("timeout_sec")

    evaluation_results = _required_file(output_root / "evaluation" / "results.json", "evaluation results")
    evaluation_rows = _load_evaluation_rows(evaluation_results)
    cfgs = _build_cfgs(evaluation_rows)
    cfgs_path, commands_path, results_path = _prepare_outputs(output_root=output_root, cfgs=cfgs, params_path=params_path)

    if args.cfg_index is not None:
        if args.cfg_index < 0 or args.cfg_index >= len(cfgs):
            raise IndexError(f"--cfg-index {args.cfg_index} out of range; generated {len(cfgs)} cfgs.")
        selected = [(args.cfg_index, cfgs[args.cfg_index])]
        results = _load_json(results_path) if results_path.exists() else []
        if not isinstance(results, list):
            raise ValueError(f"Expected results list in {results_path}")
    else:
        selected = list(enumerate(cfgs))
        results = []

    logs_dir = output_root / "verification" / "logs"
    stats_dir = output_root / "verification" / "stats"
    print(f"[ok] generated {len(cfgs)} verification cfgs -> {cfgs_path}", flush=True)
    print(f"[ok] commands: {commands_path}", flush=True)
    print(f"[ok] results: {results_path}", flush=True)

    selected_results = []
    for index, cfg in selected:
        print(f"[verify] {index + 1}/{len(cfgs)} run_id={cfg['run_id']}", flush=True)
        row = _verify_one(
            cfg,
            plaja_root=plaja_root,
            plaja_bin=plaja_bin,
            repo_root=repo_root,
            logs_dir=logs_dir,
            stats_dir=stats_dir,
            timeout_sec=timeout_sec,
        )
        results.append(row)
        selected_results.append(row)
        _write_json(results_path, results)
        if row.get("result") == "finished":
            print(f"[done] run_id={cfg['run_id']} outcome={row.get('outcome')}", flush=True)
        else:
            print(f"[failed] run_id={cfg['run_id']} error={row.get('error', row.get('result'))}", flush=True)

    finished = sum(1 for row in selected_results if row.get("result") == "finished")
    print(f"[ok] verified {finished}/{len(selected_results)} selected configuration(s)")
    return 0 if finished == len(selected_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
