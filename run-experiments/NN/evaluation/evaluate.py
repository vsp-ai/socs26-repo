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
PLAJA_TIMEOUT_GRACE_SEC = 300
PLAJA_DOCKER_IMAGE = os.environ.get(
    "PLAJA_DOCKER_IMAGE",
    "victorsputrich/plaja_dependencies-chaahat:MRv0.5.1-roundingsat",
)


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = _expand_path(value)
    return path if path.is_absolute() else base / path


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
    _required(params, ("plaja_root", "benchmarks_file", "output_root", "services"), "params")
    services = params["services"]
    if not isinstance(services, dict) or "evaluation" not in services:
        raise KeyError("Missing services.evaluation in params")
    return params


def _load_benchmarks(path: Path) -> list[dict]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    _required(obj, ("bench_root", "benchmarks"), "benchmark settings")
    bench_root = _resolve_path(path.parent, obj["bench_root"]).resolve()
    _required_file(bench_root, "benchmark root")

    by_key: dict[tuple[Path, Path, Path, Path, Path], dict] = {}
    for raw in obj["benchmarks"]:
        _required(raw, ("dataset", "bench_dir", "jani", "iface", "nnet", "prop"), "benchmark")
        bench_dir = _resolve_path(bench_root, raw["bench_dir"]).resolve()
        jani = _required_file((bench_dir / str(raw["jani"])).resolve(), "JANI model")
        iface = _required_file((bench_dir / str(raw["iface"])).resolve(), "JANI2NNET interface")
        nnet = _required_file((bench_dir / str(raw["nnet"])).resolve(), "NNET policy")
        prop = _required_file((bench_dir / str(raw["prop"])).resolve(), "JANI property")
        key = (bench_dir, jani, iface, nnet, prop)
        dataset = str(raw["dataset"])

        if key not in by_key:
            run_id = f"teacher_nn__{_slug(bench_dir.name)}__{_slug(nnet.stem)}"
            by_key[key] = {
                "job_id": f"eval::nn::{run_id}",
                "architecture": "NN",
                "run_id": run_id,
                "dataset": dataset,
                "source_datasets": [dataset],
                "benchmark": bench_dir.name,
                "problem": jani.stem,
                "task": "teacher",
                "mode": "teacher_nn",
                "structure": None,
                "lr": None,
                "bench_dir": str(bench_dir),
                "jani": str(jani),
                "iface": str(iface),
                "nnet": str(nnet),
                "prop": str(prop),
            }
        elif dataset not in by_key[key]["source_datasets"]:
            by_key[key]["source_datasets"].append(dataset)

    cfgs = sorted(by_key.values(), key=lambda cfg: (cfg["benchmark"], cfg["run_id"]))
    for cfg in cfgs:
        cfg["source_datasets"] = sorted(cfg["source_datasets"])
    if not cfgs:
        raise ValueError(f"No NN evaluation configs generated from {path}")
    return cfgs


def _resolve_plaja_bin(plaja_root: Path) -> Path:
    root = plaja_root.expanduser().resolve()
    candidates = [root] if root.is_file() else []
    candidates.extend([root / "build" / "PlaJA", root / "PlaJA"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find PlaJA binary under {root}; expected build/PlaJA")


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


def _build_plaja_cmd(
    *,
    plaja_bin: Path,
    cfg: dict,
    stats_file: Path,
    initial_state_enum: str,
    num_episodes: int | None,
    timeout_sec: int | None,
) -> list[str]:
    cmd = [
        str(plaja_bin),
        "--engine",
        "POLICY_EVALUATION",
        "--model-file",
        str(_expand_path(cfg["jani"]).resolve()),
        "--additional-properties",
        str(_expand_path(cfg["prop"]).resolve()),
        "--prop",
        str(PROP_INDEX),
        "--initial-state-enum",
        str(initial_state_enum),
        "--applicability-filtering",
        str(APPLICABILITY_FILTERING),
        "--print-stats",
        "--stats-file",
        str(stats_file),
        "--nn-interface",
        str(_expand_path(cfg["iface"]).resolve()),
        "--nn",
        str(_expand_path(cfg["nnet"]).resolve()),
    ]
    if num_episodes is not None:
        cmd.extend(["--num-episodes", str(int(num_episodes))])
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
        reader = csv.DictReader(handle)
        last_row = None
        for row in reader:
            last_row = row
    return last_row or {}


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


def _goal_reaching(goal, episodes):
    goal_value = _to_float(goal)
    episodes_value = _to_float(episodes)
    if goal_value is None or episodes_value is None or episodes_value <= 0:
        return None
    return goal_value / episodes_value


def _safety_from_goal_path(value):
    int_value = _to_int(value)
    if int_value is None:
        return None
    return "not_safe" if int_value == 1 else "safe"


def _log_excerpt(path: Path, max_lines: int = 3) -> str:
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("[cmd]")]
    return " | ".join(lines[:max_lines])


def _evaluate_one(
    cfg: dict,
    *,
    plaja_root: Path,
    plaja_bin: Path,
    repo_root: Path,
    logs_dir: Path,
    stats_dir: Path,
    initial_state_enum: str,
    num_episodes: int | None,
    timeout_sec: int | None,
) -> dict:
    run_id = str(cfg["run_id"])
    stats_file = stats_dir / f"plaja_eval_stats__{_slug(run_id)}.csv"
    log_file = logs_dir / f"plaja_eval_log__{_slug(run_id)}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    plaja_cmd = _build_plaja_cmd(
        plaja_bin=plaja_bin,
        cfg=cfg,
        stats_file=stats_file,
        initial_state_enum=initial_state_enum,
        num_episodes=num_episodes,
        timeout_sec=timeout_sec,
    )
    mount_paths = [
        plaja_root,
        repo_root,
        _expand_path(cfg["jani"]).resolve().parent,
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
    row = {
        **cfg,
        "teacher_nnet": cfg["nnet"],
        "plaja_status": stats.get("Status"),
        "has_goal_path": _to_int(stats.get("HasGoalPath")),
        "safety": _safety_from_goal_path(stats.get("HasGoalPath")),
        "goal": _to_float(stats.get("Goal")),
        "episodes": _to_float(stats.get("Episodes")),
        "goal_reaching": _goal_reaching(stats.get("Goal"), stats.get("Episodes")),
        "failure": _to_float(stats.get("Failure")),
        "undone": _to_float(stats.get("Undone")),
        "deadend": _to_float(stats.get("DeadEnd")),
        "search_time": _to_float(stats.get("SearchTime")),
        "average_score": _to_float(stats.get("AverageScore")),
        "average_length": _to_float(stats.get("AverageLength")),
        "iterations": _to_int(stats.get("Iterations")),
        "expanded_states": _to_int(stats.get("ExpandedStates")),
        "generated_states": _to_int(stats.get("GeneratedStates")),
        "global_predicates_added": _to_int(stats.get("GlobalPredicatesAdded")),
        "timeout": timed_out,
        "return_code": return_code,
        "stats_file": str(stats_file),
        "log_file": str(log_file),
    }
    if timed_out:
        row["result"] = "timeout"
    elif return_code == 0:
        row["result"] = "finished"
    else:
        error = f"PlaJA exited with code {return_code}"
        excerpt = _log_excerpt(log_file)
        if excerpt:
            error = f"{error}: {excerpt}"
        row["result"] = "error"
        row["error"] = error
    return row


def _prepare_outputs(*, output_root: Path, cfgs: list[dict], params_path: Path) -> tuple[Path, Path, Path]:
    evaluation_root = output_root / "evaluation"
    cfgs_path = evaluation_root / "cfgs.json"
    commands_path = evaluation_root / "commands.txt"
    results_path = evaluation_root / "results.json"
    _write_json(
        cfgs_path,
        {
            "meta": {
                "params": str(params_path),
                "count": len(cfgs),
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
    parser = argparse.ArgumentParser(description="Evaluate teacher NN policies with PlaJA.")
    parser.add_argument(
        "--params",
        default=str((script_dir.parent / "params-subset.json").resolve()),
        help="Path to NN params JSON.",
    )
    parser.add_argument("--cfg-index", type=int, default=None, help="Run only one generated evaluation config.")
    args = parser.parse_args()

    params_path = _expand_path(args.params).resolve()
    params = _load_params(params_path)
    output_root = _expand_path(params["output_root"]).resolve()
    benchmarks_file = _required_file(_expand_path(params["benchmarks_file"]).resolve(), "benchmark settings")
    plaja_root = _expand_path(params["plaja_root"]).resolve()
    plaja_bin = _resolve_plaja_bin(plaja_root)
    repo_root = params_path.parents[2]

    eval_params = params["services"]["evaluation"].get("eval_params", {}) or {}
    initial_state_enum = str(eval_params.get("initial_state_enum", "sample"))
    num_episodes = eval_params.get("num_episodes")
    timeout_sec = eval_params.get("timeout_sec")

    cfgs = _load_benchmarks(benchmarks_file)
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

    logs_dir = output_root / "evaluation" / "logs"
    stats_dir = output_root / "evaluation" / "stats"
    print(f"[ok] generated {len(cfgs)} NN evaluation cfgs -> {cfgs_path}", flush=True)
    print(f"[ok] commands: {commands_path}", flush=True)
    print(f"[ok] results: {results_path}", flush=True)

    selected_results = []
    for index, cfg in selected:
        print(f"[nn-eval] {index + 1}/{len(cfgs)} run_id={cfg['run_id']}", flush=True)
        row = _evaluate_one(
            cfg,
            plaja_root=plaja_root,
            plaja_bin=plaja_bin,
            repo_root=repo_root,
            logs_dir=logs_dir,
            stats_dir=stats_dir,
            initial_state_enum=initial_state_enum,
            num_episodes=num_episodes,
            timeout_sec=timeout_sec,
        )
        results.append(row)
        selected_results.append(row)
        _write_json(results_path, results)
        if row.get("result") == "finished":
            print(f"[done] run_id={cfg['run_id']} status={row.get('plaja_status')}", flush=True)
        else:
            print(f"[failed] run_id={cfg['run_id']} error={row.get('error', row.get('result'))}", flush=True)

    finished = sum(1 for row in selected_results if row.get("result") == "finished")
    print(f"[ok] evaluated {finished}/{len(selected_results)} selected NN configuration(s)")
    return 0 if finished == len(selected_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
