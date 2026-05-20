#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

from make_fnn_dataset import convert_trace_to_dataset


ALLOWED_TASKS = {"classification", "regression"}
PROP_INDEX = 1
APPLICABILITY_FILTERING = 1
TERMINATE_CYCLES = True
PLAJA_DOCKER_IMAGE = os.environ.get(
    "PLAJA_DOCKER_IMAGE",
    "victorsputrich/plaja_dependencies-chaahat:MRv0.5.1-roundingsat",
)


@dataclass(frozen=True)
class Benchmark:
    dataset_base: str
    benchmark: str
    bench_dir: Path
    data_dir: Path
    jani: Path
    iface: Path
    nnet: Path
    prop: Path


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = _expand_path(value)
    return path if path.is_absolute() else base / path


def _load_json(path: Path):
    return json.loads(path.read_text())


def _required(mapping: dict, keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing {label} keys: {missing}")


def _required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _dataset_base(dataset: str) -> str:
    if dataset.endswith("_reg") or dataset.endswith("_cls"):
        return dataset.rsplit("_", 1)[0]
    return dataset


def _dataset_name(base: str, task: str) -> str:
    suffix = "cls" if task == "classification" else "reg"
    return f"{base}_{suffix}"


def _load_params(path: Path) -> dict:
    params = _load_json(path)
    if not isinstance(params, dict):
        raise ValueError(f"Expected JSON object in {path}")
    _required(params, ("fnn_root", "plaja_root", "benchmarks_file", "output_root", "services"), "params")
    return params


def _normalize_tasks(raw_tasks) -> list[str]:
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("services.data_generation.tasks must be a non-empty list")
    tasks = []
    for value in raw_tasks:
        task = str(value).strip().lower()
        if task not in ALLOWED_TASKS:
            raise ValueError(f"Unsupported data-generation task '{value}'. Expected one of {sorted(ALLOWED_TASKS)}")
        if task not in tasks:
            tasks.append(task)
    return tasks


def _load_benchmarks(path: Path) -> list[Benchmark]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    _required(obj, ("bench_root", "benchmarks"), "benchmark settings")
    bench_root = _resolve_path(path.parent, obj["bench_root"]).resolve()
    _required_file(bench_root, "benchmark root")

    seen = {}
    for raw in obj["benchmarks"]:
        _required(raw, ("dataset", "bench_dir", "jani", "iface", "nnet", "prop"), "benchmark")
        bench_dir = _resolve_path(bench_root, raw["bench_dir"]).resolve()
        data_dir = _resolve_path(bench_dir, raw.get("data_dir", "training_dataset")).resolve()
        benchmark = Benchmark(
            dataset_base=_dataset_base(str(raw["dataset"])),
            benchmark=bench_dir.name,
            bench_dir=bench_dir,
            data_dir=data_dir,
            jani=_required_file((bench_dir / str(raw["jani"])).resolve(), "JANI model"),
            iface=_required_file((bench_dir / str(raw["iface"])).resolve(), "JANI2NNET interface"),
            nnet=_required_file((bench_dir / str(raw["nnet"])).resolve(), "NNET policy"),
            prop=_required_file((bench_dir / str(raw["prop"])).resolve(), "JANI property"),
        )
        key = (benchmark.bench_dir, benchmark.jani, benchmark.iface, benchmark.nnet, benchmark.prop, benchmark.data_dir)
        existing = seen.get(key)
        if existing is not None:
            if existing.dataset_base != benchmark.dataset_base:
                raise ValueError(
                    "Benchmark settings contain the same files with different dataset bases: "
                    f"{existing.dataset_base!r} and {benchmark.dataset_base!r}"
                )
            continue
        seen[key] = benchmark
    return list(seen.values())


def _resolve_plaja_bin(plaja_root: Path) -> Path:
    root = plaja_root.expanduser().resolve()
    candidates = [root] if root.is_file() else []
    candidates.extend([root / "build" / "PlaJA", root / "PlaJA"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find PlaJA binary under {root}; expected build/PlaJA")


def _build_plaja_cmd(
    *,
    plaja_bin: Path,
    benchmark: Benchmark,
    task: str,
    trace_file: Path,
    initial_state_enum: str,
    num_episodes: int | None,
    max_len_episode: int | None,
) -> list[str]:
    cmd = [
        str(plaja_bin),
        "--engine",
        "POLICY_EVALUATION",
        "--model-file",
        str(benchmark.jani),
        "--prop",
        str(PROP_INDEX),
        "--additional-properties",
        str(benchmark.prop),
        "--nn-interface",
        str(benchmark.iface),
        "--nn",
        str(benchmark.nnet),
        "--initial-state-enum",
        str(initial_state_enum),
        "--applicability-filtering",
        str(APPLICABILITY_FILTERING),
        "--store-traces",
        str(trace_file),
        "--store-traces-mode",
        task,
    ]
    if TERMINATE_CYCLES:
        cmd.append("--terminate-cycles")
    if num_episodes is not None:
        cmd.extend(["--num-episodes", str(int(num_episodes))])
    if max_len_episode is not None:
        cmd.extend(["--max-len-episode", str(int(max_len_episode))])
    return cmd


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


def _run_plaja(
    cmd: list[str],
    *,
    plaja_root: Path,
    repo_root: Path,
    benchmark: Benchmark,
    log_file: Path,
    timeout_sec: int | None,
) -> tuple[int, bool]:
    docker_cmd = _build_docker_cmd(
        mount_paths=[plaja_root, repo_root, benchmark.bench_dir, benchmark.data_dir],
        plaja_cmd=cmd,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as handle:
        handle.write(f"[cmd] {shlex.join(docker_cmd)}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                docker_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            )
            return int(proc.returncode), False
        except FileNotFoundError:
            handle.write("\n[error] docker executable not found\n")
            return 127, False
        except subprocess.TimeoutExpired:
            handle.write(f"\n[timeout] exceeded {timeout_sec} sec\n")
            return 124, True


def _log_excerpt(path: Path, max_lines: int = 3) -> str:
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("[cmd]")]
    return " | ".join(lines[:max_lines])


def _write_conversion_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _generate_one(
    *,
    plaja_bin: Path,
    plaja_root: Path,
    repo_root: Path,
    benchmark: Benchmark,
    task: str,
    initial_state_enum: str,
    num_episodes: int | None,
    max_len_episode: int | None,
    timeout_sec: int | None,
    logs_dir: Path,
) -> dict:
    dataset = _dataset_name(benchmark.dataset_base, task)
    trace_file = benchmark.data_dir / f"{dataset}.json"
    plaja_log = logs_dir / f"plaja_store_log__{dataset}.log"
    conversion_log = logs_dir / f"make_fnn_dataset_log__{dataset}.json"

    benchmark.data_dir.mkdir(parents=True, exist_ok=True)
    if trace_file.exists():
        trace_file.unlink()

    cmd = _build_plaja_cmd(
        plaja_bin=plaja_bin,
        benchmark=benchmark,
        task=task,
        trace_file=trace_file,
        initial_state_enum=initial_state_enum,
        num_episodes=num_episodes,
        max_len_episode=max_len_episode,
    )
    return_code, timed_out = _run_plaja(
        cmd,
        plaja_root=plaja_root,
        repo_root=repo_root,
        benchmark=benchmark,
        log_file=plaja_log,
        timeout_sec=timeout_sec,
    )
    result = {
        "dataset": dataset,
        "task": task,
        "benchmark": benchmark.benchmark,
        "bench_dir": str(benchmark.bench_dir),
        "data_dir": str(benchmark.data_dir),
        "trace_file": str(trace_file),
        "data_file": str((benchmark.data_dir / f"{dataset}.data").resolve()),
        "info_file": str((benchmark.data_dir / f"{dataset}.info").resolve()),
        "plaja_log_file": str(plaja_log),
        "conversion_log_file": str(conversion_log),
        "plaja_return_code": return_code,
        "plaja_timeout": timed_out,
    }
    if timed_out:
        result.update({"result": "timeout", "error": f"PlaJA timed out after {timeout_sec} sec"})
        return result
    if return_code != 0:
        excerpt = _log_excerpt(plaja_log)
        error = f"PlaJA exited with code {return_code}"
        if excerpt:
            error = f"{error}: {excerpt}"
        result.update({"result": "error", "error": error})
        return result

    try:
        metadata = convert_trace_to_dataset(
            trace_path=trace_file,
            jani_path=benchmark.jani,
            iface_path=benchmark.iface,
            out_dir=benchmark.data_dir,
            dataset_name=dataset,
        )
        _write_conversion_log(conversion_log, metadata)
        result.update(
            {
                "result": "finished",
                "rows_before": metadata["rows_before"],
                "rows_after": metadata["rows_after"],
                "augmentation_enabled": metadata["augmentation_enabled"],
            }
        )
    except Exception as exc:
        _write_conversion_log(conversion_log, {"result": "error", "error": str(exc)})
        result.update({"result": "error", "error": str(exc)})
    return result


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate FNN datasets from PlaJA policy traces.")
    parser.add_argument(
        "--params",
        default=str((script_dir.parent / "params-subset.json").resolve()),
        help="Path to FNN params JSON.",
    )
    args = parser.parse_args()

    params_path = _expand_path(args.params).resolve()
    params = _load_params(params_path)
    services = params["services"]
    data_cfg = services.get("data_generation", {})
    data_params = data_cfg.get("data_params", {}) or {}

    fnn_root = _expand_path(params["fnn_root"]).resolve()
    output_root = _expand_path(params["output_root"]).resolve()
    benchmarks_file = _expand_path(params["benchmarks_file"]).resolve()
    plaja_root = _expand_path(params["plaja_root"]).resolve()
    plaja_bin = _resolve_plaja_bin(plaja_root)
    repo_root = fnn_root.parent
    _required_file(fnn_root, "FNN root")
    _required_file(benchmarks_file, "benchmark settings")

    tasks = _normalize_tasks(data_cfg.get("tasks", ["classification", "regression"]))
    benchmarks = _load_benchmarks(benchmarks_file)
    logs_dir = output_root / "data_generation" / "logs"
    results_path = output_root / "data_generation" / "results.json"
    commands_path = output_root / "data_generation" / "commands.txt"

    initial_state_enum = str(data_params.get("initial_state_enum", "sample"))
    num_episodes = data_params.get("num_episodes")
    max_len_episode = data_params.get("max_len_episode")
    timeout_sec = data_params.get("timeout_sec")

    commands = []
    results = []
    for benchmark in benchmarks:
        for task in tasks:
            dataset = _dataset_name(benchmark.dataset_base, task)
            trace_file = benchmark.data_dir / f"{dataset}.json"
            commands.append(
                shlex.join(
                    _build_docker_cmd(
                        mount_paths=[plaja_root, repo_root, benchmark.bench_dir, benchmark.data_dir],
                        plaja_cmd=_build_plaja_cmd(
                            plaja_bin=plaja_bin,
                            benchmark=benchmark,
                            task=task,
                            trace_file=trace_file,
                            initial_state_enum=initial_state_enum,
                            num_episodes=num_episodes,
                            max_len_episode=max_len_episode,
                        ),
                    )
                )
            )
            print(f"[generate] benchmark={benchmark.benchmark} task={task} dataset={dataset}", flush=True)
            result = _generate_one(
                plaja_bin=plaja_bin,
                plaja_root=plaja_root,
                repo_root=repo_root,
                benchmark=benchmark,
                task=task,
                initial_state_enum=initial_state_enum,
                num_episodes=num_episodes,
                max_len_episode=max_len_episode,
                timeout_sec=timeout_sec,
                logs_dir=logs_dir,
            )
            results.append(result)
            if result.get("result") == "finished":
                print(f"[done] dataset={dataset} rows={result.get('rows_after')}", flush=True)
            else:
                print(
                    f"[failed] dataset={dataset} error={result.get('error')} log={result.get('plaja_log_file')}",
                    flush=True,
                )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("\n".join(commands) + ("\n" if commands else ""))
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    finished = sum(1 for row in results if row.get("result") == "finished")
    print(f"[ok] generated {finished}/{len(results)} datasets")
    print(f"[ok] commands: {commands_path}")
    print(f"[ok] results: {results_path}")
    return 0 if finished == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
