#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shlex
import sys
import time
from itertools import product
from pathlib import Path

sys.dont_write_bytecode = True

ALLOWED_TASKS = {"classification", "regression"}
ALLOWED_MODES = {"onehot", "predicates"}
ALLOWED_LOSSES = {"ce", "mse", "rank"}
REGRESSION_LOSSES = {"mse", "rank"}


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


def _canonical_loss(value) -> str:
    loss = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "ce" if loss == "cross_entropy" else loss


def _parse_structure(structure: str) -> tuple[int, list[int]]:
    parts = [part.strip() for part in str(structure).split("@") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid structure '{structure}'. Expected '<binarization_width>@<logical_width>[...]'.")
    values = [int(part) for part in parts]
    if any(value < 0 for value in values):
        raise ValueError(f"Invalid structure '{structure}'. Values must be non-negative.")
    return values[0], values[1:]


def _read_info(path: Path) -> tuple[str, bool]:
    task = None
    has_continuous_features = False
    has_features = False
    has_targets = False

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) == 2 and tokens[0].upper() == "TASK":
            task = tokens[1].lower()
            continue
        if len(tokens) != 3:
            raise ValueError(f"Malformed .info line in {path}: {raw!r}")
        _, typ, role = tokens
        typ = typ.lower()
        role = role.lower()
        if role == "feature":
            has_features = True
            if typ == "continuous":
                has_continuous_features = True
        elif role == "target":
            has_targets = True
        elif role != "ignore":
            raise ValueError(f"Unsupported .info role in {path}: {role!r}")

    if task not in ALLOWED_TASKS:
        raise ValueError(f"Missing or unsupported TASK in {path}. Expected one of {sorted(ALLOWED_TASKS)}.")
    if not has_features:
        raise ValueError(f"No feature columns declared in {path}.")
    if not has_targets:
        raise ValueError(f"No target columns declared in {path}.")
    return task, has_continuous_features


def _normalize_tasks(raw_tasks) -> list[str]:
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("services.data_generation.tasks must be a non-empty list")
    tasks = []
    for value in raw_tasks:
        task = str(value).strip().lower()
        if task not in ALLOWED_TASKS:
            raise ValueError(f"Unsupported task '{value}'. Expected one of {sorted(ALLOWED_TASKS)}.")
        if task not in tasks:
            tasks.append(task)
    return tasks


def _normalize_modes(raw_modes) -> list[str]:
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError("services.training.grid.modes must be a non-empty list")
    modes = []
    for value in raw_modes:
        mode = str(value).strip().lower()
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported training mode '{value}'. Expected one of {sorted(ALLOWED_MODES)}.")
        if mode not in modes:
            modes.append(mode)
    return modes


def _normalize_losses(raw_losses) -> list[str]:
    if not isinstance(raw_losses, list) or not raw_losses:
        raise ValueError("services.training.grid.losses must be a non-empty list")
    losses = []
    for value in raw_losses:
        loss = _canonical_loss(value)
        if loss not in ALLOWED_LOSSES:
            raise ValueError(f"Unsupported loss '{value}'. Expected one of {sorted(ALLOWED_LOSSES)}.")
        if loss not in losses:
            losses.append(loss)
    return losses


def _losses_for_task(task: str, configured_losses: list[str]) -> list[str]:
    if task == "classification":
        if "ce" not in configured_losses:
            raise ValueError("Classification datasets require 'ce' in services.training.grid.losses.")
        return ["ce"]
    losses = [loss for loss in configured_losses if loss in REGRESSION_LOSSES]
    if not losses:
        raise ValueError(f"Regression datasets require one of {sorted(REGRESSION_LOSSES)} in services.training.grid.losses.")
    return losses


def _validate_structure_for_mode(structure: str, *, mode: str, has_continuous_features: bool) -> None:
    binarization_width, _ = _parse_structure(structure)
    if mode == "predicates" and binarization_width != 0:
        raise ValueError(f"Predicate mode requires structures to start with 0, got '{structure}'.")
    if mode == "onehot" and not has_continuous_features and binarization_width != 0:
        raise ValueError(
            f"One-hot mode on a discrete dataset requires structures to start with 0, got '{structure}'."
        )


def _load_params(path: Path) -> dict:
    params = _load_json(path)
    if not isinstance(params, dict):
        raise ValueError(f"Expected JSON object in {path}")
    _required(params, ("fnn_root", "benchmarks_file", "output_root", "model_root", "services"), "params")
    services = params["services"]
    if not isinstance(services, dict) or "training" not in services:
        raise KeyError("Missing services.training in params")
    return params


def _load_benchmarks(path: Path) -> list[dict]:
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
        bench = {
            "dataset_base": _dataset_base(str(raw["dataset"])),
            "benchmark": bench_dir.name,
            "bench_dir": bench_dir,
            "data_dir": data_dir,
            "jani": _required_file((bench_dir / str(raw["jani"])).resolve(), "JANI model"),
            "iface": _required_file((bench_dir / str(raw["iface"])).resolve(), "JANI2NNET interface"),
            "nnet": _required_file((bench_dir / str(raw["nnet"])).resolve(), "NNET policy"),
            "prop": _required_file((bench_dir / str(raw["prop"])).resolve(), "JANI property"),
        }
        key = (bench["bench_dir"], bench["jani"], bench["iface"], bench["nnet"], bench["prop"], bench["data_dir"])
        existing = seen.get(key)
        if existing is not None:
            if existing["dataset_base"] != bench["dataset_base"]:
                raise ValueError(
                    "Benchmark settings contain the same files with different dataset bases: "
                    f"{existing['dataset_base']!r} and {bench['dataset_base']!r}"
                )
            continue
        seen[key] = bench
    return list(seen.values())


def _slug(value: object) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        elif ch in {"@", ":", "/", " "}:
            keep.append("-")
    slug = "".join(keep).strip("-_")
    return slug or "run"


def _format_lr(value: object) -> str:
    return f"{float(value):g}".replace(".", "p")


def _mode_tag(mode: str) -> str:
    return {"onehot": "oh", "predicates": "pred"}[mode]


def _run_id(cfg: dict) -> str:
    structure = _slug(str(cfg["structure"]).replace("@", "x"))
    return (
        f"{_slug(cfg['dataset'])}__{_mode_tag(cfg['mode'])}__h{structure}__"
        f"{_slug(cfg['loss'])}__lr{_format_lr(cfg['lr'])}__s{int(cfg['seed'])}"
    )


def _build_cfgs(params: dict, benchmarks: list[dict]) -> list[dict]:
    services = params["services"]
    training = services["training"]
    data_generation = services.get("data_generation")
    if not isinstance(data_generation, dict):
        raise KeyError("Missing services.data_generation. Training uses its tasks to find generated datasets.")

    _required(training, ("fixed_params", "grid"), "services.training")
    fixed = training["fixed_params"]
    grid = training["grid"]
    _required(
        fixed,
        ("kfold", "device_id", "save_best", "use_nlaf", "alpha", "beta", "gamma", "epochs", "batch_size",
         "lr_decay_rate", "lr_decay_epoch", "weight_decay", "temperature"),
        "services.training.fixed_params",
    )
    _required(grid, ("losses", "seeds", "structures", "lrs", "modes"), "services.training.grid")

    tasks = _normalize_tasks(data_generation["tasks"])
    modes = _normalize_modes(grid["modes"])
    losses = _normalize_losses(grid["losses"])
    seeds = [int(seed) for seed in grid["seeds"]]
    structures = [str(structure) for structure in grid["structures"]]
    lrs = [float(lr) for lr in grid["lrs"]]
    if not seeds or not structures or not lrs:
        raise ValueError("services.training.grid seeds, structures, and lrs must be non-empty")

    cfgs = []
    seen_run_ids = set()
    for bench in benchmarks:
        for requested_task in tasks:
            dataset = _dataset_name(bench["dataset_base"], requested_task)
            data_file = _required_file(bench["data_dir"] / f"{dataset}.data", "dataset data")
            info_file = _required_file(bench["data_dir"] / f"{dataset}.info", "dataset info")
            task, has_continuous_features = _read_info(info_file)
            if task != requested_task:
                raise ValueError(f"Dataset {dataset} has TASK {task!r}, expected {requested_task!r}.")

            for mode in modes:
                for structure in structures:
                    _validate_structure_for_mode(
                        structure,
                        mode=mode,
                        has_continuous_features=has_continuous_features,
                    )
                for structure, lr, loss, seed in product(
                    structures,
                    lrs,
                    _losses_for_task(task, losses),
                    seeds,
                ):
                    cfg = {
                        "dataset": dataset,
                        "task": task,
                        "benchmark": bench["benchmark"],
                        "bench_dir": str(bench["bench_dir"]),
                        "data_dir": str(bench["data_dir"]),
                        "data_file": str(data_file),
                        "info_file": str(info_file),
                        "jani": str(bench["jani"]),
                        "iface": str(bench["iface"]),
                        "nnet": str(bench["nnet"]),
                        "prop": str(bench["prop"]),
                        "mode": mode,
                        "structure": str(structure),
                        "loss": loss,
                        "seed": int(seed),
                        "epochs": int(fixed["epochs"]),
                        "batch_size": int(fixed["batch_size"]),
                        "lr": float(lr),
                        "lr_decay_rate": float(fixed["lr_decay_rate"]),
                        "lr_decay_epoch": int(fixed["lr_decay_epoch"]),
                        "weight_decay": float(fixed["weight_decay"]),
                        "temperature": float(fixed["temperature"]),
                        "has_continuous_features": bool(has_continuous_features),
                    }
                    cfg["run_id"] = _run_id(cfg)
                    if cfg["run_id"] in seen_run_ids:
                        raise ValueError(f"Duplicate training run_id: {cfg['run_id']}")
                    seen_run_ids.add(cfg["run_id"])
                    cfgs.append(cfg)
    return cfgs


def _resolve_device_id(raw_device_id):
    if raw_device_id is None:
        return None
    text = str(raw_device_id).strip().lower()
    if text in {"", "none", "cpu"}:
        return None

    import torch

    device_id = int(text)
    if not torch.cuda.is_available():
        raise RuntimeError(f"Configured device_id={raw_device_id!r}, but CUDA is not available.")
    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        raise RuntimeError(f"Configured device_id={device_id}, but available CUDA device ids are 0..{device_count - 1}.")
    return device_id


def _seed_all(seed: int, torch_mod, np_mod) -> None:
    random.seed(seed)
    np_mod.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _build_command(params_path: Path, cfg_index: int) -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), "--params", str(params_path), "--cfg-index", str(cfg_index)])


def _prepare_outputs(*, output_root: Path, cfgs: list[dict], params_path: Path) -> tuple[Path, Path]:
    training_root = output_root / "training"
    cfgs_path = training_root / "cfgs.json"
    commands_path = training_root / "commands.txt"
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
    commands = [_build_command(params_path, index) for index in range(len(cfgs))]
    commands_path.write_text("\n".join(commands) + ("\n" if commands else ""))
    return cfgs_path, commands_path


def _load_cfgs(path: Path) -> list[dict]:
    obj = _load_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("cfgs"), list):
        return obj["cfgs"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported cfg file format in {path}")


def _train_one(
    cfg: dict,
    *,
    fnn_root: Path,
    model_root: Path,
    fixed: dict,
    device_id,
) -> dict:
    if str(fnn_root) not in sys.path:
        sys.path.insert(0, str(fnn_root))

    import numpy as np
    import torch
    from experiment import get_data_loader, load_model, parse_structure
    from src.models import FNN
    from src.validators import FidelityValidator

    seed = int(cfg["seed"])
    _seed_all(seed, torch, np)

    run_dir = model_root / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pth"
    symbolic_path = run_dir / f"fnn_model__{_slug(cfg['dataset'])}.json"
    log_path = run_dir / "log.txt"

    binarization_width, logical_widths = parse_structure(str(cfg["structure"]))
    print(
        "[train] "
        f"dataset={cfg['dataset']} mode={cfg['mode']} structure={cfg['structure']} "
        f"loss={cfg['loss']} lr={cfg['lr']} seed={seed}",
        flush=True,
    )

    db_enc, train_loader, valid_loader, test_loader = get_data_loader(
        dataset=cfg["dataset"],
        batch_size=int(cfg["batch_size"]),
        k=int(fixed["kfold"]),
        pin_memory=device_id is not None,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        save_best=bool(fixed["save_best"]),
        binarization=str(cfg["mode"]),
        jani_path=cfg["jani"],
        iface_path=cfg["iface"],
        data_dir=cfg["data_dir"],
        binarization_width=binarization_width,
        seed=seed,
    )

    regression_mode = cfg["loss"] if db_enc.task == "regression" else "mse"
    model_binarization_width = 0 if cfg["mode"] == "predicates" else binarization_width
    fnn = FNN(
        dim_list=[(db_enc.discrete_flen, db_enc.continuous_flen)]
        + [model_binarization_width]
        + logical_widths
        + [len(db_enc.y_fname)],
        task=db_enc.task,
        device_id=device_id,
        use_not=False,
        is_rank0=True,
        log_file=str(log_path),
        save_best=bool(fixed["save_best"]),
        estimated_grad=False,
        use_skip=False,
        save_path=str(model_path),
        use_nlaf=bool(fixed["use_nlaf"]),
        alpha=float(fixed["alpha"]),
        beta=int(fixed["beta"]),
        gamma=int(fixed["gamma"]),
        temperature=float(cfg["temperature"]),
        regression_mode=regression_mode,
        run_name=str(cfg["dataset"]),
    )
    if cfg["mode"] == "predicates":
        fnn.predicate_bank = [predicate.to_dict() for predicate in db_enc.predicate_bank]

    train_start = time.perf_counter()
    fnn.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=float(cfg["lr"]),
        epoch=int(cfg["epochs"]),
        lr_decay_rate=float(cfg["lr_decay_rate"]),
        lr_decay_epoch=int(cfg["lr_decay_epoch"]),
        weight_decay=float(cfg["weight_decay"]),
        validator=FidelityValidator(valid_loader),
    )
    train_time = time.perf_counter() - train_start

    if not model_path.exists():
        raise FileNotFoundError(f"Training did not produce expected model file: {model_path}")

    trained = load_model(str(model_path), device_id, log_file=str(log_path))
    fidelity, f1 = trained.test(test_loader, set_name="Test", output_log=True)
    metrics = getattr(trained, "last_test_metrics", {}) or {}
    stats = trained.model_info(
        db_enc.X_fname,
        db_enc.y_fname,
        train_loader,
        mean=db_enc.mean,
        std=db_enc.std,
    )
    symbolic_payload = trained.export_symbolic(
        db_enc.X_fname,
        db_enc.y_fname,
        train_loader,
        file=io.StringIO(),
        mean=db_enc.mean,
        std=db_enc.std,
        atoms_type=cfg["mode"],
        display=False,
    )
    symbolic_payload["training_dataset"] = cfg["dataset"]
    symbolic_payload["training_metadata"] = {
        "run_id": cfg["run_id"],
        "dataset": cfg["dataset"],
        "task": db_enc.task,
        "mode": cfg["mode"],
        "loss": cfg["loss"],
        "structure": cfg["structure"],
        "seed": seed,
    }
    _write_json(symbolic_path, symbolic_payload)

    result = {
        **cfg,
        "model_dir": str(run_dir),
        "model_path": str(model_path),
        "symbolic_path": str(symbolic_path),
        "log_path": str(log_path),
        "fidelity": float(fidelity),
        "f1": float(f1),
        "mse": metrics.get("mse"),
        "rank_acc": metrics.get("rank_acc"),
        "train_time_sec": float(train_time),
        "binarization_time_sec": float(getattr(db_enc, "binarization_time_sec", 0.0)),
        "binarization_size": stats.get("binarization_size"),
        "rules_count": stats.get("rules_count"),
        "avg_LpR": stats.get("avg_LpR"),
        "max_LpR": stats.get("max_LpR"),
        "Eub": stats.get("Eub"),
        "Edisc": stats.get("Edisc"),
        "Ealive": stats.get("Ealive"),
        "Er1": stats.get("Er1"),
        "Er2": stats.get("Er2"),
        "Rub": stats.get("Rub"),
        "Rrelevant": stats.get("Rrelevant"),
        "n_atoms": stats.get("n_atoms"),
    }
    print(
        "[done] "
        f"run_id={cfg['run_id']} fidelity={float(fidelity):.4f} f1={float(f1):.4f}",
        flush=True,
    )
    return result


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train all FNN configurations from a SOCS26 params file.")
    parser.add_argument(
        "--params",
        default=str((script_dir.parent / "params-subset.json").resolve()),
        help="Path to FNN params JSON.",
    )
    parser.add_argument(
        "--cfg-index",
        type=int,
        default=None,
        help="Run only one generated config by index.",
    )
    args = parser.parse_args()

    params_path = _expand_path(args.params).resolve()
    params = _load_params(params_path)
    fnn_root = _required_file(_expand_path(params["fnn_root"]).resolve(), "FNN root")
    benchmarks_file = _required_file(_expand_path(params["benchmarks_file"]).resolve(), "benchmark settings")
    output_root = _expand_path(params["output_root"]).resolve()
    model_root = _expand_path(params["model_root"]).resolve()
    fixed = params["services"]["training"]["fixed_params"]

    benchmarks = _load_benchmarks(benchmarks_file)
    cfgs = _build_cfgs(params, benchmarks)
    cfgs_path, commands_path = _prepare_outputs(output_root=output_root, cfgs=cfgs, params_path=params_path)
    results_path = output_root / "training" / "results.json"

    if args.cfg_index is not None:
        if args.cfg_index < 0 or args.cfg_index >= len(cfgs):
            raise IndexError(f"--cfg-index {args.cfg_index} out of range; generated {len(cfgs)} cfgs.")
        selected = [(args.cfg_index, cfgs[args.cfg_index])]
    else:
        selected = list(enumerate(cfgs))

    device_id = _resolve_device_id(fixed["device_id"])
    print(f"[ok] generated {len(cfgs)} training cfgs -> {cfgs_path}", flush=True)
    print(f"[ok] commands: {commands_path}", flush=True)
    print(f"[ok] results: {results_path}", flush=True)

    results = _load_cfgs(results_path) if results_path.exists() and args.cfg_index is not None else []
    for index, cfg in selected:
        print(f"[cfg] {index + 1}/{len(cfgs)} run_id={cfg['run_id']}", flush=True)
        result = _train_one(
            cfg,
            fnn_root=fnn_root,
            model_root=model_root,
            fixed=fixed,
            device_id=device_id,
        )
        results.append(result)
        _write_json(results_path, results)

    print(f"[ok] trained {len(selected)} configuration(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
