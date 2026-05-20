#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(f"make_fnn_dataset.py requires numpy and pandas; missing module: {exc.name}") from None


DATASET_SEED = 0


def _resolve_bound(value, const_map):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return int(const_map[stripped])
    if isinstance(value, dict) and isinstance(value.get("ref"), str):
        return int(const_map[value["ref"]])
    raise ValueError(f"Unsupported JANI bound: {value!r}")


def load_jani_bounds(path: Path):
    obj = json.loads(path.read_text())
    constants = obj.get("constants") or (obj.get("model", {}) or {}).get("constants") or []
    const_map = {c["name"]: c.get("value") for c in constants if "name" in c}
    variables = obj.get("variables") or (obj.get("model", {}) or {}).get("variables") or []

    bounds = {}
    for var in variables:
        name = var.get("name")
        typ = var.get("type")
        lo = hi = None
        if isinstance(typ, dict) and typ.get("kind") == "bounded":
            lo = _resolve_bound(typ.get("lower-bound", typ.get("lower")), const_map)
            hi = _resolve_bound(typ.get("upper-bound", typ.get("upper")), const_map)
        bounds[name] = (lo, hi)
    return bounds


def load_jani2nnet(path: Path):
    obj = json.loads(path.read_text())
    inputs = obj["input"]
    feature_names = (
        [str(e.get("name", f"feat_{i}")) for i, e in enumerate(inputs)]
        if inputs and isinstance(inputs[0], dict)
        else [str(e) for e in inputs]
    )
    outputs = obj["output"]
    class_names = (
        [str(e.get("name", e.get("id", e.get("class", f"class_{i}")))) for i, e in enumerate(outputs)]
        if outputs and isinstance(outputs[0], dict)
        else [str(c) for c in outputs]
    )
    return feature_names, class_names


def _sanitize_col(name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not out:
        out = "target"
    if out[0].isdigit():
        out = f"t_{out}"
    return out


def _unique(names):
    seen = {}
    result = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
            continue
        seen[name] += 1
        result.append(f"{name}_{seen[name]}")
    return result


def _series_is_integral(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.all(np.isclose(values, np.round(values))))


def _coerce_feature_value(value, discrete: bool):
    numeric = float(value)
    return int(round(numeric)) if discrete else numeric


def load_trace_json(path: Path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "sample" not in data or not isinstance(data["sample"], list):
        raise ValueError(f"Trace file {path} has invalid format. Expected object with list field 'sample'.")
    return data


def convert_trace_to_dataset(
    *,
    trace_path: Path,
    jani_path: Path,
    iface_path: Path,
    out_dir: Path,
    dataset_name: str,
) -> dict:
    rng = np.random.default_rng(DATASET_SEED)
    rows = load_trace_json(trace_path)["sample"]
    if not rows:
        raise ValueError("Input sample is empty")

    feature_names, class_names = load_jani2nnet(iface_path)
    n_features = len(feature_names)

    states = []
    for row in rows:
        state = row["state"]
        if len(state) != n_features:
            raise ValueError(
                f"Trace state width does not match interface width ({len(state)} != {n_features})."
            )
        states.append(state)

    df = pd.DataFrame(states, columns=feature_names)
    feature_types = {
        col: ("discrete" if _series_is_integral(df[col]) else "continuous")
        for col in feature_names
    }
    has_continuous_features = any(kind == "continuous" for kind in feature_types.values())
    mask_cols = []

    is_regression = "y" in rows[0]
    if is_regression and any("y" not in row for row in rows):
        raise ValueError("Mixed trace format: some rows have 'y', some do not")

    if is_regression:
        y_rows = [row["y"] for row in rows]
        if any(len(y) != len(class_names) for y in y_rows):
            raise ValueError("Regression target length does not match interface output count")

        target_cols = _unique([_sanitize_col(name) for name in class_names])
        for i, target_col in enumerate(target_cols):
            df[target_col] = [float(y[i]) for y in y_rows]

        has_mask = "mask" in rows[0]
        if has_mask and any("mask" not in row for row in rows):
            raise ValueError("Mixed trace format: some regression rows have 'mask', some do not")
        if has_mask:
            mask_rows = [row["mask"] for row in rows]
            if any(not isinstance(mask, list) for mask in mask_rows):
                raise ValueError("Regression mask must be a list per sample")
            if any(len(mask) != len(class_names) for mask in mask_rows):
                raise ValueError("Regression mask length does not match interface output count")

            mask_cols = _unique([f"{target_col}_mask" for target_col in target_cols])
            for i, mask_col in enumerate(mask_cols):
                df[mask_col] = [int(mask[i]) for mask in mask_rows]
        task = "regression"
    else:
        if any("label" not in row for row in rows):
            raise ValueError("Classification trace rows must contain 'label'")
        df["class"] = [int(row["label"]) for row in rows]
        target_cols = ["class"]
        task = "classification"

    label_counts_before = _label_counts(df, target_cols, task)
    augmentation = _augment_discrete_rows(
        df=df,
        feature_names=feature_names,
        feature_types=feature_types,
        target_cols=target_cols,
        mask_cols=mask_cols,
        class_names=class_names,
        task=task,
        jani_path=jani_path,
        rng=rng,
        enabled=not has_continuous_features,
    )
    df = augmentation["df"]
    label_counts_after = _label_counts(df, target_cols, task)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{dataset_name}.data"
    info_path = out_dir / f"{dataset_name}.info"
    metadata_path = out_dir / f"{dataset_name}.auginfo.json"

    df.to_csv(data_path, header=False, index=False)
    _write_info(
        path=info_path,
        feature_names=feature_names,
        feature_types=feature_types,
        target_cols=target_cols,
        mask_cols=mask_cols,
        task=task,
    )

    metadata = {
        "task": task,
        "features": augmentation["missing_report"],
        "total_aug_rows_feature_values": augmentation["total_aug_rows"],
        "total_aug_rows_labels": augmentation["total_label_aug_rows"],
        "missing_labels_added": augmentation["label_aug_missing"],
        "rows_before": len(rows),
        "rows_after": len(df),
        "n_features": n_features,
        "augmentation_enabled": augmentation["enabled"],
        "augmentation_disabled_reason": None if augmentation["enabled"] else "continuous_features",
        "label_counts_before": label_counts_before,
        "label_counts_after": label_counts_after,
        "num_outputs_in_interface": len(class_names),
        "num_targets": len(target_cols),
        "num_masks": len(mask_cols),
        "data_path": str(data_path),
        "info_path": str(info_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _label_counts(df: pd.DataFrame, target_cols: list[str], task: str) -> dict[int, int]:
    if task == "classification":
        counts = df["class"].value_counts().sort_index()
    else:
        argmax = np.argmax(df[target_cols].to_numpy(dtype=float), axis=1)
        counts = pd.Series(argmax).value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def _augment_discrete_rows(
    *,
    df: pd.DataFrame,
    feature_names: list[str],
    feature_types: dict[str, str],
    target_cols: list[str],
    mask_cols: list[str],
    class_names: list[str],
    task: str,
    jani_path: Path,
    rng,
    enabled: bool,
) -> dict:
    missing_report = []
    label_aug_missing = []
    total_aug_rows = 0
    total_label_aug_rows = 0

    if not enabled:
        return {
            "df": df,
            "enabled": False,
            "missing_report": missing_report,
            "label_aug_missing": label_aug_missing,
            "total_aug_rows": total_aug_rows,
            "total_label_aug_rows": total_label_aug_rows,
        }

    jani_bounds = load_jani_bounds(jani_path)
    bounds_seq = [jani_bounds.get(name) for name in feature_names]
    base_values = df[feature_names].to_numpy(dtype=float)
    synthetic_rows = []

    for i, bounds in enumerate(bounds_seq):
        name = feature_names[i]
        if feature_types[name] != "discrete":
            missing_report.append({"idx": i, "name": name, "missing": 0, "note": "continuous_feature"})
            continue
        if bounds is None or bounds[0] is None or bounds[1] is None:
            missing_report.append({"idx": i, "name": name, "missing": 0, "note": "no_bounds"})
            continue

        lo, hi = bounds
        seen = set(int(v) for v in df[name].unique())
        missing_values = [v for v in range(lo, hi + 1) if v not in seen]
        missing_report.append({"idx": i, "name": name, "missing": len(missing_values)})
        for value in missing_values:
            base = base_values[rng.integers(0, len(base_values))].copy()
            base[i] = value
            row = {
                feature_names[j]: _coerce_feature_value(
                    base[j],
                    feature_types[feature_names[j]] == "discrete",
                )
                for j in range(len(feature_names))
            }
            if task == "classification":
                observed_labels = df["class"].unique().tolist()
                row["class"] = int(rng.choice(observed_labels)) if observed_labels else 0
            else:
                src_idx = int(rng.integers(0, len(df)))
                for target_col in target_cols:
                    row[target_col] = float(df.iloc[src_idx][target_col])
                for mask_col in mask_cols:
                    row[mask_col] = int(df.iloc[src_idx][mask_col])
            synthetic_rows.append(row)

    if synthetic_rows:
        total_aug_rows = len(synthetic_rows)
        df = pd.concat(
            [df, pd.DataFrame(synthetic_rows, columns=feature_names + target_cols + mask_cols)],
            ignore_index=True,
        )

    if task == "classification":
        all_label_ids = list(range(len(class_names)))
        observed_labels = set(int(v) for v in df["class"].unique())
        label_aug_missing = [label for label in all_label_ids if label not in observed_labels]
        if label_aug_missing:
            feature_pool = df[feature_names].to_numpy(dtype=float)
            synthetic_label_rows = []
            for label in label_aug_missing:
                base = feature_pool[rng.integers(0, len(feature_pool))].copy()
                row = {
                    feature_names[j]: _coerce_feature_value(
                        base[j],
                        feature_types[feature_names[j]] == "discrete",
                    )
                    for j in range(len(feature_names))
                }
                row["class"] = int(label)
                synthetic_label_rows.append(row)
            df = pd.concat(
                [df, pd.DataFrame(synthetic_label_rows, columns=feature_names + ["class"])],
                ignore_index=True,
            )
            total_label_aug_rows = len(synthetic_label_rows)

    return {
        "df": df,
        "enabled": True,
        "missing_report": missing_report,
        "label_aug_missing": label_aug_missing,
        "total_aug_rows": total_aug_rows,
        "total_label_aug_rows": total_label_aug_rows,
    }


def _write_info(
    *,
    path: Path,
    feature_names: list[str],
    feature_types: dict[str, str],
    target_cols: list[str],
    mask_cols: list[str],
    task: str,
) -> None:
    with path.open("w") as handle:
        for feature in feature_names:
            handle.write(f"{feature} {feature_types[feature]} feature\n")
        if task == "classification":
            handle.write("class discrete target\n")
            handle.write("TASK classification\n")
        else:
            for target_col in target_cols:
                handle.write(f"{target_col} continuous target\n")
            for mask_col in mask_cols:
                handle.write(f"{mask_col} discrete ignore\n")
            handle.write("TASK regression\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one PlaJA trace JSON into FNN .data/.info files.")
    parser.add_argument("--trace", required=True, help="PlaJA trace JSON path.")
    parser.add_argument("--jani", required=True, help="JANI model path.")
    parser.add_argument("--iface", required=True, help="JANI2NNET interface path.")
    parser.add_argument("--outdir", required=True, help="Output dataset directory.")
    parser.add_argument("--name", required=True, help="Dataset basename.")
    args = parser.parse_args()

    metadata = convert_trace_to_dataset(
        trace_path=Path(args.trace).expanduser().resolve(),
        jani_path=Path(args.jani).expanduser().resolve(),
        iface_path=Path(args.iface).expanduser().resolve(),
        out_dir=Path(args.outdir).expanduser().resolve(),
        dataset_name=str(args.name),
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
