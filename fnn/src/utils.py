import csv
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from pathlib import Path

from src.models import FNN

def _predicate_from_dict(predicate):
    if not isinstance(predicate, dict):
        return predicate

    from src.BinarizationLayer.jani_predicates import Predicate

    return Predicate(
        op=predicate["op"],
        vars=tuple(predicate["vars"]),
        coeffs=tuple(predicate["coeffs"]),
        const=predicate["const"],
    )

def read_info(info_path):
    """Read dataset schema from .info using the strict v2 format.

    Required format:
      - one line: `TASK classification|regression`
      - column lines: `<name> <type> <role>` where role is `feature`, `target`, or `ignore`

    Legacy .info files (e.g. `<name> <type>` + `LABEL_POS`) are rejected.
    """
    columns = []
    task = None

    with open(info_path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            head = tokens[0].upper()

            if head == "TASK":
                if len(tokens) != 2:
                    raise ValueError(f"Malformed TASK line in {info_path} at line {lineno}: '{raw.rstrip()}'")
                task = tokens[1].strip().lower()
                if task not in ("classification", "regression"):
                    raise ValueError(f"Unsupported TASK '{task}' in {info_path} at line {lineno}")
                continue

            if head == "LABEL_POS" or len(tokens) == 2:
                raise ValueError(
                    "Legacy .info format detected in "
                    f"{info_path} at line {lineno}: '{raw.rstrip()}'. "
                    "Please update the file to the new schema: "
                    "`<name> <type> <role>` and add `TASK classification|regression`."
                )

            if len(tokens) != 3:
                raise ValueError(f"Malformed column line in {info_path} at line {lineno}: '{raw.rstrip()}'")

            name, typ, role = tokens
            role = role.strip().lower()
            if role not in ("feature", "target", "ignore"):
                raise ValueError(
                    f"Unsupported role '{role}' for column '{name}' in {info_path} at line {lineno}"
                )
            columns.append({"name": name, "type": typ, "role": role})

    if task is None:
        raise ValueError(
            f"Missing TASK declaration in {info_path}. "
            "Add `TASK classification` or `TASK regression`."
        )

    if not columns:
        raise ValueError(f"No column definitions found in {info_path}")

    if not any(c["role"] == "feature" for c in columns):
        raise ValueError(f"No feature columns declared in {info_path}")

    if not any(c["role"] == "target" for c in columns):
        raise ValueError(f"No target columns declared in {info_path}")

    return {
        "columns": columns,
        "task": task,
    }


def read_csv(data_path, info_path, shuffle=False, random_state=0, return_meta=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=random_state).reset_index(drop=True)

    info = read_info(info_path)
    columns = info["columns"]
    col_names = [c["name"] for c in columns]

    if D.shape[1] != len(col_names):
        raise ValueError(
            f"Column mismatch for {data_path}: .data has {D.shape[1]} cols but .info defines {len(col_names)}"
        )

    D.columns = col_names

    feature_cols = [c["name"] for c in columns if c["role"] == "feature"]
    target_cols = [c["name"] for c in columns if c["role"] == "target"]
    ignore_cols = [c["name"] for c in columns if c["role"] == "ignore"]
    ignore_df = D[ignore_cols] if ignore_cols else pd.DataFrame(index=D.index)
    label_pos = col_names.index(target_cols[0]) if len(target_cols) == 1 else None

    X_df = D[feature_cols]
    y_df = D[target_cols]

    # DBEncoder expects DataFrame with [feature_name, feature_type]
    f_rows = [[c["name"], c["type"]] for c in columns if c["role"] == "feature"]
    f_df = pd.DataFrame(f_rows)

    if return_meta:
        meta = {
            "task": info["task"],
            "target_cols": target_cols,
            "feature_cols": feature_cols,
            "ignore_cols": ignore_cols,
            "ignore_df": ignore_df,
        }
        return X_df, y_df, f_df, label_pos, meta

    return X_df, y_df, f_df, label_pos

def dump_csv_results(csv_path: Path, fnn: FNN, acc: float, f1: float, args: dict):
    stats = getattr(fnn, "model_stats", {}) or {}
    def _round4(val):
        if isinstance(val, float):
            return round(val, 4)
        return val
    row = {
        "name": args.data_set,
        "structure": args.structure,
        "fidelity": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "batch_size": int(args.batch_size),
        "epoch": int(args.epoch),
        "learning_rate": _round4(float(args.learning_rate)),
        "lrdr": _round4(float(args.lr_decay_rate)),
        "lrde": int(args.lr_decay_epoch),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "nlaf": bool(getattr(args, "nlaf", False)),
        "alpha": _round4(float(getattr(args, "alpha", 0.0))),
        "beta": int(getattr(args, "beta", 0)),
        "gamma": int(getattr(args, "gamma", 0)),
        "temp": _round4(float(getattr(args, "temp", 0.0))),
        "Eub": stats.get("Eub"),
        "Edisc": stats.get("Edisc"),
        "Ealive": stats.get("Ealive"),
        "Er1": _round4(stats.get("Er1")),
        "Er2": _round4(stats.get("Er2")),
        "Rub": stats.get("Rub"),
        "Rrelevant": stats.get("Rrelevant"),
        "binarization_size": stats.get("binarization_size"),
        "binarization_time_sec": _round4(float(getattr(args, "binarization_time_sec", 0.0))),
        "rules_count": stats.get("rules_count"),
        "avg_LpR": _round4(stats.get("avg_LpR")),
        "max_LpR": stats.get("max_LpR"),
        "n_atoms": stats.get("n_atoms"),
    }
    # EXACT columns you asked for (with your arg names)
    fieldnames = [
        "name",
        "structure",
        "fidelity",
        "f1",
        "batch_size",
        "epoch",
        "learning_rate",
        "lrdr",
        "lrde",
        "weight_decay",
        "nlaf",
        "alpha",
        "beta",
        "gamma",
        "temp",
        "Eub",
        "Edisc",
        "Ealive",
        "Er1",
        "Er2",
        "Rub",
        "Rrelevant",
        "binarization_size",
        "binarization_time_sec",
        "rules_count",
        "avg_LpR",
        "max_LpR",
        "n_atoms",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in fieldnames})
    
    print(f"[csv] appended results to {csv_path}")

class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first', task='classification'):
        self.f_df = f_df
        self.discrete = discrete
        self.task = str(task).lower()
        self.y_one_hot = y_one_hot if self.task != 'regression' else False
        self.label_enc = None if self.task == 'regression' else (preprocessing.OneHotEncoder(categories='auto') if self.y_one_hot else preprocessing.LabelEncoder())
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None
        self.predicate_bank = None
        self.predicate_var_index = None
        self.continuous_source_names = []
        self.binarization_time_sec = 0.0

    def _load_iface_inputs(self, iface_path):
        obj = json.loads(Path(iface_path).read_text())
        inp = obj.get("input", [])
        if inp and isinstance(inp[0], dict):
            return [str(e.get("name", f"feat_{i}")) for i, e in enumerate(inp)]
        return [str(e) for e in inp]

    def _build_onehot_feature_names(self, feature_names):
        """Build deterministic one-hot atom names as `<feature> eq <value>`."""
        categories = getattr(self.feature_enc, "categories_", None)
        if categories is None:
            return list(self.feature_enc.get_feature_names_out(feature_names))

        drop_idx = getattr(self.feature_enc, "drop_idx_", None)
        names = []
        for fi, (fname, cats) in enumerate(zip(feature_names, categories)):
            dropped = None
            if drop_idx is not None and fi < len(drop_idx):
                dropped = drop_idx[fi]
            for ci, cat in enumerate(cats):
                if dropped is not None and ci == dropped:
                    continue
                names.append(f"{fname} eq {cat}")
        return names

    def _feature_names(self, iface_path=None):
        f_names = list(self.f_df.iloc[:, 0])
        if all(str(n).isdigit() for n in f_names):
            if iface_path is None:
                raise ValueError("numeric .info requires --iface for predicate binarization")
            iface_names = self._load_iface_inputs(iface_path)
            if len(iface_names) != len(f_names):
                raise ValueError("iface input length does not match dataset feature count")
            return iface_names
        return f_names

    def _numeric_feature_frame(self, X_df, *, include_discrete_numeric=False):
        if not include_discrete_numeric:
            _, continuous_data = self.split_data(X_df)
            return continuous_data

        cols = {}
        for name in X_df.columns:
            series = X_df[name]
            if series.dtype == object:
                series = series.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            numeric = pd.to_numeric(series, errors='coerce')
            if not numeric.notna().any():
                continue
            cols[name] = numeric.astype(np.float64)
        if not cols:
            return pd.DataFrame(index=X_df.index)
        return pd.DataFrame(cols, index=X_df.index)

    def _prepare_continuous_data(self, continuous_data):
        if continuous_data.empty:
            self.continuous_source_names = []
            self.mean = None
            self.std = None
            return continuous_data

        continuous_imp = pd.DataFrame(
            self.imp.transform(continuous_data.values),
            columns=continuous_data.columns,
            index=continuous_data.index,
        )
        self.continuous_source_names = list(continuous_imp.columns)
        self.mean = continuous_imp.mean()
        self.std = continuous_imp.std()
        return continuous_imp

    def _transform_continuous_frame(self, X_df, normalized=False, keep_stat=False, include_discrete_numeric=False):
        continuous_source = self._numeric_feature_frame(X_df, include_discrete_numeric=include_discrete_numeric)
        if not self.continuous_source_names:
            return pd.DataFrame(index=X_df.index)
        continuous_source = continuous_source[self.continuous_source_names]
        continuous_source = pd.DataFrame(
            self.imp.transform(continuous_source.values),
            columns=self.continuous_source_names,
            index=X_df.index,
        )
        if normalized:
            if keep_stat:
                self.mean = continuous_source.mean()
                self.std = continuous_source.std()
            std_safe = self.std.replace(0, 1.0) if self.std is not None else None
            continuous_source = continuous_source if std_safe is None else ((continuous_source - self.mean) / std_safe)
        return continuous_source

    def fit_predicates(
        self,
        jani_path,
        X_df=None,
        iface_path=None,
        prop_path=None,
        predicate_bank=None,
    ):
        from src.BinarizationLayer.jani_predicates import parse_problem_predicates, predicate_to_str
        feature_names = self._feature_names(iface_path)
        self.predicate_var_index = {name: i for i, name in enumerate(feature_names)}

        kind_by_predicate_key = {}
        if predicate_bank is not None:
            self.predicate_bank = [
                p
                for p in (_predicate_from_dict(p) for p in predicate_bank)
                if all(v in self.predicate_var_index for v in p.vars)
            ]
        else:
            jani_path = Path(jani_path)
            if prop_path is None:
                candidate = jani_path.parent / "compact-property.jani"
                prop_path = candidate if candidate.exists() else None
            parsed_bank, kind_map = parse_problem_predicates(
                jani_path,
                prop_path=prop_path,
                include_goal=True,
                include_fail=True,
                include_fail_complements=True,
            )
            filtered_keys = []
            self.predicate_bank = []
            for p, key in zip(parsed_bank, kind_map.keys()):
                if all(v in self.predicate_var_index for v in p.vars):
                    self.predicate_bank.append(p)
                    filtered_keys.append(key)
            for p, key in zip(self.predicate_bank, filtered_keys):
                kind_by_predicate_key[(p.op, p.vars, p.coeffs, p.const)] = kind_map.get(key, [])

        # build names with kind prefix for traceability
        self.X_fname = []
        for p in self.predicate_bank:
            if hasattr(p, "op") and hasattr(p, "vars") and hasattr(p, "coeffs") and hasattr(p, "const"):
                kinds = kind_by_predicate_key.get((p.op, p.vars, p.coeffs, p.const), [])
                kind = "+".join(sorted(kinds)) if kinds else None
                self.X_fname.append(predicate_to_str(p, kind=kind))
            else:
                self.X_fname.append(str(p))
        if X_df is not None:
            continuous_data = self._numeric_feature_frame(X_df.reset_index(drop=True), include_discrete_numeric=True)
            if not continuous_data.empty:
                self.imp.fit(continuous_data.values)
            self._prepare_continuous_data(continuous_data)
        else:
            self.continuous_source_names = []
        self.discrete_flen = len(self.predicate_bank)
        self.continuous_flen = len(self.continuous_source_names)
        self.X_fname.extend(self.continuous_source_names)

    def transform_predicates(self, X_df, y_df):
        from src.BinarizationLayer.jani_predicates import evaluate_predicates
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        if self.task == 'regression':
            y = y_df.values.astype(np.float32)
        else:
            y = self.label_enc.transform(y_df.values.reshape(-1, 1))
            if self.y_one_hot:
                y = y.toarray()
        X_pred = evaluate_predicates(X_df.values, self.predicate_bank, self.predicate_var_index)
        continuous_source = self._transform_continuous_frame(
            X_df,
            normalized=True,
            keep_stat=False,
            include_discrete_numeric=True,
        )
        if not continuous_source.empty:
            X = np.concatenate([X_pred, continuous_source.to_numpy(dtype=np.float32)], axis=1)
        else:
            X = X_pred
        return X, y

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float64)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        if self.task == 'regression':
            self.y_fname = list(y_df.columns)
        else:
            self.label_enc.fit(y_df)
            # self.y_fname = list(self.label_enc.get_feature_names(y_df.columns)) if self.y_one_hot else y_df.columns
            if self.y_one_hot:
                try:
                    self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns))
                except AttributeError:
                    self.y_fname = list(self.label_enc.get_feature_names(y_df.columns))
            else:
                self.y_fname = y_df.columns


        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
            self._prepare_continuous_data(continuous_data)
        else:
            self.continuous_source_names = []
            self.mean = None
            self.std = None
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = self._build_onehot_feature_names(feature_names)
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(self.continuous_source_names)
        else:
            self.X_fname = list(self.continuous_source_names)
            self.discrete_flen = 0
        self.continuous_flen = len(self.continuous_source_names)

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        if self.task == 'regression':
            y = y_df.values.astype(np.float32)
        else:
            # Encode string value to int index.
            y = self.label_enc.transform(y_df.values.reshape(-1, 1))
            if self.y_one_hot:
                y = y.toarray()

        continuous_source = self._transform_continuous_frame(
            X_df,
            normalized=normalized,
            keep_stat=keep_stat,
            include_discrete_numeric=False,
        )
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_source], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_source
        return X_df.values, y
