from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SymbolicBuildMeta:
    edges_upper_bound: int
    edges_alive_raw: int
    edges_exported: int
    exported_rule_names: list
    rub: int
    rrelevant: int


@dataclass
class SymbolicModel:
    atoms: list
    rules: dict
    linear: dict
    meta: SymbolicBuildMeta


def _ensure_activation_counts(fnn, train_loader):
    if fnn.net.layer_list[1] is None and train_loader is None:
        raise Exception("Need train_loader for the dead nodes detection.")
    if fnn.net.layer_list[1].node_activation_cnt is None:
        fnn.detect_dead_node(train_loader)


def build_symbolic_model(fnn, feature_name, label_name, train_loader, mean=None, std=None):
    _ensure_activation_counts(fnn, train_loader)

    layers = fnn.net.layer_list
    bin_layer = layers[0]
    logical_layers = layers[1:-1]
    lr_layer = layers[-1]

    bin_layer.get_bound_name(feature_name, mean, std)
    atoms = [str(s) for s in bin_layer.rule_name]

    rules_json = {}
    prev = bin_layer

    edges_upper_bound = 0
    edges_alive_raw = 0
    edges_exported = 0

    def resolve_operand(layer_idx: int, key):
        k, rid = key
        is_not = k > 0
        src = abs(k)

        def maybe_not(payload):
            return {"NOT": payload} if is_not else payload

        if src == 2:
            return maybe_not(f"L{layer_idx - 1}_{rid}")

        if layer_idx == 0:
            base = atoms[rid] if 0 <= rid < len(atoms) else f"atom[{rid}]"
            return maybe_not(base)

        return maybe_not(f"L{layer_idx}_{rid}")

    for i, layer in enumerate(logical_layers, start=1):
        edges_upper_bound += int(layer.con_layer.W.numel() + layer.dis_layer.W.numel())
        edges_alive_raw += int(layer.edge_count())
        layer.get_rules(prev, layer.conn.skip_from_layer)
        con_list, dis_list = layer.rule_list

        for j, rule_tuple in enumerate(con_list):
            operands = [resolve_operand(i - 1, k) for k in rule_tuple]
            rules_json[f"L{i}_{j}"] = {"AND": operands}
            edges_exported += len(rule_tuple)

        base = len(con_list)
        for j, rule_tuple in enumerate(dis_list):
            operands = [resolve_operand(i - 1, k) for k in rule_tuple]
            rules_json[f"L{i}_{base + j}"] = {"OR": operands}
            edges_exported += len(rule_tuple)

        prev = layer

    lr_layer.get_rule2weights(lr_layer.conn.prev_layer, lr_layer.conn.skip_from_layer)
    lr_weights = lr_layer.fc1.weight

    edges_upper_bound += int(lr_weights.numel())
    edges_alive_raw += int((lr_weights != 0).sum().item())
    edges_exported += int(len(lr_layer.rule2weights) * len(label_name))

    last_idx = len(logical_layers)

    def name_for_rid(rid):
        shift, j = rid
        if shift == -1:
            return f"L{last_idx}_{j}"
        if shift == -2:
            return f"L{last_idx - 1}_{j}"
        return f"RID{shift}_{j}"

    exported_rids = [rid for rid, _ in lr_layer.rule2weights]
    exported_rule_names = [name_for_rid(rid) for rid in exported_rids]

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if hasattr(lr_layer, "bl") and lr_layer.bl is not None:
        bias = to_numpy(lr_layer.bl)
    else:
        bias = to_numpy(lr_layer.fc1.bias)

    linear = {
        "bias": {label_name[a]: float(bias[a]) for a in range(len(label_name))},
        "weights": {label_name[li]: {} for li in range(len(label_name))},
    }

    for rid, weights in lr_layer.rule2weights:
        rule_name = str(name_for_rid(rid))
        for li, label in enumerate(label_name):
            linear["weights"][label][rule_name] = float(weights[li])

    meta = SymbolicBuildMeta(
        edges_upper_bound=int(edges_upper_bound),
        edges_alive_raw=int(edges_alive_raw),
        edges_exported=int(edges_exported),
        exported_rule_names=exported_rule_names,
        rub=int(lr_layer.fc1.weight.shape[1]),
        rrelevant=int(len(lr_layer.rule2weights)),
    )
    return SymbolicModel(atoms=atoms, rules=rules_json, linear=linear, meta=meta)


def compute_symbolic_stats(fnn, symbolic_model: SymbolicModel):
    rules_json = symbolic_model.rules
    atoms = symbolic_model.atoms
    meta = symbolic_model.meta

    def count_atomic_literals(expr, memo):
        if isinstance(expr, str):
            if expr.startswith("L"):
                if expr in memo:
                    return memo[expr]
                if expr not in rules_json:
                    memo[expr] = 0
                    return 0
                val = count_atomic_literals(rules_json[expr], memo)
                memo[expr] = val
                return val
            return 1

        if isinstance(expr, dict):
            if "NOT" in expr:
                return count_atomic_literals(expr["NOT"], memo)
            if "AND" in expr:
                return sum(count_atomic_literals(x, memo) for x in expr["AND"])
            if "OR" in expr:
                return sum(count_atomic_literals(x, memo) for x in expr["OR"])
            return 0

        raise ValueError("Invalid expression type for counting atomic literals.")

    memo = {}
    exported_rule_lens = [count_atomic_literals(rule_name, memo) for rule_name in meta.exported_rule_names]
    n_rules = len(exported_rule_lens)
    total_literals = sum(exported_rule_lens)
    max_literals_per_rule = max(exported_rule_lens, default=0)
    avg_literals_per_rule = (total_literals / n_rules) if n_rules else 0.0
    edge_ratio_disc = (meta.edges_alive_raw / meta.edges_upper_bound) if meta.edges_upper_bound else 0.0
    edge_ratio_alive = (meta.edges_exported / meta.edges_upper_bound) if meta.edges_upper_bound else 0.0

    def r4(x):
        return round(float(x), 4)

    return {
        "Eub": int(meta.edges_upper_bound),
        "Edisc": int(meta.edges_alive_raw),
        "Ealive": int(meta.edges_exported),
        "Er1": r4(edge_ratio_disc),
        "Er2": r4(edge_ratio_alive),
        "Rub": int(meta.rub),
        "Rrelevant": int(meta.rrelevant),
        "rules_count": int(n_rules),
        "avg_LpR": r4(avg_literals_per_rule),
        "max_LpR": int(max_literals_per_rule),
        "n_atoms": int(len(atoms)),
        "binarization_size": int(fnn.net.layer_list[0].summary_size()),
    }


def serialize_symbolic(symbolic_model: SymbolicModel, atoms_type=None):
    atoms = symbolic_model.atoms
    return {
        "atoms": {
            "type": atoms_type or "unknown",
            "len": int(len(atoms)),
            "content": atoms,
        },
        "rules": symbolic_model.rules,
        "linear": symbolic_model.linear,
    }


def model_info(fnn, feature_name, label_name, train_loader, mean=None, std=None):
    symbolic_model = build_symbolic_model(fnn, feature_name, label_name, train_loader, mean=mean, std=std)
    stats = compute_symbolic_stats(fnn, symbolic_model)
    fnn.model_stats = stats
    return stats


def export_symbolic(
    fnn,
    feature_name,
    label_name,
    train_loader,
    file,
    mean=None,
    std=None,
    atoms_type=None,
):
    symbolic_model = build_symbolic_model(fnn, feature_name, label_name, train_loader, mean=mean, std=std)
    out = serialize_symbolic(symbolic_model, atoms_type=atoms_type)
    print(json.dumps(out, indent=2, ensure_ascii=False), file=file)
    return out
