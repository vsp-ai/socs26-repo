from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_CMP_OP_MAP = {
    "=": "eq",
    "!=": "neq",
    "≠": "neq",
    "<": "lt",
    "<=": "leq",
    "≤": "leq",
    ">": "gt",
    ">=": "geq",
    "≥": "geq",
}

_LOGICAL_OPS = {"∧", "∨", "and", "or"}
_FLOAT_EPS = 1e-9


@dataclass(frozen=True)
class Predicate:
    op: str
    vars: Tuple[str, ...]
    coeffs: Tuple[float, ...]
    const: float

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "vars": list(self.vars),
            "coeffs": [_normalize_number(v) for v in self.coeffs],
            "const": _normalize_number(self.const),
        }


_OP_SWAP = {
    "lt": "gt",
    "leq": "geq",
    "gt": "lt",
    "geq": "leq",
}


def _normalize_predicate(pred: Predicate) -> Predicate:
    # Ensure leading coefficient is positive by flipping sign if needed.
    # For inequalities, swap direction when flipping sign.
    if not pred.coeffs:
        return pred
    lead = pred.coeffs[0]
    if lead >= 0:
        return pred
    new_coeffs = tuple(-c for c in pred.coeffs)
    new_const = -pred.const
    new_op = _OP_SWAP.get(pred.op, pred.op)
    return Predicate(op=new_op, vars=pred.vars, coeffs=new_coeffs, const=new_const)


_COMPLEMENT_OP = {
    "lt": "geq",
    "leq": "gt",
    "gt": "leq",
    "geq": "lt",
}


def _complement_predicate(pred: Predicate) -> Predicate | None:
    # Only complement inequalities (keeps size controlled).
    comp = _COMPLEMENT_OP.get(pred.op)
    if comp is None:
        return None
    return Predicate(op=comp, vars=pred.vars, coeffs=pred.coeffs, const=pred.const)


def _to_int_if_possible(x):
    if isinstance(x, (int, float)):
        if int(x) == x:
            return int(x)
    return x


def _normalize_number(x: float):
    xf = float(x)
    if abs(xf) < _FLOAT_EPS:
        return 0
    nearest = round(xf)
    if abs(xf - nearest) < _FLOAT_EPS:
        return int(nearest)
    return round(xf, 12)


def _is_zero(x: float) -> bool:
    return abs(float(x)) < _FLOAT_EPS


def _add_coeffs(a: Dict[str, float], b: Dict[str, float], scale: float = 1.0) -> Dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + scale * v
    return {k: _normalize_number(v) for k, v in out.items() if not _is_zero(v)}


def _scale_coeffs(a: Dict[str, float], scale: float) -> Dict[str, float]:
    return {k: _normalize_number(v * scale) for k, v in a.items() if not _is_zero(v * scale)}


def _resolve_numeric_constants(jani_obj: dict) -> Dict[str, float]:
    model = jani_obj.get("model", jani_obj)
    raw = {}
    for const in model.get("constants", []):
        name = const.get("name")
        if isinstance(name, str) and "value" in const:
            raw[name] = const["value"]

    resolved: Dict[str, float] = {}
    visiting: set[str] = set()

    def eval_value(exp):
        if isinstance(exp, bool):
            return exp
        if isinstance(exp, (int, float)):
            return _normalize_number(exp)
        if isinstance(exp, str):
            return resolve_name(exp)
        if isinstance(exp, dict):
            if "value" in exp:
                return eval_value(exp["value"])
            if "ref" in exp and isinstance(exp["ref"], str):
                return resolve_name(exp["ref"])
            if "exp" in exp:
                return eval_value(exp["exp"])
            op = exp.get("op")
            if op in ("+", "-", "*", "/"):
                left = eval_value(exp["left"])
                right = eval_value(exp["right"])
                if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                    raise ValueError(f"Unsupported non-numeric constant expression: {exp}")
                if op == "+":
                    return _normalize_number(left + right)
                if op == "-":
                    return _normalize_number(left - right)
                if op == "*":
                    return _normalize_number(left * right)
                if _is_zero(right):
                    raise ValueError(f"Division by zero in constant expression: {exp}")
                return _normalize_number(left / right)
        raise ValueError(f"Unsupported constant expression: {exp}")

    def resolve_name(name: str):
        if name in resolved:
            return resolved[name]
        if name not in raw:
            raise KeyError(name)
        if name in visiting:
            raise ValueError(f"Cyclic constant definition involving '{name}'")
        visiting.add(name)
        value = eval_value(raw[name])
        visiting.remove(name)
        resolved[name] = value
        return value

    for name in list(raw):
        try:
            value = resolve_name(name)
        except Exception:
            continue
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            resolved[name] = _normalize_number(value)

    return resolved


def _expr_to_linear(exp, consts: Dict[str, float] | None = None) -> Tuple[Dict[str, float], float]:
    # Returns (coeffs, const) for linear expressions.
    consts = consts or {}
    if isinstance(exp, (int, float)):
        return {}, float(exp)
    if isinstance(exp, str):
        if exp in consts:
            return {}, float(consts[exp])
        return {exp: 1}, 0
    if isinstance(exp, dict):
        if "ref" in exp and isinstance(exp["ref"], str):
            if exp["ref"] in consts:
                return {}, float(consts[exp["ref"]])
            return {exp["ref"]: 1}, 0
        if "exp" in exp:
            return _expr_to_linear(exp["exp"], consts)
        op = exp.get("op")
        if op is None:
            # Sometimes value literals are wrapped.
            if "value" in exp:
                return _expr_to_linear(exp["value"], consts)
            raise ValueError(f"Unsupported expression dict: {exp}")
        if op in ("+", "-"):
            left_c, left_k = _expr_to_linear(exp["left"], consts)
            right_c, right_k = _expr_to_linear(exp["right"], consts)
            if op == "+":
                return _add_coeffs(left_c, right_c), _normalize_number(left_k + right_k)
            return _add_coeffs(left_c, right_c, scale=-1), _normalize_number(left_k - right_k)
        if op == "*":
            left_c, left_k = _expr_to_linear(exp["left"], consts)
            right_c, right_k = _expr_to_linear(exp["right"], consts)
            # Allow only constant * linear
            if left_c and right_c:
                raise ValueError(f"Non-linear multiplication: {exp}")
            if not left_c and not right_c:
                return {}, _normalize_number(left_k * right_k)
            if left_c:
                return _scale_coeffs(left_c, right_k), _normalize_number(left_k * right_k)
            return _scale_coeffs(right_c, left_k), _normalize_number(left_k * right_k)
        if op == "/":
            left_c, left_k = _expr_to_linear(exp["left"], consts)
            right_c, right_k = _expr_to_linear(exp["right"], consts)
            if right_c:
                raise ValueError(f"Non-linear division: {exp}")
            if _is_zero(right_k):
                raise ValueError(f"Division by zero in expression: {exp}")
            if not left_c:
                return {}, _normalize_number(left_k / right_k)
            return _scale_coeffs(left_c, 1.0 / right_k), _normalize_number(left_k / right_k)
        # If this is a comparison node, caller should handle it.
        return _expr_to_linear(exp.get("left"), consts) if "left" in exp else ({}, 0)
    raise ValueError(f"Unsupported expression type: {type(exp)}")


def _collect_atoms(exp, out: List[Predicate], consts: Dict[str, float] | None = None):
    if isinstance(exp, dict) and "op" in exp and "left" in exp and "right" in exp:
        op = exp["op"]
        if op in _LOGICAL_OPS:
            _collect_atoms(exp["left"], out, consts)
            _collect_atoms(exp["right"], out, consts)
            return
        if op in _CMP_OP_MAP:
            left_c, left_k = _expr_to_linear(exp["left"], consts)
            right_c, right_k = _expr_to_linear(exp["right"], consts)
            # Move variables to LHS, constants to RHS
            coeffs = _add_coeffs(left_c, right_c, scale=-1)
            const = _normalize_number(right_k - left_k)
            if not coeffs:
                return
            items = sorted(coeffs.items())
            vars_ = tuple(k for k, _ in items)
            coeffs_ = tuple(_normalize_number(v) for _, v in items)
            pred = Predicate(op=_CMP_OP_MAP[op], vars=vars_, coeffs=coeffs_, const=const)
            pred = _normalize_predicate(pred)
            out.append(pred)
            return
    if isinstance(exp, dict) and "exp" in exp:
        _collect_atoms(exp["exp"], out, consts)


def load_jani(path: Path) -> dict:
    return json.loads(path.read_text())


def _expand_file_refs(obj, base_dir: Path, visited: set[Path] | None = None):
    """Recursively resolve include nodes like {"file": "child.jani"}."""
    if visited is None:
        visited = set()

    if isinstance(obj, dict):
        # Only treat a pure {"file": "..."} object as an include node.
        if set(obj.keys()) == {"file"} and isinstance(obj.get("file"), str):
            include_path = (base_dir / obj["file"]).resolve()
            if include_path in visited:
                raise ValueError(f"Cyclic property include detected: {include_path}")
            if not include_path.exists():
                raise FileNotFoundError(f"Included property file not found: {include_path}")

            visited.add(include_path)
            included_obj = json.loads(include_path.read_text())
            resolved = _expand_file_refs(included_obj, include_path.parent, visited)
            visited.remove(include_path)
            return resolved

        return {k: _expand_file_refs(v, base_dir, visited) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_expand_file_refs(item, base_dir, visited) for item in obj]

    return obj


def iter_guard_exprs(jani_obj: dict) -> Iterable[dict]:
    model = jani_obj.get("model", jani_obj)
    for automaton in model.get("automata", []):
        for edge in automaton.get("edges", []):
            guard = edge.get("guard")
            if guard:
                yield guard.get("exp", guard)


def _iter_property_exprs(prop_obj: dict, key: str) -> Iterable[dict]:
    if isinstance(prop_obj, dict):
        if key in prop_obj and isinstance(prop_obj[key], dict):
            exp = prop_obj[key].get("exp")
            if exp is not None:
                yield exp
        for v in prop_obj.values():
            yield from _iter_property_exprs(v, key)
    elif isinstance(prop_obj, list):
        for item in prop_obj:
            yield from _iter_property_exprs(item, key)


def iter_property_exprs(prop_obj: dict) -> Iterable[tuple[str, dict]]:
    # goal: objective/goal expressions
    for exp in _iter_property_exprs(prop_obj, "goal"):
        yield "goal", exp
    # fail/budget-like: reach or avoid expressions
    for exp in _iter_property_exprs(prop_obj, "reach"):
        yield "fail", exp
    for exp in _iter_property_exprs(prop_obj, "avoid"):
        yield "fail", exp


def parse_guard_predicates(jani_path: Path) -> List[Predicate]:
    j = load_jani(jani_path)
    consts = _resolve_numeric_constants(j)
    atoms: List[Predicate] = []
    for exp in iter_guard_exprs(j):
        _collect_atoms(exp, atoms, consts)
    # Deduplicate by structural key
    uniq = {}
    for p in atoms:
        key = (p.op, p.vars, p.coeffs, p.const)
        uniq[key] = p
    return [uniq[k] for k in sorted(uniq.keys())]


def parse_problem_predicates(
    jani_path: Path,
    prop_path: Path | None = None,
    include_goal: bool = True,
    include_fail: bool = True,
    include_fail_complements: bool = True,
) -> tuple[List[Predicate], Dict[Tuple[str, Tuple[str, ...], Tuple[int, ...], int], List[str]]]:
    j = load_jani(jani_path)
    consts = _resolve_numeric_constants(j)
    atoms: List[tuple[Predicate, str]] = []

    # Guards
    for exp in iter_guard_exprs(j):
        tmp: List[Predicate] = []
        _collect_atoms(exp, tmp, consts)
        atoms.extend((p, "guard") for p in tmp)

    # Properties (goal/fail)
    prop_obj = None
    if prop_path is not None and Path(prop_path).exists():
        prop_path = Path(prop_path)
        prop_obj = json.loads(prop_path.read_text())
        prop_obj = _expand_file_refs(prop_obj, prop_path.parent)
    elif "properties" in j:
        prop_obj = _expand_file_refs(j, jani_path.parent)

    if prop_obj:
        for kind, exp in iter_property_exprs(prop_obj):
            if kind == "goal" and not include_goal:
                continue
            if kind == "fail" and not include_fail:
                continue
            tmp: List[Predicate] = []
            _collect_atoms(exp, tmp, consts)
            atoms.extend((p, kind) for p in tmp)
            if kind == "fail" and include_fail_complements:
                for p in tmp:
                    comp = _complement_predicate(p)
                    if comp is not None:
                        atoms.append((comp, "fail"))

    # Deduplicate with kind tracking
    uniq: Dict[Tuple[str, Tuple[str, ...], Tuple[int, ...], int], Predicate] = {}
    kinds: Dict[Tuple[str, Tuple[str, ...], Tuple[int, ...], int], List[str]] = {}
    for p, kind in atoms:
        key = (p.op, p.vars, p.coeffs, p.const)
        uniq[key] = p
        kinds.setdefault(key, [])
        if kind not in kinds[key]:
            kinds[key].append(kind)
    ordered_keys = sorted(uniq.keys())
    return [uniq[k] for k in ordered_keys], {k: kinds[k] for k in ordered_keys}


def predicate_to_str(p: Predicate, kind: str | None = None) -> str:
    def fmt_num(x):
        x = _normalize_number(x)
        return str(x)

    # Build a readable linear expression like "x - y + 2"
    parts = []
    for var, coeff in zip(p.vars, p.coeffs):
        if coeff == 1:
            term = var
        elif coeff == -1:
            term = f"-{var}"
        else:
            term = f"{fmt_num(coeff)}*{var}"
        parts.append(term)
    if not parts:
        lhs = "0"
    else:
        lhs = parts[0]
        for term in parts[1:]:
            if term.startswith("-"):
                lhs += f" - {term[1:]}"
            else:
                lhs += f" + {term}"
    expr = f"{lhs} {p.op} {fmt_num(p.const)}"
    if kind:
        return f"{kind}:{expr}"
    return expr


def evaluate_predicates(X, predicates: List[Predicate], var_index: Dict[str, int]):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros((X.shape[0], len(predicates)), dtype=np.float32)
    for i, p in enumerate(predicates):
        vals = np.zeros(X.shape[0], dtype=np.float64)
        for var, coeff in zip(p.vars, p.coeffs):
            idx = var_index[var]
            vals = vals + coeff * X[:, idx]
        if p.op == "eq":
            mask = np.isclose(vals, p.const, atol=_FLOAT_EPS, rtol=_FLOAT_EPS)
        elif p.op == "neq":
            mask = ~np.isclose(vals, p.const, atol=_FLOAT_EPS, rtol=_FLOAT_EPS)
        elif p.op == "leq":
            mask = vals <= (p.const + _FLOAT_EPS)
        elif p.op == "geq":
            mask = vals >= (p.const - _FLOAT_EPS)
        elif p.op == "lt":
            mask = vals < (p.const - _FLOAT_EPS)
        elif p.op == "gt":
            mask = vals > (p.const + _FLOAT_EPS)
        else:
            raise ValueError(f"Unknown predicate op: {p.op}")
        out[:, i] = mask.astype(np.float32)
    return out
