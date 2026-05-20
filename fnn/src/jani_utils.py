from __future__ import annotations

import json
from pathlib import Path


def resolve_bounds(t, const_map):
    def to_int(x):
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s.lstrip("-").isdigit():
                return int(s)
            return int(const_map[s])
        if isinstance(x, dict) and isinstance(x.get("ref"), str):
            return int(const_map[x["ref"]])
        raise ValueError(f"Unsupported bound: {x!r}")
    lo_raw = t.get("lower-bound", t.get("lower"))
    hi_raw = t.get("upper-bound", t.get("upper"))
    return to_int(lo_raw), to_int(hi_raw)


def load_jani_bounds(p: Path):
    j = json.loads(p.read_text())
    const_list = j.get("constants") or (j.get("model", {}) or {}).get("constants") or []
    const_map = {c["name"]: c.get("value") for c in const_list if "name" in c}

    vars_list = j.get("variables") or (j.get("model", {}) or {}).get("variables") or []
    out = {}

    for v in vars_list:
        name, t = v.get("name"), v.get("type")
        lo = hi = None
        if isinstance(t, dict) and t.get("kind") == "bounded":
            lo, hi = resolve_bounds(t, const_map)
            if lo is None and hi is None:
                raise SystemExit(f"Cannot resolve bounds for variable {name!r} in {p}")

        # elif (isinstance(t, dict) and t.get("kind") == "bool") or t == "bool":
        #     lo, hi = 0, 1
        out[name] = (lo, hi)

    return out
