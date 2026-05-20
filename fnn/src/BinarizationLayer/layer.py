from collections import defaultdict

import torch
import torch.nn as nn


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, use_not=False, left=None, right=None):
        super(BinarizeLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.layer_type = 'binarization'
        self.rule_name = None

        self.register_buffer('left', left)
        self.register_buffer('right', right)

        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.dim2id = {i: i for i in range(self.output_dim)}

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)
            else:
                cl = torch.randn(self.n, self.input_dim[1])
            self.register_buffer('cl', cl)

    def __str__(self):
        lines = []
        lines.append(f"BinarizeLayer(n={self.n}, input_dim={tuple(self.input_dim)}, use_not={self.use_not})")
        disc_cnt, cont_cnt = int(self.input_dim[0]), int(self.input_dim[1])

        # --- Discrete predicates ---
        if disc_cnt > 0:
            lines.append("Discrete:")
            for i in range(disc_cnt):
                # try to use pretty names if rule_name already built by get_bound_name(...)
                name = None
                if self.rule_name is not None and len(self.rule_name) > i:
                    name = str(self.rule_name[i])
                else:
                    name = f"x[{i}]"
                lines.append(f"  - {name} == 1")
            if self.use_not:
                for i in range(disc_cnt):
                    # rule_name (if present) already contains '~name' entries for NOT
                    if self.rule_name is not None and len(self.rule_name) > disc_cnt + i:
                        lines.append(f"  - {self.rule_name[disc_cnt + i]}")
                    else:
                        lines.append(f"  - ¬x[{i}] (i.e., x[{i}] == 0)")

        # --- continuous predicates (threshold splits) ---
        if cont_cnt > 0:
            lines.append("Continuous:")
            # centers buffer exists only if cont_cnt > 0
            cl = getattr(self, "cl", None)
            if cl is None:
                lines.append("  (no centers registered)")
            else:
                cl_np = cl.detach().cpu().numpy()  # shape [n, cont_cnt]
                for j in range(cont_cnt):
                    # best-effort pretty feature name from rule_name tail (optional)
                    # otherwise use c[j]
                    fname = f"c[{j}]"
                    if self.rule_name is not None:
                        # rule_name for continuous was appended after discrete (and NOTs) as: all '>' then all '<='
                        # names carry the original feature name; extract it if available
                        for s in self.rule_name[disc_cnt * (2 if self.use_not else 1):]:
                            if isinstance(s, str) and s.split() and s.split()[0] != '~':
                                fname = s.split()[0]
                                break
                    centers = sorted(float(v) for v in cl_np[:, j].tolist())
                    lines.append(f"  - {fname}: cuts at [{', '.join(f'{c:.3f}' for c in centers)}]")
                    for c in centers:
                        lines.append(f"      >  {c:.3f}")
                    for c in centers:
                        lines.append(f"      ≤ {c:.3f}")
            # optional per-feature bounds if provided
            if self.left is not None and self.right is not None:
                L = self.left.detach().cpu().numpy()
                R = self.right.detach().cpu().numpy()
                if L.ndim == 1 and R.ndim == 1 and len(L) == cont_cnt and len(R) == cont_cnt:
                    for j in range(cont_cnt):
                        lines.append(f"    bounds[{j}]: [{float(L[j]):.3f}, {float(R[j]):.3f}]")
                else:
                    lines.append("    (global bounds present)")
        lines.append(f"output_dim = {self.output_dim}")
        return "\n".join(lines)

    def summary_size(self):
        size = int(self.disc_num)
        if self.input_dim[1] > 0:
            size += int(self.n) * int(self.input_dim[1])
        return int(size)

    def forward(self, x):
        if self.input_dim[1] > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
            x = x.unsqueeze(-1)
            binarize_res = Binarize.apply(x - self.cl.t()).view(x.shape[0], -1)
            return torch.cat((x_disc, binarize_res, 1. - binarize_res), dim=1)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(self.cl.data > self.right, self.right, self.cl.data)
            self.cl.data = torch.where(self.cl.data < self.left, self.left, self.cl.data)

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append('~' + feature_name[i])
        if self.input_dim[1] > 0:
            for c, op in [(self.cl, 'gt'), (self.cl, 'leq')]:
                c = c.detach().cpu().numpy()
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i]
                    for j in ci:
                        if mean is not None and std is not None:
                            j = j * std[fi_name] + mean[fi_name]
                        bound_name.append('{} {} {:.3f}'.format(fi_name, op, j))
        self.rule_name = bound_name
