import os
import logging
import json
import time
from pathlib import Path

import numpy as np
import torch
from src.utils import dump_csv_results

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from src.utils import read_csv, DBEncoder
from src.models import FNN
from src.validators import FidelityValidator

DEFAULT_DATASET_DIR = "./dataset"


def parse_structure(structure: str) -> tuple[int, list[int]]:
    parts = [int(p) for p in str(structure).split("@") if str(p).strip()]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid structure '{structure}'. Expected '<binarization_width>@<logical_width>[...]'."
        )
    binarization_width = int(parts[0])
    if binarization_width < 0:
        raise ValueError("binarization width must be >= 0")
    return binarization_width, parts[1:]


def _resolve_dataset_dir(data_dir=None) -> str:
    base = DEFAULT_DATASET_DIR if data_dir is None else data_dir
    return str(Path(os.path.expandvars(str(base))).expanduser())


def _load_iface_output_names(iface_path):
    obj = json.loads(Path(iface_path).read_text())
    out = obj.get("output", [])
    if out and isinstance(out[0], dict):
        return [str(e.get("name", e.get("id", e.get("class", f"class_{i}")))) for i, e in enumerate(out)]
    return [str(x) for x in out]


def get_data_loader(
    dataset,
    batch_size,
    world_size=1,
    rank=0,
    k=0,
    pin_memory=False,
    num_workers=0,
    persistent_workers=False,
    prefetch_factor=2,
    save_best=True,
    binarization="onehot",
    jani_path=None,
    iface_path=None,
    data_dir=None,
    binarization_width=0,
    predicate_bank=None,
    seed=42,
):
    """Load one generated training dataset from <dataset>.data and <dataset>.info."""
    _ = (world_size, rank)  # kept for backward-compatible callers
    base_data_dir = _resolve_dataset_dir(data_dir)
    data_path = os.path.join(base_data_dir, dataset + ".data")
    info_path = os.path.join(base_data_dir, dataset + ".info")
    X_df, y_df, f_df, label_pos, data_meta = read_csv(
        data_path,
        info_path,
        shuffle=True,
        random_state=seed,
        return_meta=True,
    )
    task = str(data_meta.get("task", "classification")).lower()

    bin_t0 = time.perf_counter()
    db_enc = DBEncoder(f_df, discrete=False, y_one_hot=(task != "regression"), task=task)
    db_enc.fit(X_df, y_df)
    if binarization == "onehot" and int(binarization_width) > 0 and int(db_enc.continuous_flen or 0) == 0:
        logging.warning(
            "binarization mode=onehot with a discrete problem ignores the first structure parameter (%s).",
            int(binarization_width),
        )

    if iface_path is not None:
        iface_outputs = _load_iface_output_names(iface_path)
        if len(iface_outputs) != len(db_enc.y_fname):
            raise ValueError(
                f"Interface outputs ({len(iface_outputs)}) != model outputs ({len(db_enc.y_fname)}). "
                "Cannot export symbolic model with action labels."
            )
        db_enc.y_fname = iface_outputs

    if binarization == "predicates":
        db_enc.fit_predicates(
            jani_path,
            X_df=X_df,
            iface_path=iface_path,
            predicate_bank=predicate_bank,
        )
        X, y = db_enc.transform_predicates(X_df, y_df)
    elif binarization == "onehot":
        X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    else:
        raise ValueError(f"Unknown binarization mode: {binarization}")
    db_enc.binarization_time_sec = time.perf_counter() - bin_t0

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    if task == "regression":
        train_set = TensorDataset(
            torch.tensor(X_train.astype(np.float32)),
            torch.tensor(y_train.astype(np.float32)),
        )
        test_set = TensorDataset(
            torch.tensor(X_test.astype(np.float32)),
            torch.tensor(y_test.astype(np.float32)),
        )
    else:
        train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)),
                                  torch.tensor(y_train.astype(np.float32)))
        test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)),
                                 torch.tensor(y_test.astype(np.float32)))

    train_len = int(len(train_set) * 0.80)
    split_generator = torch.Generator().manual_seed(seed)
    train_sub, valid_set = random_split(
        train_set,
        [train_len, len(train_set) - train_len],
        generator=split_generator,
    )

    if save_best:  # use validation set for model selections.
        train_set = train_sub

    use_persistent = persistent_workers and num_workers > 0
    loader_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = use_persistent
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, **loader_kwargs
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, **loader_kwargs
    )

    return db_enc, train_loader, valid_loader, test_loader


def train_model(args):
    use_gpu = torch.cuda.is_available() and getattr(args, "device_id", None) is not None
    seed = int(getattr(args, "seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_id = args.device_id if use_gpu else None
    if use_gpu:
        torch.cuda.set_device(device_id)
    if getattr(args, "debug_device", False):
        print(f"[device] cuda_available={torch.cuda.is_available()} "
              f"use_gpu={use_gpu} device_ids={args.device_ids} "
              f"device_id={device_id}")

    dataset = args.data_set
    binarization_width, logical_widths = parse_structure(args.structure)
    db_enc, train_loader, valid_loader, _ = get_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        world_size=args.world_size,
        rank=0,
        k=args.ith_kfold,
        pin_memory=use_gpu,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        save_best=args.save_best,
        binarization=args.binarization,
        jani_path=args.jani,
        iface_path=args.iface,
        data_dir=getattr(args, "data_dir", None),
        binarization_width=binarization_width,
        seed=seed,
    )
    args.binarization_time_sec = float(getattr(db_enc, "binarization_time_sec", 0.0))

    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen
    y_fname = db_enc.y_fname

    model_binarization_width = 0 if args.binarization == "predicates" else binarization_width
    fnn = FNN(
        dim_list=[(discrete_flen, continuous_flen)]
        + [model_binarization_width]
        + logical_widths
        + [len(y_fname)],
        task=db_enc.task,
        device_id=device_id,   # None => CPU
        use_not=args.use_not,
        is_rank0=True,
        log_file=args.log,
        save_best=args.save_best,
        estimated_grad=args.estimated_grad,
        use_skip=args.skip,
        save_path=args.model,
        use_nlaf=args.nlaf,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temp,
        regression_mode=getattr(args, "regression_mode", "mse"),
        run_name=dataset,
    )
    if args.binarization == "predicates":
        fnn.predicate_bank = [p.to_dict() for p in db_enc.predicate_bank]
    fnn.debug_device = bool(getattr(args, "debug_device", False))
    logging.info(
        "binarization: mode=%s time=%.3fs size=%s",
        args.binarization,
        float(getattr(db_enc, "binarization_time_sec", 0.0)),
        int(fnn.net.layer_list[0].summary_size()),
    )

    validator = FidelityValidator(valid_loader)

    fnn.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        validator=validator,
    )


def load_model(path, device_id, log_file=None):
    checkpoint = torch.load(path, map_location="cpu")
    saved_args = checkpoint["fnn_args"]

    fnn = FNN(
        dim_list=saved_args["dim_list"],
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args["use_not"],
        log_file=log_file,
        estimated_grad=saved_args["estimated_grad"],
        use_skip=saved_args["use_skip"],
        use_nlaf=saved_args["use_nlaf"],
        alpha=saved_args["alpha"],
        beta=saved_args["beta"],
        gamma=saved_args["gamma"],
        task=saved_args.get("task", "classification"),
        regression_mode=saved_args.get("regression_mode", "mse"),
        run_name=saved_args.get("run_name"),
    )

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    fnn.net.load_state_dict(new_state_dict, strict=True)
    fnn.predicate_bank = checkpoint.get("predicate_bank")
    return fnn


def test_model(args):
    dev = args.device_id
    fnn = load_model(args.model, dev, log_file=args.test_res)
    use_gpu = torch.cuda.is_available() and (dev is not None)
    predicate_bank = getattr(fnn, "predicate_bank", None)

    dataset = args.data_set
    db_enc, train_loader, _, test_loader = get_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        world_size=1,
        rank=0,
        k=args.ith_kfold,
        pin_memory=use_gpu,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        save_best=False,
        binarization=args.binarization,
        jani_path=args.jani,
        iface_path=args.iface,
        data_dir=getattr(args, "data_dir", None),
        predicate_bank=predicate_bank,
        seed=int(getattr(args, "seed", 42)),
    )
    args.binarization_time_sec = float(getattr(db_enc, "binarization_time_sec", 0.0))

    test_metrics = FidelityValidator(
        test_loader,
        set_name="Test",
        output_log=True,
    ).validate(-1, fnn)
    acc, f1 = test_metrics["val_acc"], test_metrics["val_f1"]
    fnn.model_info(
        db_enc.X_fname,
        db_enc.y_fname,
        train_loader,
        mean=db_enc.mean,
        std=db_enc.std,
    )
    dump_csv_results(Path(args.folder_path).resolve().parent.parent/"overall_validation.csv", fnn, acc, f1, args)

    if getattr(args, "export_symbolic", False):
        with open(args.symbolic_file, "w") as sym_file:
            fnn.export_symbolic(
                db_enc.X_fname, db_enc.y_fname, train_loader,
                file=sym_file, mean=db_enc.mean, std=db_enc.std,
                atoms_type=args.binarization
            )

if __name__ == "__main__":
    from args import fnn_args
    train_model(fnn_args)
    test_model(fnn_args)
