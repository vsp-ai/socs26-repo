import os
import argparse
import torch

from src.losses import regression_loss_names


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_default_device = "cuda" if torch.cuda.is_available() else "cpu"
parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument(
    '--data-dir',
    type=str,
    default='./dataset',
    help='Directory containing <dataset>.data/.info/.auginfo.json files.'
)
parser.add_argument('-i', '--device_ids', type=str, default=None, help='Set the device (GPU ids). Split by @.'
                                                                       ' E.g., 0 or 0@2@3 (only the first id is used per run).')
parser.add_argument('--device', type=str, default=_default_device,
                    help='Device to use: "cuda", "cuda:<id>", "cpu", or "auto". '
                         'Defaults to cuda if available else cpu.')
parser.add_argument('--no-gpu', dest='no_gpu', action="store_true",
                    help='Force CPU-only execution (ignore CUDA even if available).')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-e', '--epoch', type=int, default=41, help='Set the total epoch.')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Set the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Set the initial learning rate.')
parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=10, help='Set the learning rate decay epoch.')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Set the weight decay (L2 penalty).')
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset shuffling, splits, and training.')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1', help='Set the master address.')
parser.add_argument('-mp', '--master_port', type=str, default='0', help='Set the master port.')
parser.add_argument('--nlaf', action="store_true",
                    help='Use novel logical activation functions to take less time and GPU memory usage. We recommend trying (alpha, beta, gamma) in {(0.999, 8, 1), (0.999, 8, 3), (0.9, 3, 3)}')
parser.add_argument('--alpha', type=float, default=0.999, help='Set the alpha for NLAF.')
parser.add_argument('--beta', type=int, default=8, help='Set the beta for NLAF.')
parser.add_argument('--gamma', type=int, default=1, help='Set the gamma for NLAF.')

parser.add_argument('--temp', type=float, default=1.0, help='Set the temperature.')

parser.add_argument('--use_not', action="store_true",
                    help='Use the NOT (~) operator in logical rules. '
                         'It will enhance model capability but make the FNN more complex.')
parser.add_argument('--save_best', action="store_true",
                    help='Save the model with best performance on the validation set.')
parser.add_argument('--skip', action="store_true",
                    help='Use skip connections when the number of logical layers is greater than 2.')
parser.add_argument('--estimated_grad', action="store_true",
                    help='Use estimated gradient.')
parser.add_argument('--weighted', action="store_true",
                    help='Use weighted loss for imbalanced data.')
parser.add_argument('--regression-mode', type=str, default='mse',
                    choices=regression_loss_names(),
                    help='Regression training objective.')
parser.add_argument('--export_symbolic', action="store_true",
                    help='Export the symbolic model as JSON.')
parser.add_argument('--debug_device', '--debug-device', dest='debug_device', action="store_true",
                    help='Print device and CUDA memory info for the first batch.')
parser.add_argument('-s', '--structure', type=str, default='5@64',
                    help='Set the one-hot threshold width and logical-layer widths. '
                         'E.g., 10@64 or 10@64@32@16 means 10 threshold centers per continuous feature, then logical widths 64, 32, 16. '
                         'Use 0 as the first value with --binarization predicates.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader worker processes. Use 0 for in-process loading.')
parser.add_argument('--prefetch_factor', type=int, default=2,
                    help='DataLoader prefetch factor per worker (only when num_workers > 0).')
parser.add_argument('--persistent_workers', action="store_true",
                    help='Keep DataLoader workers alive between epochs (requires num_workers > 0).')

parser.add_argument(
    "--folder_name",
    type=str,
    default=None,
    help="Override the auto-generated run folder name (subdir inside log_folder/<dataset>/)."
)

parser.add_argument(
    "--validator",
    type=str,
    default="fidelity",
    help="Deprecated. Training uses fidelity validation only."
)
parser.add_argument(
    "--binarization",
    type=str,
    default="onehot",
    choices=["onehot", "predicates"],
    help="Input binarization mode. 'onehot' uses the standard encoder; 'predicates' uses predicates parsed from the JANI model/property."
)
parser.add_argument("--jani", type=str, default=None,
                    help="Path to model JANI file for PlaJA.")
parser.add_argument("--iface", type=str, default=None,
                    help="Path to JANI2NNET interface file for PlaJA.")
parser.add_argument("--prop", type=str, default=None,
                    help="Deprecated. No longer used for training-time validation.")


fnn_args = parser.parse_args()
fnn_args.data_dir = os.path.expandvars(os.path.expanduser(fnn_args.data_dir))
fnn_args.regression_mode = str(fnn_args.regression_mode).strip().lower().replace("-", "_").replace(" ", "_")

if fnn_args.binarization == "predicates":
    if not (fnn_args.jani and fnn_args.iface):
        parser.error("--binarization predicates requires --jani and --iface")

if fnn_args.validator != "fidelity":
    print(
        f"[warn] --validator={fnn_args.validator!r} is deprecated. "
        "Training uses fidelity validation only."
    )
    fnn_args.validator = "fidelity"

_auto_name = '{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_ki{}_seed{}_rc{}_useNOT{}_saveBest{}_useNLAF{}_estimatedGrad{}_useSkip{}_alpha{}_beta{}_gamma{}_temp{}_regMode{}'.format(
    fnn_args.data_set, fnn_args.epoch, fnn_args.batch_size, fnn_args.learning_rate, fnn_args.lr_decay_rate,
    fnn_args.lr_decay_epoch, fnn_args.weight_decay, fnn_args.ith_kfold, fnn_args.seed, fnn_args.round_count, fnn_args.use_not,
    fnn_args.save_best, fnn_args.nlaf, fnn_args.estimated_grad, fnn_args.skip,
    fnn_args.alpha, fnn_args.beta, fnn_args.gamma, fnn_args.temp, fnn_args.regression_mode)

# If user passed --folder_name, use it as-is; else append structure to the auto name
fnn_args.folder_name = fnn_args.folder_name if fnn_args.folder_name else (_auto_name + '_L' + fnn_args.structure)


if not os.path.exists('log_folder'):
    os.mkdir('log_folder')
fnn_args.set_folder_path = os.path.join('log_folder', fnn_args.data_set)
if not os.path.exists(fnn_args.set_folder_path):
    os.mkdir(fnn_args.set_folder_path)
fnn_args.folder_path = os.path.join(fnn_args.set_folder_path, fnn_args.folder_name)
if not os.path.exists(fnn_args.folder_path):
    os.mkdir(fnn_args.folder_path)
fnn_args.model = os.path.join(fnn_args.folder_path, 'model.pth')
fnn_args.symbolic_file = os.path.join(fnn_args.folder_path, 'sym_model.json')
fnn_args.log = os.path.join(fnn_args.folder_path, 'log.txt')
fnn_args.test_res = os.path.join(fnn_args.folder_path, 'test_res.txt')
#fnn_args.device_ids = list(map(int, fnn_args.device_ids.strip().split('@')))


# ---- Device config (single-process / single-GPU-per-run) ----
device_opt = (fnn_args.device or "").strip().lower()
if fnn_args.no_gpu:
    device_opt = "cpu"
if device_opt in ("", "auto"):
    device_opt = _default_device

if isinstance(fnn_args.device_ids, str) and fnn_args.device_ids.strip():
    parsed_ids = list(map(int, fnn_args.device_ids.strip().split('@')))
else:
    parsed_ids = []

if len(parsed_ids) > 1:
    print(
        f"[warn] Multiple GPU ids were provided ({parsed_ids}). "
        "Only the first id is used per run. "
        "Launch separate processes for independent multi-GPU training."
    )
    parsed_ids = [parsed_ids[0]]

if device_opt.startswith("cpu"):
    fnn_args.device_ids = []
elif device_opt.startswith("cuda"):
    if not torch.cuda.is_available():
        fnn_args.device_ids = []
    elif parsed_ids:
        fnn_args.device_ids = parsed_ids
    else:
        dev_idx = 0
        if ":" in device_opt:
            try:
                dev_idx = int(device_opt.split(":", 1)[1])
            except ValueError:
                dev_idx = 0
        fnn_args.device_ids = [dev_idx]
else:
    # fallback to parsed ids or default device if available
    if parsed_ids:
        fnn_args.device_ids = parsed_ids
    elif torch.cuda.is_available():
        fnn_args.device_ids = [0]
    else:
        fnn_args.device_ids = []

# validate selected GPU id; fallback to CPU on invalid/unavailable GPU
if fnn_args.device_ids:
    selected_id = fnn_args.device_ids[0]
    if not torch.cuda.is_available():
        print(
            f"[warn] Requested GPU id {selected_id} but CUDA is not available. "
            "Falling back to CPU."
        )
        fnn_args.device_ids = []
    else:
        dev_count = torch.cuda.device_count()
        if selected_id < 0 or selected_id >= dev_count:
            avail = "none" if dev_count <= 0 else f"0..{dev_count - 1}"
            print(
                f"[warn] Requested GPU id {selected_id} is not available "
                f"(available: {avail}). Falling back to CPU."
            )
            fnn_args.device_ids = []

# number of GPUs requested on CLI
fnn_args.device_id = fnn_args.device_ids[0] if fnn_args.device_ids else None
fnn_args.gpus = 1 if fnn_args.device_id is not None else 0
fnn_args.world_size = 1
