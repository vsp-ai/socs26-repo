import sys
import logging
import numpy as np
import torch
import torch.nn as nn

import time

from src.BinarizationLayer import BinarizeLayer
from src.components import LogicalLayer, LRLayer
from src.losses import get_classification_loss, get_regression_loss
from src import symbolic_model
from src.validators import BaseValidator


class Net(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, use_nlaf=False, estimated_grad=False, use_skip=True, alpha=0.999, beta=8, gamma=1, temperature=0.01):
        super(Net, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])
        self.use_skip = use_skip
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            
            skip_from_layer = None
            if self.use_skip and i >= 4:
                skip_from_layer = self.layer_list[-2]
                num += skip_from_layer.output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            else:
                # The first logical layer does not use NOT if the binarization layer has already used NOT
                #layer_use_not = True if i != 2 else False
                layer_use_not = use_not
                layer = LogicalLayer(dim_list[i], num, use_nlaf=use_nlaf, estimated_grad=estimated_grad, use_not=layer_use_not, alpha=alpha, beta=beta, gamma=gamma)
                layer_name = 'logical{}'.format(i)
            
            layer.conn = lambda: None  # create an empty class to save the connections
            layer.conn.prev_layer = self.layer_list[-1] if len(self.layer_list) > 0 else None
            layer.conn.is_skip_to_layer = False
            layer.conn.skip_from_layer = skip_from_layer
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True

            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
        return x
    
    def bi_forward(self, x, count=False):
        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer.binarized_forward(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
            if count and layer.layer_type != 'linear':
                layer.node_activation_cnt += torch.sum(x, dim=0)
                layer.forward_tot += x.shape[0]
        return x


class FNN:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, left=None,
                 right=None, save_best=False, estimated_grad=False, save_path=None, use_skip=False, 
                 use_nlaf=False, alpha=0.999, beta=8, gamma=1, temperature=0.01, task="classification",
                 regression_mode="mse", run_name=None):


        super().__init__()
        # hyperparameters
        self.dim_list = dim_list
        self.use_not = use_not
        self.use_skip = use_skip
        self.use_nlaf = use_nlaf
        self.alpha =alpha
        self.beta = beta
        self.gamma = gamma
        self.best_f1 = -1.
        self.best_loss = 1e20
        self.task = str(task).lower()
        self.run_name = str(run_name) if run_name is not None else "unknown"
        self.regression_mode = str(regression_mode).strip().lower().replace("-", "_").replace(" ", "_")
        self.classification_loss = get_classification_loss("ce")
        self.regression_loss = get_regression_loss(self.regression_mode)
        self.use_rank_metrics = self.task == "regression" and self.regression_mode == "rank"
        self.last_test_metrics = {}
        # model structure statistics
        self.total_edges = 0
        self.alive_edges = 0
        self.edge_ratio = 0.0
        self.logical_rules = 0
        self.max_literals_per_rule = 0
        self.avg_literals_per_rule = 0.0
                
        # device setup
        self.device_id = device_id
        # dealing with cpu setting
        self.device = torch.device(f"cuda:{device_id}") if device_id is not None else torch.device("cpu")
        self.use_cuda = (self.device.type == "cuda")
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        # Temperature scaling is only used for classification logits.
        net_temperature = temperature if self.task != "regression" else 1.0
        self.net = Net(dim_list, use_not=use_not, left=left, right=right, use_nlaf=use_nlaf,
                       estimated_grad=estimated_grad, use_skip=use_skip, alpha=alpha, beta=beta,
                       gamma=gamma, temperature=net_temperature)
        self.net.to(self.device)

    def _pairwise_rank_counts(self, y_pred, y_true):
        return self.regression_loss.rank_counts(y_pred, y_true)

    def _pairwise_rank_accuracy(self, y_pred, y_true):
        correct, total = self._pairwise_rank_counts(y_pred, y_true)
        return correct / total.clamp(min=1.0)

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()
    
    def edge_penalty(self):
        edge_penalty = 0.0
        for layer in self.net.layer_list[1: -1]:
            edge_penalty += layer.edge_count()
        return edge_penalty
    
    def l1_penalty(self):
        l1_penalty = 0.0
        for layer in self.net.layer_list[1: ]:
            l1_penalty += layer.l1_norm()
        return l1_penalty
    
    def l2_penalty(self):
        l2_penalty = 0.0
        for layer in self.net.layer_list[1: ]:
            l2_penalty += layer.l2_norm()
        return l2_penalty
    
    def mixed_penalty(self):
        penalty = 0.0
        for layer in self.net.layer_list[1: -1]:
            penalty += layer.l2_norm()
        penalty += self.net.layer_list[-1].l1_norm()
        return penalty

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(
        self,
        validator: BaseValidator,
        data_loader=None,
        valid_loader=None,
        epoch=50,
        lr=0.01,
        lr_decay_epoch=100,
        lr_decay_rate=0.75,
        weight_decay=0.0,
    ):
        if data_loader is None:
            raise Exception("Data loader is unavailable!")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0)

        for epo in range(epoch):
            start_epoch_time = time.perf_counter()

            # ensure training mode
            self.net.train()
            sampler = getattr(data_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epo)

            optimizer = self.exp_lr_scheduler(
                optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch
            )

            # epoch aggregates (training set)
            epoch_loss_sum = 0.0      # loss * n_samples
            epoch_train_correct = 0.0
            epoch_train_total = 0.0
            epoch_train_rank_correct = 0.0
            epoch_train_rank_total = 0.0

            # profiling
            ba_cnt = 0
            tmove = 0.0
            tforward = 0.0
            tbackward = 0.0
            topt = 0.0

            for batch in data_loader:
                X, y = batch[:2]
                ba_cnt += 1
                batch_start = time.perf_counter()

                # move batch to correct device (CPU/GPU safe)
                X = X.to(self.device, non_blocking=self.use_cuda)
                y = y.to(self.device, non_blocking=self.use_cuda)
                move_end = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)

                forward_start = time.perf_counter()
                y_bar = self.net.forward(X)
                
                
                if self.task == "regression":
                    y_target = y
                    y_arg = torch.argmax(y_target, dim=1)
                    loss_main = self.regression_loss(y_bar, y_target, epoch=epo)
                    loss_fnn = loss_main + weight_decay * self.l2_penalty()

                else:
                    # trainable softmax temperature (classification only)
                    y_bar = y_bar / torch.exp(self.net.t)
                    y_arg = torch.argmax(y, dim=1)
                    loss_fnn = self.classification_loss(y_bar, y_arg) + weight_decay * self.l2_penalty()
                forward_end = time.perf_counter()

                if self.is_rank0 and getattr(self, "debug_device", False) and not getattr(self, "_debug_device_done", False):
                    param_device = next(self.net.parameters()).device
                    msg = (f"[debug-device] model_device={param_device} "
                           f"X={X.device} y={y.device} loss_device={loss_fnn.device}")
                    if self.use_cuda:
                        alloc = torch.cuda.memory_allocated(self.device)
                        reserved = torch.cuda.memory_reserved(self.device)
                        msg += f" cuda_alloc={alloc} cuda_reserved={reserved}"
                    print(msg)
                    self._debug_device_done = True

                ba_loss = float(loss_fnn.item())

                # accuracy/loss accumulators (per sample)
                with torch.no_grad():
                    pred = torch.argmax(y_bar, dim=1)
                    bs = float(y_arg.numel())
                    epoch_train_total += bs
                    epoch_train_correct += float((pred == y_arg).sum().item())
                    epoch_loss_sum += ba_loss * bs
                    if self.task == "regression" and self.use_rank_metrics:
                        batch_rank_correct, batch_rank_total = self._pairwise_rank_counts(y_bar, y_target)
                        epoch_train_rank_correct += float(batch_rank_correct.item())
                        epoch_train_rank_total += float(batch_rank_total.item())

                loss_fnn.backward()
                backward_end = time.perf_counter()
                optimizer.step()
                self.clip()
                step_end = time.perf_counter()

                batch_end = time.perf_counter()
                tmove  += move_end - batch_start
                topt   += step_end - backward_end
                tforward  += forward_end - forward_start
                tbackward += backward_end - forward_end

            # --------- epoch validation ---------
            val_start = time.perf_counter()
            metrics = validator.validate(epo, self)
            val_time = time.perf_counter() - val_start
            train_loss = epoch_loss_sum / max(epoch_train_total, 1.0)
            metrics["train_loss"] = train_loss
            if self.save_best and self.is_rank0:
                if validator.is_better(metrics, getattr(self, "best_metrics", {})):
                    self.best_metrics = dict(metrics)
                    self.save_model()

            # --------- epoch logging ---------
            if self.is_rank0:
                train_acc = epoch_train_correct / max(epoch_train_total, 1.0)
                val_acc = metrics.get("val_acc", -1.0)
                val_f1 = metrics.get("val_f1", -1.0)
                val_mse = metrics.get("val_mse")
                val_rank_acc = metrics.get("val_rank_acc")
                train_rank_acc = (
                    epoch_train_rank_correct / max(epoch_train_rank_total, 1.0)
                    if self.task == "regression" and self.use_rank_metrics else None
                )
                cur_lr = optimizer.param_groups[0]["lr"]
                end_epoch_time = time.perf_counter()
                total_time = end_epoch_time - start_epoch_time
                avg_batch = total_time / max(len(data_loader), 1)
                mode_text = self.regression_mode if self.task == "regression" else self.task
                log_lines = [
                    f"benchmark\t{self.run_name}",
                    f"epoch\t{epo}\tmode\t{mode_text}\tlr\t{cur_lr:.6g}",
                    f"train\tloss\t{train_loss:.6f}\tacc\t{train_acc:.6f}",
                    f"valid\tacc\t{val_acc:.6f}\tf1\t{val_f1:.6f}\ttime\t{val_time:.2f}s",
                ]
                if self.task == "regression":
                    if self.use_rank_metrics:
                        train_rank_text = f"{train_rank_acc:.6f}" if train_rank_acc is not None else "n/a"
                        val_rank_text = f"{float(val_rank_acc):.6f}" if val_rank_acc is not None else "n/a"
                        log_lines.append(f"train-reg\trank_acc\t{train_rank_text}")
                        log_lines.append(f"valid-reg\trank_acc\t{val_rank_text}")
                    else:
                        val_mse_text = f"{float(val_mse):.6f}" if val_mse is not None else "n/a"
                        log_lines.append(f"valid-reg\tmse\t{val_mse_text}")
                log_lines.append(
                    f"time\tepoch\t{total_time:.2f}s\tavg_batch\t{avg_batch:.3f}s\tbatches\t{ba_cnt:04d}"
                )
                log_lines.append(
                    f"time-detail\tmove\t{tmove:.3f}s\tforward\t{tforward:.3f}s\t"
                    f"backward\t{tbackward:.3f}s\topt_step\t{topt:.3f}s"
                )
                logging.info("\n" + "\n".join(log_lines))

        if self.is_rank0 and not self.save_best:
            if valid_loader is not None:
                self.test(test_loader=valid_loader, set_name='Validation', output_log=True)
            else:
                self.test(test_loader=data_loader, set_name='Training', output_log=True)
            self.save_model()

        return None

    @torch.inference_mode()
    def test(self, test_loader=None, set_name='Validation', output_log=True):
        if test_loader is None:
            raise Exception("Data loader is unavailable!")
        
        was_training = self.net.training
        # put model in eval mode for consistent testing
        self.net.eval()

        n_classes = self.dim_list[-1]
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        correct = 0
        total = 0

        mse_sum = 0.0
        mse_count = 0.0
        rank_correct_sum = 0.0
        rank_total_sum = 0.0

        # ONE pass over loader: get y_true and y_pred for each batch
        for batch in test_loader:
            X, y = batch[:2]
            X = X.to(self.device, non_blocking=self.use_cuda)
            y_dev = y.to(self.device, non_blocking=self.use_cuda)
            # y is one-hot (classification) or action-score vector (regression)
            if self.task == "regression":
                y_true = torch.argmax(y_dev, dim=1).cpu().numpy()
            else:
                y_true = torch.argmax(y, dim=1).cpu().numpy()
            # forward pass
            out = self.net.forward(X)
            y_pred = torch.argmax(out, dim=1).cpu().numpy()

            if self.task == "regression":
                if self.use_rank_metrics:
                    batch_rank_correct, batch_rank_total = self._pairwise_rank_counts(out, y_dev)
                    rank_correct_sum += float(batch_rank_correct.item())
                    rank_total_sum += float(batch_rank_total.item())
                else:
                    diff = out - y_dev
                    mse_sum += float(torch.sum(diff * diff).item())
                    mse_count += float(y_dev.numel())
            
            total += y_true.shape[0]
            correct += (y_pred == y_true).sum()

            # update confusion matrix
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1

        acc = correct / max(total, 1)
        mse = (
            (mse_sum / max(mse_count, 1.0))
            if self.task == "regression" and not self.use_rank_metrics
            else None
        )
        rank_acc = (
            (rank_correct_sum / max(rank_total_sum, 1.0))
            if self.task == "regression" and self.use_rank_metrics
            else None
        )

        # macro-F1 from confusion matrix (avoids sklearn heavy report)
        f1s = []
        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1 = 0.0 if (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
            f1s.append(f1)
        f1_macro = float(np.mean(f1s))

        self.last_test_metrics = {
            "acc": float(acc),
            "f1": float(f1_macro),
            "mse": None if mse is None else float(mse),
            "rank_acc": None if rank_acc is None else float(rank_acc),
        }

        if output_log:
            logging.info('-' * 60)
            logging.info(
                f"On {set_name} Set:\n\tAccuracy of FNN Model: {acc}"
                f"\n\tF1 Score (macro) of FNN Model: {f1_macro}"
            )
            if mse is not None:
                logging.info(f"\tMSE (target scores): {mse}")
            if rank_acc is not None:
                logging.info(f"\tPairwise Rank Accuracy: {rank_acc}")
            logging.info(f"Confusion matrix:\n{cm}")
            logging.info('-' * 60)
        
        if was_training:
            self.net.train()

        return acc, f1_macro

    def save_model(self):
        fnn_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'use_skip': self.use_skip, 'estimated_grad': self.estimated_grad, 
                    'use_nlaf': self.use_nlaf, 'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma, 'task': self.task,
                    'regression_mode': self.regression_mode, 'run_name': self.run_name}
        checkpoint = {'model_state_dict': self.net.state_dict(), 'fnn_args': fnn_args}
        predicate_bank = getattr(self, "predicate_bank", None)
        if predicate_bank is not None:
            checkpoint["predicate_bank"] = predicate_bank
        torch.save(checkpoint, self.save_path)

    def detect_dead_node(self, data_loader=None):
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device)
                layer.forward_tot = 0

            for batch in data_loader:
                x = batch[0]
                x_bar = x.to(self.device, non_blocking=self.use_cuda)
                self.net.bi_forward(x_bar, count=True)

    def model_info(self, feature_name, label_name, train_loader, mean=None, std=None):
        return symbolic_model.model_info(self, feature_name, label_name, train_loader, mean=mean, std=std)

    def export_symbolic(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None, atoms_type=None, display=True):
        _ = display
        return symbolic_model.export_symbolic(
            self,
            feature_name,
            label_name,
            train_loader,
            file=file,
            mean=mean,
            std=std,
            atoms_type=atoms_type,
        )
