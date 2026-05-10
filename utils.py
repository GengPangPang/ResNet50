import csv
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """
    Initialize DDP if launched by torchrun.

    torchrun provides:
      RANK
      WORLD_SIZE
      LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int):
    rank = get_rank()
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def append_csv(path: str, row: Dict):
    if not is_main_process():
        return

    ensure_dir(os.path.dirname(path))
    write_header = not os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """
    Average a scalar tensor across DDP processes.
    """
    if not is_dist_avail_and_initialized():
        return value

    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= get_world_size()
    return value


@torch.no_grad()
def compute_lfw_similarities(model, dataloader, device: torch.device, use_flip: bool = True):
    """
    Compute cosine similarities for LFW pairs.

    Run this only on rank 0 in DDP training.
    """
    model.eval()

    similarities = []
    labels = []

    for img1, img2, same in dataloader:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        emb1 = model(img1)
        emb2 = model(img2)

        if use_flip:
            emb1_flip = model(torch.flip(img1, dims=[3]))
            emb2_flip = model(torch.flip(img2, dims=[3]))
            emb1 = torch.nn.functional.normalize(emb1 + emb1_flip, p=2, dim=1)
            emb2 = torch.nn.functional.normalize(emb2 + emb2_flip, p=2, dim=1)

        sim = torch.sum(emb1 * emb2, dim=1)

        similarities.append(sim.detach().cpu().numpy())
        labels.append(same.numpy())

    return np.concatenate(similarities), np.concatenate(labels)


def evaluate_lfw_10fold(similarities: np.ndarray, labels: np.ndarray, n_splits: int = 10) -> Tuple[float, float, List[float]]:
    thresholds = np.arange(-1.0, 1.0, 0.001)
    indices = np.arange(len(labels))

    kfold = KFold(n_splits=n_splits, shuffle=False)

    fold_accs = []
    fold_thresholds = []

    for train_idx, test_idx in kfold.split(indices):
        train_sims = similarities[train_idx]
        train_labels = labels[train_idx]

        best_threshold = 0.0
        best_acc = -1.0

        for threshold in thresholds:
            pred = (train_sims > threshold).astype(np.int64)
            acc = accuracy_score(train_labels, pred)

            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        test_pred = (similarities[test_idx] > best_threshold).astype(np.int64)
        test_acc = accuracy_score(labels[test_idx], test_pred)

        fold_accs.append(float(test_acc))
        fold_thresholds.append(float(best_threshold))

    return float(np.mean(fold_accs)), float(np.mean(fold_thresholds)), fold_accs


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, arcface, optimizer, scheduler, epoch, best_lfw_acc, args):
    if not is_main_process():
        return

    ensure_dir(os.path.dirname(path))

    model_to_save = unwrap_model(model)
    arcface_to_save = unwrap_model(arcface) if arcface is not None else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "arcface_state_dict": arcface_to_save.state_dict() if arcface_to_save is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_lfw_acc": best_lfw_acc,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }

    torch.save(checkpoint, path)


def load_backbone_checkpoint(path: str, model, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return ckpt
