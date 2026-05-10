import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets import LFWDataset, WebFaceFolder, eval_transform, train_transform
from losses import ArcFaceHead
from models import build_model
from utils import (
    append_csv,
    cleanup_distributed,
    compute_lfw_similarities,
    ensure_dir,
    evaluate_lfw_10fold,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_mean,
    save_checkpoint,
    set_seed,
    setup_distributed,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser("DDP train ResNet50 + ArcFace on WebFace 112x112 and evaluate on LFW.")

    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--lfw_root", type=str, required=True)
    parser.add_argument("--lfw_pairs", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--easy_margin", action="store_true")

    # In DDP this is per-GPU batch size.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=int, nargs="+", default=[6, 11, 16])
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_flip_eval", action="store_true")

    return parser.parse_args()


def build_loader(dataset, batch_size, workers, is_train, distributed):
    sampler = None
    shuffle = is_train

    if distributed and is_train:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
        drop_last=is_train,
    ), sampler


def run_lfw_eval(model, lfw_loader, device, use_flip: bool):
    eval_model = unwrap_model(model)
    similarities, labels = compute_lfw_similarities(eval_model, lfw_loader, device, use_flip=use_flip)
    return evaluate_lfw_10fold(similarities, labels, n_splits=10)


def main():
    args = parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    set_seed(args.seed)

    if is_main_process():
        ensure_dir(args.output_dir)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    train_set = WebFaceFolder(
        root=args.train_root,
        transform=train_transform(args.image_size),
    )

    train_loader, train_sampler = build_loader(
        train_set,
        batch_size=args.batch_size,
        workers=args.workers,
        is_train=True,
        distributed=distributed,
    )

    # Only rank 0 loads LFW for validation to avoid duplicate evaluation.
    lfw_set = None
    lfw_loader = None

    if is_main_process():
        lfw_set = LFWDataset(
            lfw_root=args.lfw_root,
            pairs_path=args.lfw_pairs,
            transform=eval_transform(args.image_size),
        )
        lfw_loader, _ = build_loader(
            lfw_set,
            batch_size=args.batch_size,
            workers=args.workers,
            is_train=False,
            distributed=False,
        )

    model = build_model(args.backbone, args.embedding_size, args.dropout).to(device)

    arcface = ArcFaceHead(
        embedding_size=args.embedding_size,
        num_classes=train_set.num_classes,
        scale=args.scale,
        margin=args.margin,
        easy_margin=args.easy_margin,
    ).to(device)

    if args.compile:
        model = torch.compile(model)
        arcface = torch.compile(arcface)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        arcface = DDP(arcface, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    global_batch_size = args.batch_size * world_size

    if is_main_process():
        print("=" * 80)
        print("Environment")
        print(f"Runtime torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Distributed: {distributed}")
        print(f"World size: {world_size}")
        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Global batch size: {global_batch_size}")
        print(f"Device on rank 0: {device}")
        print("=" * 80)
        print(f"Train root: {args.train_root}")
        print(f"Train images: {len(train_set)}")
        print(f"Train identities/classes: {train_set.num_classes}")
        print(f"LFW pairs: {len(lfw_set) if lfw_set is not None else 0}")
        print(f"LR: {args.lr}")
        print(f"AMP: {use_amp}")
        print(f"torch.compile: {args.compile}")
        print(f"Output dir: {args.output_dir}")
        print("=" * 80)

    best_lfw_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        arcface.train()

        epoch_loss_sum = 0.0
        epoch_correct_sum = 0.0
        epoch_total_sum = 0.0

        start_time = time.time()

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                embeddings = model(images)
                logits = arcface(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                batch_correct = (pred == labels).sum().float()
                batch_total = torch.tensor(float(labels.size(0)), device=device)
                batch_loss_sum = loss.detach() * batch_total

                if distributed:
                    dist.all_reduce(batch_correct, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_loss_sum, op=dist.ReduceOp.SUM)

                epoch_loss_sum += batch_loss_sum.item()
                epoch_correct_sum += batch_correct.item()
                epoch_total_sum += batch_total.item()

            if is_main_process():
                avg_loss = epoch_loss_sum / max(epoch_total_sum, 1.0)
                cls_acc = epoch_correct_sum / max(epoch_total_sum, 1.0)
                iterator.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "cls_acc": f"{cls_acc:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

        scheduler.step()

        epoch_seconds = time.time() - start_time
        train_loss = epoch_loss_sum / max(epoch_total_sum, 1.0)
        train_acc = epoch_correct_sum / max(epoch_total_sum, 1.0)

        if is_main_process():
            append_csv(
                os.path.join(args.output_dir, "train_log.csv"),
                {
                    "epoch": epoch,
                    "loss": f"{train_loss:.8f}",
                    "classification_acc": f"{train_acc:.8f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                    "seconds": f"{epoch_seconds:.2f}",
                    "world_size": world_size,
                    "batch_size_per_gpu": args.batch_size,
                    "global_batch_size": global_batch_size,
                },
            )

            print(f"[Epoch {epoch}] loss={train_loss:.6f}, cls_acc={train_acc:.6f}, time={epoch_seconds:.1f}s")

        if distributed:
            dist.barrier()

        if is_main_process() and epoch % args.eval_every == 0:
            use_flip = not args.no_flip_eval
            lfw_acc, lfw_threshold, fold_accs = run_lfw_eval(model, lfw_loader, device, use_flip=use_flip)

            print(f"[LFW] accuracy={lfw_acc * 100:.4f}%, threshold={lfw_threshold:.4f}, flip_eval={use_flip}")
            print("[LFW] fold_accs:", " ".join(f"{x * 100:.2f}" for x in fold_accs))

            append_csv(
                os.path.join(args.output_dir, "lfw_log.csv"),
                {
                    "epoch": epoch,
                    "lfw_acc": f"{lfw_acc * 100:.8f}",
                    "threshold": f"{lfw_threshold:.8f}",
                    "flip_eval": int(use_flip),
                },
            )

            if lfw_acc > best_lfw_acc:
                best_lfw_acc = lfw_acc
                save_checkpoint(
                    os.path.join(args.output_dir, "best_model.pth"),
                    model,
                    arcface,
                    optimizer,
                    scheduler,
                    epoch,
                    best_lfw_acc,
                    args,
                )
                print(f"[SAVE] best_model.pth, best_lfw_acc={best_lfw_acc * 100:.4f}%")

        if distributed:
            dist.barrier()

        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.output_dir, "last_model.pth"),
                model,
                arcface,
                optimizer,
                scheduler,
                epoch,
                best_lfw_acc,
                args,
            )

        if distributed:
            dist.barrier()

    if is_main_process():
        print(f"Training finished. Best LFW accuracy: {best_lfw_acc * 100:.4f}%")

        try:
            import plot_curves
            plot_curves.plot_all(args.output_dir)
            print("Curves saved.")
        except Exception as exc:
            print(f"Curve plotting skipped: {exc}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
