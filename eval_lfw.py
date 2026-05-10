import argparse

import torch
from torch.utils.data import DataLoader

from datasets import LFWDataset, eval_transform
from models import build_model
from utils import compute_lfw_similarities, evaluate_lfw_10fold, load_backbone_checkpoint


def parse_args():
    parser = argparse.ArgumentParser("Evaluate trained ResNet50 face model on LFW.")
    parser.add_argument("--lfw_root", type=str, required=True)
    parser.add_argument("--lfw_pairs", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_flip_eval", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model(args.backbone, args.embedding_size, args.dropout).to(device)
    ckpt = load_backbone_checkpoint(args.checkpoint, model, device)
    model.eval()

    lfw_set = LFWDataset(
        lfw_root=args.lfw_root,
        pairs_path=args.lfw_pairs,
        transform=eval_transform(args.image_size),
    )

    lfw_loader = DataLoader(
        lfw_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )

    use_flip = not args.no_flip_eval

    similarities, labels = compute_lfw_similarities(model, lfw_loader, device, use_flip=use_flip)
    mean_acc, mean_threshold, fold_accs = evaluate_lfw_10fold(similarities, labels, n_splits=10)

    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown') if isinstance(ckpt, dict) else 'unknown'}")
    print(f"Flip eval: {use_flip}")
    print(f"LFW mean accuracy: {mean_acc * 100:.4f}%")
    print(f"LFW mean threshold: {mean_threshold:.4f}")
    print("Fold accuracies:")
    for i, acc in enumerate(fold_accs, start=1):
        print(f"  Fold {i}: {acc * 100:.4f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
