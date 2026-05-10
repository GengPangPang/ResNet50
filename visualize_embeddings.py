import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import build_model
from utils import load_backbone_checkpoint


IMG_EXTENSIONS = (".jpg",)


def eval_transform(image_size=112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def pil_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class SampledWebFaceDataset(Dataset):
    def __init__(self, root, num_classes=20, images_per_class=30, transform=None, seed=42):
        self.root = root
        self.transform = transform

        random.seed(seed)

        identities = sorted([
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ])

        if len(identities) == 0:
            raise RuntimeError(f"No identity folders found in: {root}")

        selected_ids = random.sample(identities, min(num_classes, len(identities)))

        self.samples = []

        for label, identity in enumerate(selected_ids):
            identity_dir = os.path.join(root, identity)

            images = sorted([
                name for name in os.listdir(identity_dir)
                if name.lower().endswith(IMG_EXTENSIONS)
            ])

            if len(images) == 0:
                continue

            images = random.sample(images, min(images_per_class, len(images)))

            for image_name in images:
                image_path = os.path.join(identity_dir, image_name)
                self.samples.append((image_path, label))

        if len(self.samples) == 0:
            raise RuntimeError("No .jpg images found for t-SNE visualization.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = pil_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels


def plot_tsne(embeddings, labels, output_path, title):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    points = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(np.unique(labels))

    for label in unique_labels:
        idx = labels == label
        plt.scatter(
            points[idx, 0],
            points[idx, 1],
            s=12,
            alpha=0.75,
            label=str(label),
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(
        markerscale=2,
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved t-SNE figure to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./results/tsne_embeddings.png")

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--images_per_class", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    dataset = SampledWebFaceDataset(
        root=args.data_root,
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        transform=eval_transform(args.image_size),
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(
        backbone=args.backbone,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
    ).to(device)

    load_backbone_checkpoint(args.checkpoint, model, device)

    embeddings, labels = extract_embeddings(model, dataloader, device)

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)

    plot_tsne(
        embeddings=embeddings,
        labels=labels,
        output_path=args.output,
        title=f"t-SNE of ResNet50 + ArcFace Face Embeddings"
    )


if __name__ == "__main__":
    main()