import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_all(log_dir: str):
    train_log = os.path.join(log_dir, "train_log.csv")
    lfw_log = os.path.join(log_dir, "lfw_log.csv")

    if os.path.isfile(train_log):
        df = pd.read_csv(train_log)

        plt.figure()
        plt.plot(df["epoch"], df["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(df["epoch"], df["classification_acc"])
        plt.xlabel("Epoch")
        plt.ylabel("Classification Accuracy")
        plt.title("Training Classification Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "train_accuracy_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()

    if os.path.isfile(lfw_log):
        df = pd.read_csv(lfw_log)

        plt.figure()
        plt.plot(df["epoch"], df["lfw_acc"])
        plt.xlabel("Epoch")
        plt.ylabel("LFW Accuracy (%)")
        plt.title("LFW Verification Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "lfw_accuracy_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    plot_all(args.log_dir)


if __name__ == "__main__":
    main()
