#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

TRAIN_ROOT="/scratch/$USER/datasets/webface_112x112"
LFW_ROOT="/scratch/$USER/datasets/lfw_funneled"
LFW_PAIRS="/scratch/$USER/datasets/lfw_funneled/pairs.txt"
OUTPUT_DIR="./results"

torchrun --standalone --nproc_per_node=4 train_webface_arcface_ddp.py \
  --train_root "$TRAIN_ROOT" \
  --lfw_root "$LFW_ROOT" \
  --lfw_pairs "$LFW_PAIRS" \
  --output_dir "$OUTPUT_DIR" \
  --backbone resnet50 \
  --image_size 112 \
  --embedding_size 512 \
  --batch_size 64 \
  --epochs 18 \
  --lr 0.2 \
  --workers 8 \
  --amp \
  --eval_every 1 \
  --device cuda

python plot_curves.py \
  --log_dir "$OUTPUT_DIR"

python visualize_embeddings.py \
  --data_root "$TRAIN_ROOT" \
  --checkpoint "$OUTPUT_DIR/best_model.pth" \
  --output "$OUTPUT_DIR/tsne_embeddings.png" \
  --backbone resnet50 \
  --image_size 112 \
  --embedding_size 512 \
  --num_classes 20 \
  --images_per_class 30 \
  --batch_size 128 \
  --workers 4 \
  --device cuda

echo "All done."
echo "Results saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"