#!/bin/bash
#SBATCH --job-name=resnet50_arcface_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

cd ~/projects/ResNet50

mkdir -p logs

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# 按你的超算环境修改 module 名称；如果不用 module，可以注释掉。
# module purge
# module load cuda/12.1
# module load anaconda/3

source ~/.bashrc
conda activate face_arcface

python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count())"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SLURM 通常会自动设置可见 GPU；这里不要强制 CUDA_VISIBLE_DEVICES，除非你的集群要求。
torchrun --standalone --nproc_per_node=4 train_webface_arcface_ddp.py \
  --train_root /scratch/$USER/datasets/webface_112x112 \
  --lfw_root /scratch/$USER/datasets/lfw_funneled \
  --lfw_pairs /scratch/$USER/datasets/lfw_funneled/pairs.txt \
  --output_dir ./results \
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
  --log_dir ./results

python visualize_embeddings.py \
  --data_root /scratch/$USER/datasets/webface_112x112 \
  --checkpoint ./results/best_model.pth \
  --output ./results/tsne_embeddings.png \
  --backbone resnet50 \
  --image_size 112 \
  --embedding_size 512 \
  --num_classes 20 \
  --images_per_class 30 \
  --batch_size 128 \
  --workers 4 \
  --device cuda