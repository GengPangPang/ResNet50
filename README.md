# ResNet50 + ArcFace + WebFace 112x112 + LFW：超算多卡 DDP 训练版

这是一套针对超算/服务器多 GPU 训练的完整代码。核心配置：

- Backbone：ResNet50
- Loss：ArcFace
- 训练集：WebFace 112x112，folder-style，每个文件夹一个身份
- 验证集：LFW / lfw_funneled，使用 pairs.txt 做 10-fold verification
- 图片格式：只读取 `.jpg`
- 多卡方式：PyTorch DistributedDataParallel，推荐用 `torchrun`
- 适配：Python 3.10，PyTorch 2.4.0+cu121，CUDA 12.1

---

## 1. 文件说明

```text
resnet50_arcface_webface_lfw_ddp_hpc/
  train_webface_arcface_ddp.py      # 多卡/单卡训练入口
  eval_lfw.py                       # 单独 LFW 验证
  datasets.py                       # WebFace / LFW 数据集，只读取 jpg
  models.py                         # ResNet50 backbone
  losses.py                         # ArcFace head
  utils.py                          # DDP、评估、保存、日志工具
  plot_curves.py                    # 画曲线
  run_train_single.sh               # 单卡训练脚本
  run_train_4gpu.sh                 # 单节点 4 卡训练脚本
  run_train.slurm                   # SLURM 示例
  requirements.txt
```

---

## 2. 数据目录要求

WebFace：

```text
webface_112x112/
  id_0/
    xxx.jpg
  id_1/
    xxx.jpg
```

LFW：

```text
lfw_funneled/
  AJ_Cook/
    AJ_Cook_0001.jpg
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg
  pairs.txt
```

你的本地路径示例：

```text
WebFace:
/home/rum/.cache/kagglehub/datasets/yakhyokhuja/webface-112x112/versions/1/webface_112x112

LFW:
/mnt/e/Desktop/Job/Intern/lfw_funneled
```

上超算后建议放到高速盘，例如：

```text
/scratch/$USER/datasets/webface_112x112
/scratch/$USER/datasets/lfw_funneled
```

---

## 3. 安装依赖

```bash
conda create -n face_arcface python=3.10 -y
conda activate face_arcface

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## 4. 单卡训练

```bash
bash run_train_single.sh
```

或者手动执行：

```bash
CUDA_VISIBLE_DEVICES=0 python train_webface_arcface_ddp.py \
  --train_root /scratch/$USER/datasets/webface_112x112 \
  --lfw_root /scratch/$USER/datasets/lfw_funneled \
  --lfw_pairs /scratch/$USER/datasets/lfw_funneled/pairs.txt \
  --output_dir ./results \
  --batch_size 128 \
  --epochs 18 \
  --lr 0.1 \
  --workers 8 \
  --amp \
  --device cuda
```

---

## 5. 单节点 4 卡训练

这里的 `--batch_size` 是 **每张 GPU 的 batch size**，不是总 batch size。

4 卡时：

```text
per-GPU batch_size = 64
global batch size = 64 × 4 = 256
lr = 0.2
```

```bash
bash run_train_4gpu.sh
```

或者手动执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_webface_arcface_ddp.py \
  --train_root /scratch/$USER/datasets/webface_112x112 \
  --lfw_root /scratch/$USER/datasets/lfw_funneled \
  --lfw_pairs /scratch/$USER/datasets/lfw_funneled/pairs.txt \
  --output_dir ./results \
  --batch_size 64 \
  --epochs 18 \
  --lr 0.2 \
  --workers 8 \
  --amp \
  --device cuda
```

---

## 6. SLURM 提交

修改 `run_train.slurm` 里的路径、partition、module 名称，然后：

```bash
sbatch run_train.slurm
```

查看：

```bash
squeue -u $USER
tail -f logs/resnet50_arcface_ddp_<jobid>.out
```

---

## 7. 推荐 batch size 和学习率

`--batch_size` 是每张 GPU 的 batch size。

```text
1 GPU: per-GPU batch 128, global batch 128, lr 0.1
2 GPU: per-GPU batch 64,  global batch 128, lr 0.1  稳妥
2 GPU: per-GPU batch 128, global batch 256, lr 0.2  更快
4 GPU: per-GPU batch 64,  global batch 256, lr 0.2  推荐
4 GPU: per-GPU batch 128, global batch 512, lr 0.4  显存充足时
```

建议超算第一次测试：

```text
--epochs 1
--batch_size 64
--lr 0.2
4 GPU
```

跑通后改：

```text
--epochs 18
```

---

## 8. 输出结果

```text
results/
  best_model.pth
  last_model.pth
  train_log.csv
  lfw_log.csv
  loss_curve.png
  train_accuracy_curve.png
  lfw_accuracy_curve.png
```

DDP 只会在 rank 0 保存模型和日志，避免多个进程同时写文件。
