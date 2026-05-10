# ResNet50 + ArcFace + WebFace 112x112 + LFW：超算多卡 DDP 训练版
这是一套针对超算 / 服务器多 GPU 训练的完整人脸识别代码。核心配置：
Backbone：ResNet50
Loss：ArcFace
训练集：WebFace 112x112，folder-style，每个文件夹一个身份
验证集：LFW / lfw_funneled，使用 `pairs.txt` 做 10-fold verification
图片格式：只读取 `.jpg`
多卡方式：PyTorch DistributedDataParallel，推荐用 `torchrun`
适配环境：Python 3.10，PyTorch 2.4.0+cu121，CUDA 12.1
---
1. 文件说明
```text
resnet50_arcface_webface_lfw_ddp_hpc/
  download.py                       # 下载 WebFace 112x112 训练集
  train_webface_arcface_ddp.py      # 多卡 / 单卡训练入口
  eval_lfw.py                       # 单独 LFW 验证
  datasets.py                       # WebFace / LFW 数据集，只读取 jpg
  models.py                         # ResNet50 backbone
  losses.py                         # ArcFace head
  utils.py                          # DDP、评估、保存、日志工具
  plot_curves.py                    # 画 loss / accuracy 曲线
  visualize_embeddings.py           # t-SNE 特征可视化
  run_train_single.sh               # 单卡训练脚本
  run_train_4gpu.sh                 # 单节点 4 卡训练脚本
  run_train.slurm                   # SLURM 超算提交脚本
  requirements.txt
  README.md
```
---
2. 数据集准备
本项目需要两个数据集：
```text
1. WebFace 112x112：训练集
2. LFW / lfw_funneled：验证集
```
---
2.1 下载 WebFace 112x112 训练集
本项目使用 WebFace 112x112 作为训练集，来源为 Kaggle 数据集：
```text
https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112
```
项目中已经提供了 `download.py`，可以直接运行下载训练集：
```bash
python download.py
```
下载完成后，数据通常位于 KaggleHub 缓存目录，例如：
```text
/home/rum/.cache/kagglehub/datasets/yakhyokhuja/webface-112x112/versions/1/webface_112x112
```
可以使用下面命令检查是否下载成功：
```bash
ls /home/rum/.cache/kagglehub/datasets/yakhyokhuja/webface-112x112/versions/1/webface_112x112 | head
```
正常情况下会看到类似：
```text
id_0
id_1
id_10
id_100
id_1000
```
WebFace 目录结构应为：
```text
webface_112x112/
  id_0/
    xxx.jpg
    xxx.jpg
  id_1/
    xxx.jpg
    xxx.jpg
  id_2/
    xxx.jpg
```
每个 `id_x` 文件夹代表一个身份类别，代码会自动将文件夹映射为整数类别标签，不需要额外的标签文件。
也可以从 Kaggle 网页手动下载：
```text
https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112
```
下载并解压后，只需要保证最终目录结构为：
```text
webface_112x112/
  id_0/
  id_1/
  id_2/
  ...
```
---
2.2 准备 LFW 验证集
本项目使用 LFW / lfw_funneled 作为验证集。LFW 不参与模型参数更新，只用于每个 epoch 结束后的 verification accuracy 评估。
LFW 目录结构应为：
```text
lfw_funneled/
  AJ_Cook/
    AJ_Cook_0001.jpg
  AJ_Lamas/
    AJ_Lamas_0001.jpg
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg
  ...
  pairs.txt
```
其中 `pairs.txt` 是 LFW 官方验证协议文件，格式类似：
```text
10      300
Abel_Pacheco    1       4
Akhmed_Zakayev  1       3
...
```
代码会根据 `pairs.txt` 构造 6000 对人脸图片，并进行 10-fold verification。
需要注意：
```text
LFW 中很多身份只有一张图片，这是正常现象。
LFW 是验证集，不是训练集。
只要 pairs.txt 中指定的图片都能在 lfw_funneled 中找到即可。
```
可以检查 LFW 路径：
```bash
ls /path/to/lfw_funneled | head
head /path/to/lfw_funneled/pairs.txt
```
---
2.3 超算数据路径建议
上超算训练时，建议将数据放到高速盘，例如：
```text
/scratch/$USER/datasets/webface_112x112
/scratch/$USER/datasets/lfw_funneled
```
如果 WebFace 已经通过 `download.py` 下载到了本地缓存，可以复制到超算高速盘：
```bash
mkdir -p /scratch/$USER/datasets

cp -r /home/rum/.cache/kagglehub/datasets/yakhyokhuja/webface-112x112/versions/1/webface_112x112 \
  /scratch/$USER/datasets/
```
如果 LFW 已经准备好，也复制到高速盘：
```bash
cp -r /path/to/lfw_funneled /scratch/$USER/datasets/
```
最终训练脚本中的路径应类似：
```bash
--train_root /scratch/$USER/datasets/webface_112x112
--lfw_root /scratch/$USER/datasets/lfw_funneled
--lfw_pairs /scratch/$USER/datasets/lfw_funneled/pairs.txt
```
---
3. 安装依赖
建议使用 Conda 环境：
```bash
conda create -n face_arcface python=3.10 -y
conda activate face_arcface
```
安装 PyTorch：
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
安装项目依赖：
```bash
pip install -r requirements.txt
```
检查环境：
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```
期望输出类似：
```text
2.4.0+cu121
True
12.1
```
---
4. 单卡训练
可以直接运行：
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
  --backbone resnet50 \
  --image_size 112 \
  --embedding_size 512 \
  --batch_size 128 \
  --epochs 18 \
  --lr 0.1 \
  --workers 8 \
  --amp \
  --eval_every 1 \
  --device cuda
```
单卡训练时：
```text
per-GPU batch size = 128
global batch size = 128
learning rate = 0.1
```
---
5. 单节点 4 卡训练
可以直接运行：
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
```
注意：DDP 版本中，`--batch_size` 是 每张 GPU 的 batch size，不是总 batch size。
4 卡时：
```text
per-GPU batch size = 64
global batch size = 64 × 4 = 256
learning rate = 0.2
```
---
6. SLURM 超算提交
修改 `run_train.slurm` 中的以下配置：
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
```
根据实际超算环境修改 partition、GPU 数量、CPU 数量、内存和运行时间。
如果超算需要 module，也需要修改脚本中的 module 部分：
```bash
# module purge
# module load cuda/12.1
# module load anaconda/3
```
然后提交：
```bash
sbatch run_train.slurm
```
查看任务：
```bash
squeue -u $USER
```
查看日志：
```bash
tail -f logs/resnet50_arcface_ddp_<jobid>.out
```
---
7. 推荐 batch size 和学习率
DDP 版本中：
```text
--batch_size 是每张 GPU 的 batch size
global batch size = per-GPU batch size × GPU 数量
```
推荐配置：
```text
1 GPU:
  per-GPU batch size = 128
  global batch size  = 128
  lr = 0.1

2 GPU 稳妥：
  per-GPU batch size = 64
  global batch size  = 128
  lr = 0.1

2 GPU 更快：
  per-GPU batch size = 128
  global batch size  = 256
  lr = 0.2

4 GPU 推荐：
  per-GPU batch size = 64
  global batch size  = 256
  lr = 0.2

4 GPU 显存充足：
  per-GPU batch size = 128
  global batch size  = 512
  lr = 0.4
```
学习率按 global batch size 线性缩放：
```text
lr = 0.1 × global_batch_size / 128
```
首次在超算上测试时，建议先用：
```text
4 GPU
--epochs 1
--batch_size 64
--lr 0.2
```
确认能完整跑通后，再改为：
```text
--epochs 18
```
---
8. 训练过程记录
训练过程会自动在 `results/` 目录下保存日志和模型。
```text
results/
  train_log.csv
  lfw_log.csv
  best_model.pth
  last_model.pth
```
8.1 train_log.csv
每个 epoch 记录一次训练指标：
```text
epoch
loss
classification_acc
lr
seconds
world_size
batch_size_per_gpu
global_batch_size
```
含义：
```text
epoch：当前训练轮数
loss：当前 epoch 的平均训练损失
classification_acc：训练分类准确率
lr：当前学习率
seconds：当前 epoch 耗时
world_size：DDP 进程数 / GPU 数
batch_size_per_gpu：每张 GPU 的 batch size
global_batch_size：总 batch size
```
8.2 lfw_log.csv
每次 LFW 验证记录一次：
```text
epoch
lfw_acc
threshold
flip_eval
```
含义：
```text
epoch：当前训练轮数
lfw_acc：LFW 10-fold verification accuracy
threshold：该次验证的平均最佳阈值
flip_eval：是否使用水平翻转增强
```
当前默认：
```bash
--eval_every 1
```
因此每个 epoch 结束后都会在 LFW 上验证一次。
---
9. 输出结果
训练完成后，`results/` 中会包含：
```text
results/
  best_model.pth
  last_model.pth
  train_log.csv
  lfw_log.csv
  loss_curve.png
  train_accuracy_curve.png
  lfw_accuracy_curve.png
  tsne_embeddings.png
```
其中：
```text
best_model.pth：LFW accuracy 最好的 checkpoint
last_model.pth：最后一次保存的 checkpoint
loss_curve.png：训练 loss 曲线
train_accuracy_curve.png：训练分类准确率曲线
lfw_accuracy_curve.png：LFW 验证准确率曲线
tsne_embeddings.png：人脸 embedding 的 t-SNE 二维可视化图
```
DDP 只会在 rank 0 主进程保存模型和日志，避免多个进程同时写文件。
---
10. 单独验证 LFW
训练完成后，可以单独使用 `best_model.pth` 在 LFW 上验证：
```bash
python eval_lfw.py \
  --lfw_root /scratch/$USER/datasets/lfw_funneled \
  --lfw_pairs /scratch/$USER/datasets/lfw_funneled/pairs.txt \
  --checkpoint ./results/best_model.pth \
  --backbone resnet50 \
  --image_size 112 \
  --embedding_size 512 \
  --batch_size 128 \
  --workers 8 \
  --device cuda
```
输出示例：
```text
LFW mean accuracy: 98.xx%
LFW mean threshold: 0.xxxx
Fold accuracies:
  Fold 1: xx.xxxx%
  ...
```
---
11. 生成训练曲线
训练结束后会自动生成曲线。如果需要手动生成：
```bash
python plot_curves.py --log_dir ./results
```
会生成：
```text
loss_curve.png
train_accuracy_curve.png
lfw_accuracy_curve.png
```
---
12. 生成 t-SNE 特征可视化图
训练完成后，可以使用 `visualize_embeddings.py` 对模型提取的人脸 embedding 做 t-SNE 可视化：
```bash
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
```
该脚本会随机选取若干身份，每个身份采样若干图片，提取 512 维 embedding 后使用 t-SNE 降到二维空间并绘图。
默认参数：
```text
num_classes = 20
images_per_class = 30
```
输出：
```text
results/tsne_embeddings.png
```
---
13. 任务说明
本项目的训练和验证流程为：
```text
WebFace 112x112 作为训练集
LFW / lfw_funneled 作为验证集
```
LFW 不参与模型参数更新，只用于每个 epoch 结束后的验证准确率评估和 best checkpoint 选择。
如果需要更严格的最终测试，可以使用：
```text
last_model.pth
```
在训练完成后只测试一次 LFW；或者额外准备独立测试集。
---
14. 常见问题
14.1 LFW 中很多身份只有一张图，是否正常？
正常。
LFW 是 verification 数据集，不是分类训练集。`pairs.txt` 会指定 6000 对图片：
```text
3000 对 same person
3000 对 different person
```
很多只有一张图的身份会用于 different-person pair，这是正常现象。
---
14.2 WebFace 是否需要标签文件？
不需要。
WebFace 目录中每个 `id_x` 文件夹代表一个身份，代码会自动扫描文件夹并生成标签：
```text
id_0 -> label 0
id_1 -> label 1
...
```
---
14.3 只支持 jpg 吗？
是的，本项目的 `datasets.py` 设置为：
```python
IMG_EXTENSIONS = (".jpg",)
```
因此只读取 `.jpg` 文件。
---
14.4 4 卡训练为什么 batch size 是 64？
DDP 中 `--batch_size` 是每张 GPU 的 batch size。
```text
4 GPU × 64 = global batch size 256
```
因此学习率设置为：
```text
lr = 0.2
```
如果写：
```text
--batch_size 128
```
在 4 卡上就是：
```text
4 GPU × 128 = global batch size 512
```
对应学习率通常设置为：
```text
lr = 0.4
```
---
14.5 每个 epoch 都会验证 LFW 吗？
默认会，因为训练命令中设置了：
```bash
--eval_every 1
```
如果想每 2 个 epoch 验证一次，可以设置：
```bash
--eval_every 2
```
---
14.6 训练结果保存在哪里？
默认保存在：
```text
./results
```
可以通过参数修改：
```bash
--output_dir ./results
```