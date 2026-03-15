## Kaggle CIFAR-10

这个目录用于实战 Kaggle `CIFAR-10 - Object Recognition in Images` 比赛：
[https://www.kaggle.com/c/cifar-10](https://www.kaggle.com/c/cifar-10)

### 模型选择

默认使用 `EfficientNet-B0` 的 ImageNet 预训练权重。

原因：

- 相比 `EfficientNetV2-S` 更轻，适合 `M2 Pro / 16GB` 这类本机训练环境
- 相比 `ResNet18/34`，通常仍能保持较强的迁移学习效果
- 对 CIFAR-10 这种小图像分类任务，作为本地可训练的强 baseline 很合适

### 目录结构

推荐先把 Kaggle 原始文件放到 `raw/` 目录，再用预处理脚本自动整理。

原始文件结构：

```text
ch02/cifar10_kaggle/
  raw/
    cifar-10.zip
```

执行预处理脚本后，会得到下面结构：

```text
ch02/cifar10_kaggle/
  data/
    cifar-10/
      train/
        1.png
        2.png
        ...
      test/
        1.png
        2.png
        ...
      trainLabels.csv
      sampleSubmission.csv
```

### 运行方式

先激活项目虚拟环境：

```bash
source .venv/bin/activate
```

安装新增依赖：

```bash
pip install -r requirements.txt
```

先准备数据：

```bash
python ch02/cifar10_kaggle/prepare_data.py
```

如果要覆盖重建：

```bash
python ch02/cifar10_kaggle/prepare_data.py --force
```

训练并生成提交文件：

```bash
python ch02/cifar10_kaggle/train.py
```

训练完成后，如需单独生成 Kaggle 提交文件：

```bash
python ch02/cifar10_kaggle/train.py --predict-only
```

如果想先快速试跑：

```bash
python ch02/cifar10_kaggle/train.py --epochs 4 --image-size 96 --batch-size 96 --model mobilenet_v3_large
```

只做脚本自检：

```bash
python ch02/cifar10_kaggle/train.py --dry-run
```

### 输出文件

训练阶段会在当前目录下生成：

- `outputs/best_model.pt`
- `outputs/metrics.json`
- `outputs/training_curves.png`

执行 `--predict-only` 后会额外生成：

- `outputs/submission.csv`

### macOS / MPS

脚本会自动按以下优先级选择设备：

1. `cuda`
2. `mps`
3. `cpu`

在 Apple Silicon 上会优先使用 `mps`。
