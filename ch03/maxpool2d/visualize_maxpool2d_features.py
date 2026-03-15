"""
Conv + MaxPool 多层特征图可视化
================================

将一张真实图片（CIFAR-10）依次通过 Conv2D + ReLU + MaxPool 的交替管道，
可视化每一层输出的特征图变化过程，直观感受池化层的下采样效果。

管道结构（模拟经典 CNN）：
    Input (3, 32, 32)
      → Conv1 (3→8, k3, p1)  + ReLU    →  (8, 32, 32)
      → MaxPool (2x2, s2)               →  (8, 16, 16)   ← 空间减半
      → Conv2 (8→16, k3, p1) + ReLU    →  (16, 16, 16)
      → MaxPool (2x2, s2)               →  (16, 8, 8)    ← 再减半
      → Conv3 (16→32, k3, p1) + ReLU   →  (32, 8, 8)
      → MaxPool (2x2, s2)               →  (32, 4, 4)    ← 再减半

输出:
    pool_features/pool_feature_maps_overview.png  — 总览图
    pool_features/pool_feature_maps_detail.png    — 详细多通道图
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 导入我们自己实现的算子
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CONV2D_DIR = SCRIPT_DIR.parent / "conv2d"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CONV2D_DIR))
from conv2d import Conv2D
from maxpool2d import MaxPooling2D

# matplotlib 配置
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib.pyplot as plt

# ---- 配置中文字体 ----
_CJK_CANDIDATES = [
    "PingFang HK", "PingFang SC", "STHeiti",
    "Heiti TC", "Songti SC", "Arial Unicode MS",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei",
]
for _font in _CJK_CANDIDATES:
    if _font in [f.name for f in plt.matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
#  辅助函数
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def load_image(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def select_representative_channels(feature_map: np.ndarray, n: int = 1) -> list[int]:
    C = feature_map.shape[0]
    variances = [feature_map[c].var() for c in range(C)]
    indices = sorted(range(C), key=lambda i: variances[i], reverse=True)
    return indices[:n]


# ============================================================
#  构建 Conv + MaxPool 管道
# ============================================================

def build_pipeline() -> list[dict]:
    """
    构建 Conv + ReLU + MaxPool 交替的管道，模拟经典 CNN 特征提取过程。
    每一步都是一个独立的层，方便逐层可视化。
    """
    np.random.seed(2024)

    pipeline = [
        {
            "name": "Conv1 (3->8, k3, p1) + ReLU",
            "type": "conv",
            "layer": Conv2D(3, 8, kernel_size=3, stride=1, padding=1),
        },
        {
            "name": "MaxPool (2x2, s2)",
            "type": "pool",
            "layer": MaxPooling2D(kernel_size=2, stride=2),
        },
        {
            "name": "Conv2 (8->16, k3, p1) + ReLU",
            "type": "conv",
            "layer": Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
        },
        {
            "name": "MaxPool (2x2, s2)",
            "type": "pool",
            "layer": MaxPooling2D(kernel_size=2, stride=2),
        },
        {
            "name": "Conv3 (16->32, k3, p1) + ReLU",
            "type": "conv",
            "layer": Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
        },
        {
            "name": "MaxPool (2x2, s2)",
            "type": "pool",
            "layer": MaxPooling2D(kernel_size=2, stride=2),
        },
    ]
    return pipeline


# ============================================================
#  前向推理 & 收集特征图
# ============================================================

def forward_through_pipeline(x: np.ndarray, pipeline: list[dict]) -> list[dict]:
    results = []
    current = x

    for info in pipeline:
        layer = info["layer"]
        out = layer.forward(current)
        if info["type"] == "conv":
            out = relu(out)

        results.append({
            "name": info["name"],
            "type": info["type"],
            "output": out,
        })
        current = out
        tag = "Conv+ReLU" if info["type"] == "conv" else "MaxPool"
        print(f"  [{tag:10s}] {info['name']:35s}  ->  {out.shape}")

    return results


# ============================================================
#  可视化 1: 总览图
# ============================================================

def plot_overview(
        input_img: np.ndarray,
        results: list[dict],
        label: str,
        save_path: str | Path,
):
    n_stages = 1 + len(results)
    fig, axes = plt.subplots(1, n_stages, figsize=(2.8 * n_stages, 3.8))

    # 原始图像
    img_hwc = input_img[0].transpose(1, 2, 0)
    axes[0].imshow(np.clip(img_hwc, 0, 1))
    axes[0].set_title(f"Input ({label})\n{input_img.shape[1:]}",
                      fontsize=8, fontweight="bold")
    axes[0].axis("off")

    for i, result in enumerate(results):
        feat = result["output"][0]
        ch = select_representative_channels(feat, n=1)[0]
        vis = normalize(feat[ch])

        # 池化层用不同 colormap 区分
        cmap = "viridis" if result["type"] == "pool" else "magma"

        axes[i + 1].imshow(vis, cmap=cmap)

        # 标题颜色区分: 池化层蓝色, 卷积层默认黑色
        title_color = "#1565C0" if result["type"] == "pool" else "#333333"
        axes[i + 1].set_title(
            f"{result['name']}\nch={ch}, {feat.shape}",
            fontsize=7, color=title_color, fontweight="bold",
        )
        axes[i + 1].axis("off")

        # 池化层加蓝色边框
        if result["type"] == "pool":
            for spine in axes[i + 1].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(2)

    fig.suptitle(
        "Conv + MaxPool 特征提取管道 (NumPy 实现)\n"
        "紫色=卷积层输出, 绿色=池化层输出",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 总览图已保存: {save_path}")


# ============================================================
#  可视化 2: 详细多通道图
# ============================================================

def plot_detail(
        input_img: np.ndarray,
        results: list[dict],
        label: str,
        save_path: str | Path,
        max_channels: int = 16,
):
    n_layers = len(results)
    fig, axes = plt.subplots(
        n_layers, max_channels + 1,
        figsize=(max_channels * 1.1 + 1.5, n_layers * 1.3 + 1.0),
    )

    for row, result in enumerate(results):
        feat = result["output"][0]
        C = feat.shape[0]
        is_pool = result["type"] == "pool"

        # 第一列：层名
        ax_label = axes[row, 0]
        color = "#1565C0" if is_pool else "#333333"
        ax_label.text(
            0.5, 0.5, result["name"],
            transform=ax_label.transAxes,
            ha="center", va="center",
            fontsize=6, fontweight="bold",
            color=color, wrap=True,
        )
        ax_label.axis("off")

        # 特征图通道
        cmap = "viridis" if is_pool else "magma"
        for col in range(max_channels):
            ax = axes[row, col + 1]
            if col < C:
                vis = normalize(feat[col])
                ax.imshow(vis, cmap=cmap, aspect="equal")
                ax.set_title(f"ch{col}", fontsize=4, pad=1)
            ax.axis("off")

    fig.suptitle(
        "Conv + MaxPool 各层各通道特征图 (NumPy)\n"
        "紫色=卷积层, 绿色=池化层",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ 详细图已保存: {save_path}")


# ============================================================
#  可视化 3: 池化前后对比图
# ============================================================

def plot_pool_comparison(
        results: list[dict],
        save_path: str | Path,
        n_channels: int = 4,
):
    """
    对比每个 MaxPool 层前后的特征图，直观展示池化的下采样效果。
    """
    # 找出所有 (conv, pool) 对
    pairs = []
    for i, result in enumerate(results):
        if result["type"] == "pool" and i > 0:
            pairs.append((results[i - 1], result))

    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        n_pairs, n_channels * 2 + 1,
        figsize=(n_channels * 2 * 1.5 + 2, n_pairs * 2.2 + 1),
    )
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for row, (before, after) in enumerate(pairs):
        feat_before = before["output"][0]
        feat_after = after["output"][0]

        # 选取代表通道
        channels = select_representative_channels(feat_before, n=n_channels)

        # 标签列
        ax_label = axes[row, 0]
        ax_label.text(
            0.5, 0.5,
            f"{before['name']}\n     ↓\n{after['name']}",
            transform=ax_label.transAxes,
            ha="center", va="center",
            fontsize=6, fontweight="bold", color="#333333",
        )
        ax_label.axis("off")

        for j, ch in enumerate(channels):
            # 池化前
            ax_before = axes[row, 1 + j * 2]
            vis_b = normalize(feat_before[ch])
            ax_before.imshow(vis_b, cmap="magma", aspect="equal")
            h_b, w_b = feat_before.shape[1], feat_before.shape[2]
            ax_before.set_title(f"ch{ch} ({h_b}x{w_b})", fontsize=6, pad=2)
            ax_before.axis("off")

            # 池化后
            ax_after = axes[row, 2 + j * 2]
            vis_a = normalize(feat_after[ch])
            ax_after.imshow(vis_a, cmap="viridis", aspect="equal")
            h_a, w_a = feat_after.shape[1], feat_after.shape[2]
            ax_after.set_title(f"ch{ch} ({h_a}x{w_a})", fontsize=6, pad=2)
            ax_after.axis("off")

            # 蓝色边框标识池化后
            for spine in ax_after.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(1.5)

    fig.suptitle(
        "MaxPool 前后对比：每对左(紫)=池化前, 右(绿)=池化后\n"
        "注意空间尺寸减半，但保留了主要特征",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ 池化前后对比图已保存: {save_path}")


# ============================================================
#  反向传播 & 收集梯度
# ============================================================

def backward_through_pipeline(
        pipeline: list[dict],
        results: list[dict],
) -> list[np.ndarray]:
    """
    从最后一层向前反向传播，收集每层的 grad_input。
    使用全 1 的 grad_output 作为起点（等价于 sum 损失函数）。

    Returns
    -------
    grad_maps : list[np.ndarray]
        每层的 grad_input (N, C, H, W)，顺序与 pipeline 一致。
    """
    grad = np.ones_like(results[-1]["output"])
    grad_maps = [None] * len(pipeline)

    for i in reversed(range(len(pipeline))):
        info = pipeline[i]
        layer = info["layer"]

        if info["type"] == "conv":
            # ReLU 的梯度: 前向输出 > 0 的位置保留梯度，否则置 0
            relu_mask = (results[i]["output"] > 0).astype(grad.dtype)
            grad = grad * relu_mask

        # 层的反向传播
        grad = layer.backward(grad)
        grad_maps[i] = grad

    return grad_maps


# ============================================================
#  可视化 4: 反向传播梯度热力图
# ============================================================

def plot_gradient_overview(
        input_img: np.ndarray,
        pipeline: list[dict],
        results: list[dict],
        grad_maps: list[np.ndarray],
        label: str,
        save_path,
):
    """
    一行展示各层的梯度幅值热力图，用颜色区分 Conv 和 Pool 层。
    """
    n_stages = 1 + len(results)
    fig, axes = plt.subplots(1, n_stages, figsize=(2.8 * n_stages, 3.8))

    # 第一列: 输入图像上的梯度热力图
    grad_input = grad_maps[0]  # (N, C, H, W)
    grad_heatmap = np.abs(grad_input[0]).mean(axis=0)  # (H, W)
    grad_heatmap = grad_heatmap / (grad_heatmap.max() + 1e-8)

    img_hwc = input_img[0].transpose(1, 2, 0)
    axes[0].imshow(np.clip(img_hwc, 0, 1))
    axes[0].imshow(grad_heatmap, cmap="hot", alpha=0.5)
    axes[0].set_title(f"Input + grad\n({label})", fontsize=8, fontweight="bold")
    axes[0].axis("off")

    # 各层梯度热力图
    for i in range(len(results)):
        ax = axes[i + 1]
        gm = grad_maps[i][0]  # (C, H, W)
        is_pool = results[i]["type"] == "pool"

        # 选梯度幅值最大的通道
        ch_grad_mag = [np.abs(gm[c]).sum() for c in range(gm.shape[0])]
        ch = int(np.argmax(ch_grad_mag))
        vis = np.abs(gm[ch])
        vis = vis / (vis.max() + 1e-8)

        ax.imshow(vis, cmap="hot")

        title_color = "#1565C0" if is_pool else "#333333"
        ax.set_title(
            f"{results[i]['name']}\n|grad| ch={ch}, {gm.shape}",
            fontsize=7, color=title_color, fontweight="bold",
        )
        ax.axis("off")

        if is_pool:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(2)

    fig.suptitle(
        "Conv + MaxPool 反向传播梯度流 (NumPy)\n"
        "梯度幅值热力图 — 亮色=梯度大",
        fontsize=12, fontweight="bold", y=1.05,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ 梯度热力图已保存: {save_path}")


# ============================================================
#  Main
# ============================================================

def main():
    # ---------- 加载图片 ----------
    cifar_dir = (SCRIPT_DIR / ".." / ".." / "ch02" / "cifar10_kaggle"
                 / "data" / "cifar-10" / "train").resolve()

    if cifar_dir.exists():
        labels_csv = cifar_dir.parent / "trainLabels.csv"
        label_name = "cat"
        image_id = None
        if labels_csv.exists():
            import csv
            with labels_csv.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["label"] == label_name:
                        image_id = row["id"]
                        break
        if image_id is not None:
            image_path = cifar_dir / f"{image_id}.png"
        else:
            image_path = sorted(cifar_dir.glob("*.png"))[0]
            label_name = "unknown"

        print(f"📷 加载图片: {image_path}")
        x = load_image(image_path)
    else:
        print("📷 CIFAR-10 数据不可用，使用合成图片")
        label_name = "synthetic"
        np.random.seed(42)
        h, w = 32, 32
        r = np.linspace(0, 1, h).reshape(h, 1) * np.ones((1, w))
        g = np.linspace(1, 0, w).reshape(1, w) * np.ones((h, 1))
        b = np.abs(np.sin(np.linspace(0, 4 * np.pi, h).reshape(h, 1)
                          + np.linspace(0, 4 * np.pi, w).reshape(1, w)))
        img = np.stack([r, g, b], axis=0)
        x = img[np.newaxis, ...]

    print(f"   输入张量形状: {x.shape}")

    # ---------- 构建 & 推理 ----------
    print("\n🔧 构建 Conv + MaxPool 管道...")
    pipeline = build_pipeline()

    print("\n🚀 前向传播...")
    results = forward_through_pipeline(x, pipeline)

    # ---------- 反向传播 ----------
    print("\n🔙 反向传播...")
    grad_maps = backward_through_pipeline(pipeline, results)
    for i, gm in enumerate(grad_maps):
        tag = "Conv+ReLU" if pipeline[i]["type"] == "conv" else "MaxPool"
        print(f"  [{tag:10s}] {pipeline[i]['name']:35s}  <-  grad shape: {gm.shape}")

    # ---------- 可视化 ----------
    output_dir = SCRIPT_DIR / "maxpool2d_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_overview(x, results, label_name,
                  output_dir / "maxpool2d_feature_maps_overview.png")
    plot_detail(x, results, label_name,
                output_dir / "maxpool2d_feature_maps_detail.png")
    plot_pool_comparison(results,
                         output_dir / "maxpool2d_before_after.png")
    plot_gradient_overview(x, pipeline, results, grad_maps, label_name,
                           output_dir / "maxpool2d_gradient_overview.png")

    print("\n🎉 可视化完成！")


if __name__ == "__main__":
    main()
