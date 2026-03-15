"""
Conv2D 多层卷积特征图可视化
============================

将一张真实图片（CIFAR-10）依次通过多个 Conv2D 层 + ReLU，
可视化每一层输出的特征图变化过程。

输出:
    feature_maps_overview.png  — 总览图（每层选取代表性通道）
    feature_maps_detail.png    — 详细图（每层展示所有通道的网格）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 兼容直接运行和模块导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conv2d import Conv2D

# matplotlib 配置（避免 macOS 缓存目录权限问题）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
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
    """ReLU 激活函数"""
    return np.maximum(0, x)


def normalize(arr: np.ndarray) -> np.ndarray:
    """将数组归一化到 [0, 1]"""
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    加载图片并转为 (1, 3, H, W) 的 float64 张量，像素值归一化到 [0, 1]。
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float64) / 255.0  # (H, W, 3)
    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis, ...]  # (1, 3, H, W)


def select_representative_channels(feature_map: np.ndarray, n: int = 1) -> list[int]:
    """
    从 (C, H, W) 特征图中，按每个通道的方差排序，选出最具代表性的 n 个通道。
    """
    C = feature_map.shape[0]
    variances = [feature_map[c].var() for c in range(C)]
    indices = sorted(range(C), key=lambda i: variances[i], reverse=True)
    return indices[:n]


# ============================================================
#  构建多层卷积管道
# ============================================================

def build_layers() -> list[dict]:
    """
    构建一个 4 层卷积管道，模拟典型 CNN 的特征提取过程：
      Layer 1: 3  -> 8  通道, kernel=3, padding=1  (边缘 / 颜色检测)
      Layer 2: 8  -> 16 通道, kernel=3, padding=1  (纹理检测)
      Layer 3: 16 -> 32 通道, kernel=3, stride=2, padding=1  (下采样 + 复杂特征)
      Layer 4: 32 -> 64 通道, kernel=3, stride=2, padding=1  (高级特征 + 进一步下采样)
    """
    np.random.seed(2024)

    layers = [
        {
            "name": "Conv1 (3→8, k3, p1)",
            "conv": Conv2D(3, 8, kernel_size=3, stride=1, padding=1),
        },
        {
            "name": "Conv2 (8→16, k3, p1)",
            "conv": Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
        },
        {
            "name": "Conv3 (16→32, k3, s2, p1)",
            "conv": Conv2D(16, 32, kernel_size=3, stride=2, padding=1),
        },
        {
            "name": "Conv4 (32→64, k3, s2, p1)",
            "conv": Conv2D(32, 64, kernel_size=3, stride=2, padding=1),
        },
    ]
    return layers


# ============================================================
#  前向推理 & 收集特征图
# ============================================================

def forward_through_layers(x: np.ndarray, layers: list[dict]) -> list[dict]:
    """
    依次通过各层卷积 + ReLU，收集每层的输出特征图。

    Returns
    -------
    results: list[dict]
        每项包含 name, output (N,C,H,W)
    """
    results = []
    current = x

    for layer_info in layers:
        conv: Conv2D = layer_info["conv"]
        out = conv.forward(current)
        out = relu(out)
        results.append({
            "name": layer_info["name"],
            "output": out,
        })
        current = out
        print(f"  {layer_info['name']:30s}  →  output shape: {out.shape}")

    return results


# ============================================================
#  可视化 1: 总览图 (每层选一个代表通道)
# ============================================================

def plot_overview(
        input_img: np.ndarray,
        results: list[dict],
        label: str,
        save_path: str | Path,
):
    """
    一行展示：原图 → 各层最具代表性的通道特征图。
    """
    n_stages = 1 + len(results)
    fig, axes = plt.subplots(1, n_stages, figsize=(3.2 * n_stages, 3.6))

    # 原始图像 (CHW -> HWC)
    img_hwc = input_img[0].transpose(1, 2, 0)  # (H, W, 3)
    axes[0].imshow(np.clip(img_hwc, 0, 1))
    axes[0].set_title(f"Input ({label})\n{input_img.shape[1:]}",
                      fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # 各层特征图
    for i, result in enumerate(results):
        feat = result["output"][0]  # (C, H, W) — 取 batch 中的第一张
        ch = select_representative_channels(feat, n=1)[0]
        vis = normalize(feat[ch])

        axes[i + 1].imshow(vis, cmap="magma")
        axes[i + 1].set_title(f"{result['name']}\nch={ch}, {feat.shape}",
                              fontsize=8)
        axes[i + 1].axis("off")

    fig.suptitle("Conv2D Feature Extraction Pipeline (NumPy)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 总览图已保存: {save_path}")


# ============================================================
#  可视化 2: 详细图 (每层展示多通道网格)
# ============================================================

def plot_detail(
        input_img: np.ndarray,
        results: list[dict],
        label: str,
        save_path: str | Path,
        max_channels: int = 16,
):
    """
    为每一层绘制一行，展示前 max_channels 个通道的特征图网格。
    """
    n_layers = len(results)
    fig, axes = plt.subplots(
        n_layers, max_channels + 1,
        figsize=(max_channels * 1.1 + 1.5, n_layers * 1.4 + 0.8),
    )

    for row, result in enumerate(results):
        feat = result["output"][0]  # (C, H, W)
        C = feat.shape[0]

        # 第一列：层名标签（用空白图占位）
        ax_label = axes[row, 0]
        ax_label.text(
            0.5, 0.5, result["name"],
            transform=ax_label.transAxes,
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            rotation=0, wrap=True,
        )
        ax_label.axis("off")

        # 后续列：各通道特征图
        for col in range(max_channels):
            ax = axes[row, col + 1]
            if col < C:
                vis = normalize(feat[col])
                ax.imshow(vis, cmap="magma", aspect="equal")
                ax.set_title(f"ch{col}", fontsize=5, pad=1)
            ax.axis("off")

    fig.suptitle("Conv2D Per-Channel Feature Maps (NumPy)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ 详细图已保存: {save_path}")


# ============================================================
#  反向传播 & 收集梯度
# ============================================================

def backward_through_layers(
        layers: list[dict],
        results: list[dict],
) -> list[np.ndarray]:
    """
    从最后一层向前反向传播，收集每层的 grad_input。
    使用全 1 的 grad_output 作为起点（等价于 sum 损失函数）。

    Returns
    -------
    grad_maps : list[np.ndarray]
        每层的 grad_input (N, C, H, W)，顺序与 layers 一致。
    """
    # 从最后一层的输出开始，grad = 全 1
    grad = np.ones_like(results[-1]["output"])

    grad_maps = [None] * len(layers)

    for i in reversed(range(len(layers))):
        conv: Conv2D = layers[i]["conv"]
        # ReLU 的梯度: 前向输出 > 0 的位置保留梯度，否则置 0
        relu_mask = (results[i]["output"] > 0).astype(grad.dtype)
        grad = grad * relu_mask
        # Conv2D 反向传播
        grad = conv.backward(grad)
        grad_maps[i] = grad

    return grad_maps


# ============================================================
#  可视化 3: 反向传播梯度热力图
# ============================================================

def plot_gradient_overview(
        input_img: np.ndarray,
        layers: list[dict],
        results: list[dict],
        grad_maps: list[np.ndarray],
        label: str,
        save_path,
):
    """
    一行展示：原图梯度 ← 各层梯度热力图（反向方向）。
    """
    # grad_maps[0] 是最接近输入的梯度
    n_stages = 1 + len(results)
    fig, axes = plt.subplots(1, n_stages, figsize=(3.2 * n_stages, 3.6))

    # 最后一列: 输入图像上的梯度热力图 (grad_maps[0] 的梯度幅值)
    grad_input = grad_maps[0]  # (N, C, H, W)
    # 对通道取绝对值平均，得到单通道热力图
    grad_heatmap = np.abs(grad_input[0]).mean(axis=0)  # (H, W)
    grad_heatmap = grad_heatmap / (grad_heatmap.max() + 1e-8)

    # 在原图上叠加梯度热力图
    img_hwc = input_img[0].transpose(1, 2, 0)
    axes[0].imshow(np.clip(img_hwc, 0, 1))
    axes[0].imshow(grad_heatmap, cmap="hot", alpha=0.5)
    axes[0].set_title(f"Input + grad\n({label})", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # 各层梯度热力图
    for i in range(len(results)):
        ax = axes[i + 1]
        if i < len(grad_maps) and grad_maps[i] is not None:
            gm = grad_maps[i][0]  # (C, H, W)
        else:
            gm = np.abs(results[i]["output"][0])

        # 选一个梯度最大的通道
        ch_grad_mag = [np.abs(gm[c]).sum() for c in range(gm.shape[0])]
        ch = int(np.argmax(ch_grad_mag))
        vis = np.abs(gm[ch])
        vis = vis / (vis.max() + 1e-8)

        ax.imshow(vis, cmap="hot")
        ax.set_title(
            f"{results[i]['name']}\n|grad| ch={ch}, {gm.shape}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        "Conv2D 反向传播梯度流 (NumPy)\n梯度幅值热力图 — 亮色=梯度大",
        fontsize=13, fontweight="bold", y=1.02,
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
    # 优先使用 CIFAR-10 真实图片
    cifar_dir = Path(__file__).resolve().parents[1] / ".." / "ch02" / "cifar10_kaggle" / "data" / "cifar-10" / "train"
    cifar_dir = cifar_dir.resolve()

    if cifar_dir.exists():
        # 取一张猫的图片 (id=3 在 CIFAR-10 trainLabels 中是 cat)
        # 读 trainLabels.csv 找到一张 cat
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
            # fallback: 取第一张图
            image_path = sorted(cifar_dir.glob("*.png"))[0]
            label_name = "unknown"

        print(f"📷 加载图片: {image_path}")
        x = load_image(image_path)
    else:
        # 如果没有 CIFAR-10 数据，生成一张合成图片
        print("📷 CIFAR-10 数据不可用，使用合成图片")
        label_name = "synthetic"
        np.random.seed(42)
        # 生成一个彩色渐变 + 棋盘格叠加的 32x32 图像
        h, w = 32, 32
        r = np.linspace(0, 1, h).reshape(h, 1) * np.ones((1, w))
        g = np.linspace(1, 0, w).reshape(1, w) * np.ones((h, 1))
        b = np.abs(np.sin(np.linspace(0, 4 * np.pi, h).reshape(h, 1)
                          + np.linspace(0, 4 * np.pi, w).reshape(1, w)))
        checkerboard = ((np.indices((h, w)).sum(axis=0) // 4) % 2).astype(np.float64) * 0.3
        r = np.clip(r + checkerboard, 0, 1)
        g = np.clip(g + checkerboard, 0, 1)
        b = np.clip(b, 0, 1)
        img = np.stack([r, g, b], axis=0)  # (3, H, W)
        x = img[np.newaxis, ...]  # (1, 3, H, W)

    print(f"   输入张量形状: {x.shape}")

    # ---------- 构建 & 推理 ----------
    print("\n🔧 构建卷积管道...")
    layers = build_layers()

    print("\n🚀 前向传播...")
    results = forward_through_layers(x, layers)

    # ---------- 反向传播 ----------
    print("\n🔙 反向传播...")
    grad_maps = backward_through_layers(layers, results)
    for i, gm in enumerate(grad_maps):
        print(f"  {layers[i]['name']:30s}  ←  grad shape: {gm.shape}")

    # ---------- 可视化 ----------
    output_dir = SCRIPT_DIR / "conv2d_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(x, results, label_name, output_dir / "conv2d_feature_maps_overview.png")
    plot_detail(x, results, label_name, output_dir / "conv2d_feature_maps_detail.png")
    plot_gradient_overview(x, layers, results, grad_maps, label_name,
                           output_dir / "conv2d_gradient_overview.png")

    print("\n🎉 可视化完成！")


if __name__ == "__main__":
    main()
