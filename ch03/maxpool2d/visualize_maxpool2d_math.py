"""
最大池化运算数学过程可视化
===========================

用最直观的方式展示 MaxPooling2D 的每一步计算：
  1. 输入矩阵（4×4）与池化窗口（2×2, stride=2）
  2. 滑窗在输入上移动，每个位置的：
     - 池化窗口提取
     - 取最大值（标星号 ★）
     - 写入输出位置
  3. 反向传播：梯度仅回传到最大值位置

输出:
    pool_math/pool_forward_steps.png  — 逐步前向传播
    pool_math/pool_summary.png        — 全局总览
    pool_math/pool_backward.png       — 反向传播梯度回传可视化
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from maxpool2d import MaxPooling2D

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

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
#  绘制带数值标注的矩阵网格
# ============================================================

def draw_matrix(
        ax,
        matrix: np.ndarray,
        title: str = "",
        highlight_cells: list[tuple[int, int]] | None = None,
        highlight_color: str = "#FFEB3B",
        star_cells: list[tuple[int, int]] | None = None,
        star_color: str = "#E53935",
        cmap_name: str = "Blues",
        fmt: str = ".0f",
        fontsize: int = 14,
        title_fontsize: int = 11,
):
    """
    绘制带数值标注的矩阵网格。

    star_cells : 需要标注星号 ★ 的 (row, col) 列表（用于标记最大值）
    """
    rows, cols = matrix.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=6)

    vmin, vmax = matrix.min(), matrix.max()
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1
    cmap = plt.get_cmap(cmap_name)

    for r in range(rows):
        for c in range(cols):
            is_hl = highlight_cells and (r, c) in highlight_cells
            is_star = star_cells and (r, c) in star_cells

            if is_star:
                bg = "#FFCDD2"  # 浅红色 — 最大值
            elif is_hl:
                bg = highlight_color
            else:
                norm_val = (matrix[r, c] - vmin) / (vmax - vmin)
                bg = cmap(norm_val * 0.6 + 0.1)

            rect = patches.FancyBboxPatch(
                (c - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor="#555555", linewidth=1.0,
            )
            ax.add_patch(rect)

            val = matrix[r, c]
            text = f"{val:{fmt}}"
            color = star_color if is_star else "#222222"
            weight = "bold"
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=fontsize, fontweight=weight, color=color)

            # 星号标记
            if is_star:
                ax.text(c + 0.35, r - 0.35, "max", ha="center", va="center",
                        fontsize=6, fontweight="bold", color=star_color)

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_operator(ax, symbol: str, fontsize: int = 22):
    """在 Axes 中间绘制运算符号。"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, symbol, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#333333")
    ax.axis("off")


# ============================================================
#  可视化 1: 逐步前向传播
# ============================================================

def plot_forward_steps(
        input_matrix: np.ndarray,
        pool: MaxPooling2D,
        save_path: str | Path,
):
    """
    使用 MaxPooling2D 算子执行前向传播，然后逐步可视化每个输出位置：
      [输入(高亮窗口+标记最大值)] → max() → [输出值]
    """
    iH, iW = input_matrix.shape
    kH = kW = pool.kernel_size
    s = pool.stride

    # ---- 使用我们自己的 MaxPooling2D 执行前向传播 ----
    x_4d = input_matrix.reshape(1, 1, iH, iW)  # 2D → 4D
    out_4d = pool.forward(x_4d)  # (1, 1, oH, oW)
    output_matrix = out_4d[0, 0]  # 取回 2D
    oH, oW = output_matrix.shape

    # 从 pool._argmax_mask 中提取每个输出位置对应的最大值坐标
    max_positions = {}  # (oh, ow) -> (abs_h, abs_w)
    for oh in range(oH):
        for ow in range(oW):
            h_max = int(pool._argmax_mask[0, 0, oh, ow, 0])
            w_max = int(pool._argmax_mask[0, 0, oh, ow, 1])
            max_positions[(oh, ow)] = (h_max, w_max)

    # ---- 逐步可视化 ----
    n_steps = oH * oW
    fig = plt.figure(figsize=(18, n_steps * 2.8 + 1.5))
    gs = GridSpec(
        n_steps, 5, figure=fig,
        width_ratios=[4, 0.6, 2, 0.6, 1.5],
        wspace=0.15, hspace=0.5,
    )

    step = 0
    for oh in range(oH):
        for ow in range(oW):
            h_start = oh * s
            w_start = ow * s

            # 从输入中提取窗口（仅用于可视化展示）
            window = input_matrix[h_start:h_start + kH, w_start:w_start + kW]
            max_val = output_matrix[oh, ow]
            max_abs = max_positions[(oh, ow)]
            max_rel = (max_abs[0] - h_start, max_abs[1] - w_start)

            # 高亮和星号坐标
            highlight = [(r, c) for r in range(h_start, h_start + kH)
                         for c in range(w_start, w_start + kW)]

            # --- col0: 输入矩阵 + 高亮 + 最大值标记 ---
            ax_input = fig.add_subplot(gs[step, 0])
            draw_matrix(
                ax_input, input_matrix,
                title=(f"Step {step + 1}: 输入 (池化窗口高亮, 最大值标红)"
                       if step == 0 else f"Step {step + 1}"),
                highlight_cells=highlight,
                highlight_color="#FFF9C4",
                star_cells=[max_abs],
                cmap_name="Greys",
                fontsize=13,
                title_fontsize=9,
            )
            # 窗口边框
            rect_border = patches.Rectangle(
                (w_start - 0.5, h_start - 0.5), kW, kH,
                linewidth=2.5, edgecolor="#1565C0", facecolor="none",
                linestyle="-", zorder=10,
            )
            ax_input.add_patch(rect_border)

            # --- col1: → max ---
            ax_op1 = fig.add_subplot(gs[step, 1])
            draw_operator(ax_op1, "max\n  →", fontsize=13)

            # --- col2: 池化窗口内容 ---
            ax_window = fig.add_subplot(gs[step, 2])
            draw_matrix(
                ax_window, window,
                title="池化窗口" if step == 0 else "",
                star_cells=[max_rel],
                cmap_name="YlOrBr",
                fontsize=14,
                title_fontsize=9,
            )
            # 下方标注 max 计算
            vals = [f"{window[r, c]:.0f}" for r in range(kH) for c in range(kW)]
            calc_str = f"max({', '.join(vals)}) = {max_val:.0f}"
            ax_window.text(
                0.5, -0.2, calc_str,
                transform=ax_window.transAxes,
                ha="center", va="top",
                fontsize=8, color="#555555", style="italic",
            )

            # --- col3: = ---
            ax_op2 = fig.add_subplot(gs[step, 3])
            draw_operator(ax_op2, "=", fontsize=20)

            # --- col4: 输出值 ---
            ax_result = fig.add_subplot(gs[step, 4])
            ax_result.set_xlim(0, 1)
            ax_result.set_ylim(0, 1)
            result_box = patches.FancyBboxPatch(
                (0.1, 0.2), 0.8, 0.6,
                boxstyle="round,pad=0.08",
                facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2,
            )
            ax_result.add_patch(result_box)
            ax_result.text(
                0.5, 0.5, f"{max_val:.0f}",
                ha="center", va="center",
                fontsize=18, fontweight="bold", color="#1565C0",
            )
            ax_result.text(
                0.5, 0.05, f"Y[{oh},{ow}]",
                ha="center", va="bottom",
                fontsize=9, color="#666666",
            )
            if step == 0:
                ax_result.set_title("输出值", fontsize=9, fontweight="bold", pad=6)
            ax_result.axis("off")

            step += 1

    fig.suptitle(
        "最大池化逐步计算过程  |  Y[i,j] = max(窗口内所有元素)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 前向逐步计算图已保存: {save_path}")

    return pool, output_matrix, max_positions


# ============================================================
#  可视化 2: 全局总览
# ============================================================

def plot_summary(
        input_matrix: np.ndarray,
        output_matrix: np.ndarray,
        max_positions: dict,
        kernel_size: int,
        stride: int,
        save_path: str | Path,
):
    """输入 → MaxPool → 输出 总览图。"""
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[4, 0.8, 2.5], wspace=0.25)

    # 所有最大值位置
    all_max = list(max_positions.values())

    # 输入矩阵 — 标记所有最大值位置
    ax_input = fig.add_subplot(gs[0, 0])
    iH, iW = input_matrix.shape
    draw_matrix(
        ax_input, input_matrix,
        title=f"输入矩阵 X  ({iH}x{iW})",
        star_cells=all_max,
        cmap_name="YlOrBr",
        fontsize=15, title_fontsize=13,
    )

    # 画出所有池化窗口
    oH, oW = output_matrix.shape
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
    for idx, (oh, ow) in enumerate(
            [(oh, ow) for oh in range(oH) for ow in range(oW)]
    ):
        h_start = oh * stride
        w_start = ow * stride
        rect = patches.Rectangle(
            (w_start - 0.5, h_start - 0.5), kernel_size, kernel_size,
            linewidth=2, edgecolor=colors[idx % len(colors)],
            facecolor="none", linestyle="--", zorder=10,
        )
        ax_input.add_patch(rect)

    # MaxPool 符号
    ax_op = fig.add_subplot(gs[0, 1])
    ax_op.set_xlim(0, 1)
    ax_op.set_ylim(0, 1)
    ax_op.text(0.5, 0.55, "MaxPool", ha="center", va="center",
               fontsize=14, fontweight="bold", color="#333333")
    ax_op.text(0.5, 0.35, f"{kernel_size}x{kernel_size}, s={stride}",
               ha="center", va="center", fontsize=10, color="#666666")
    # 箭头
    ax_op.annotate("", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                   arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax_op.axis("off")

    # 输出矩阵
    ax_output = fig.add_subplot(gs[0, 2])
    draw_matrix(
        ax_output, output_matrix,
        title=f"输出矩阵 Y  ({oH}x{oW})",
        cmap_name="Blues",
        fontsize=15, title_fontsize=13,
    )

    # 公式
    fig.text(
        0.5, 0.02,
        r"$Y[i,j] = \max\; X[i \cdot s + m,\; j \cdot s + n]$"
        r"$\quad (0 \leq m < kH,\; 0 \leq n < kW)$"
        f"     (kernel={kernel_size}x{kernel_size}, stride={stride})",
        ha="center", va="bottom",
        fontsize=13, color="#333333",
    )

    fig.suptitle("最大池化运算总览", fontsize=16, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 总览图已保存: {save_path}")


# ============================================================
#  可视化 3: 反向传播梯度回传
# ============================================================

def plot_backward(
        input_matrix: np.ndarray,
        pool: MaxPooling2D,
        output_matrix: np.ndarray,
        max_positions: dict,
        grad_output_2d: np.ndarray,
        save_path: str | Path,
):
    """
    使用 MaxPooling2D.backward() 执行反向传播，然后可视化梯度回传过程。
    左: grad_output    中: argmax 掩码    右: grad_input
    """
    iH, iW = input_matrix.shape
    oH, oW = output_matrix.shape

    # ---- 使用我们自己的 MaxPooling2D 执行反向传播 ----
    grad_output_4d = grad_output_2d.reshape(1, 1, oH, oW)
    grad_input_4d = pool.backward(grad_output_4d)  # (1, 1, iH, iW)
    grad_input = grad_input_4d[0, 0]  # 取回 2D

    # ---- 可视化 ----
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 7, figure=fig,
                  width_ratios=[2.5, 0.6, 4, 0.6, 4, 0.6, 4], wspace=0.15)

    # --- col0: grad_output ---
    ax_grad_out = fig.add_subplot(gs[0, 0])
    draw_matrix(ax_grad_out, grad_output_2d,
                title=f"grad_output ({oH}x{oW})",
                cmap_name="Oranges", fontsize=15, title_fontsize=11)

    # --- col1: → ---
    ax_op1 = fig.add_subplot(gs[0, 1])
    draw_operator(ax_op1, "→", fontsize=22)

    # --- col2: 输入矩阵 + argmax 标记 ---
    ax_mask = fig.add_subplot(gs[0, 2])
    all_max = list(max_positions.values())
    draw_matrix(ax_mask, input_matrix,
                title=f"输入 X + argmax 位置 ({iH}x{iW})",
                star_cells=all_max,
                cmap_name="Greys", fontsize=14, title_fontsize=10)

    # 画箭头: 从 grad_output 的每个位置指向 argmax 位置
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
    idx = 0
    for oh in range(oH):
        for ow in range(oW):
            h_max, w_max = max_positions[(oh, ow)]
            color = colors[idx % len(colors)]

            # 在 argmax 位置画一个高亮框
            rect = patches.FancyBboxPatch(
                (w_max - 0.48, h_max - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor="none", edgecolor=color, linewidth=2.5, zorder=15,
            )
            ax_mask.add_patch(rect)

            # 标注梯度值
            ax_mask.text(
                w_max, h_max + 0.38, f"+{grad_output_2d[oh, ow]:.0f}",
                ha="center", va="center",
                fontsize=7, fontweight="bold", color=color,
            )
            idx += 1

    # --- col3: = ---
    ax_op2 = fig.add_subplot(gs[0, 3])
    draw_operator(ax_op2, "=", fontsize=22)

    # --- col4: grad_input (由 MaxPooling2D.backward() 计算) ---
    ax_grad_in = fig.add_subplot(gs[0, 4])
    nonzero_cells = [(r, c) for r in range(iH) for c in range(iW)
                     if grad_input[r, c] != 0]
    draw_matrix(ax_grad_in, grad_input,
                title=f"grad_input ({iH}x{iW})",
                highlight_cells=nonzero_cells,
                highlight_color="#C8E6C9",
                cmap_name="Greys", fontsize=14, title_fontsize=11)

    # --- col5: 对比 ---
    ax_op3 = fig.add_subplot(gs[0, 5])
    draw_operator(ax_op3, "vs", fontsize=14)

    # --- col6: 输入矩阵（对比参考） ---
    ax_ref = fig.add_subplot(gs[0, 6])
    draw_matrix(ax_ref, input_matrix,
                title=f"原始输入 X (对比)",
                cmap_name="YlOrBr", fontsize=14, title_fontsize=11)

    # 说明文字
    fig.text(
        0.5, 0.01,
        "反向传播规则:  dX[h,w] = dY[i,j] (若 X[h,w] 是窗口最大值)，否则为 0"
        "    —— 梯度只流向最大值位置，其余位置为 0",
        ha="center", va="bottom",
        fontsize=11, color="#333333",
    )

    fig.suptitle(
        "最大池化反向传播：梯度回传过程",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 反向传播图已保存: {save_path}")


# ============================================================
#  Main
# ============================================================

def main():
    # ---------- 定义输入 ----------
    input_matrix = np.array([
        [1, 3, 2, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 0],
        [1, 2, 3, 4],
    ], dtype=np.float64)

    kernel_size = 2
    stride = 2

    # ---- 使用我们自己的 MaxPooling2D 算子 ----
    pool = MaxPooling2D(kernel_size=kernel_size, stride=stride)

    print("=" * 50)
    print(" 最大池化运算数学过程可视化")
    print(f" 使用算子: {pool}")
    print("=" * 50)
    print(f"\n📐 输入矩阵 X ({input_matrix.shape[0]}x{input_matrix.shape[1]}):")
    print(input_matrix.astype(int))
    print(f"\n📐 池化窗口: {kernel_size}x{kernel_size}, stride={stride}")

    oH = (input_matrix.shape[0] - kernel_size) // stride + 1
    oW = (input_matrix.shape[1] - kernel_size) // stride + 1
    print(f"📐 输出尺寸: {oH}x{oW}")
    print(f"   公式: ({input_matrix.shape[0]} - {kernel_size}) / {stride} + 1"
          f" = {oH}\n")

    # ---- 输出目录 ----
    output_dir = SCRIPT_DIR / "maxpool2d_math"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图 1: 逐步前向传播（内部调用 pool.forward()）
    print("🎨 绘制逐步前向传播...")
    pool, output_matrix, max_positions = plot_forward_steps(
        input_matrix, pool,
        output_dir / "maxpool2d_math_steps.png",
    )

    print(f"\n📐 输出矩阵 Y:")
    print(output_matrix.astype(int))

    # 图 2: 全局总览
    print("\n🎨 绘制全局总览...")
    plot_summary(
        input_matrix, output_matrix, max_positions,
        kernel_size, stride,
        output_dir / "maxpool2d_math_summary.png",
    )

    # 图 3: 反向传播（内部调用 pool.backward()）
    print("\n🎨 绘制反向传播...")
    grad_output = np.array([
        [1, 1],
        [1, 1],
    ], dtype=np.float64)
    plot_backward(
        input_matrix, pool, output_matrix, max_positions, grad_output,
        output_dir / "maxpool2d_math_backward.png",
    )

    print("\n🎉 所有可视化完成！")
    print(f"   📁 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
