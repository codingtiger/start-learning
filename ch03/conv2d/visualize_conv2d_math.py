"""
卷积运算数学过程可视化
=======================

用最直观的方式展示二维卷积的每一步数学计算：
  1. 输入矩阵（5×5）与卷积核（3×3）
  2. 卷积核在输入上滑动，每个位置的：
     - 感受野提取
     - 逐元素相乘（⊙ Hadamard 积）
     - 求和得到输出值
  3. 最终输出矩阵（3×3）

所有数值直接标注在格子中，让每一步运算一目了然。

输出:
    conv_math_steps.png    — 9 步滑窗计算过程详解
    conv_math_summary.png  — 全局总览（输入 → 卷积核 → 输出）
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

from conv2d import Conv2D

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- 配置中文字体 ----
# macOS 优先使用 PingFang / STHeiti；Linux 回退到 Noto Sans CJK
_CJK_CANDIDATES = [
    "PingFang HK", "PingFang SC", "STHeiti",
    "Heiti TC", "Songti SC", "Arial Unicode MS",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei",
]
for _font in _CJK_CANDIDATES:
    if _font in [f.name for f in plt.matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# ============================================================
#  绘制带数值标注的矩阵网格
# ============================================================

def draw_matrix(
        ax,
        matrix: np.ndarray,
        title: str = "",
        highlight_cells: list[tuple[int, int]] | None = None,
        highlight_color: str = "#FFEB3B",
        cell_colors: np.ndarray | None = None,
        cmap_name: str = "Blues",
        fmt: str = ".0f",
        fontsize: int = 13,
        title_fontsize: int = 11,
        show_border: bool = True,
):
    """
    在指定 Axes 上绘制一个带数值标注的矩阵网格。

    Parameters
    ----------
    matrix          : 2D numpy array
    title           : 标题
    highlight_cells : 需要高亮的 (row, col) 列表
    highlight_color : 高亮颜色
    cell_colors     : 自定义每个单元格的颜色值（用于 colormap）
    cmap_name       : colormap 名称
    fmt             : 数值格式化字符串
    fontsize        : 数值字号
    """
    rows, cols = matrix.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=6)

    # 颜色映射
    if cell_colors is not None:
        color_data = cell_colors
    else:
        color_data = matrix

    vmin, vmax = color_data.min(), color_data.max()
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1

    cmap = plt.get_cmap(cmap_name)

    for r in range(rows):
        for c in range(cols):
            # 决定背景色
            is_highlighted = highlight_cells and (r, c) in highlight_cells
            if is_highlighted:
                bg = highlight_color
            else:
                norm_val = (color_data[r, c] - vmin) / (vmax - vmin)
                bg = cmap(norm_val * 0.6 + 0.1)  # 偏淡一些

            rect = patches.FancyBboxPatch(
                (c - 0.48, r - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor="#555555", linewidth=1.0,
            )
            ax.add_patch(rect)

            # 数值文本
            val = matrix[r, c]
            # 根据背景亮度选择文字颜色
            text_color = "#222222"
            ax.text(
                c, r, f"{val:{fmt}}",
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color,
            )

    if show_border:
        for spine in ax.spines.values():
            spine.set_visible(False)


def draw_operator(ax, symbol: str, fontsize: int = 22):
    """在一个 Axes 中间绘制运算符号。"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, symbol, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#333333")
    ax.axis("off")


# ============================================================
#  可视化 1: 逐步滑窗计算过程
# ============================================================

def plot_conv_steps(
        input_matrix: np.ndarray,
        conv: Conv2D,
        save_path: str | Path,
):
    """
    使用 Conv2D 算子执行前向传播，然后逐步可视化每个输出位置：
      [输入(高亮感受野)] ⊙ [卷积核] = [逐元素乘积] → sum = 输出值
    """
    iH, iW = input_matrix.shape
    kernel = conv.weight[0, 0]  # (kH, kW) — 单通道单输出
    kH, kW = kernel.shape

    # ---- 使用我们自己的 Conv2D 执行前向传播 ----
    x_4d = input_matrix.reshape(1, 1, iH, iW)  # 2D → 4D
    out_4d = conv.forward(x_4d)  # (1, 1, oH, oW)
    output_matrix = out_4d[0, 0]  # 取回 2D
    oH, oW = output_matrix.shape

    n_steps = oH * oW

    # 布局: 每个 step 一行, 每行 7 列:
    #   col0: 输入矩阵(高亮)  col1: ⊙  col2: 卷积核  col3: =
    #   col4: 逐元素乘积  col5: →  col6: 求和结果
    fig = plt.figure(figsize=(24, n_steps * 2.5 + 1.5))

    # 用 GridSpec 精确控制列宽比
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        n_steps, 7, figure=fig,
        width_ratios=[5, 0.8, 3, 0.8, 3, 0.8, 1.5],
        wspace=0.15, hspace=0.45,
    )

    step = 0

    for oh in range(oH):
        for ow in range(oW):
            # --- 提取感受野（仅用于可视化展示） ---
            receptive_field = input_matrix[oh:oh + kH, ow:ow + kW]

            # --- 逐元素相乘（展示计算过程） ---
            elementwise = receptive_field * kernel

            # --- 输出值来自 Conv2D.forward() 的结果 ---
            output_val = output_matrix[oh, ow]

            # --- 高亮坐标 ---
            highlight = [(r, c) for r in range(oh, oh + kH) for c in range(ow, ow + kW)]

            # ---- 绘制这一行 ----
            # col0: 输入矩阵 + 高亮
            ax_input = fig.add_subplot(gs[step, 0])
            draw_matrix(
                ax_input, input_matrix,
                title=f"Step {step + 1}: 输入 (感受野高亮)" if step == 0 else f"Step {step + 1}",
                highlight_cells=highlight,
                highlight_color="#FFD54F",
                cmap_name="Greys",
                fontsize=12,
                title_fontsize=10,
            )
            # 画感受野外框
            rect_border = patches.Rectangle(
                (ow - 0.5, oh - 0.5), kW, kH,
                linewidth=2.5, edgecolor="#E53935", facecolor="none",
                linestyle="-", zorder=10,
            )
            ax_input.add_patch(rect_border)

            # col1: ⊙
            ax_op1 = fig.add_subplot(gs[step, 1])
            draw_operator(ax_op1, "⊙", fontsize=20)

            # col2: 卷积核
            ax_kernel = fig.add_subplot(gs[step, 2])
            draw_matrix(
                ax_kernel, kernel,
                title="卷积核 W" if step == 0 else "",
                cmap_name="Oranges",
                fontsize=12,
                title_fontsize=10,
            )

            # col3: =
            ax_op2 = fig.add_subplot(gs[step, 3])
            draw_operator(ax_op2, "=", fontsize=20)

            # col4: 逐元素乘积
            ax_prod = fig.add_subplot(gs[step, 4])
            draw_matrix(
                ax_prod, elementwise,
                title="逐元素乘积" if step == 0 else "",
                cmap_name="Greens",
                fmt=".0f",
                fontsize=12,
                title_fontsize=10,
            )

            # col5: → Σ =
            ax_op3 = fig.add_subplot(gs[step, 5])
            draw_operator(ax_op3, "Σ→", fontsize=16)

            # col6: 求和结果
            ax_sum = fig.add_subplot(gs[step, 6])
            ax_sum.set_xlim(0, 1)
            ax_sum.set_ylim(0, 1)
            # 背景圆角框
            result_box = patches.FancyBboxPatch(
                (0.1, 0.2), 0.8, 0.6,
                boxstyle="round,pad=0.08",
                facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2,
            )
            ax_sum.add_patch(result_box)
            ax_sum.text(
                0.5, 0.5, f"{output_val:.0f}",
                ha="center", va="center",
                fontsize=16, fontweight="bold", color="#1565C0",
            )
            if step == 0:
                ax_sum.set_title("输出值", fontsize=10, fontweight="bold", pad=6)
            ax_sum.axis("off")

            # --- 在逐元素乘积下方标注计算过程 ---
            terms = []
            for kr in range(kH):
                for kc in range(kW):
                    terms.append(f"{receptive_field[kr, kc]:.0f}×{kernel[kr, kc]:.0f}")
            calc_str = " + ".join(terms) + f" = {output_val:.0f}"
            ax_prod.text(
                0.5, -0.18, calc_str,
                transform=ax_prod.transAxes,
                ha="center", va="top",
                fontsize=7, color="#555555",
                style="italic",
            )

            step += 1

    fig.suptitle(
        "二维卷积逐步计算过程  |  Y[i,j] = Σ X[i+m, j+n] · W[m, n]",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 逐步计算图已保存: {save_path}")

    return output_matrix


# ============================================================
#  可视化 2: 全局总览 (输入 → 核 → 输出)
# ============================================================

def plot_summary(
        input_matrix: np.ndarray,
        kernel: np.ndarray,
        output_matrix: np.ndarray,
        save_path: str | Path,
):
    """
    单张总览图: 输入 ★ 卷积核 = 输出，并在输出矩阵上用颜色对应各区域。
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 5.5))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[5, 0.8, 3, 0.8, 3], wspace=0.2)

    # 输入矩阵
    ax_input = fig.add_subplot(gs[0, 0])
    draw_matrix(ax_input, input_matrix, title="输入矩阵 X  (5×5)",
                cmap_name="YlOrBr", fontsize=15, title_fontsize=13)

    # ★ 符号
    ax_star = fig.add_subplot(gs[0, 1])
    draw_operator(ax_star, "*", fontsize=28)

    # 卷积核
    ax_kernel = fig.add_subplot(gs[0, 2])
    draw_matrix(ax_kernel, kernel, title="卷积核 W  (3×3)",
                cmap_name="Oranges", fontsize=15, title_fontsize=13)

    # = 符号
    ax_eq = fig.add_subplot(gs[0, 3])
    draw_operator(ax_eq, "=", fontsize=28)

    # 输出矩阵
    ax_output = fig.add_subplot(gs[0, 4])
    draw_matrix(ax_output, output_matrix, title="输出矩阵 Y  (3×3)",
                cmap_name="Blues", fontsize=15, title_fontsize=13)

    # --- 在下方添加公式 ---
    fig.text(
        0.5, 0.02,
        r"$Y[i,\,j] \;=\; \sum_{m=0}^{kH-1}\;\sum_{n=0}^{kW-1}\; X[\,i+m,\; j+n\,] \;\cdot\; W[\,m,\; n\,]$"
        r"$\qquad$"
        r"(stride=1, padding=0)",
        ha="center", va="bottom",
        fontsize=14, color="#333333",
    )

    fig.suptitle(
        "二维卷积运算总览",
        fontsize=16, fontweight="bold", y=0.99,
    )
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 总览图已保存: {save_path}")


# ============================================================
#  可视化 3: 动态滑窗过程 (3×3 网格，对应输出每个位置)
# ============================================================

def plot_sliding_window(
        input_matrix: np.ndarray,
        kernel: np.ndarray,
        output_matrix: np.ndarray,
        save_path: str | Path,
):
    """
    用 3×3 网格展示卷积核在输入上的 9 个滑窗位置，
    每个子图显示当前位置的感受野高亮和对应输出值。
    """
    from matplotlib.gridspec import GridSpec

    iH, iW = input_matrix.shape
    kH, kW = kernel.shape
    oH = iH - kH + 1
    oW = iW - kW + 1

    fig = plt.figure(figsize=(oW * 4.2, oH * 4.2))
    gs = GridSpec(oH, oW, figure=fig, wspace=0.3, hspace=0.4)

    # 9 种不同的感受野边框颜色
    colors = [
        "#E53935", "#1E88E5", "#43A047",
        "#FB8C00", "#8E24AA", "#00ACC1",
        "#F4511E", "#3949AB", "#7CB342",
    ]

    idx = 0
    for oh in range(oH):
        for ow in range(oW):
            ax = fig.add_subplot(gs[oh, ow])

            # 高亮感受野
            highlight = [(r, c) for r in range(oh, oh + kH) for c in range(ow, ow + kW)]
            draw_matrix(
                ax, input_matrix,
                title=f"Y[{oh},{ow}] = {output_matrix[oh, ow]:.0f}",
                highlight_cells=highlight,
                highlight_color="#FFF9C4",
                cmap_name="Greys",
                fontsize=11,
                title_fontsize=11,
            )

            # 画感受野边框
            color = colors[idx % len(colors)]
            rect = patches.Rectangle(
                (ow - 0.5, oh - 0.5), kW, kH,
                linewidth=3, edgecolor=color, facecolor="none",
                linestyle="-", zorder=10,
            )
            ax.add_patch(rect)

            # 在感受野右上角标注输出值
            ax.text(
                ow + kW - 0.5, oh - 0.55,
                f"→ {output_matrix[oh, ow]:.0f}",
                ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.9),
            )

            idx += 1

    fig.suptitle(
        "卷积核滑窗过程：每个位置对应一个输出值",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 滑窗过程图已保存: {save_path}")


# ============================================================
#  Main
# ============================================================

# ============================================================
#  可视化 4: 反向传播 — dW 和 dX 的计算过程
# ============================================================

def plot_backward(
        input_matrix: np.ndarray,
        conv: Conv2D,
        output_matrix: np.ndarray,
        grad_output_2d: np.ndarray,
        save_path: str | Path,
):
    """
    使用 Conv2D.backward() 执行反向传播，可视化 grad_input 和 grad_weight。

    布局:
      第 1 行: grad_output → backward → grad_input  (与输入同尺寸)
      第 2 行: grad_output → backward → grad_weight  (与卷积核同尺寸)
    """
    from matplotlib.gridspec import GridSpec

    iH, iW = input_matrix.shape
    kernel = conv.weight[0, 0]
    kH, kW = kernel.shape
    oH, oW = output_matrix.shape

    # ---- 使用 Conv2D.backward() ----
    grad_output_4d = grad_output_2d.reshape(1, 1, oH, oW)
    grad_input_4d = conv.backward(grad_output_4d)
    grad_input = grad_input_4d[0, 0]  # (iH, iW)
    grad_weight = conv.grad_weight[0, 0]  # (kH, kW)
    grad_bias = conv.grad_bias[0]  # 标量

    # ---- 可视化 ----
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 7, figure=fig,
                  width_ratios=[3, 0.5, 5, 0.5, 5, 0.5, 3],
                  hspace=0.5, wspace=0.15)

    # ========== 第 1 行: grad_output → grad_input ==========
    # grad_output
    ax_go1 = fig.add_subplot(gs[0, 0])
    draw_matrix(ax_go1, grad_output_2d,
                title=f"grad_output ({oH}x{oW})",
                cmap_name="Oranges", fontsize=14, title_fontsize=10)

    ax_op1 = fig.add_subplot(gs[0, 1])
    draw_operator(ax_op1, "→", fontsize=20)

    # 卷积核 (旋转180°概念)
    ax_kernel = fig.add_subplot(gs[0, 2])
    draw_matrix(ax_kernel, kernel,
                title=f"卷积核 W ({kH}x{kW})\n(梯度通过 W 散布回输入)",
                cmap_name="Oranges", fontsize=13, title_fontsize=9)

    ax_op2 = fig.add_subplot(gs[0, 3])
    draw_operator(ax_op2, "→", fontsize=20)

    # grad_input
    ax_gi = fig.add_subplot(gs[0, 4])
    # 高亮非零位置
    nonzero = [(r, c) for r in range(iH) for c in range(iW)
               if abs(grad_input[r, c]) > 1e-8]
    draw_matrix(ax_gi, grad_input,
                title=f"grad_input ({iH}x{iW})",
                highlight_cells=nonzero,
                highlight_color="#C8E6C9",
                cmap_name="RdBu_r", fmt=".1f",
                fontsize=11, title_fontsize=10)

    ax_op3 = fig.add_subplot(gs[0, 5])
    draw_operator(ax_op3, "vs", fontsize=14)

    # 原始输入 (对比)
    ax_ref = fig.add_subplot(gs[0, 6])
    draw_matrix(ax_ref, input_matrix,
                title=f"原始输入 X ({iH}x{iW})",
                cmap_name="YlOrBr", fontsize=14, title_fontsize=10)

    # ========== 第 2 行: grad_output → grad_weight ==========
    ax_go2 = fig.add_subplot(gs[1, 0])
    draw_matrix(ax_go2, grad_output_2d,
                title=f"grad_output ({oH}x{oW})",
                cmap_name="Oranges", fontsize=14, title_fontsize=10)

    ax_op4 = fig.add_subplot(gs[1, 1])
    draw_operator(ax_op4, "→", fontsize=20)

    # 输入 (用于计算 dW)
    ax_x = fig.add_subplot(gs[1, 2])
    draw_matrix(ax_x, input_matrix,
                title=f"输入 X ({iH}x{iW})\n(dW = grad_output 与 X 的互相关)",
                cmap_name="YlOrBr", fontsize=13, title_fontsize=9)

    ax_op5 = fig.add_subplot(gs[1, 3])
    draw_operator(ax_op5, "→", fontsize=20)

    # grad_weight
    ax_gw = fig.add_subplot(gs[1, 4])
    draw_matrix(ax_gw, grad_weight,
                title=f"grad_weight ({kH}x{kW})",
                cmap_name="Purples", fmt=".1f",
                fontsize=13, title_fontsize=10)

    ax_op6 = fig.add_subplot(gs[1, 5])
    draw_operator(ax_op6, "vs", fontsize=14)

    # 原始核 (对比)
    ax_wref = fig.add_subplot(gs[1, 6])
    draw_matrix(ax_wref, kernel,
                title=f"原始核 W ({kH}x{kW})",
                cmap_name="Oranges", fontsize=14, title_fontsize=10)

    # 公式说明
    fig.text(
        0.5, 0.02,
        "dX[i,j] = (grad_output 与 W 旋转 180° 的 full 卷积)    |    "
        "dW[m,n] = (grad_output 与 X 的互相关)    |    "
        f"dBias = sum(grad_output) = {grad_bias:.1f}",
        ha="center", va="bottom", fontsize=10, color="#333333",
    )

    fig.suptitle(
        "卷积层反向传播：梯度计算过程",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 反向传播图已保存: {save_path}")


def main():
    # ---------- 定义一个简单的输入和卷积核 ----------
    # 使用小整数，便于手动验证
    input_matrix = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 3, 1, 0, 2],
        [2, 1, 0, 1, 3],
        [0, 2, 1, 2, 1],
    ], dtype=np.float64)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ], dtype=np.float64)

    # ---- 使用我们自己的 Conv2D 算子 ----
    # 单通道输入、单通道输出、无 padding，手动设置权重和偏置
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
    conv.weight[0, 0] = kernel  # 设置卷积核
    conv.bias[:] = 0.0  # 偏置置 0

    print("=" * 50)
    print(" 卷积运算数学过程可视化")
    print(f" 使用算子: {conv}")
    print("=" * 50)
    print(f"\n📐 输入矩阵 X ({input_matrix.shape[0]}×{input_matrix.shape[1]}):")
    print(input_matrix.astype(int))
    print(f"\n📐 卷积核 W ({kernel.shape[0]}×{kernel.shape[1]}):")
    print(kernel.astype(int))

    kH, kW = kernel.shape
    oH = input_matrix.shape[0] - kH + 1
    oW = input_matrix.shape[1] - kW + 1

    print(f"\n📐 输出尺寸: {oH}×{oW}")
    print(f"   公式: ({input_matrix.shape[0]} - {kH} + 1) × ({input_matrix.shape[1]} - {kW} + 1)\n")

    # ---- 生成可视化 ----
    output_dir = SCRIPT_DIR / "conv2d_math"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图 1: 逐步计算过程（内部调用 conv.forward()）
    print("🎨 绘制逐步计算过程...")
    output_matrix = plot_conv_steps(
        input_matrix, conv,
        output_dir / "conv2d_math_steps.png",
    )

    print(f"\n📐 输出矩阵 Y:")
    print(output_matrix.astype(int))

    # 图 2: 全局总览
    print("\n🎨 绘制全局总览...")
    plot_summary(
        input_matrix, kernel, output_matrix,
        output_dir / "conv2d_math_summary.png",
    )

    # 图 3: 滑窗过程
    print("\n🎨 绘制滑窗过程...")
    plot_sliding_window(
        input_matrix, kernel, output_matrix,
        output_dir / "conv2d_math_sliding.png",
    )

    # 图 4: 反向传播（内部调用 conv.backward()）
    print("\n🎨 绘制反向传播...")
    grad_output = np.ones_like(output_matrix)
    plot_backward(
        input_matrix, conv, output_matrix, grad_output,
        output_dir / "conv2d_math_backward.png",
    )

    print("\n🎉 所有可视化完成！")
    print(f"   📁 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
