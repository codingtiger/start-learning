"""
Conv2D 测试脚本
================

1. 基础维度验证：初始化 Conv2D(3, 16, 3)，传入 (2, 3, 32, 32) 随机张量，
   执行 forward / backward，打印各梯度维度。
2. 数值梯度检查：用有限差分法验证 backward 计算的正确性。
"""

import numpy as np

from conv2d import Conv2D


def test_basic_shapes():
    """测试 1: 基础维度验证"""
    print("=" * 60)
    print(" 测试 1: 基础维度验证")
    print("=" * 60)

    np.random.seed(42)

    # 初始化卷积层
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
    print(f"卷积层: {conv}")
    print(f"权重维度: {conv.weight.shape}")  # (16, 3, 3, 3)
    print(f"偏置维度: {conv.bias.shape}")  # (16,)

    # 随机输入
    x = np.random.randn(2, 3, 32, 32)
    print(f"\n输入维度: {x.shape}")

    # 前向传播
    out = conv.forward(x)
    print(f"输出维度: {out.shape}")  # 期望 (2, 16, 30, 30)

    # 模拟上游梯度
    grad_output = np.random.randn(*out.shape)

    # 反向传播
    grad_input = conv.backward(grad_output)

    print(f"\ngrad_input  维度: {grad_input.shape}")  # 期望 (2, 3, 32, 32)
    print(f"grad_weight 维度: {conv.grad_weight.shape}")  # 期望 (16, 3, 3, 3)
    print(f"grad_bias   维度: {conv.grad_bias.shape}")  # 期望 (16,)

    # 断言维度正确
    assert grad_input.shape == x.shape, f"grad_input 维度错误: {grad_input.shape}"
    assert conv.grad_weight.shape == conv.weight.shape, (
        f"grad_weight 维度错误: {conv.grad_weight.shape}"
    )
    assert conv.grad_bias.shape == conv.bias.shape, (
        f"grad_bias 维度错误: {conv.grad_bias.shape}"
    )

    print("\n✅ 基础维度验证通过！")


def test_with_stride_and_padding():
    """测试 2: 带 stride 和 padding 的维度验证"""
    print("\n" + "=" * 60)
    print(" 测试 2: stride=2, padding=1")
    print("=" * 60)

    np.random.seed(123)

    conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
    print(f"卷积层: {conv}")

    x = np.random.randn(4, 3, 16, 16)
    print(f"输入维度: {x.shape}")

    out = conv.forward(x)
    # H_out = (16 + 2*1 - 3) / 2 + 1 = 15/2 + 1 = 7 + 1 = 8
    print(f"输出维度: {out.shape}")  # 期望 (4, 8, 8, 8)

    grad_output = np.random.randn(*out.shape)
    grad_input = conv.backward(grad_output)

    print(f"grad_input  维度: {grad_input.shape}")  # 期望 (4, 3, 16, 16)
    print(f"grad_weight 维度: {conv.grad_weight.shape}")
    print(f"grad_bias   维度: {conv.grad_bias.shape}")

    assert grad_input.shape == x.shape
    assert conv.grad_weight.shape == conv.weight.shape
    assert conv.grad_bias.shape == conv.bias.shape

    print("\n✅ stride + padding 维度验证通过！")


def test_numerical_gradient():
    """测试 3: 数值梯度检查 (finite difference)"""
    print("\n" + "=" * 60)
    print(" 测试 3: 数值梯度检查")
    print("=" * 60)

    np.random.seed(0)

    # 使用较小的尺寸加速数值检查
    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(1, 2, 5, 5)

    # 前向传播
    out = conv.forward(x)
    # 使用 sum 作为标量损失
    grad_output = np.ones_like(out)
    grad_input = conv.backward(grad_output)

    # ---- 检查 grad_input ----
    eps = 1e-5
    numerical_grad_input = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_plus[idx] += eps
        out_plus = conv.forward(x_plus)

        x_minus = x.copy()
        x_minus[idx] -= eps
        out_minus = conv.forward(x_minus)

        numerical_grad_input[idx] = (out_plus.sum() - out_minus.sum()) / (2 * eps)

    diff_input = np.max(np.abs(grad_input - numerical_grad_input))
    print(f"grad_input  最大误差: {diff_input:.2e}")

    # ---- 检查 grad_weight ----
    # 需要重新前向传播以恢复缓存
    conv.forward(x)
    conv.backward(grad_output)

    numerical_grad_weight = np.zeros_like(conv.weight)
    for idx in np.ndindex(conv.weight.shape):
        w_orig = conv.weight[idx]

        conv.weight[idx] = w_orig + eps
        out_plus = conv.forward(x)

        conv.weight[idx] = w_orig - eps
        out_minus = conv.forward(x)

        numerical_grad_weight[idx] = (out_plus.sum() - out_minus.sum()) / (2 * eps)
        conv.weight[idx] = w_orig  # 恢复

    diff_weight = np.max(np.abs(conv.grad_weight - numerical_grad_weight))
    print(f"grad_weight 最大误差: {diff_weight:.2e}")

    # ---- 检查 grad_bias ----
    conv.forward(x)
    conv.backward(grad_output)

    numerical_grad_bias = np.zeros_like(conv.bias)
    for i in range(conv.bias.shape[0]):
        b_orig = conv.bias[i]

        conv.bias[i] = b_orig + eps
        out_plus = conv.forward(x)

        conv.bias[i] = b_orig - eps
        out_minus = conv.forward(x)

        numerical_grad_bias[i] = (out_plus.sum() - out_minus.sum()) / (2 * eps)
        conv.bias[i] = b_orig

    diff_bias = np.max(np.abs(conv.grad_bias - numerical_grad_bias))
    print(f"grad_bias   最大误差: {diff_bias:.2e}")

    tol = 1e-6
    assert diff_input < tol, f"grad_input 数值检查失败: {diff_input:.2e}"
    assert diff_weight < tol, f"grad_weight 数值检查失败: {diff_weight:.2e}"
    assert diff_bias < tol, f"grad_bias 数值检查失败: {diff_bias:.2e}"

    print("\n✅ 数值梯度检查通过！所有梯度误差 < 1e-6")


if __name__ == "__main__":
    test_basic_shapes()
    test_with_stride_and_padding()
    test_numerical_gradient()
    print("\n" + "=" * 60)
    print(" 🎉 所有测试通过！")
    print("=" * 60)
