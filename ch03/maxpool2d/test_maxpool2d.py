"""
MaxPooling2D 测试脚本
=====================

1. 手工验证：用一个 (1, 1, 4, 4) 矩阵，验证前向传播和反向传播的正确性。
2. 多通道测试：验证通道独立性。
3. 数值梯度检查：用有限差分法验证 backward 的精确性。
"""

import numpy as np

from maxpool2d import MaxPooling2D


def test_basic():
    """测试 1: 手工验证 (1, 1, 4, 4) 输入"""
    print("=" * 60)
    print(" 测试 1: 手工验证 — MaxPooling2D(kernel_size=2, stride=2)")
    print("=" * 60)

    pool = MaxPooling2D(kernel_size=2, stride=2)
    print(f"池化层: {pool}")

    # 构造一个可以手动验证的输入
    x = np.array([
        [1, 3, 2, 4],
        [5, 6, 7, 8],
        [3, 2, 1, 0],
        [1, 2, 3, 4],
    ], dtype=np.float64).reshape(1, 1, 4, 4)

    print(f"\n输入 X (1, 1, 4, 4):")
    print(x[0, 0])

    # 前向传播
    out = pool.forward(x)
    print(f"\n输出 Y (最大池化):")
    print(out[0, 0])
    # 期望:
    #   max(1,3,5,6)=6   max(2,4,7,8)=8
    #   max(3,2,1,2)=3   max(1,0,3,4)=4

    expected_out = np.array([[6, 8], [3, 4]], dtype=np.float64)
    assert np.allclose(out[0, 0], expected_out), f"前向传播结果错误: {out[0, 0]}"
    print("✅ 前向传播结果正确！")

    # 反向传播 — 用全 1 的梯度
    grad_output = np.ones_like(out)
    grad_input = pool.backward(grad_output)

    print(f"\ngrad_output (全 1):")
    print(grad_output[0, 0])
    print(f"\ngrad_input (梯度只流向最大值位置):")
    print(grad_input[0, 0])

    # 期望: 梯度只出现在最大值位置
    # 6 在 (1,1), 8 在 (1,3), 3 在 (2,0), 4 在 (3,3)
    expected_grad = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    assert np.allclose(grad_input[0, 0], expected_grad), (
        f"反向传播结果错误:\n{grad_input[0, 0]}"
    )
    print("✅ 反向传播结果正确！梯度只流向最大值位置。")


def test_multichannel():
    """测试 2: 多通道独立性"""
    print("\n" + "=" * 60)
    print(" 测试 2: 多通道独立性 (2, 3, 6, 6)")
    print("=" * 60)

    np.random.seed(42)
    pool = MaxPooling2D(kernel_size=2, stride=2)

    x = np.random.randn(2, 3, 6, 6)
    print(f"输入维度: {x.shape}")

    out = pool.forward(x)
    # H_out = (6 - 2) / 2 + 1 = 3
    print(f"输出维度: {out.shape}")  # 期望 (2, 3, 3, 3)

    grad_output = np.random.randn(*out.shape)
    grad_input = pool.backward(grad_output)
    print(f"grad_input 维度: {grad_input.shape}")  # 期望 (2, 3, 6, 6)

    assert out.shape == (2, 3, 3, 3), f"输出维度错误: {out.shape}"
    assert grad_input.shape == x.shape, f"grad_input 维度错误: {grad_input.shape}"

    # 验证通道数保持一致
    assert out.shape[1] == x.shape[1], "通道数应保持一致"
    print("✅ 多通道维度验证通过！")


def test_numerical_gradient():
    """测试 3: 数值梯度检查"""
    print("\n" + "=" * 60)
    print(" 测试 3: 数值梯度检查")
    print("=" * 60)

    np.random.seed(123)
    pool = MaxPooling2D(kernel_size=2, stride=2)

    # 使用不重复的数值避免 argmax 不稳定
    x = np.arange(1, 37, dtype=np.float64).reshape(1, 1, 6, 6)
    x = x + np.random.randn(*x.shape) * 0.1  # 加小扰动保证不重复

    # 前向 + 反向
    out = pool.forward(x)
    grad_output = np.ones_like(out)
    grad_input = pool.backward(grad_output)

    # 有限差分法
    eps = 1e-5
    numerical_grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_plus[idx] += eps
        out_plus = pool.forward(x_plus)

        x_minus = x.copy()
        x_minus[idx] -= eps
        out_minus = pool.forward(x_minus)

        numerical_grad[idx] = (out_plus.sum() - out_minus.sum()) / (2 * eps)

    diff = np.max(np.abs(grad_input - numerical_grad))
    print(f"grad_input 最大误差: {diff:.2e}")

    tol = 1e-6
    assert diff < tol, f"数值梯度检查失败: {diff:.2e}"
    print("✅ 数值梯度检查通过！")


def test_stride_not_equal_kernel():
    """测试 4: stride ≠ kernel_size (有重叠的池化)"""
    print("\n" + "=" * 60)
    print(" 测试 4: stride=1, kernel_size=2 (重叠池化)")
    print("=" * 60)

    pool = MaxPooling2D(kernel_size=2, stride=1)
    print(f"池化层: {pool}")

    x = np.array([
        [1, 5, 3],
        [4, 2, 6],
        [7, 8, 9],
    ], dtype=np.float64).reshape(1, 1, 3, 3)

    print(f"\n输入 X:")
    print(x[0, 0])

    out = pool.forward(x)
    # H_out = (3 - 2) / 1 + 1 = 2
    print(f"\n输出 Y:")
    print(out[0, 0])

    expected = np.array([[5, 6], [8, 9]], dtype=np.float64)
    assert np.allclose(out[0, 0], expected), f"结果错误: {out[0, 0]}"

    grad_output = np.ones_like(out)
    grad_input = pool.backward(grad_output)
    print(f"\ngrad_input:")
    print(grad_input[0, 0])

    # 5 被用于 Y[0,0]，6 被用于 Y[0,1]，8 被用于 Y[1,0]，9 被用于 Y[1,1]
    expected_grad = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ], dtype=np.float64)
    assert np.allclose(grad_input[0, 0], expected_grad), (
        f"结果错误:\n{grad_input[0, 0]}"
    )
    print("✅ 重叠池化验证通过！")


if __name__ == "__main__":
    test_basic()
    test_multichannel()
    test_numerical_gradient()
    test_stride_not_equal_kernel()
    print("\n" + "=" * 60)
    print(" 🎉 所有测试通过！")
    print("=" * 60)
