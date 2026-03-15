"""
MaxPooling2D — 使用 NumPy 实现的最大池化层
============================================

模仿 PyTorch nn.MaxPool2d 的接口设计，支持：
  • 4‑D 张量  (N, C, H, W)
  • kernel_size / stride
  • 完整的前向传播 & 反向传播
  • 基于 argmax 掩码的精确梯度回传

数学推导（反向传播）
--------------------

前向传播:
    Y[n, c, oh, ow] = max{ X[n, c, oh*s+m, ow*s+n] | 0 ≤ m < kH, 0 ≤ n < kW }

其中 (m*, n*) 是取到最大值的坐标（argmax），记为:
    (m*, n*) = argmax_{m,n} X[n, c, oh*s+m, ow*s+n]

反向传播:
    max 运算的梯度性质：梯度仅流向取到最大值的那个位置，其余位置梯度为 0。

    dL/dX[n, c, oh*s+m, ow*s+n] =
        dL/dY[n, c, oh, ow]    如果 (m, n) == (m*, n*)
        0                       否则

    实现中，我们在 forward 阶段记录每个滑窗的 argmax 坐标（掩码），
    backward 阶段根据掩码将上游梯度精确分发到对应的输入位置。
"""

import numpy as np


class MaxPooling2D:
    """二维最大池化层，纯 NumPy 实现。"""

    def __init__(self, kernel_size: int, stride: int | None = None):
        """
        Parameters
        ----------
        kernel_size : 池化窗口尺寸（假设为正方形）
        stride      : 步幅，默认等于 kernel_size（无重叠池化）
        """
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

        # 缓存（前向传播保存，供反向传播使用）
        self._input_shape: tuple | None = None
        self._argmax_mask: np.ndarray | None = None  # (N, C, H_out, W_out, 2)

    # -----------------------------------------------------------------
    #  前向传播
    # -----------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, C, H_in, W_in)

        Returns
        -------
        out : np.ndarray, shape (N, C, H_out, W_out)
        """
        N, C, H_in, W_in = x.shape
        kH = kW = self.kernel_size
        s = self.stride

        # ---------- 输出尺寸 ----------
        H_out = (H_in - kH) // s + 1
        W_out = (W_in - kW) // s + 1

        # ---------- 保存输入形状 ----------
        self._input_shape = x.shape

        # ---------- 分配输出和 argmax 掩码 ----------
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        # 掩码存储每个输出位置对应最大值在输入中的 (h, w) 坐标
        self._argmax_mask = np.zeros((N, C, H_out, W_out, 2), dtype=np.int64)

        # ---------- 最大池化运算（嵌套循环，清晰易读） ----------
        for n in range(N):  # batch
            for c in range(C):  # 通道（逐通道池化）
                for oh in range(H_out):  # 输出高度
                    for ow in range(W_out):  # 输出宽度
                        # 确定滑窗在输入上的起始位置
                        h_start = oh * s
                        w_start = ow * s

                        # 提取池化窗口
                        window = x[
                            n, c,
                            h_start: h_start + kH,
                            w_start: w_start + kW,
                        ]  # (kH, kW)

                        # 取最大值
                        out[n, c, oh, ow] = window.max()

                        # 记录最大值在窗口内的相对坐标
                        max_idx = np.unravel_index(window.argmax(), window.shape)
                        # 转换为在原始输入中的绝对坐标
                        self._argmax_mask[n, c, oh, ow, 0] = h_start + max_idx[0]
                        self._argmax_mask[n, c, oh, ow, 1] = w_start + max_idx[1]

        return out

    # -----------------------------------------------------------------
    #  反向传播
    # -----------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        grad_output : np.ndarray, shape (N, C, H_out, W_out)
                      从上一层传回的梯度（与 forward 输出维度一致）。

        Returns
        -------
        grad_input : np.ndarray, shape (N, C, H_in, W_in)
                     关于原始输入的梯度。
                     只有 forward 中取到最大值的位置有梯度，其余为 0。
        """
        N, C, H_out, W_out = grad_output.shape

        # 创建与输入同形状的全零梯度张量
        grad_input = np.zeros(self._input_shape, dtype=grad_output.dtype)

        # 根据 argmax 掩码，将梯度精确分发回对应位置
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        # 获取该输出位置对应的输入最大值坐标
                        h_max = self._argmax_mask[n, c, oh, ow, 0]
                        w_max = self._argmax_mask[n, c, oh, ow, 1]

                        # 梯度只流向最大值位置
                        grad_input[n, c, h_max, w_max] += grad_output[n, c, oh, ow]

        return grad_input

    def __repr__(self) -> str:
        return (
            f"MaxPooling2D(kernel_size={self.kernel_size}, "
            f"stride={self.stride})"
        )
