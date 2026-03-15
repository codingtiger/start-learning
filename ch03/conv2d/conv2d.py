"""
Conv2D — 使用 NumPy 实现的工业标准二维卷积层
==============================================

模仿 PyTorch nn.Conv2d 的接口设计，支持：
  • 4‑D 张量  (N, C, H, W)
  • stride / padding
  • 完整的前向传播 & 反向传播
  • He (Kaiming) 权重初始化

数学推导（反向传播）
--------------------

前向传播:
    Y[n, co, oh, ow] = Σ_{ci, kh, kw} X_pad[n, ci, oh*s+kh, ow*s+kw] * W[co, ci, kh, kw] + b[co]

其中 X_pad 是对输入 X 做零填充后的张量。

1) dW (权重梯度):
   对损失函数 L 关于 W 求偏导:
       dL/dW[co, ci, kh, kw] = Σ_{n, oh, ow} dL/dY[n, co, oh, ow] * X_pad[n, ci, oh*s+kh, ow*s+kw]

   即 dW 是 dY 与 X_pad 之间的互相关（cross-correlation）。

2) db (偏置梯度):
       dL/db[co] = Σ_{n, oh, ow} dL/dY[n, co, oh, ow]

   即 dY 在 batch / 空间维度上的求和。

3) dX (输入梯度):
   对损失函数 L 关于 X_pad 求偏导:
       dL/dX_pad[n, ci, ih, iw] = Σ_{co, kh, kw} dL/dY[n, co, oh, ow] * W[co, ci, kh, kw]
       其中 oh, ow 满足 oh*s + kh = ih, ow*s + kw = iw

   这等价于将 dY 按 stride 散布（dilate）后，与 W 旋转 180° 进行卷积，
   最后裁剪掉 padding 区域得到 dX。

   实现中我们直接在 X_pad 的梯度上做累加，然后裁剪 padding 即可。
"""

import numpy as np


class Conv2D:
    """二维卷积层，纯 NumPy 实现。"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
    ):
        """
        Parameters
        ----------
        in_channels  : 输入通道数 C_in
        out_channels : 输出通道数 C_out
        kernel_size  : 卷积核尺寸 (假设为正方形)
        stride       : 步幅
        padding      : 零填充大小
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # ------ He (Kaiming) 初始化 ------
        # fan_in = C_in * kH * kW
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)

        # 权重: (C_out, C_in, kH, kW)
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float64) * std

        # 偏置: (C_out,)
        self.bias = np.zeros(out_channels, dtype=np.float64)

        # 梯度占位
        self.grad_weight = None
        self.grad_bias = None

        # 缓存 (前向传播保存，供反向传播使用)
        self._x = None
        self._x_padded = None

    # -----------------------------------------------------------------
    #  前向传播
    # -----------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, C_in, H_in, W_in)

        Returns
        -------
        out : np.ndarray, shape (N, C_out, H_out, W_out)
        """
        N, C_in, H_in, W_in = x.shape
        kH = kW = self.kernel_size
        s = self.stride
        p = self.padding

        assert C_in == self.in_channels, (
            f"输入通道数不匹配: 期望 {self.in_channels}, 实际 {C_in}"
        )

        # ---------- padding ----------
        if p > 0:
            x_padded = np.pad(
                x,
                pad_width=((0, 0), (0, 0), (p, p), (p, p)),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x

        # ---------- 输出尺寸 ----------
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        H_out = (H_padded - kH) // s + 1
        W_out = (W_padded - kW) // s + 1

        # ---------- 保存缓存 ----------
        self._x = x
        self._x_padded = x_padded

        # ---------- 卷积运算 (嵌套循环，清晰易读) ----------
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=x.dtype)

        for n in range(N):  # batch
            for co in range(self.out_channels):  # 输出通道
                for oh in range(H_out):  # 输出高度
                    for ow in range(W_out):  # 输出宽度
                        # 提取感受野区域
                        h_start = oh * s
                        w_start = ow * s
                        receptive_field = x_padded[
                            n, :, h_start: h_start + kH, w_start: w_start + kW
                        ]  # (C_in, kH, kW)

                        # 逐元素相乘再求和 + 偏置
                        out[n, co, oh, ow] = (
                                np.sum(receptive_field * self.weight[co]) + self.bias[co]
                        )

        return out

    # -----------------------------------------------------------------
    #  反向传播
    # -----------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        grad_output : np.ndarray, shape (N, C_out, H_out, W_out)
                      从上一层传回的梯度（与 forward 输出维度一致）。

        Returns
        -------
        grad_input : np.ndarray, shape (N, C_in, H_in, W_in)
                     关于原始输入 x 的梯度。
        """
        x_padded = self._x_padded
        N, C_in, H_in, W_in = self._x.shape
        _, C_out, H_out, W_out = grad_output.shape
        kH = kW = self.kernel_size
        s = self.stride
        p = self.padding

        # ===================== 1) grad_weight =====================
        # dW[co, ci, kh, kw] = Σ_{n, oh, ow} grad_output[n,co,oh,ow]
        #                       * x_padded[n, ci, oh*s+kh, ow*s+kw]
        self.grad_weight = np.zeros_like(self.weight)

        for co in range(C_out):
            for ci in range(C_in):
                for kh in range(kH):
                    for kw in range(kW):
                        # 收集所有 (n, oh, ow) 的贡献
                        patch = x_padded[
                            :, ci, kh: kh + H_out * s: s, kw: kw + W_out * s: s
                        ]  # (N, H_out, W_out)
                        self.grad_weight[co, ci, kh, kw] = np.sum(
                            grad_output[:, co, :, :] * patch
                        )

        # ===================== 2) grad_bias =====================
        # db[co] = Σ_{n, oh, ow} grad_output[n, co, oh, ow]
        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3))  # (C_out,)

        # ===================== 3) grad_input =====================
        # dX_pad[n, ci, oh*s+kh, ow*s+kw] += grad_output[n,co,oh,ow] * W[co,ci,kh,kw]
        # 最后裁剪 padding 得到 dX
        grad_x_padded = np.zeros_like(x_padded)

        for n in range(N):
            for co in range(C_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        h_start = oh * s
                        w_start = ow * s
                        # 将梯度散布回对应的感受野
                        grad_x_padded[
                            n, :, h_start: h_start + kH, w_start: w_start + kW
                        ] += (
                                grad_output[n, co, oh, ow] * self.weight[co]
                        )

        # 裁剪 padding 区域，恢复原始输入的尺寸
        if p > 0:
            grad_input = grad_x_padded[:, :, p:-p, p:-p]
        else:
            grad_input = grad_x_padded

        return grad_input

    def __repr__(self) -> str:
        return (
            f"Conv2D(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )
