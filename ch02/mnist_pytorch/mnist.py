"""
PyTorch MNIST 手写数字识别
使用卷积神经网络 (CNN) 在 MNIST 数据集上训练并评估模型。
"""

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")  # 非交互式后端，直接保存图片
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchviz import make_dot
import os
import ssl
import certifi

# 修复 macOS 下 SSL 证书验证问题
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

# ── 基准目录（确保所有输出文件保存在脚本所在目录下） ────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 超参数 ──────────────────────────────────────────────
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 10
LEARNING_RATE = 0.001
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── 数据预处理 ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值和标准差
])


# ── CNN 模型定义 ────────────────────────────────────────
class MNISTNet(nn.Module):
    """简洁而高效的卷积神经网络"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 32, 28, 28]
        x = self.pool(x)  # -> [B, 32, 14, 14]
        x = F.relu(self.conv2(x))  # -> [B, 64, 14, 14]
        x = self.pool(x)  # -> [B, 64, 7, 7]
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))  # -> [B, 128]
        x = self.dropout2(x)
        x = self.fc2(x)  # -> [B, 10]
        return x


# ── 训练函数 ────────────────────────────────────────────
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(
                f"  Epoch {epoch} [{total:>5d}/{len(train_loader.dataset)}] "
                f"Loss: {running_loss / total:.4f}  "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = running_loss / total
    avg_acc = 100.0 * correct / total
    print(f"  Epoch {epoch} 训练完成 — Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    return avg_loss, avg_acc


# ── 测试函数 ────────────────────────────────────────────
def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            correct += output.argmax(dim=1).eq(target).sum().item()

    total = len(test_loader.dataset)
    test_loss /= total
    accuracy = 100.0 * correct / total
    print(f"  测试集 — Loss: {test_loss:.4f}, Acc: {correct}/{total} ({accuracy:.2f}%)\n")
    return test_loss, accuracy


# ── 可视化函数 ──────────────────────────────────────────
def plot_training_curves(history):
    """绘制训练/测试的 Loss 和准确率曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history["train_loss"], "o-", color="#2196F3", label="Train Loss")
    ax1.plot(epochs, history["test_loss"], "s-", color="#F44336", label="Test Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curve", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history["train_acc"], "o-", color="#2196F3", label="Train Accuracy")
    ax2.plot(epochs, history["test_acc"], "s-", color="#F44336", label="Test Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 训练曲线已保存至 training_curves.png")


def plot_sample_predictions(model, device, test_loader):
    """展示模型对测试样本的预测结果（正确为绿色，错误为红色）"""
    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            images.append(data.cpu())
            labels.append(target.cpu())
            preds.append(pred.cpu())
            if len(labels) * test_loader.batch_size >= 100:
                break

    images = torch.cat(images)[:100]
    labels = torch.cat(labels)[:100]
    preds = torch.cat(preds)[:100]

    # 找出一些错误预测用于展示
    wrong_mask = preds != labels
    wrong_indices = wrong_mask.nonzero(as_tuple=True)[0].tolist()
    right_indices = (~wrong_mask).nonzero(as_tuple=True)[0].tolist()

    # 优先展示一些错误样本，其余用正确样本填充
    n_show = 25
    n_wrong = min(len(wrong_indices), 10)
    show_indices = wrong_indices[:n_wrong] + right_indices[: n_show - n_wrong]
    show_indices = show_indices[:n_show]

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(show_indices):
            idx = show_indices[i]
            img = images[idx].squeeze().numpy()
            true_label = labels[idx].item()
            pred_label = preds[idx].item()
            correct = true_label == pred_label

            ax.imshow(img, cmap="gray")
            color = "#4CAF50" if correct else "#F44336"
            symbol = "OK" if correct else "X"
            ax.set_title(
                f"True:{true_label} Pred:{pred_label} [{symbol}]",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "sample_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 样本预测图已保存至 sample_predictions.png")


def plot_confusion_matrix(model, device, test_loader):
    """绘制混淆矩阵"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.append(output.argmax(dim=1).cpu().numpy())
            all_labels.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    classes = list(range(10))
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # 在格子中标注数值
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> 混淆矩阵已保存至 confusion_matrix.png")


def plot_computation_graph(model, device):
    """绘制 PyTorch 计算图（前向 + 反向传播）"""
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(dummy_input)

    dot = make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    dot.attr(rankdir="TB")  # 从上到下布局
    dot.attr("node", fontsize="12")
    dot.attr(dpi="300")  # 高分辨率输出
    dot.render(os.path.join(BASE_DIR, "computation_graph"), format="png", cleanup=True)
    dot.render(os.path.join(BASE_DIR, "computation_graph"), format="pdf", cleanup=True)
    print("  -> 计算图已保存至 computation_graph.png / computation_graph.pdf")


def _select_representative_channel(feature_tensor):
    """从特征图中选择响应最强的通道索引。"""
    # feature_tensor: [1, C, H, W]
    channel_score = feature_tensor[0].abs().mean(dim=(1, 2))
    return int(channel_score.argmax().item())


def _normalize_feature_map(feature_map):
    """将特征图归一化到 [0, 1]，便于可视化。"""
    feature_map = feature_map.astype(np.float32)
    min_v, max_v = feature_map.min(), feature_map.max()
    if max_v - min_v < 1e-8:
        return np.zeros_like(feature_map)
    return (feature_map - min_v) / (max_v - min_v)


def plot_cnn_feature_extraction(model, device, test_dataset, output_dir="feature_maps", sample_idx=0):
    """展示一张图片在 CNN 各层的特征提取过程，并保存到独立目录。"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    image, label = test_dataset[sample_idx]
    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        feat_conv1 = F.relu(model.conv1(x))
        feat_pool1 = model.pool(feat_conv1)
        feat_conv2 = F.relu(model.conv2(feat_pool1))
        feat_pool2 = model.pool(feat_conv2)
        pred = model(x).argmax(dim=1).item()

    # 恢复输入图像到可视化范围
    input_img = image.squeeze().cpu().numpy() * 0.3081 + 0.1307
    input_img = np.clip(input_img, 0.0, 1.0)

    layers = [
        ("input", input_img, None),
        ("conv1", feat_conv1, _select_representative_channel(feat_conv1)),
        ("pool1", feat_pool1, _select_representative_channel(feat_pool1)),
        ("conv2", feat_conv2, _select_representative_channel(feat_conv2)),
        ("pool2", feat_pool2, _select_representative_channel(feat_pool2)),
    ]

    # 单图保存：每一层一张图片
    for name, tensor_or_img, channel in layers:
        if name == "input":
            vis = tensor_or_img
            title = f"input (label={label}, pred={pred})"
        else:
            vis = tensor_or_img[0, channel].cpu().numpy()
            vis = _normalize_feature_map(vis)
            title = f"{name} (channel={channel})"

        plt.figure(figsize=(4, 4))
        plt.imshow(vis, cmap="gray")
        plt.title(title, fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 总览图保存：一行展示全过程
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for i, (name, tensor_or_img, channel) in enumerate(layers):
        if name == "input":
            vis = tensor_or_img
            axes[i].set_title(f"input\nlabel={label}, pred={pred}", fontsize=9)
        else:
            vis = tensor_or_img[0, channel].cpu().numpy()
            vis = _normalize_feature_map(vis)
            axes[i].set_title(f"{name}\nch={channel}", fontsize=9)

        axes[i].imshow(vis, cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_extraction_overview.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  -> CNN 特征提取过程已保存至 {output_dir}/")


def plot_forward_matrix_flow(model, device, test_dataset, output_dir="forward_flow", sample_idx=0):
    """针对单个输入样本，展示 CNN 前向计算的矩阵化流程图。"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    image, label = test_dataset[sample_idx]
    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        conv1 = F.relu(model.conv1(x))
        pool1 = model.pool(conv1)
        conv2 = F.relu(model.conv2(pool1))
        pool2 = model.pool(conv2)
        flat = pool2.view(1, -1)
        fc1 = F.relu(model.fc1(flat))
        logits = model.fc2(fc1)
        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())

    conv1_ch = _select_representative_channel(conv1)
    pool1_ch = _select_representative_channel(pool1)
    conv2_ch = _select_representative_channel(conv2)
    pool2_ch = _select_representative_channel(pool2)

    input_map = image.squeeze().cpu().numpy() * 0.3081 + 0.1307
    input_map = np.clip(input_map, 0.0, 1.0)
    conv1_map = _normalize_feature_map(conv1[0, conv1_ch].cpu().numpy())
    pool1_map = _normalize_feature_map(pool1[0, pool1_ch].cpu().numpy())
    conv2_map = _normalize_feature_map(conv2[0, conv2_ch].cpu().numpy())
    pool2_map = _normalize_feature_map(pool2[0, pool2_ch].cpu().numpy())
    flat_map = _normalize_feature_map(flat.cpu().numpy().reshape(56, 56))
    fc1_map = _normalize_feature_map(fc1.cpu().numpy().reshape(8, 16))
    logits_map = _normalize_feature_map(logits.cpu().numpy().reshape(1, 10))
    probs_map = probs.cpu().numpy().reshape(1, 10)

    stages = [
        (
            "Input\n1x28x28",
            "x in R^(1x28x28)",
            input_map,
            "gray",
        ),
        (
            f"Conv1+ReLU\nch={conv1_ch} (28x28)",
            "y1(c,i,j)=ReLU(sum_u,v w1(c,u,v)*x(i+u,j+v)+b1(c))",
            conv1_map,
            "magma",
        ),
        (
            f"MaxPool1\nch={pool1_ch} (14x14)",
            "p1(c,i,j)=max(y1(c,2i+m,2j+n)), m,n in {0,1}",
            pool1_map,
            "magma",
        ),
        (
            f"Conv2+ReLU\nch={conv2_ch} (14x14)",
            "y2(k,i,j)=ReLU(sum_c,u,v w2(k,c,u,v)*p1(c,i+u,j+v)+b2(k))",
            conv2_map,
            "magma",
        ),
        (
            f"MaxPool2\nch={pool2_ch} (7x7)",
            "p2(k,i,j)=max(y2(k,2i+m,2j+n)), m,n in {0,1}",
            pool2_map,
            "magma",
        ),
        (
            "Flatten\n1x3136 -> 56x56",
            "f = vec(p2),   f in R^3136",
            flat_map,
            "viridis",
        ),
        (
            "FC1+ReLU\n1x128 -> 8x16",
            "h = ReLU(W1*f + b1),   h in R^128",
            fc1_map,
            "viridis",
        ),
        (
            "Logits\n1x10",
            "z = W2*h + b2,   z in R^10",
            logits_map,
            "coolwarm",
        ),
        (
            "Softmax Prob\n1x10",
            "p_i = exp(z_i) / sum_j exp(z_j)",
            probs_map,
            "YlGnBu",
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (title, formula, data, cmap) in zip(axes.flat, stages):
        ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5,
            -0.16,
            formula,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"CNN Forward Matrix Flow (sample={sample_idx}, label={label}, pred={pred})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "forward_matrix_flow.png"), dpi=260, bbox_inches="tight")
    plt.close()
    print(f"  -> 前向矩阵流程图已保存至 {output_dir}/forward_matrix_flow.png")


# ── 主流程 ──────────────────────────────────────────────
def main():
    print(f"使用设备: {DEVICE}")
    print(f"PyTorch 版本: {torch.__version__}\n")

    # 加载数据
    print("正在下载/加载 MNIST 数据集...")
    data_dir = os.path.join(BASE_DIR, "data")
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}\n")

    # 初始化模型
    model = MNISTNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"模型结构:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}\n")

    # 训练循环
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print("=" * 50)
    print("开始训练")
    print("=" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, DEVICE, test_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "mnist_best.pth"))

    print("=" * 50)
    print(f"训练完成！最佳测试准确率: {best_acc:.2f}%")
    print("模型已保存至 mnist_best.pth")
    print("=" * 50)

    # 生成可视化图表
    print("\n正在生成可视化图表...")
    plot_training_curves(history)

    # 加载最优模型进行预测展示
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "mnist_best.pth"), weights_only=True))
    plot_sample_predictions(model, DEVICE, test_loader)
    plot_confusion_matrix(model, DEVICE, test_loader)
    plot_computation_graph(model, DEVICE)
    plot_cnn_feature_extraction(model, DEVICE, test_dataset, output_dir=os.path.join(BASE_DIR, "feature_maps"),
                                sample_idx=0)
    plot_forward_matrix_flow(model, DEVICE, test_dataset, output_dir=os.path.join(BASE_DIR, "forward_flow"),
                             sample_idx=0)
    print("\n所有图表已生成完毕！")


if __name__ == "__main__":
    main()
