"""
Fashion-MNIST + ResNet18：鞋类子集预训练 → 全类服装迁移学习。

流程：
1. 在鞋类（Sandal / Sneaker / Ankle boot）三分类上训练 ResNet18，保存骨干+鞋类头。
2. 在 10 类服装上对比：迁移微调（加载鞋类骨干）vs 从零训练（随机初始化）。
3. 特征复用：冻结骨干仅训分类头若干 epoch，比较初始验证准确率（线性探测）。

依赖：torch, torchvision, matplotlib, numpy, scikit-learn, certifi
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import time
from pathlib import Path
from typing import Callable

import certifi
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

# Fashion-MNIST: 鞋类标签 5=Sandal, 7=Sneaker, 9=Ankle boot
SHOE_LABELS = (5, 7, 9)
SHOE_REMAP = {5: 0, 7: 1, 9: 2}

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fashion_gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """供 transforms.Lambda 使用（须为模块级函数，便于 DataLoader 多进程 pickle）。"""
    return x.repeat(3, 1, 1)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """单通道复制为三通道，按 ImageNet 统计量归一化（与 ResNet 预训练一致）。"""
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(_fashion_gray_to_rgb),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(_fashion_gray_to_rgb),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return train_tf, eval_tf


def build_resnet18(num_classes: int, imagenet_pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_backbone_skip_fc(model: nn.Module, state_path: Path, device: torch.device) -> None:
    """加载除 fc 外的权重（用于鞋类 → 服装迁移）。"""
    ckpt = torch.load(state_path, map_location=device, weights_only=True)
    model_sd = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in ckpt.items():
        if k.startswith("fc."):
            skipped += 1
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_sd)
    print(f"Loaded backbone keys: {loaded}, skipped: {skipped}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def shoe_indices_from_dataset(train_ds: datasets.FashionMNIST) -> list[int]:
    targets = train_ds.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return [i for i, t in enumerate(targets) if t in SHOE_LABELS]


class RemapShoeLabels(torch.utils.data.Dataset):
    """将原始 Fashion 标签 5/7/9 映射为 0/1/2。"""

    def __init__(self, base: datasets.FashionMNIST, indices: list[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        y_new = SHOE_REMAP[int(y)]
        return x, y_new


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[dict[str, list[float]], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val:
            best_val = va_acc
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.2f}%"
        )

    return history, best_val


def head_only_probe(
    model_init_fn: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[float, float]:
    """冻结骨干，只训练 fc；返回 (首轮 val_acc, 末轮 best val_acc)。"""
    model = model_init_fn().to(device)
    set_backbone_trainable(model, trainable=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    first_val_acc = 0.0
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, va_acc = evaluate(model, val_loader, criterion, device)
        if epoch == 1:
            first_val_acc = va_acc
        if va_acc > best_val:
            best_val = va_acc
    return first_val_acc, best_val


def plot_comparison(
    hist_transfer: dict[str, list[float]],
    hist_scratch: dict[str, list[float]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1 = range(1, len(hist_transfer["val_acc"]) + 1)
    e2 = range(1, len(hist_scratch["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(e1, hist_transfer["val_acc"], "o-", label="Fine-tune (shoe-pretrained backbone)")
    ax.plot(e2, hist_scratch["val_acc"], "s-", label="Train from scratch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Fashion-MNIST 10-class: validation accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fashion-MNIST ResNet18 鞋类 → 服装迁移学习实验")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=min(2, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--epochs-shoe", type=int, default=12)
    p.add_argument("--epochs-clothing", type=int, default=20)
    p.add_argument("--lr-shoe", type=float, default=3e-4)
    p.add_argument("--lr-clothing", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-only-epochs", type=int, default=5, help="冻结骨干时只训分类头的 epoch 数（特征复用）")
    p.add_argument("--lr-head-only", type=float, default=5e-3)
    p.add_argument("--skip-shoe", action="store_true", help="若已有 shoe_best.pt 则跳过鞋类训练")
    p.add_argument("--dry-run", action="store_true", help="只检查数据与模型构建")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size)

    print(f"设备: {device}")
    print("加载 Fashion-MNIST …")
    full_train = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )
    test_ds = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=eval_tf,
    )

    # 鞋类子集：仅在训练集上选取标签 ∈ {5,7,9}
    shoe_idx = shoe_indices_from_dataset(full_train)
    print(f"训练集中鞋类样本数: {len(shoe_idx)} / {len(full_train)}")

    s_train, s_val = train_test_split(
        shoe_idx,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    s_train_list = [int(i) for i in np.asarray(s_train).ravel()]
    s_val_list = [int(i) for i in np.asarray(s_val).ravel()]
    shoe_train_ds = RemapShoeLabels(full_train, s_train_list)
    shoe_val_ds = RemapShoeLabels(
        datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=eval_tf),
        s_val_list,
    )

    shoe_train_loader = DataLoader(
        shoe_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    shoe_val_loader = DataLoader(
        shoe_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 10 类：从训练集划分 val
    n_total = len(full_train)
    indices = np.arange(n_total)
    tr_idx, va_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    tr_list = [int(i) for i in np.asarray(tr_idx).ravel()]
    va_list = [int(i) for i in np.asarray(va_idx).ravel()]

    train_only_ds = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )
    train_split = Subset(train_only_ds, tr_list)
    val_split = Subset(
        datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=eval_tf),
        va_list,
    )

    clothing_train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    clothing_val_loader = DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.dry_run:
        m = build_resnet18(10, imagenet_pretrained=True)
        print(f"Dry-run OK, ResNet18 参数量: {sum(p.numel() for p in m.parameters()):,}")
        return

    shoe_ckpt = args.output_dir / "shoe_best.pt"

    # ----- 阶段 1：鞋类三分类 -----
    if not (args.skip_shoe and shoe_ckpt.exists()):
        print("\n=== 阶段 1：鞋类（Sandal / Sneaker / Ankle boot）三分类 ===")
        model_shoe = build_resnet18(3, imagenet_pretrained=True).to(device)
        hist_shoe, best_shoe = run_training_loop(
            model_shoe,
            shoe_train_loader,
            shoe_val_loader,
            device,
            args.epochs_shoe,
            args.lr_shoe,
            args.weight_decay,
        )
        torch.save(model_shoe.state_dict(), shoe_ckpt)
        with (args.output_dir / "shoe_history.json").open("w", encoding="utf-8") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist_shoe.items()}, f, ensure_ascii=False, indent=2)
        print(f"鞋类验证集最佳准确率: {best_shoe:.2f}%，已保存 {shoe_ckpt}")
    else:
        print(f"跳过鞋类训练，使用已有检查点: {shoe_ckpt}")

    # ----- 特征复用：冻结骨干，只训头 -----
    print("\n=== 特征复用：冻结 ResNet 骨干，仅训练 10 类分类头 ===")

    def make_transfer_model() -> nn.Module:
        m = build_resnet18(10, imagenet_pretrained=False)
        load_backbone_skip_fc(m, shoe_ckpt, device)
        return m

    def make_scratch_model() -> nn.Module:
        return build_resnet18(10, imagenet_pretrained=False)

    t_first, t_last = head_only_probe(
        make_transfer_model,
        clothing_train_loader,
        clothing_val_loader,
        device,
        args.head_only_epochs,
        args.lr_head_only,
    )
    s_first, s_last = head_only_probe(
        make_scratch_model,
        clothing_train_loader,
        clothing_val_loader,
        device,
        args.head_only_epochs,
        args.lr_head_only,
    )
    print(
        f"迁移骨干 — 线性探测: 第 1 个 epoch 后 val_acc={t_first:.2f}%, "
        f"{args.head_only_epochs} epoch 后最佳 val_acc={t_last:.2f}%"
    )
    print(
        f"随机初始化 — 线性探测: 第 1 个 epoch 后 val_acc={s_first:.2f}%, "
        f"{args.head_only_epochs} epoch 后最佳 val_acc={s_last:.2f}%"
    )

    # ----- 阶段 2a：迁移微调 -----
    print("\n=== 阶段 2a：服装 10 类 — 迁移微调（加载鞋类骨干）===")
    model_tr = build_resnet18(10, imagenet_pretrained=False).to(device)
    load_backbone_skip_fc(model_tr, shoe_ckpt, device)
    hist_tr, best_tr = run_training_loop(
        model_tr,
        clothing_train_loader,
        clothing_val_loader,
        device,
        args.epochs_clothing,
        args.lr_clothing,
        args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    _, test_acc_tr = evaluate(model_tr, test_loader, criterion, device)
    torch.save(model_tr.state_dict(), args.output_dir / "clothing_transfer_best.pt")

    # ----- 阶段 2b：从零训练 -----
    print("\n=== 阶段 2b：服装 10 类 — 从零训练（随机初始化）===")
    model_sc = build_resnet18(10, imagenet_pretrained=False).to(device)
    hist_sc, best_sc = run_training_loop(
        model_sc,
        clothing_train_loader,
        clothing_val_loader,
        device,
        args.epochs_clothing,
        args.lr_clothing,
        args.weight_decay,
    )
    _, test_acc_sc = evaluate(model_sc, test_loader, criterion, device)
    torch.save(model_sc.state_dict(), args.output_dir / "clothing_scratch_best.pt")

    plot_comparison(hist_tr, hist_sc, args.output_dir / "clothing_val_compare.png")

    summary = {
        "device": str(device),
        "seed": args.seed,
        "epochs_clothing": args.epochs_clothing,
        "head_only_epochs": args.head_only_epochs,
        "linear_probe_transfer_epoch1_val_acc": t_first,
        "linear_probe_transfer_best_val_acc": t_last,
        "linear_probe_scratch_epoch1_val_acc": s_first,
        "linear_probe_scratch_best_val_acc": s_last,
        "clothing_transfer_best_val_acc": best_tr,
        "clothing_transfer_test_acc": test_acc_tr,
        "clothing_scratch_best_val_acc": best_sc,
        "clothing_scratch_test_acc": test_acc_sc,
        "note": (
            "线性探测：相同冻结策略下，鞋类预训练骨干在第 1 epoch 的 val 更高则说明底层特征更可复用。"
        ),
    }
    with (args.output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(f"迁移微调 — 验证最佳: {best_tr:.2f}% | 测试集: {test_acc_tr:.2f}%")
    print(f"从零训练 — 验证最佳: {best_sc:.2f}% | 测试集: {test_acc_sc:.2f}%")
    print(f"指标已写入: {args.output_dir / 'comparison_metrics.json'}")
    print(f"曲线图: {args.output_dir / 'clothing_val_compare.png'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"总耗时: {(time.time() - t0) / 60.0:.1f} 分钟")
