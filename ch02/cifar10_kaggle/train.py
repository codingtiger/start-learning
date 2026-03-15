"""
Kaggle CIFAR-10 image classification with a transfer-learning baseline.

Default model:
- EfficientNetV2-S pretrained on ImageNet

Dataset layout expected under this directory:
    data/cifar-10/
        train/
        test/
        trainLabels.csv
        sampleSubmission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cifar-10"
OUTPUT_DIR = BASE_DIR / "outputs"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import certifi
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    efficientnet_v2_s,
    mobilenet_v3_large,
)

# 修复 macOS / pyenv 环境下的 SSL 证书问题，避免下载预训练权重失败。
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
MODEL_SPECS = {
    "mobilenet_v3_large": {
        "builder": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.DEFAULT,
        "classifier_attr": "classifier",
        "classifier_index": -1,
        "backbone_attr": "features",
    },
    "efficientnet_b0": {
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.DEFAULT,
        "classifier_attr": "classifier",
        "classifier_index": 1,
        "backbone_attr": "features",
    },
    "efficientnet_v2_s": {
        "builder": efficientnet_v2_s,
        "weights": EfficientNet_V2_S_Weights.DEFAULT,
        "classifier_attr": "classifier",
        "classifier_index": 1,
        "backbone_attr": "features",
    },
}


@dataclass
class Sample:
    image_path: Path
    label_idx: int


class KaggleCIFAR10Dataset(Dataset):
    def __init__(
            self,
            samples: list[Sample] | None = None,
            image_ids: list[int] | None = None,
            image_dir: Path | None = None,
            transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.image_ids or [])

    def __getitem__(self, index: int):
        if self.samples is not None:
            sample = self.samples[index]
            image = Image.open(sample.image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, sample.label_idx

        if self.image_ids is None or self.image_dir is None:
            raise RuntimeError("Test dataset is not configured correctly.")

        image_id = self.image_ids[index]
        image_path = self.image_dir / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kaggle CIFAR-10 classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS.keys()),
        default="efficientnet_b0",
        help="Balanced default for Apple Silicon is efficientnet_b0.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and generate submission.csv from an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path used by --predict-only. Defaults to outputs/best_model.pt.",
    )
    parser.add_argument("--scratch", action="store_true", help="Train without pretrained weights.")
    parser.add_argument("--dry-run", action="store_true", help="Only build model and print config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_train_samples(data_dir: Path) -> list[Sample]:
    labels_path = data_dir / "trainLabels.csv"
    train_dir = data_dir / "train"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    samples: list[Sample] = []
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["id"]
            label_name = row["label"]
            if label_name not in CLASS_TO_IDX:
                raise ValueError(f"Unknown label in trainLabels.csv: {label_name}")
            samples.append(
                Sample(
                    image_path=train_dir / f"{image_id}.png",
                    label_idx=CLASS_TO_IDX[label_name],
                )
            )
    return samples


def load_test_image_ids(data_dir: Path) -> list[int]:
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    image_ids = []
    for image_path in test_dir.glob("*.png"):
        image_ids.append(int(image_path.stem))
    return sorted(image_ids)


def build_transforms(image_size: int):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    spec = MODEL_SPECS[model_name]
    weights = spec["weights"] if pretrained else None
    model = spec["builder"](weights=weights)
    classifier = cast(nn.Sequential, getattr(model, cast(str, spec["classifier_attr"])))
    classifier_index = cast(int, spec["classifier_index"])
    classifier_layer = cast(nn.Linear, classifier[classifier_index])
    classifier[classifier_index] = nn.Linear(classifier_layer.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    spec = MODEL_SPECS[model_name]
    features = cast(nn.Module, getattr(model, cast(str, spec["backbone_attr"])))
    for param in features.parameters():
        param.requires_grad = trainable


def build_optimizer(
        model: nn.Module,
        model_name: str,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
):
    spec = MODEL_SPECS[model_name]
    features = cast(nn.Module, getattr(model, cast(str, spec["backbone_attr"])))
    classifier = cast(nn.Module, getattr(model, cast(str, spec["classifier_attr"])))
    backbone_params = [p for p in features.parameters() if p.requires_grad]
    head_params = [p for p in classifier.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        total_epochs: int,
        log_interval: int,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if step % log_interval == 0 or step == len(loader):
            avg_loss = loss_sum / total
            avg_acc = 100.0 * correct / total
            print(
                f"Epoch {epoch:02d}/{total_epochs} Step {step:04d}/{len(loader):04d} | "
                f"train_loss={avg_loss:.4f} train_acc={avg_acc:.2f}%"
            )

    return loss_sum / total, 100.0 * correct / total


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

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return loss_sum / total, 100.0 * correct / total


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "s-", label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"], "s-", label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def predict_test(model: nn.Module, loader: DataLoader, device: torch.device) -> list[tuple[int, str]]:
    model.eval()
    predictions: list[tuple[int, str]] = []

    for images, image_ids in loader:
        images = images.to(device)
        logits = model(images)
        pred_indices = logits.argmax(dim=1).cpu().tolist()
        for image_id, pred_idx in zip(image_ids.tolist(), pred_indices):
            predictions.append((image_id, CLASS_NAMES[pred_idx]))

    predictions.sort(key=lambda item: item[0])
    return predictions


def save_submission(predictions: Iterable[tuple[int, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(predictions)


def run_prediction_only(
        model: nn.Module,
        args: argparse.Namespace,
        device: torch.device,
        eval_transform: transforms.Compose,
) -> None:
    checkpoint_path = args.checkpoint or (args.output_dir / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for prediction: {checkpoint_path}")

    test_image_ids = load_test_image_ids(args.data_dir)
    test_dataset = KaggleCIFAR10Dataset(
        image_ids=test_image_ids,
        image_dir=args.data_dir / "test",
        transform=eval_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    predictions = predict_test(model, test_loader, device)
    submission_path = args.output_dir / "submission.csv"
    save_submission(predictions, submission_path)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved submission: {submission_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")

    model = build_model(model_name=args.model, num_classes=len(CLASS_NAMES), pretrained=not args.scratch)
    model = model.to(device)

    if args.dry_run:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_transform, eval_transform = build_transforms(args.image_size)

    if args.predict_only:
        run_prediction_only(model, args, device, eval_transform)
        return

    train_samples = load_train_samples(args.data_dir)

    labels = [sample.label_idx for sample in train_samples]
    train_records, val_records = train_test_split(
        train_samples,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    train_dataset = KaggleCIFAR10Dataset(samples=train_records, transform=train_transform)
    val_dataset = KaggleCIFAR10Dataset(samples=val_records, transform=eval_transform)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 先只训练分类头，缩短预热时间；随后解冻骨干网络进行全量微调。
    set_backbone_trainable(model, args.model, trainable=False)
    optimizer = build_optimizer(model, args.model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = args.output_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            set_backbone_trainable(model, args.model, trainable=True)
            optimizer = build_optimizer(model, args.model, args.lr_backbone, args.lr_head, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
            )
            print("Backbone unfrozen, start full fine-tuning.")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.epochs,
            args.log_interval,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time
    print(f"Best val acc: {best_val_acc:.2f}%")
    print(f"Training finished in {elapsed / 60.0:.1f} min")

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "best_val_acc": best_val_acc,
        "training_minutes": elapsed / 60.0,
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_history(history, args.output_dir / "training_curves.png")

    print(f"Saved model: {best_model_path}")
    print(f"Saved curves: {args.output_dir / 'training_curves.png'}")
    print("Training finished without test prediction.")
    print(
        "Run prediction later with: "
        "python ch02/cifar10_kaggle/train.py --predict-only"
    )


if __name__ == "__main__":
    main()
