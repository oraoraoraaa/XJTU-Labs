import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VGGClassifier(nn.Module):
    CFGS = {
        "vgg9": [64, "M", 128, "M", 256, 256, "M"],
        "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"],
    }

    def __init__(self, variant: str = "vgg9", num_classes: int = 10):
        super().__init__()
        if variant not in self.CFGS:
            raise ValueError(f"Unsupported VGG variant: {variant}")
        self.features = self._make_layers(self.CFGS[variant])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_ch = 3
        last_out_ch = 3
        for layer in cfg:
            if layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, layer, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(layer))
                layers.append(nn.ReLU(inplace=True))
                in_ch = layer
                last_out_ch = layer
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        if last_out_ch < 512:
            layers.append(nn.Conv2d(last_out_ch, 512, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, depth=20, num_classes=10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("ResNet depth must satisfy 6n+2 for CIFAR (e.g., 14, 20, 32)")
        n = (depth - 2) // 6
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def get_dataloaders(data_root: Path, batch_size: int, num_workers: int):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def choose_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion, device, collect_predictions=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

            if collect_predictions:
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

    result = {
        "loss": total_loss / total,
        "acc": total_correct / total,
    }
    if collect_predictions:
        result["labels"] = all_labels
        result["preds"] = all_preds
    return result


def save_confusion_matrix(labels, preds, classes, out_path: Path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_and_evaluate(model_name: str, args, train_loader, test_loader, device, classes, output_root: Path):
    if model_name == "vgg":
        model = VGGClassifier(variant=args.vgg_variant).to(device)
        model_tag = args.vgg_variant
    elif model_name == "resnet":
        model = ResNetCifar(depth=args.resnet_depth).to(device)
        model_tag = f"resnet{args.resnet_depth}"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    run_dir = output_root / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    best_acc = 0.0
    history = []
    best_path = run_dir / "best_model.pth"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, test_loader, criterion, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": val_result["loss"],
            "test_acc": val_result["acc"],
        }
        history.append(epoch_result)

        if val_result["acc"] > best_acc:
            best_acc = val_result["acc"]
            torch.save(model.state_dict(), best_path)

        print(
            f"[{model_tag}] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Acc {train_acc * 100:.2f}% | Test Acc {val_result['acc'] * 100:.2f}%"
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_result = evaluate(model, test_loader, criterion, device, collect_predictions=True)
    save_confusion_matrix(
        final_result["labels"],
        final_result["preds"],
        classes,
        run_dir / "confusion_matrix.png",
    )

    summary = {
        "model": model_tag,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "best_test_acc": best_acc,
        "final_test_acc": final_result["acc"],
        "final_test_loss": final_result["loss"],
        "train_seconds": time.time() - start,
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: CIFAR-10 classification with VGG/ResNet")
    parser.add_argument("--model", choices=["vgg", "resnet", "both"], default="both")
    parser.add_argument("--vgg_variant", choices=["vgg9", "vgg11"], default="vgg11")
    parser.add_argument("--resnet_depth", type=int, default=20, help="Valid examples: 14, 20, 32")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./cifar-10-batches-py")
    parser.add_argument("--output_dir", type=str, default="./outputs/task1")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(data_root, args.batch_size, args.num_workers)

    models_to_run = [args.model] if args.model != "both" else ["vgg", "resnet"]
    summaries = []
    for model_name in models_to_run:
        summaries.append(train_and_evaluate(model_name, args, train_loader, test_loader, device, classes, output_root))

    if len(summaries) == 2:
        comparison = {
            "task": "cifar10_vgg_resnet_comparison",
            "vgg": summaries[0],
            "resnet": summaries[1],
            "better_model": summaries[0]["model"] if summaries[0]["final_test_acc"] >= summaries[1]["final_test_acc"] else summaries[1]["model"],
        }
        with (output_root / "comparison.json").open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison written to {output_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
