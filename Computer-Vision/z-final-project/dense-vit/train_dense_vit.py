import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from helperT import get_img_dataloaders, test_cel

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DATASET"))


class DenseNetAgeNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self._initialize_classifier()

    def forward(self, x):
        return self.backbone(x)

    def _initialize_classifier(self):
        nn.init.kaiming_normal_(
            self.backbone.classifier.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.backbone.classifier.bias is not None:
            nn.init.constant_(self.backbone.classifier.bias, 0)


class ViTAgeNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.backbone.heads.head.weight, std=0.02)
        if self.backbone.heads.head.bias is not None:
            nn.init.constant_(self.backbone.heads.head.bias, 0)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(backbone, num_classes=101):
    backbone = backbone.lower()
    if backbone == "densenet":
        return DenseNetAgeNet(num_classes=num_classes)
    if backbone == "vit":
        return ViTAgeNet(num_classes=num_classes)
    raise ValueError(f"Unsupported backbone: {backbone}")


def make_optimizer(backbone, model):
    backbone = backbone.lower()
    if backbone == "vit":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    return torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0)


def train_model(backbone, base_dir=DEFAULT_DATA_DIR, epochs=100):
    train_loader, val_loader, test_loader = get_img_dataloaders(base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(backbone).to(device)
    optimizer = make_optimizer(backbone, model)
    criterion = nn.CrossEntropyLoss()
    best_mae = float("inf")
    best_path = os.path.join(SCRIPT_DIR, f"best_{backbone.lower()}.pth")
    ages = torch.arange(0, 101, device=device).float()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for _, (y, x) in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).long()

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        preds_val = []
        gts_val = []
        with torch.no_grad():
            for _, (y, x) in enumerate(val_loader):
                x = x.to(device).float()
                y = y.to(device).long()
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                pred_age = (probs * ages).sum(dim=1)
                preds_val.append(pred_age.cpu().numpy())
                gts_val.append(y.cpu().numpy())

        preds_val = np.concatenate(preds_val, axis=0)
        gts_val = np.concatenate(gts_val, axis=0)
        mae = np.mean(np.abs(preds_val - gts_val))

        print(
            f"[{backbone}] Epoch {epoch + 1}/{epochs} loss={running_loss:.4f} val_mae={mae:.4f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), best_path)
            print("Saved best model to", best_path)

    return model, best_path, test_loader


def run_test(backbone, model, test_loader):
    output_path = os.path.join(SCRIPT_DIR, f"custom_{backbone.lower()}.txt")
    prediction = test_cel(model, test_loader, output_path)
    print("Test results saved to", output_path)
    print(prediction[:10])
    return prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DenseNet or ViT age predictors on APPA-REAL."
    )
    parser.add_argument(
        "--backbone",
        default="densenet",
        choices=["densenet", "vit", "both"],
        help="Model backbone to train.",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Dataset root directory."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backbones = [args.backbone] if args.backbone != "both" else ["densenet", "vit"]

    for backbone in backbones:
        model, best_path, test_loader = train_model(
            backbone, base_dir=args.data_dir, epochs=args.epochs
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        run_test(backbone, model, test_loader)


if __name__ == "__main__":
    main()
