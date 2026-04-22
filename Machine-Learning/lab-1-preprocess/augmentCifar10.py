import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def add_noise(tensor, sigma=0.1):
    noise = torch.randn_like(tensor) * sigma
    return torch.clamp(tensor + noise, 0.0, 1.0)


def to_numpy_chw(image_tensor):
    return image_tensor.detach().cpu().permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 augmentation lab task")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Folder containing cifar-10-batches-py",
    )
    parser.add_argument(
        "--rows", type=int, default=10, help="How many images to visualize"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    workdir = Path(__file__).resolve().parent
    data_root = (workdir / args.data_root).resolve()
    out_dir = workdir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_transform = transforms.ToTensor()
    rotate_transform = transforms.RandomRotation(30)
    flip_transform = transforms.RandomHorizontalFlip(p=1.0)
    crop_transform = transforms.RandomCrop(32, padding=4)

    testset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=False,
        transform=base_transform,
    )

    rows = min(args.rows, len(testset))

    fig, axes = plt.subplots(rows, 5, figsize=(15, max(6, rows * 1.8)))
    plt.subplots_adjust(wspace=0.08, hspace=0.2)

    if rows == 1:
        axes = axes.reshape(1, -1)

    for r in range(rows):
        original, _ = testset[r]
        rotated = rotate_transform(original)
        flipped = flip_transform(original)
        cropped = crop_transform(original)
        noisy = add_noise(original, sigma=0.1)

        images = [original, rotated, flipped, cropped, noisy]
        titles = ["Original", "Rotated", "Flipped", "Cropped", "Noisy"]

        for c in range(5):
            axes[r, c].imshow(to_numpy_chw(images[c]))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)

    plt.suptitle("Task 3: CIFAR-10 Data Augmentations", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "task3_augmentations_grid.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(out_dir / "task3_augmentations_grid.png")


if __name__ == "__main__":
    main()
