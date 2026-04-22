import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def compute_dataset_stats(dataset, batch_size=512, num_workers=0):
    """Compute channel-wise mean/std for tensors in [0, 1]."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        # images shape: [N, C, H, W]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images**2).sum(dim=(0, 2, 3))
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / total_pixels
    var = channel_sq_sum / total_pixels - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean, std


def to_numpy_chw(image_tensor):
    return image_tensor.detach().cpu().permute(1, 2, 0).numpy()


def minmax_for_display(image_tensor):
    t = image_tensor.detach().cpu()
    t_min = t.min()
    t_max = t.max()
    if torch.isclose(t_max, t_min):
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 preprocessing lab tasks")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Folder containing cifar-10-batches-py",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for loading"
    )
    parser.add_argument(
        "--target-size", type=int, default=32, help="Unified image size"
    )
    parser.add_argument(
        "--sample-index", type=int, default=123, help="Index for task 2 visualization"
    )
    parser.add_argument(
        "--show-count",
        type=int,
        default=None,
        help="Number of images shown in task 1 (default: full batch)",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    data_root = (workdir / args.data_root).resolve()
    out_dir = workdir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.target_size + 4),
            transforms.CenterCrop(args.target_size),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # Task 1: read one batch and print a grid image.
    images, _ = next(iter(trainloader))
    if args.show_count is None:
        show_count = images.size(0)
    else:
        show_count = max(1, min(args.show_count, images.size(0)))

    nrow = min(10, show_count)
    nrows = math.ceil(show_count / nrow)
    selected_images = images[:show_count]
    grid = torchvision.utils.make_grid(selected_images, nrow=nrow, padding=2)

    plt.figure(figsize=(15, max(2.5, 2.2 * nrows)))
    plt.imshow(to_numpy_chw(grid))
    plt.axis("off")
    plt.title("Task 1: One Batch From CIFAR-10 (Unified Size)")
    plt.tight_layout()
    plt.savefig(out_dir / "task1_batch_grid.png", dpi=180)
    plt.close()

    # Task 2: compute mean, subtract mean, then z-score normalize.
    mean, std = compute_dataset_stats(trainset)

    sample_index = max(0, min(args.sample_index, len(trainset) - 1))
    original, _ = trainset[sample_index]

    mean_3d = mean.view(3, 1, 1)
    std_3d = std.view(3, 1, 1)

    centered = original - mean_3d
    normalized = centered / std_3d

    # For centered/normalized visualization, map values to [0, 1] only for plotting.
    centered_vis = minmax_for_display(centered)
    normalized_vis = minmax_for_display(normalized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_numpy_chw(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(to_numpy_chw(centered_vis))
    axes[1].set_title("Minus Mean (vis min-max)")
    axes[1].axis("off")

    axes[2].imshow(to_numpy_chw(normalized_vis))
    axes[2].set_title("Z-score (vis min-max)")
    axes[2].axis("off")

    fig.suptitle(
        f"Task 2: idx={sample_index}  mean={mean.tolist()}  std={std.tolist()}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "task2_normalization_compare.png", dpi=180)
    plt.close()

    print("Saved:")
    print(out_dir / "task1_batch_grid.png")
    print(out_dir / "task2_normalization_compare.png")


if __name__ == "__main__":
    main()
