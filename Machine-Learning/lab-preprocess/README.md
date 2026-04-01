# CIFAR-10 Dataset Preprocessing Lab

This folder contains implementations for the three required tasks:

1. Read a batch from CIFAR-10, unify image size, and visualize in one figure.
2. Compute dataset mean/std, subtract mean, perform z-score normalization, and compare one chosen sample.
3. Apply data augmentation (rotation, flip, crop, noise) and visualize original vs transformed images.

## Files

- `preprocessCifar10.py`: Task 1 + Task 2
- `augmentCifar10.py`: Task 3
- `results/`: output images generated after running scripts

## Run

From this directory:

```bash
python preprocessCifar10.py --batch-size 64 --target-size 32 --sample-index 123
python augmentCifar10.py --rows 10
```

If your dataset path differs, set `--data-root` to the folder that contains `cifar-10-batches-py`.
