# Age Prediction

> OPTIONS
>
> 1. Use the feature-based pipeline (quick, ready): call get_dataloaders('DATASET/'). This uses feature_\*.npy and\*.txt directly and requires no extra preprocessing.
>
> 2. Use the image-based pipeline (full image training): prepare gt_avg_train.csv, gt_avg_valid.csv, gt_avg_test.csv and ensure train/, valid/, test/ folders contain the corresponding _face.jpg images. Then call get_img_dataloaders('DATASET/'). This allows end-to-end training on images (but needs extra setup and more compute/GPU).

## How to Run (Option 2)

1. Put the APPA-REAL dataset under network-publish/DATASET/ as described in the notebook (the folder appa-real-release/ and related files must be present).

2. Install dependencies.

3. Run training (GPU recommended — helper functions call .cuda() in several places):

```python
python train_ageNet.py
```

Output: `network-publish/best_age_net.pth` and `network-publish/custom.txt` (test predictions).
