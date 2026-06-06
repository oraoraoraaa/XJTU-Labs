# Age Prediction

This project implements the age-prediction lab for the APPA-REAL dataset:

Image-based pipeline for end-to-end CNN training. Prepare `gt_avg_train.csv`, `gt_avg_valid.csv`, and `gt_avg_test.csv`, and make sure `train/`, `valid/`, and `test/` contain the matching `_face.jpg` images. Then call `get_img_dataloaders('DATASET/')`.

## How to Run

### AgeNet

1. Put the APPA-REAL dataset under `DATASET/` so that `appa-real-release/` and the CSV files required by the image pipeline are present. Download link: [appa-real-release](https://chalearnlap.cvc.uab.cat/dataset/26/description/).

2. Create and activate a Python environment, then install dependencies. The project expects PyTorch, torchvision, pandas, Pillow, OpenCV, imgaug, and numpy.

3. Run training. GPU is recommended, but the data helpers now work on CPU too.

```bash
python train_ageNet.py
```

Outputs:

- Terminal output: [Computer-Vision/z-final-project/terminal_output.txt](agenet/terminal_output.txt)
- Best checkpoint saved by validation MAE (Stored in google drive because of its large size): [Computer-Vision/z-final-project/best_age_net.pth](https://drive.google.com/file/d/1Z8bHwR7482lGMzuX4g3M57c-ZIEwyWpI/view?usp=sharing)
- Test predictions written after training finishes: [Computer-Vision/z-final-project/custom.txt](agenet/custom.txt)

## Project Layout

- `age_net.py`: AgeNet model definition.
- `helperT.py`: dataset loaders, augmentation, and testing helpers.
- `train_ageNet.py`: training entry point.
- `README.md`: run instructions.

### DenseNet and VIT

Navigate to `Computer-Vision/z-final-project/dense-vit`, and run:

```bash
python trainn_dense_vit.py --backbone both
```

Best checkpoint:

- [DenseNet](https://drive.google.com/file/d/1NBemYm7YQObmzZg1SeEZ6n6IfDw9mDs0/view?usp=sharing)
- [VIT](https://drive.google.com/file/d/1vLN7IxB_pKKsog9XzYVsj0cu_r9dUqfD/view?usp=sharing)

## Notes

- The model architecture in `age_net.py:13` follows the required 5 conv blocks plus 3 fully connected layers and outputs 101 classes, matching the [guidance](./instruction/guidance.md).
- The training procedure in `train_ageNet.py:24` uses AgeNet, moves model/data to device, uses SGD, and runs 100 epochs.
- The logged loss is an epoch sum, not an average: see `train_ageNet.py:47` and `train_ageNet.py:67`.
- The best MAE around 7.34 is decent for a from-scratch VGG-style baseline.
