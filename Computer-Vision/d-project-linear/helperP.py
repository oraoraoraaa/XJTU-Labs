import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# Training constants (fixed, not to be changed by students)
epoch = 100
epoch_sgd = 100
momentum = False


def prepare_data(split, base_dir):
    if split == 'train':
        features = np.load(os.path.join(base_dir, 'feature_train.npy'))
        ages = np.loadtxt(os.path.join(base_dir, 'train.txt'))
        return ages, features
    elif split == 'val':
        features = np.load(os.path.join(base_dir, 'feature_val.npy'))
        ages = np.loadtxt(os.path.join(base_dir, 'val.txt'))
        return ages, features
    elif split == 'test':
        features = np.load(os.path.join(base_dir, 'feature_test.npy'))
        return None, features
    else:
        raise ValueError(f"Unknown split: {split}")


def show_data(base_dir):
    val_images = sorted(glob.glob(os.path.join(base_dir, 'valid', '*.jpg')))
    ages = np.loadtxt(os.path.join(base_dir, 'val.txt'))

    n_show = min(len(val_images), 8)
    if n_show == 0:
        return

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for i, (img_path, ax) in enumerate(zip(val_images[:n_show], axes)):
        img = Image.open(img_path)
        ax.imshow(np.array(img))
        ax.set_title(f"Age: {ages[i]:.0f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate(w, b, age, features):
    w = np.array(w).flatten()
    b = float(np.array(b).flatten()[0])
    pred = features @ w + b
    loss = float(np.mean(np.abs(pred - age)))
    return loss, pred


def test(w, b, features, filename):
    w = np.array(w).flatten()
    b = float(np.array(b).flatten()[0])
    pred = features @ w + b
    np.savetxt(filename, pred)
    return pred


def load_image(file_names, size=(32, 32)):
    images = []
    reduced = []
    labels = []

    for fname in sorted(file_names):
        img = Image.open(fname).convert('RGBA')
        images.append(np.array(img))

        gray = img.convert('L').resize(size, Image.LANCZOS)
        reduced.append(np.array(gray, dtype=float) / 255.0)

        if 'train_smile' in fname:
            labels.append(1.0)
        else:
            labels.append(-1.0)

    return np.array(reduced), images, np.array(labels)


def visualize_results(images, preds, labels, ax=None):
    preds_flat = np.array(preds).flatten()
    y_hat = np.sign(preds_flat)
    n = len(images)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        sign_val = int(y_hat[i]) if y_hat[i] != 0 else 1
        axes[0, i].text(0.5, 0.5, str(sign_val),
                        ha='center', va='center', fontsize=14, fontweight='bold')
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        axes[0, i].axis('off')

        axes[1, i].imshow(images[i])
        axes[1, i].axis('off')

    plt.tight_layout()

    try:
        from IPython.display import display, clear_output
        clear_output(wait=True)
        display(fig)
        plt.close(fig)
    except ImportError:
        plt.show()
