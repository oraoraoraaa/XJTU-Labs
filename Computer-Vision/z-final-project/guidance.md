# Project Summary: Online Computer Vision and Pattern Recognition

* **Project Title:** Project I: Neural Network (Age Prediction Model)
* **Author:** Yuanqi Su
* **Date:** May 31, 2020

---

## 1. Project Overview & Objectives

The goal of this project is to build an end-to-end convolutional neural network (CNN) named `AgeNet` to perform multi-class classification for age prediction. The model processes a fixed-size \(128 \times 128\) RGB image on one side and produces a 101-way age classification score on the other side.

---

## 2. Environment & Directory Structure

The project relies on downloading and unzipping the **APPA-REAL DATABASE** into a specific directory structure:

```text
Network/
├── DATASET/
│   ├── appa-real-release/
│   ├── valid/
│   ├── feature_test.npy
│   └── feature_train.npy
├── helperT.py
├── ignore_list.csv
└── networkCustom.ipynb
```

---

## 3. Network Architecture

The network consists of a series of convolutional layers (with ReLU activations and max-pooling), followed by three fully-connected (FC) layers.

### Figure 1: Spatial Block Pipeline Visual Representation

```text
[128x128 Input] -> [Conv/ReLU/Pool] -> [Conv/Conv/ReLU/Pool] -> [Conv/Conv/Conv/ReLU/Pool] -> [Conv/Conv/Conv/ReLU/Pool] -> [Conv/Conv/Conv/ReLU/Pool] -> [FC/FC/FC] -> [Output]
```

### Table 1: Detailed Layer Specifications

Spatial resolution is preserved after each convolution layer using specific padding sizes. Max-pooling layers use a $2 \times 2$ pixel window with a stride of 2.

| Layer Type | Kernel Size | Channels (`fan_in`, `fan_out`) | Stride | Padding |
| :--- | :---: | :---: | :---: | :---: |
| **Convolution** | 7 | (3, 64) | 1 | 3 |
| **ReLU** | - | - | - | - |
| **Max Pooling** | 2 | - | 2 | - |
| **Convolution** | 5 | (64, 128) | 1 | 2 |
| **ReLU** | - | - | - | - |
| **Convolution** | 5 | (128, 128) | 1 | 2 |
| **ReLU** | - | - | - | - |
| **Max Pooling** | 2 | - | 2 | - |
| **Convolution** | 3 | (128, 256) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (256, 256) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (256, 256) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Max Pooling** | 2 | - | 2 | - |
| **Convolution** | 3 | (256, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (512, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (512, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Max Pooling** | 2 | - | 2 | - |
| **Convolution** | 3 | (512, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (512, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Convolution** | 3 | (512, 512) | 1 | 1 |
| **ReLU** | - | - | - | - |
| **Max Pooling** | 2 | - | 2 | - |
| **Linear (FC)** | - | ($512 \times 4 \times 4$, 4096) | - | - |
| **ReLU** | - | - | - | - |
| **Linear (FC)** | - | (4096, 4096) | - | - |
| **ReLU** | - | - | - | - |
| **Linear (FC)** | - | (4096, 101) | - | - |

*Note: Every convolutional and fully connected layer is followed by a ReLU activation function except for the final linear layer, which connects to a soft-max function.*

---

## 4. Implementation Requirements (`torch.nn`)

Students must implement the model within the `networkCustom.ipynb` notebook by completing the following components:

### Class: `AgeNet(nn.Module)`

* **`__init__` function:** Define the layer parameters, dimensions, and whether bias terms are utilized.
* **`forward` function:** Outline how the output is sequentially computed from the input.

### Function: `train_ageNet`

Tasks to implement inside the training function:

1. Instantiate the `AgeNet` model.
2. Move the network parameters to the target device (**CPU** or **GPU**/Colab).
3. Define the evaluation criterion (loss function).
4. Set up the **SGD optimizer**.
5. Execute the forward and backward training passes to update the network weights.

### Training Constraints

* **Validation:** Evaluate the network performance on the validation set during optimization and save the parameters yielding the best validation accuracy.
* **Epoch Limit:** To maintain fairness across comparisons, the maximum training duration is fixed at **100 epochs**.
