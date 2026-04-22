# lab-cnn-rnn

This folder implements both required tasks without modifying the `template` folder.

## Task 1: Image Classification (CIFAR-10)

Script: `task1_cifar_vgg_resnet.py`

- Required models: **VGG** and **ResNet**
- Supports training one model or both for comparison
- Tunable parameters: model depth/layers, optimizer, learning rate, batch size, weight decay, epochs
- Outputs:
  - Best model checkpoint for each model
  - `metrics.json` for each model
  - Confusion matrix image
  - `comparison.json` when running both models

### Run examples

```bash
cd /home/rin/Local/Github/XJTU-Labs/Machine-Learning/lab-cnn-rnn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train and compare VGG + ResNet:

```bash
python task1_cifar_vgg_resnet.py \
  --model both \
  --epochs 20 \
  --batch_size 128 \
  --optimizer adam \
  --lr 0.001 \
  --vgg_variant vgg11 \
  --resnet_depth 20
```

Train only VGG (deeper variant):

```bash
python task1_cifar_vgg_resnet.py --model vgg --vgg_variant vgg11 --epochs 30
```

Train only ResNet (deeper depth):

```bash
python task1_cifar_vgg_resnet.py --model resnet --resnet_depth 32 --epochs 30
```

## Task 2: Time-Series Multi-Step Forecasting (ECG5000)

Script: `task2_ecg_lstm_gru.py`

- Required models: **LSTM** and **GRU**
- Uses ECG5000 dataset in this folder
- Supports **single-step** (`--pred_len 1`) and **multi-step** (`--pred_len > 1`) prediction
- Tunable parameters: optimizer, learning rate, input window size, prediction horizon, hidden size, layers, batch size, epochs
- Outputs:
  - Best model checkpoint for each model
  - `metrics.json` for each model (MSE and MAE)
  - Forecast plots (`first_step_curve.png`, `horizon_sample.png`)
  - `comparison.json` when running both models

### Run examples

Multi-step comparison (LSTM + GRU):

```bash
python task2_ecg_lstm_gru.py \
  --model both \
  --input_len 60 \
  --pred_len 10 \
  --epochs 30 \
  --batch_size 128 \
  --optimizer adam \
  --lr 0.001
```

Single-step baseline:

```bash
python task2_ecg_lstm_gru.py --model lstm --pred_len 1 --epochs 20
```

GRU with different time step and optimizer:

```bash
python task2_ecg_lstm_gru.py \
  --model gru \
  --input_len 80 \
  --pred_len 12 \
  --optimizer sgd \
  --lr 0.01 \
  --epochs 40
```

## Output structure

Generated files are saved under:

- `outputs/task1/<model>/...`
- `outputs/task2/<model>/...`
- `outputs/task1/comparison.json` (if `--model both`)
- `outputs/task2/comparison.json` (if `--model both`)
