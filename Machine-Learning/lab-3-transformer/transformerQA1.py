import json
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import argparse
import os
from tqdm import tqdm
from transformers import BertTokenizerFast


class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 3e-5
        self.epochs = 3
        self.max_length = 384
        self.model_dir = "./model"
        self.train_path = "SQuAD-train-v2.0.json"
        self.dev_path = "SQuAD-dev-v2.0.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 768
        self.nhead = 12
        self.dim_feedforward = 3072
        self.dropout = 0.1
        self.num_layers = 7
        self.vocab_size = 30522
        self.num_workers = max(1, (os.cpu_count() or 1) - 1)
        self.pin_memory = torch.cuda.is_available()
        self.prefetch_factor = 4
        self.use_amp = torch.cuda.is_available()
        # torch.compile may be slower/noisier on smaller GPUs; keep it opt-in.
        self.use_compile = False


def optimize_runtime(config):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


class SQuADProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def load_data(self, path):
        with open(path, "r") as f:
            return json.load(f)["data"]

    def process(self, data):
        examples = []
        for article in data:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if not qa["answers"]:
                        continue
                    example = {
                        "context": context,
                        "question": qa["question"],
                        "answer": qa["answers"][0],
                    }
                    examples.append(example)
        return examples

    def create_features(self, examples):
        input_ids, masks = [], []
        start_pos, end_pos = [], []

        for ex in examples:
            encoding = self.tokenizer(
                ex["question"],
                ex["context"],
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )

            ans_start = ex["answer"]["answer_start"]
            ans_end = ans_start + len(ex["answer"]["text"])

            start_char = ans_start
            end_char = ans_end
            sequence_ids = encoding.sequence_ids()

            # Find token positions
            start_token, end_token = -1, -1
            for i, (idx, (s, e)) in enumerate(
                zip(sequence_ids, encoding.offset_mapping)
            ):
                if idx != 1:
                    continue  # Only look at context tokens
                if s <= start_char < e:
                    start_token = i
                if s < end_char <= e:
                    end_token = i

            if start_token != -1 and end_token != -1:
                input_ids.append(encoding["input_ids"])
                masks.append(encoding["attention_mask"])
                start_pos.append(start_token)
                end_pos.append(end_token)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
            "start_pos": torch.tensor(start_pos),
            "end_pos": torch.tensor(end_pos),
        }


class SQuADDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.features.items()}


class TransformerQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(config.max_length, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.start_fc = nn.Linear(config.d_model, 1)
        self.end_fc = nn.Linear(config.d_model, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) + self.pos_encoder[: input_ids.size(1)]
        x = self.dropout(x)

        output = self.encoder(x, src_key_padding_mask=~attention_mask.bool())

        start_logits = self.start_fc(output).squeeze(-1)
        end_logits = self.end_fc(output).squeeze(-1)
        return start_logits, end_logits


class QATrainer:
    def __init__(self, config, model, train_loader, dev_loader=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler("cuda", enabled=(config.use_amp and config.device.type == "cuda"))

        if config.use_compile:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"torch.compile disabled: {e}")
                self.config.use_compile = False

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.config.device, non_blocking=True)
            start = batch["start_pos"].to(self.config.device, non_blocking=True)
            end = batch["end_pos"].to(self.config.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=(self.config.use_amp and self.config.device.type == "cuda")):
                s_logits, e_logits = self.model(input_ids, mask)
                loss = self.criterion(s_logits, start) + self.criterion(e_logits, end)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_start_acc = 0.0
        total_end_acc = 0.0
        with torch.inference_mode():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.config.device, non_blocking=True)
                mask = batch["attention_mask"].to(self.config.device, non_blocking=True)
                start = batch["start_pos"].to(self.config.device, non_blocking=True)
                end = batch["end_pos"].to(self.config.device, non_blocking=True)

                with autocast("cuda", enabled=(self.config.use_amp and self.config.device.type == "cuda")):
                    s_logits, e_logits = self.model(input_ids, mask)
                    loss = self.criterion(s_logits, start) + self.criterion(e_logits, end)
                total_loss += loss.item()

                pred_start = torch.argmax(s_logits, dim=-1)
                pred_end = torch.argmax(e_logits, dim=-1)
                total_start_acc += (pred_start == start).float().mean().item()
                total_end_acc += (pred_end == end).float().mean().item()

        metrics = {
            "loss": total_loss / len(self.dev_loader),
            "start_acc": total_start_acc / len(self.dev_loader),
            "end_acc": total_end_acc / len(self.dev_loader),
        }
        return metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (may help on powerful GPUs, but can be slower on small GPUs).",
    )
    args = parser.parse_args()

    # 训练命令：python script.py --mode train
    # 测试命令：python script.py --mode test --model_path /path/to/model

    config = Config()
    config.use_compile = args.compile or os.environ.get("USE_TORCH_COMPILE", "0") == "1"
    optimize_runtime(config)
    processor = SQuADProcessor(config)

    if args.mode == "train":
        os.makedirs(config.model_dir, exist_ok=True)
        train_data = processor.load_data(config.train_path)
        train_examples = processor.process(train_data)
        train_features = processor.create_features(train_examples)
        train_dataset = SQuADDataset(train_features)
        train_loader_kwargs = {
            "batch_size": config.batch_size,
            "shuffle": True,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        if config.num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = config.prefetch_factor
        train_loader = DataLoader(train_dataset, **train_loader_kwargs)

        dev_data = processor.load_data(config.dev_path)
        dev_examples = processor.process(dev_data)
        dev_features = processor.create_features(dev_examples)
        dev_loader_kwargs = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        if config.num_workers > 0:
            dev_loader_kwargs["persistent_workers"] = True
            dev_loader_kwargs["prefetch_factor"] = config.prefetch_factor
        dev_loader = DataLoader(SQuADDataset(dev_features), **dev_loader_kwargs)

        model = TransformerQA(config)
        trainer = QATrainer(config, model, train_loader, dev_loader)

        for epoch in range(config.epochs):
            avg_loss = trainer.train_epoch()
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            metrics = trainer.evaluate()
            print(
                f"Eval loss={metrics['loss']:.4f}, start_acc={metrics['start_acc']:.4f}, end_acc={metrics['end_acc']:.4f}"
            )
            trainer.save_model(f"{config.model_dir}/epoch_{epoch+1}.pt")
            if args.model_path:
                trainer.save_model(args.model_path)

    elif args.mode == "test":
        # 测试流程
        dev_data = processor.load_data(config.dev_path)
        dev_examples = processor.process(dev_data)
        dev_features = processor.create_features(dev_examples)
        dev_loader_kwargs = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        if config.num_workers > 0:
            dev_loader_kwargs["persistent_workers"] = True
            dev_loader_kwargs["prefetch_factor"] = config.prefetch_factor
        dev_loader = DataLoader(SQuADDataset(dev_features), **dev_loader_kwargs)

        model = TransformerQA(config)
        trainer = QATrainer(config, model, None, dev_loader)
        trainer.load_model(args.model_path)
        metrics = trainer.evaluate()
        print(f"Test Results - {metrics}")


if __name__ == "__main__":
    main()
