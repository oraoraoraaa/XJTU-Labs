import math
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformerRaw import TransformerEncoder, TransformerDecoder


class QADataset(Dataset):
    """SQuAD格式数据处理"""

    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path) as f:
            squad_data = json.load(f)

        for article in squad_data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    if not qa["answers"]:
                        continue
                    answer = qa["answers"][0]["text"]

                    # 构造模型输入输出
                    input_text = f"[CLS] {question} [SEP] {context} [SEP]"
                    output_text = f"[CLS] {answer} [SEP]"

                    # 编码文本
                    inputs = tokenizer(
                        input_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    targets = tokenizer(
                        output_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    self.data.append(
                        {
                            "input_ids": inputs["input_ids"].squeeze(0),
                            "attention_mask": inputs["attention_mask"].squeeze(0),
                            "labels": targets["input_ids"].squeeze(0),
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class TransformerQA(nn.Module):
    """问答模型（含自定义编码器/解码器）"""

    def __init__(
        self,
        vocab_size,
        tokenizer,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, nhead, dim_feedforward, dropout
        )

        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, nhead, dim_feedforward, dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器处理
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_mask)

        # 解码器处理
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask)

        return self.fc_out(output)

    def generate_mask(self, src, tgt):
        """生成注意力掩码"""
        # 源序列填充掩码
        src_pad_mask = (src == 0).unsqueeze(1).unsqueeze(2)

        # 目标序列掩码（自回归+填充）
        tgt_pad_mask = (tgt == 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        tgt_sub_mask = (
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(src.device)
        )
        tgt_mask = tgt_pad_mask | tgt_sub_mask

        return src_pad_mask, tgt_mask

    # 实现自回归生成过程
    def generate_answer(self, src, max_len=50):
        memory = self.encoder(src)
        outputs = torch.LongTensor([[self.tokenizer.cls_token_id]]).to(src.device)

        for _ in range(max_len):
            out = self.decoder(outputs, memory)
            next_token = out.argmax(-1)[:, -1:]
            outputs = torch.cat([outputs, next_token], dim=-1)

            if next_token.item() == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class QATrainer:
    """训练管理类"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def prepare_batch(self, batch):
        """数据预处理"""
        src = batch["input_ids"].to(self.device)  # 源序列（问题+上下文）
        tgt = batch["labels"].to(self.device)  # 目标序列（答案）
        # 注意：batch是否在前。

        # 构造解码器输入（去尾）和输出（去头）
        decoder_input = tgt[:, :-1]  # 移除最后一个token
        decoder_output = tgt[:, 1:]  # 移除第一个token

        # 生成注意力掩码
        src_mask, tgt_mask = self.model.generate_mask(src, decoder_input)
        return src, decoder_input, decoder_output, src_mask, tgt_mask

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            # 数据准备
            src, decoder_in, decoder_out, src_mask, tgt_mask = self.prepare_batch(batch)

            optimizer.zero_grad()

            # 前向传播 (B, S) -> (B, S, V)
            # 直接使用目标序列来训练
            outputs = self.model(
                src.transpose(0, 1),  # (S, B) 时间维在前
                decoder_in.transpose(0, 1),
                src_mask,
                tgt_mask,
            ).transpose(
                0, 1
            )  # 转回(B, S, V)

            # 计算损失
            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[-1]), decoder_out.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 每100批次打印进度
            if batch_idx % 100 == 0:
                current_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx} | Loss: {current_loss:.4f}")

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                src, decoder_in, decoder_out, src_mask, tgt_mask = self.prepare_batch(
                    batch
                )
                outputs = self.model(
                    src.transpose(0, 1), decoder_in.transpose(0, 1), src_mask, tgt_mask
                ).transpose(0, 1)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]), decoder_out.reshape(-1)
                )
                total_loss += loss.item()
        return total_loss / len(dataloader)


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--train_path", type=str, default="SQuAD-train-v2.0.json")
    parser.add_argument("--dev_path", type=str, default="SQuAD-dev-v2.0.json")
    parser.add_argument("--model_path", type=str, default="transformer_seq2seq.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_dataset = QADataset(
            args.train_path, tokenizer, max_length=args.max_length
        )
        dev_dataset = QADataset(args.dev_path, tokenizer, max_length=args.max_length)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        model = TransformerQA(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
        trainer = QATrainer(model, tokenizer, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss = trainer.train_epoch(train_loader, optimizer)
            eval_loss = trainer.evaluate(dev_loader)
            print(
                f"Epoch {epoch+1} | train_loss={train_loss:.4f} | dev_loss={eval_loss:.4f}"
            )
        torch.save(model.state_dict(), args.model_path)
        print(f"Saved model to {args.model_path}")
    else:
        dev_dataset = QADataset(args.dev_path, tokenizer, max_length=args.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        model = TransformerQA(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        trainer = QATrainer(model, tokenizer, device=device)
        test_loss = trainer.evaluate(dev_loader)
        print(f"Test loss: {test_loss:.4f}")
