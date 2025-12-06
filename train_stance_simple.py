import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np


# ===========================
# 설정
# ===========================
CSV_PATH = "stance_dataset.csv"  # 같은 폴더에 있다고 가정
BASE_MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
NUM_LABELS = 3  # 0=찬성, 1=반대, 2=중립
MAX_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 5e-5


# ===========================
# 데이터셋 클래스 정의
# ===========================
class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.labels = list(labels)
        self.max_len = max_len

        # 미리 토크나이즈해서 저장
        self.encodings = self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def main():
        # ===========================
    # 1. 데이터 로드
    # ===========================
    df = pd.read_csv(CSV_PATH)

    # 혹시 모를 공백/이상행 제거
    df = df.dropna(subset=["text", "label"])

    # label 컬럼을 숫자로 강제 변환 (문자 → NaN 처리)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # 숫자로 바꾸지 못한 행(label이 NaN인 행) 제거
    df = df.dropna(subset=["label"])

    # 다시 int로 캐스팅
    df["label"] = df["label"].astype(int)

    print(df["label"].value_counts())  # 라벨 분포 한 번 확인용 (원하면 지워도 됨)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # train / val 나누기
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42
    )

    print(f"train size = {len(train_texts)}, val size = {len(val_texts)}")

    # ===========================
    # 2. 토크나이저 & 데이터셋 생성
    # ===========================
    tokenizer = ElectraTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_dataset = StanceDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = StanceDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===========================
    # 3. 모델 준비
    # ===========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    model = ElectraForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ===========================
    # 4. 학습 루프
    # ===========================
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ===========================
        # 5. 검증 루프
        # ===========================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] train_loss = {avg_train_loss:.4f}, val_acc = {val_acc:.4f}")

        # ===========================
    # 6. 모델 저장
    # ===========================
    import os

    save_dir = "./stance_model"
    os.makedirs(save_dir, exist_ok=True)

    # GPU를 안 쓰고 있긴 하지만, 혹시 모를 이슈 방지를 위해 CPU로 옮겨두자
    model.to("cpu")

    # safetensors 대신 일반 PyTorch 포맷으로 저장
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    print(f"모델 저장 완료: {save_dir}")


if __name__ == "__main__":
    main()
