import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ========================
# 1. LOAD DATASET
# ========================

train_df = pd.read_csv("data/train.csv", header=None)
test_df = pd.read_csv("data/test.csv", header=None)

# ========================
# 🔥 CLEAN DATA
# ========================

train_df = train_df.dropna()
test_df = test_df.dropna()

train_df[0] = pd.to_numeric(train_df[0], errors="coerce") - 1
test_df[0] = pd.to_numeric(test_df[0], errors="coerce") - 1

train_df = train_df.dropna(subset=[0])
test_df = test_df.dropna(subset=[0])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ========================
# 🔥 REDUCE DATASET (FAST TRAINING)
# ========================

train_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)
test_df = test_df.sample(n=1000, random_state=42).reset_index(drop=True)

# ========================
# TEXT PREPARATION
# ========================

train_df["text"] = train_df[1].astype(str) + " " + train_df[2].astype(str)
test_df["text"] = test_df[1].astype(str) + " " + test_df[2].astype(str)

train_texts = train_df["text"].tolist()
test_texts = test_df["text"].tolist()

train_labels = train_df[0].astype(int).tolist()
test_labels = test_df[0].astype(int).tolist()

# ========================
# 2. TOKENIZER
# ========================

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# ========================
# 3. DATASET CLASS
# ========================

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# ========================
# 4. MODEL
# ========================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)

# ========================
# 5. METRICS
# ========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# ========================
# 6. TRAINING ARGS (FAST VERSION)
# ========================

training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available()  # ⚡ speed boost if GPU available
)

# ========================
# 7. TRAINER
# ========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ========================
# 8. TRAIN
# ========================

trainer.train()

# ========================
# 9. SAVE MODEL
# ========================

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ Training completed successfully!")