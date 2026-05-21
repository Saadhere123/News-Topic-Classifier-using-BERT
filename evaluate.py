import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================
# LOAD MODEL
# ========================

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

model.eval()

# ========================
# LOAD TEST DATA
# ========================

test_df = pd.read_csv("data/test.csv", header=None)

# Clean dataset
test_df = test_df.dropna()

# Prepare text
test_df["text"] = test_df[1].astype(str) + " " + test_df[2].astype(str)

# Labels (convert 1-4 → 0-3)
test_df[0] = pd.to_numeric(test_df[0], errors="coerce") - 1
test_df = test_df.dropna(subset=[0])

texts = test_df["text"].tolist()

labels = test_df[0].astype(int).tolist()

# ========================
# PREDICTION (BATCH MODE)
# ========================

predictions = []

BATCH_SIZE = 32

for i in range(0, len(texts), BATCH_SIZE):

    batch_texts = texts[i:i+BATCH_SIZE]

    encodings = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encodings)

    preds = torch.argmax(outputs.logits, dim=1)

    predictions.extend(preds.numpy())

# ========================
# METRICS
# ========================

acc = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")

print("\n✅ Evaluation Results")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")