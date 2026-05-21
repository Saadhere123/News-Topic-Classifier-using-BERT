import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ========================
# LOAD MODEL
# ========================

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# ========================
# LABELS
# ========================

labels = ["World", "Sports", "Business", "Sci/Tech"]

# ========================
# STREAMLIT UI
# ========================

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Topic Classifier (BERT)")
st.write("Enter a news headline and predict its category.")

# ========================
# INPUT
# ========================

text = st.text_area(
    "Enter News Headline",
    placeholder="Example: Apple launches new AI-powered iPhone..."
)

# ========================
# PREDICTION
# ========================

if st.button("Predict Topic"):

    if text.strip() == "":
        st.warning("Please enter a headline.")
    else:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # ========================
        # RESULT
        # ========================

        st.success(f"Predicted Category: {labels[pred]}")

        st.subheader("Prediction Confidence")

        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[0][i].item() * 100:.2f}%")