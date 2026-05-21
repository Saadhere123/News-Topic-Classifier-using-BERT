import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "saadhere4705/news-topic-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

st.title("News Topic Classifier 📰")

text = st.text_area("Enter news text:")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    st.success(f"Predicted Topic: {labels[prediction]}")
