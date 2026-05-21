import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from HuggingFace
model_name = "saadhere4705/news-topic-bert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

st.title("News Topic Classifier 📰")

text = st.text_area("Enter news text:")

if st.button("Predict"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    st.write("Predicted Class:", prediction)
