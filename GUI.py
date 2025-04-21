import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import re
import numpy as np
import string
import spacy
import nltk
from better_profanity import profanity
profanity.load_censor_words()

# === Load models ===
tokenizer_deberta = AutoTokenizer.from_pretrained("deberta")
model_deberta = AutoModelForSequenceClassification.from_pretrained("deberta")
label_encoder = joblib.load("deberta/label_encoder.pkl")
rf_model = joblib.load("rf_model.pkl")
rf_label_encoder = joblib.load("rf_label_encoder.pkl")

nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
stopwords = set(nltk.corpus.stopwords.words("english"))

def count_profanity(text):
    return sum(1 for word in text.split() if profanity.contains_profanity(word))

def extract_features(text):
    blob = TextBlob(text)
    vader_scores = vader.polarity_scores(text)
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    capital_words = [w for w in words if w.isupper() and len(w) > 1]
    exclamations = text.count("!")
    questions = text.count("?")
    mentions = text.count("@")
    hashtags = text.count("#")
    emojis = emoji.emoji_count(text)
    badword_hits = count_profanity(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count if char_count else 0
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / word_count if word_count else 0
    repeated_chars = len(re.findall(r"(.)\1{2,}", text))
    
    # POS counts
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    noun_count = pos_counts.get(nlp.vocab.strings["NOUN"], 0)
    verb_count = pos_counts.get(nlp.vocab.strings["VERB"], 0)
    adj_count = pos_counts.get(nlp.vocab.strings["ADJ"], 0)
    adv_count = pos_counts.get(nlp.vocab.strings["ADV"], 0)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": len(set(words)),
        "punctuation_count": punctuation_count,
        "capital_word_count": len(capital_words),
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamations,
        "question_count": questions,
        "mention_count": mentions,
        "hashtag_count": hashtags,
        "emoji_count": emojis,
        "badword_count": badword_hits,
        "avg_word_length": avg_word_len,
        "stopword_ratio": stopword_ratio,
        "repeated_char_sequences": repeated_chars,
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],
        "vader_pos": vader_scores["pos"],
        "vader_compound": vader_scores["compound"],
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adj_count": adj_count,
        "adv_count": adv_count,
    }

def predict_text():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter text for classification.")
        return

    model_choice = model_var.get()

    try:
        if model_choice == "DeBERTa":
            inputs = tokenizer_deberta(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model_deberta(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]

        elif model_choice == "Random Forest":
            new_row = pd.Series(extract_features(text)).to_frame().T
            new_row.fillna(0, inplace=True)  # in case any features are missing
            # Predict
            label = rf_label_encoder.inverse_transform(rf_model.predict(new_row))[0]

        else:
            messagebox.showerror("Invalid Selection", "Please select a model.")
            return

        result_label.config(text=f"Predicted Class: {label}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# === GUI Layout ===
root = tk.Tk()
root.title("Cyberbullying Classifier")
root.geometry("500x400")

tk.Label(root, text="Enter text to classify:", font=("Arial", 12)).pack(pady=5)
input_box = tk.Text(root, height=6, width=60, wrap="word")
input_box.pack(pady=5)

tk.Label(root, text="Choose Model:", font=("Arial", 12)).pack(pady=5)
model_var = tk.StringVar(value="DeBERTa")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["DeBERTa", "Random Forest"], state="readonly")
model_dropdown.pack()

tk.Button(root, text="Predict", command=predict_text, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()
