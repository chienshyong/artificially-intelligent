# artificially-intelligent
AI project 2025

---

# Cyberbullying Detection Toolkit

This repository contains all notebooks, scripts, and data used for training, evaluating, and deploying various machine learning models to detect and classify different types of cyberbullying in text data.

---

## 📁 Dataset Files

| File | Description |
|------|-------------|
| `cyberbullying.csv` | The original dataset containing labeled tweet texts. |
| `cleaned_cyberbullying.csv` | The dataset after preprocessing and cleaning, used across all model pipelines. |
| `cyberbullying_jigsaw_scored.csv` | Dataset enriched with scores from Google's Jigsaw Perspective API for toxicity detection. |
| `cyberbullying_tweets_enriched.csv` | Dataset with engineered features used specifically for training the Random Forest model. |

---

## 📒 Notebooks and Scripts

| File | Purpose |
|------|---------|
| `cleaning_data.ipynb` | Notebook for preprocessing raw tweet data. Outputs `cleaned_cyberbullying.csv`. |
| `data-visualization + ELECTRA.ipynb` | Generates plots used in the report and trains the ELECTRA model. |
| `deberta.ipynb` | Fine-tunes the DeBERTa model for multi-class classification of cyberbullying types. |
| `electra/` | Folder containing ELECTRA model configuration and outputs. |
| `randomForest.ipynb` | Extracts linguistic, sentiment, and syntactic features for training a Random Forest model. |
| `robertabase.ipynb` | Trains a RoBERTa-based model for cyberbullying classification. |
| `sentimental_analysis.ipynb` | Contains visualizations related to sentiment analysis, included in the final report. |
| `stacking.ipynb` | Combines DeBERTa, ELECTRA, and Random Forest outputs to train a logistic regression meta-model. |
| `jigsaw-api.ipynb` | Uses Google’s Jigsaw Perspective API to generate toxicity scores for tweets. |
| `unitary_toxicbert.ipynb` | (Optional) Experimentation with Unitary AI’s ToxicBERT model. |

---

## 🖥 GUI

| File | Description |
|------|-------------|
| `GUI.py` | Launches a user interface that allows users to input text and select a model (DeBERTa, ELECTRA, or Random Forest) for classification. Outputs predicted cyberbullying type. |

---

## 📦 Other

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies needed to run the project. |
| `.gitignore` | Git ignore rules for the project. |
| `README.md` | You’re reading it! |
| `demo.mkv` | Video demonstration of the GUI. |
| `50.021 AI Final Presentation.pdf` | Final Presentation slides. |
| `50.021 AI Project Final Report.pdf` | Final Report document. |

---

## 🧠 Models Used

- **DeBERTa**: For deep semantic classification  
- **ELECTRA**: Lightweight transformer model fine-tuned on the dataset  
- **RoBERTa**: Strong performance on ambiguous and nuanced language cases  
- **Random Forest**: Trained on handcrafted features for interpretability  
- **Stacking**: A logistic regression meta-model trained on outputs from DeBERTa, ELECTRA, and Random Forest  

---