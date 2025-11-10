# Toxic Comment Detection using DistilBERT (Jigsaw Dataset)

This repository implements a **multi-label toxic comment classification** model trained on the **Jigsaw Toxic Comments Dataset** from Kaggle.  
It uses **DistilBERT**, a lightweight Transformer model, fine-tuned using the **Transformers** and **TensorFlow** libraries.

---

## 📘 Overview

The project demonstrates **text classification** for online toxicity detection across multiple categories.  
Each input comment can belong to one or more labels (e.g., toxic, obscene, insult, threat, etc.).  
The model leverages **transfer learning** with DistilBERT for efficient fine-tuning on the Jigsaw dataset.

---

## 🧠 Problem Statement

Given a comment text, predict whether it belongs to one or more of the following classes:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

---

## ⚙️ Tech Stack

- **Python 3.10+**  
- **TensorFlow 2.x**  
- **Transformers (Hugging Face)**  
- **Pandas, NumPy, Matplotlib**  
- **scikit-learn**  
- **Jupyter Notebook**

---

## 📂 Project Structure

```
toxic-comment-detection/
│
├── main_run.ipynb           # Training, evaluation, and prediction workflow
├── fine_tune.ipynb          # Fine-tuning DistilBERT on the Jigsaw dataset
├── data/                    # Dataset folder (Jigsaw Toxic Comments - from Kaggle)
├── models/                  # Saved model checkpoints
└── README.md
```

---

## 🚀 Features

- Fine-tuned **DistilBERT** for multi-label text classification  
- TensorFlow-based implementation  
- Handles class imbalance  
- Evaluation with accuracy, precision, recall, and F1-score 
- Easily extendable to other text toxicity datasets  

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ramcharan2905/toxic-comment-detection.git
   cd toxic-comment-detection
   ```

2. **Set up the environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run notebooks**
   ```bash
   jupyter notebook
   ```
   Open `main_run.ipynb` to train or evaluate the model.

---

## 📊 Results

- Model: **DistilBERT (base-uncased)**  
- Framework: **TensorFlow**  
- Task: **Multi-label Toxic Comment Detection**  
- Dataset: **Jigsaw Toxic Comments (Kaggle)**  

The model achieved strong multi-label classification performance with improved efficiency due to DistilBERT’s reduced size compared to BERT.

---

## 📈 Learning Purpose

This project aims to help practitioners understand:
- Fine-tuning Transformer models for NLP tasks  
- Handling multi-label binary classification in text  
- Model evaluation using appropriate multi-label metrics  

---

## 👤 Author

**Gudala Geeta Ramcharan**  
Machine Learning Enthusiast | NLP Researcher  
GitHub: [Ramcharan2905](https://github.com/Ramcharan2905)

---

## ⭐ Acknowledgments

- **Jigsaw Toxic Comment Dataset** — Kaggle  
- **Hugging Face Transformers** — model and tokenizer tools  
- **TensorFlow** — deep learning framework
