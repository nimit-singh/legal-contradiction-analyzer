# ⚖️ Legal Document Contradiction Detection using Transformer-Based NLP

## 📌 Overview

This project focuses on detecting contradictions in legal documents using Transformer-based Natural Language Processing (NLP) and Deep Learning techniques.

The system analyzes semantic relationships between legal sentence pairs and classifies them into:

- Contradiction
- Entailment
- Neutral

The project uses pretrained transformer embeddings with HuggingFace Transformers and TensorFlow for contextual understanding and semantic classification.

---

# 🚀 Features

- Transformer-based legal contradiction detection
- Semantic relationship classification
- Deep learning classification model
- Real-time prediction system
- Streamlit web application
- Confusion matrix visualization
- Manual prediction interface

---

# 🧠 Technologies Used

- Python
- TensorFlow
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

# 📂 Dataset

The project uses the ANLI (Adversarial Natural Language Inference) dataset.

Dataset contains:

- Sentence 1 (Premise)
- Sentence 2 (Hypothesis)
- Relationship Label

Labels:

- Contradiction
- Entailment
- Neutral

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Transformer Tokenization
   ↓
Input IDs + Attention Masks
   ↓
Transformer Model (MiniLM/BERT)
   ↓
Dense Neural Network
   ↓
Softmax Classification
   ↓
Prediction Output
```

---

# 🏗️ Model Architecture

The system uses:

- Transformer Embedding Layer
- CLS Token Extraction
- Dropout Layers
- Dense Neural Network
- Softmax Output Layer

Model Used:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

# 📊 Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🌐 Streamlit Application

The project includes a Streamlit web application for real-time contradiction prediction.

Features:

- Manual prediction
- Real-time output
- User-friendly interface
- Legal sentence analysis

---

# 📸 Example Predictions

### Contradiction

Sentence 1:
The contract is valid until 2026.

Sentence 2:
The contract expired in 2023.

Output:
Contradiction

---

### Entailment

Sentence 1:
The judge approved the merger.

Sentence 2:
The merger received approval.

Output:
Entailment

---

### Neutral

Sentence 1:
The company signed the agreement.

Sentence 2:
The agreement contains tax clauses.

Output:
Neutral

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/legal-contradiction-detection.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- LegalBERT integration
- Full document contradiction analysis
- Larger legal datasets
- Multilingual support
- Advanced interpretation systems

---

# 📄 Conclusion

This project demonstrates how Transformer-based NLP models can effectively detect contradictions and semantic relationships in legal documents using deep learning techniques.

---
