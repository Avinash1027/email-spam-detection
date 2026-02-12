# 📧 Intelligent Email Spam Detection System

A production-ready NLP-powered email classification system that detects spam messages with high confidence using TF-IDF feature engineering and a PyTorch neural network model.

Designed as a deployable REST API, the system performs real-time email classification with structured confidence scoring and scalable architecture.

---

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for email spam detection, including:

- Text preprocessing using NLP techniques (NLTK)
- TF-IDF vectorization with 5,000 engineered features
- Neural network classification using PyTorch
- Real-time prediction API built with FastAPI
- Structured confidence scoring for decision support
- Modular architecture for easy retraining and deployment

### Deployment Use Cases

- Email filtering systems
- Security pipelines
- Automated moderation tools
- Backend spam detection services

---

## 🏗️ Architecture Overview

### 1️⃣ Text Preprocessing Pipeline

**Purpose:** Normalize and clean email text before feature extraction.

**Techniques Used:**
- Lowercasing
- Regex-based noise removal
- Stopword filtering (NLTK)
- Lemmatization (WordNetLemmatizer)
- Token normalization

**Example Transformation**

| | Text |
|---|---|
| **Input** | `"URGENT: Your corporate account has been compromised. Verify credentials immediately!!!"` |
| **Output** | `"urgent corporate account compromised verify credential immediately"` |

This ensures vocabulary consistency between training and inference.

---

### 2️⃣ Feature Engineering (TF-IDF)

**Purpose:** Convert cleaned text into numerical representations.

**Configuration:**
```python
TfidfVectorizer(max_features=5000)
```

- Trained on preprocessed email dataset
- Vocabulary fixed and persisted using `joblib`

**Key Benefits:**
- Captures word importance
- Reduces noise
- Handles sparse feature space efficiently
- Produces consistent 5000-dimensional feature vectors

---

### 3️⃣ Neural Network Model (PyTorch)

**Purpose:** Classify emails into spam or not spam.

**Model Characteristics:**
- Fully connected feedforward architecture
- Input dimension: 5000 (TF-IDF features)
- Sigmoid output activation
- Binary classification (Spam vs Not Spam)
- Inference optimized with `torch.inference_mode()`

**Prediction Logic:**
```python
confidence = sigmoid(output_logit)
if confidence > 0.7:
    prediction = "spam"
else:
    prediction = "not spam"
```

> The threshold can be tuned based on business requirements.

---

### 4️⃣ FastAPI Deployment Layer

**Purpose:** Serve predictions through a scalable REST API.

**Features:**
- `/predict` endpoint
- JSON-based request/response
- Pydantic schema validation
- Stateless inference
- Lightweight ASGI server via Uvicorn

**Example Request:**
```json
{
  "text": "Action Required: Your IT helpdesk password expires today. Click here to reset your credentials and avoid losing access."
}
```

**Example Response:**
```json
{
  "text": "action required it helpdesk password expire today click reset credential avoid lose access",
  "confidence_score": 0.94,
  "prediction": "spam"
}
```

---

## 🔄 End-to-End Flow

```
Email Input
    ↓
Text Preprocessing (NLTK + Regex + Lemmatization)
    ↓
TF-IDF Transformation (5000 features)
    ↓
PyTorch Neural Network
    ↓
Sigmoid Confidence Score
    ↓
JSON API Response
```

---

## 🛠️ Tech Stack

### Core ML & NLP
| Library | Purpose |
|---|---|
| PyTorch `2.10.0` | Neural network model |
| Scikit-learn | TF-IDF vectorization |
| NLTK | Stopwords & lemmatization |
| Regex | Custom text cleaning |

### API & Backend
| Library | Purpose |
|---|---|
| FastAPI `0.128.8` | REST API framework |
| Pydantic `2.12.5` | Schema validation |
| Uvicorn `0.40.0` | ASGI server |

### Data & Utilities
| Library | Purpose |
|---|---|
| Pandas | Data handling |
| NumPy | Numerical operations |
| Joblib | Model persistence |

---

## 📁 Project Structure

```
email-spam-detection/
├── src/
│   ├── model/
│   │   ├── model_definition.py
│   │   ├── model.pth
│   │   └── tfidf.pkl
│   ├── schema/
│   │   └── predict.py
│   └── api/
│       └── routes.py
├── notebooks/
│   └── training_pipeline.ipynb
├── requirements.txt
└── README.md
```

---
