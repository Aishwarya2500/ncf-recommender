# Neural Collaborative Filtering Recommender System

## 🚀 Overview
This project implements a deep learning-based recommendation system using Neural Collaborative Filtering (NCF) with PyTorch.

It learns user-item interactions and generates personalized recommendations.

---

## 🧠 Features
- User & Item Embeddings
- Negative Sampling
- Mini-batch Training (PyTorch DataLoader)
- Top-K Recommendation API (FastAPI)
- Evaluation Metrics (Precision@K, Recall@K)

---

## 🏗️ Architecture
Offline Training:
- Data preprocessing (NumPy, Pandas)
- Negative sampling
- Model training (PyTorch)

Online Inference:
- FastAPI service
- Returns top-K recommendations

---

## 🛠️ Tech Stack
- Python
- NumPy, Pandas
- PyTorch
- FastAPI

---

## ▶️ How to Run

### 1. Install dependencies

### 2. Train model

### 3. Run API

---

## 📊 Future Improvements
- Add real dataset (MovieLens)
- Improve ranking loss
- Deploy using Docker
