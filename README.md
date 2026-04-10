# 🚀 Neural Collaborative Filtering Recommender System

## 📌 Overview
This project implements a **Neural Collaborative Filtering (NCF)** based recommendation system using PyTorch.

It learns user-item interaction patterns from implicit feedback data and generates **personalized top-K recommendations** via a FastAPI service.

---

## 🧠 Problem Statement
Traditional recommendation systems rely on similarity or heuristics.  
This system uses **deep learning** to learn complex user item relationships.

---

## 🏗️ Architecture

### 🔹 Offline Pipeline (Training)
1. Load interaction data
2. Apply **ID mapping** (convert raw IDs → embeddings)
3. Perform **negative sampling**
4. Train NCF model using PyTorch
5. Evaluate using **Precision@K**
6. Save:
   - `model.pth` (weights)
   - `mappings.pkl` (ID mappings)

---

### 🔹 Online Pipeline (Inference)
1. FastAPI loads trained model
2. Accepts `user_id`
3. Maps to embedding index
4. Scores all items (vectorized)
5. Returns **Top-K recommendations**

---

## 🧩 Model Architecture

- User Embedding
- Item Embedding
- Concatenation
- Multi-Layer Perceptron (MLP)

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Pandas, NumPy
- FastAPI
---

### 4️⃣ Train Model
- Scikit-learn

---

## 📂 Project Structure
---

## ▶️ How to Run

### 1️⃣ Clone Repository

---

### 2️⃣ Create Virtual Environment

---

### 3️⃣ Install Dependencies

---

### 4️⃣ Train Model

---
### 5️⃣ Run API
---

### 6️⃣ Test API
Open:

---

## 📊 Evaluation

- Metric: **Precision@K**
- Evaluates how many recommended items are relevant

---

## ⚡ Key Features

✅ Neural Collaborative Filtering  
✅ Negative Sampling  
✅ Embedding based learning  
✅ Vectorized inference (fast)  
✅ FastAPI deployment  
✅ Precision evaluation  

---

## 🚧 Limitations

- Small dataset
- Cold start problem for new users
- No temporal/context features

---

## 🚀 Future Improvements

- Use real dataset (MovieLens)
- Add GMF + MLP hybrid (paper-based NCF)
- Deploy using Docker + AWS
- Add ranking loss (BPR Loss)

---

## 🧠 Key Learnings

- Handling implicit feedback
- Importance of negative sampling
- Embedding-based recommendation systems
- Serving ML models via APIs

---

## 👩‍💻 Author

Aishwarya Kottapalli


