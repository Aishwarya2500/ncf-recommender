from fastapi import FastAPI
import torch
from model.ncf import NCF

app = FastAPI()

NUM_USERS = 5
NUM_ITEMS = 7

import os

model = NCF(NUM_USERS, NUM_ITEMS)

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "model.pth")

    print("Loading model from:", model_path)  # debug line

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully")

except Exception as e:
    print("Error loading model:", e)
@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    scores = []

    for item_id in range(NUM_ITEMS):
        user = torch.tensor([user_id])
        item = torch.tensor([item_id])

        score = model(user, item).item()
        scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return {"recommendations": scores[:10]}