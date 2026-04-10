from fastapi import FastAPI
import torch
import pickle
import os

from model.ncf import NCF
from utils.preprocess import load_data

app = FastAPI()

# Load data info
df, NUM_USERS, NUM_ITEMS, _, _ = load_data("data/interactions.csv")

# Load mappings
with open("mappings.pkl", "rb") as f:
    user2id, item2id = pickle.load(f)

# Device
device = torch.device("cpu")

# Load model
model = NCF(NUM_USERS, NUM_ITEMS).to(device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pth")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):

    if user_id not in user2id:
        return {"message": "New user - no recommendations available"}

    user_idx = user2id[user_id]

    user_tensor = torch.tensor([user_idx] * NUM_ITEMS)
    item_tensor = torch.arange(NUM_ITEMS)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor).numpy()

    top_items = scores.argsort()[::-1][:k]

    return {"recommended_item_ids": top_items.tolist()}