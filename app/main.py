from fastapi import FastAPI
import torch
from model.ncf import NCF

app = FastAPI()

NUM_USERS = 1000
NUM_ITEMS = 1000

model = NCF(NUM_USERS, NUM_ITEMS)

try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
except:
    print("Model not found. Please train first.")

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
