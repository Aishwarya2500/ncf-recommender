import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle

from sklearn.model_selection import train_test_split

from model.ncf import NCF
from model.dataset import InteractionDataset
from utils.preprocess import load_data, negative_sampling

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df, num_users, num_items, user2id, item2id = load_data("data/interactions.csv")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Negative sampling
train_df = negative_sampling(train_df, num_items)

# Dataset
dataset = InteractionDataset(train_df)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Model
model = NCF(num_users, num_items).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
for epoch in range(5):
    total_loss = 0

    for user, item, label in loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

        pred = model(user, item)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")

# Save mappings
with open("mappings.pkl", "wb") as f:
    pickle.dump((user2id, item2id), f)

print("Model and mappings saved!")

# Evaluation
def precision_at_k(model, test_df, num_items, k=5):
    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for user in test_df['user'].unique():
            user_tensor = torch.tensor([user] * num_items).to(device)
            item_tensor = torch.arange(num_items).to(device)

            scores = model(user_tensor, item_tensor).cpu().numpy()
            top_k = scores.argsort()[::-1][:k]

            actual = test_df[test_df['user'] == user]['item'].values

            hits += len(set(top_k) & set(actual))
            total += k

    return hits / total

print("Precision@5:", precision_at_k(model, test_df, num_items))