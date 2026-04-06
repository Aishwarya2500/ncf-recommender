import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model.ncf import NCF
from model.dataset import InteractionDataset
from utils.preprocess import load_data, negative_sampling

# Load data
df, num_users, num_items, _ = load_data("data/interactions.csv")

# Create training data
train_df = negative_sampling(df, num_items)

dataset = InteractionDataset(train_df)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Model
model = NCF(num_users, num_items)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    total_loss = 0

    for user, item, label in loader:
        pred = model(user, item)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")