import pandas as pd
import random

def load_data(path):
    df = pd.read_csv(path)

    # Fix column names
    df = df.rename(columns={
        "user_id": "user",
        "item_id": "item",
        "interaction": "label"
    })

    # Create mappings
    user_ids = df['user'].unique()
    item_ids = df['item'].unique()

    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {i: j for j, i in enumerate(item_ids)}

    df['user'] = df['user'].map(user2id)
    df['item'] = df['item'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)

    return df, num_users, num_items, user2id, item2id


def negative_sampling(df, num_items, num_negatives=2):
    user_item_set = set(zip(df['user'], df['item']))
    rows = []

    for (user, item) in user_item_set:
        rows.append((user, item, 1))

        for _ in range(num_negatives):
            neg_item = random.randint(0, num_items - 1)

            while (user, neg_item) in user_item_set:
                neg_item = random.randint(0, num_items - 1)

            rows.append((user, neg_item, 0))

    return pd.DataFrame(rows, columns=['user', 'item', 'label'])