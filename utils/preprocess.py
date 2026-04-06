import numpy as np
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()

    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}

    df['user'] = df['user_id'].map(user_map)
    df['item'] = df['item_id'].map(item_map)

    return df, len(user_ids), len(item_ids), item_map


def negative_sampling(df, num_items, num_neg=2):
    data = []

    user_item_set = set(zip(df['user'], df['item']))

    for (u, i) in user_item_set:
        data.append((u, i, 1))  # positive

        for _ in range(num_neg):
            j = np.random.randint(0, num_items)
            while (u, j) in user_item_set:
                j = np.random.randint(0, num_items)
            data.append((u, j, 0))  # negative

    return pd.DataFrame(data, columns=['user', 'item', 'label'])