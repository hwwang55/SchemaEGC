import os
import json


def read_instances(node2idx):
    data = []
    for file_name in os.listdir('../data/pandemic'):
        matched_nodes = json.load(open('../data/pandemic/' + file_name))
        matched_ids = [node2idx[i] for i in matched_nodes]
        data.append(matched_ids)

    train_data = data[: int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test_data = data[int(0.89 * len(data)):]

    return train_data, val_data, test_data
