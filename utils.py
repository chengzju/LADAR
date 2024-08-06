import torch
import numpy as np
import os
import json


def load_group(dataset, group_tree=0):
    if dataset == 'Wiki-500K':
        return np.load(f'./data/Wiki-500K/label_group{group_tree}.npy', allow_pickle=True)
    elif dataset == 'Amazon-670K':
        return np.load(f'./data/Amazon-670K/label_group{group_tree}.npy', allow_pickle=True)

def get_label2group(group_y, label_num, label2group_path):
    if os.path.exists(label2group_path):
        label2group = json.load(open(label2group_path))
        return label2group
    label2group = [0] * label_num
    for i, one_group_ids in enumerate(group_y):
        for ids in one_group_ids:
            label2group[ids] = i
    print('label2group finish')
    json.dump(label2group, open(label2group_path, 'w'), indent=2, ensure_ascii=False)
    return label2group