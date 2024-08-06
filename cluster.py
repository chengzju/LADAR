# cluster from AttentionXML

import os
from tqdm import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
import argparse


def get_labels(label_save_path, ori_label_file):
    if os.path.exists(label_save_path):
        print('label npy exist')
        labels = np.load(label_save_path, allow_pickle=True)
    else:
        print('label npy not exist')
        labels = None
        if ori_label_file is not None:
            with open(ori_label_file) as fp:
                labels = np.asarray([[label for label in line.split()]
                                     for line in tqdm(fp, desc='Converting labels', leave=False)])
        np.save(label_save_path, labels)
    return labels


def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        print('mlb exist')
        return joblib.load(mlb_path)
    print('mlb exist')
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    sparse_labels = [i.replace('\n', '').split() for i in open(label_file)]
    return normalize(sparse_x), np.array(sparse_labels)


def build_tree_by_level(sparse_data_x, sparse_data_y, eps: float, max_leaf: int, levels: list, groups_path, label_save_path, mlb_path):
    train_labels = get_labels(label_save_path, sparse_data_y)
    mlb = get_mlb(mlb_path, train_labels)
    print('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    sparse_y = mlb.transform(sparse_labels)
    print('Getting Labels Feature')
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))
    print(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    for i in range(len(levels)-1, -1, -1):
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            print(F'{groups_path}-Level-{i}.npy')
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break
    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    print(levels)
    while q:
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'in: Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            print(F'out: Finish Clustering {len(labels_list)}')
        next_q = []

        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
            else:
                np.save(F'{groups_path}-last.npy', np.asarray(labels_list))
        q = next_q
    print('Finish Clustering')
    return mlb


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--id', type=str, default='0')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == '670k':
        label_save_path = './data/Amazon-670K/train_labels.npy'
        mlb_path = './data/Amazon-670K/labels_binarizer'
        mlb = build_tree_by_level('./data/Amazon-670K/train_v1.txt',
                                  './data/Amazon-670K/train_labels.txt',
                                  1e-4, 100, [], './data/Amazon-670K/label_group' + args.id,
                                  label_save_path, mlb_path
                                  )
        groups = np.load(f'./data/Amazon-670K/label_group{args.id}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'./data/Amazon-670K/label_group{args.id}.npy', np.array(new_group))
    elif dataset == '500k':
        label_save_path = './data/Wiki-500K/train_labels.npy'
        mlb_path = './data/Wiki-500K/labels_binarizer'
        mlb = build_tree_by_level('./data/Wiki-500K/train.txt',
                                  './data/Wiki-500K/train_labels.txt',
                                  1e-4, 100, [], './data/Wiki-500K/groups',
                                  label_save_path, mlb_path)
        groups = np.load(f'./data/Wiki-500K/groups-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'./data/Wiki-500K/label_group{args.id}.npy', np.array(new_group))


if __name__ == '__main__':
    main()
