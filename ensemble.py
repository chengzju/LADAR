import argparse
import random
import os
import json
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import numpy as np
from datasets.XMTCDataset import *
from models.Model import convert_weights
from apex import amp
from networks.modules import *
from metric import *
from tqdm import tqdm
from collections import defaultdict
from models.MADModel import *

def label2list(label):
    outputs = [[] for _ in range(label.shape[0])]
    x,y = np.where(label==1)
    for xx,yy in zip(x,y):
        outputs[xx].append(yy)
    return outputs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path for the data folders')
    parser.add_argument('--test_file_path')
    parser.add_argument('--ensemble_file_name', type=str)
    parser.add_argument('--ps_metric', action='store_true')
    args = parser.parse_args()

    model_file_list = [args.ensemble_file_name.format(idx) for idx in [1, 2, 3]]
    ddp_datasets = ['AmazonCat-13K', 'Amazon-670K', 'Wiki-500K']
    dataset = args.data_dir.split('/')[-1]

    label_freq_path = os.path.join(args.data_dir, 'label_freq.json')
    label_freq = json.load(open(label_freq_path))
    label_freq_desc = sorted(label_freq.items(), key=lambda x: x[1], reverse=True)
    label_index = [x[0] for x in label_freq_desc]
    labels = label_index
    label2id = {j: i for i, j in enumerate(labels)}

    train_labels = []
    if args.ps_metric:
        train_file_name = 'total_clf_train_data.json'
        train_data_path = os.path.join(args.data_dir, train_file_name)
        train_data = json.load(open(train_data_path, 'r'))
        train_labels = [[label2id[x] for x in item['labels']] for item in train_data]

    en_labels, en_scores = [], []
    ensemble_labels, ensemble_scores = [], []
    y_true_out = None

    for file in model_file_list:

        save_path = os.path.join(args.test_file_path, file)
        print(file, os.path.exists(save_path))
        data = np.load(os.path.join(save_path, 'outputs.npz'), allow_pickle=True)
        y_true = data['y_true']
        if dataset in ddp_datasets:
            y_true_list = y_true
        else:
            y_true_list = label2list(y_true)
        y_pred = data['y_pred']
        y_prob = data['y_prob']
        en_labels.append(y_pred)
        en_scores.append(y_prob)
        if y_true_out is None:
            y_true_out = y_true_list

        total_str = ''
        p1, p3, p5, n1, n3, n5 = base1_metric(y_true_list, y_pred, np.arange(len(labels)))
        log_str = '\n' + '\t'.join(['p1', 'p3', 'p5', 'n1', 'n3', 'n5'])
        total_str += log_str
        log_str = '\n' + '\t'.join(['%.6f'] * 6)
        log_str = log_str % (p1, p3, p5, n1, n3, n5)
        total_str += log_str
        if args.ps_metric:
            psp1, psp3, psp5, psn1, psn3, psn5 = ps_metric_pred(y_true_list, y_pred, np.arange(len(label2id)),
                                                                train_labels)
            log_str = '\n' + '\t'.join(['psp1', 'psp3', 'psp5', 'psn1', 'psn3', 'psn5'])
            total_str += log_str
            log_str = '\n' + '\t'.join(['%.6f'] * 6)
            log_str = log_str % (psp1, psp3, psp5, psn1, psn3, psn5)
            total_str += log_str
        print(total_str)

    for i in tqdm(range(len(en_labels[0]))):
        s = defaultdict(float)
        for j in range(len(en_labels[0][i])):
            for k in range(len(model_file_list)):
                s[en_labels[k][i][j]] = max(en_scores[k][i][j], s[en_labels[k][i][j]])
        s = sorted(s.items(), key=lambda x: x[1], reverse=True)
        ensemble_labels.append([x[0] for x in s[:len(en_labels[0][i])]])
        ensemble_scores.append([x[1] for x in s[:len(en_labels[0][i])]])
    ensemble_labels = np.array(ensemble_labels)
    p1, p3, p5, n1, n3, n5 = base1_metric(y_true_out, ensemble_labels, np.arange(len(labels)))
    final_str = ''
    log_str = '\n' + '\t'.join(['p1', 'p3', 'p5', 'n1', 'n3', 'n5'])
    final_str+=log_str
    log_str = '\n' + '\t'.join(['%.6f'] * 6)
    log_str = log_str % (p1, p3, p5, n1, n3, n5)
    final_str += log_str
    if args.ps_metric:
        psp1, psp3, psp5, psn1, psn3, psn5 = ps_metric_pred(y_true_out, ensemble_labels, np.arange(len(label2id)),
                                                            train_labels)
        log_str = '\n' + '\t'.join(['psp1', 'psp3', 'psp5', 'psn1', 'psn3', 'psn5'])
        final_str += log_str
        log_str = '\n' + '\t'.join(['%.6f'] * 6)
        log_str = log_str % (psp1, psp3, psp5, psn1, psn3, psn5)
        final_str += log_str
    print(final_str)


if __name__ == '__main__':
    main()