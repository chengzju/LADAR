from torch.utils.data import Dataset
import numpy as np
import torch


class XmtcDataset(Dataset):
    def __init__(self, args, data, tokenizer, label2id, maxlength=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.maxlength = maxlength
        self.all_labels_num = len(label2id)
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        true_labels = list(set(item['labels']) & set(self.label2id.keys()) )
        labels = np.zeros(self.all_labels_num)
        true_labels_id = [self.label2id[x] for x in true_labels]
        labels[true_labels_id] = 1
        inputs_list = [labels]

        input_ids = item['input_ids']
        input_ids_entire = [self.tokenizer.cls_token_id] + input_ids[:self.maxlength - 2] + [
            self.tokenizer.sep_token_id]
        input_type_ids_entire = [0] * len(input_ids_entire)
        input_mask_entire = [1] * len(input_ids_entire)
        # padding
        input_ids_entire += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids_entire))
        input_type_ids_entire += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids_entire))
        input_mask_entire += [0] * (self.maxlength - len(input_mask_entire))
        # to tensor
        input_ids_entire = torch.tensor(input_ids_entire, dtype=torch.long)
        input_type_ids_entire = torch.tensor(input_type_ids_entire, dtype=torch.long)
        input_mask_entire = torch.tensor(input_mask_entire, dtype=torch.long)

        inputs_entire = [input_ids_entire, input_type_ids_entire, input_mask_entire]
        inputs_list.append(inputs_entire)


        input_ids_list_frag = []
        input_type_ids_list_frag = []
        input_mask_list_frag = []
        for i in range(self.args.num_frag):
            idx_s = i * self.args.len_frag
            idx_e = (i + 1) * self.args.len_frag
            one_input_ids = [self.tokenizer.cls_token_id] + input_ids[idx_s: idx_e] + [self.tokenizer.sep_token_id]
            one_input_type_ids = [0] * len(one_input_ids)
            one_input_mask = [1] * len(one_input_ids)
            # padding
            one_input_ids += [self.tokenizer.pad_token_id] * (self.args.len_frag + 2 - len(one_input_ids))
            one_input_type_ids += [self.tokenizer.pad_token_type_id] * (
                    self.args.len_frag + 2 - len(one_input_type_ids))
            one_input_mask += [0] * (self.args.len_frag + 2 - len(one_input_mask))

            input_ids_list_frag.append(one_input_ids)
            input_type_ids_list_frag.append(one_input_type_ids)
            input_mask_list_frag.append(one_input_mask)
        # to tensor
        input_ids_list_frag = torch.tensor(input_ids_list_frag, dtype=torch.long)
        input_type_ids_list_frag = torch.tensor(input_type_ids_list_frag, dtype=torch.long)
        input_mask_list_frag = torch.tensor(input_mask_list_frag, dtype=torch.long)

        inputs_frag = [input_ids_list_frag, input_type_ids_list_frag, input_mask_list_frag]
        inputs_list.append(inputs_frag)

        return inputs_list



class XmtcDataset_PLT(Dataset):
    def __init__(self, args, data, tokenizer, label2id, maxlength=512, mode='train', group_y=None, device=None):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.maxlength = maxlength
        self.all_labels_num = len(label2id)
        self.args = args
        self.device = device

        self.mode = mode

        self.candidates_num = args.candidates_num
        self.group_y = group_y
        if group_y is not None:
            self.group_y = []
            self.group_num = group_y.shape[0]
            self.map_label_2_group = np.empty(self.all_labels_num, dtype=np.long)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    if label in self.label2id.keys():
                        self.group_y[-1].append(self.label2id[label])
                self.map_label_2_group[self.group_y[-1]] = idx
                self.group_y[-1] = np.array(self.group_y[-1])
            self.group_y = np.array((self.group_y))

    def get_group_y(self):
        return self.group_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        true_labels = list(set(item['labels']) & set(self.label2id.keys()) )
        labels = np.zeros(self.all_labels_num)
        true_labels_id = [self.label2id[x] for x in true_labels]
        labels[true_labels_id] = 1

        if self.group_y is not None:
            true_group_labels_id = self.map_label_2_group[true_labels_id]
            group_labels = np.zeros(self.group_num)
            group_labels[true_group_labels_id] = 1
            if len(true_group_labels_id) > 0:
                candidates = np.concatenate(self.group_y[true_group_labels_id], axis=0)
            else:
                candidates = np.array([], dtype=np.long)
            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.all_labels_num, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)
            if self.mode == 'train':
                labels_list = [labels[candidates], group_labels, candidates]
            else:
                labels_list = [labels, group_labels, candidates]
        inputs_list = [labels_list]

        input_ids = item['input_ids']

        input_ids_entire = [self.tokenizer.cls_token_id] + input_ids[:self.maxlength - 2] + [
            self.tokenizer.sep_token_id]
        input_type_ids_entire = [0] * len(input_ids_entire)
        input_mask_entire = [1] * len(input_ids_entire)
        input_ids_entire += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids_entire))
        input_type_ids_entire += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids_entire))
        input_mask_entire += [0] * (self.maxlength - len(input_mask_entire))
        input_ids_entire = torch.tensor(input_ids_entire, dtype=torch.long)
        input_type_ids_entire = torch.tensor(input_type_ids_entire, dtype=torch.long)
        input_mask_entire = torch.tensor(input_mask_entire, dtype=torch.long)
        inputs_entire = [input_ids_entire, input_type_ids_entire, input_mask_entire]
        inputs_list.append(inputs_entire)

        input_ids_list_frag = []
        input_type_ids_list_frag = []
        input_mask_list_frag = []
        for i in range(self.args.num_frag):
            idx_s = i * self.args.len_frag
            idx_e = (i + 1) * self.args.len_frag
            one_input_ids = [self.tokenizer.cls_token_id] + input_ids[idx_s: idx_e] + [self.tokenizer.sep_token_id]
            one_input_type_ids = [0] * len(one_input_ids)
            one_input_mask = [1] * len(one_input_ids)
            one_input_ids += [self.tokenizer.pad_token_id] * (self.args.len_frag + 2 - len(one_input_ids))
            one_input_type_ids += [self.tokenizer.pad_token_type_id] * (
                    self.args.len_frag + 2 - len(one_input_type_ids))
            one_input_mask += [0] * (self.args.len_frag + 2 - len(one_input_mask))
            input_ids_list_frag.append(one_input_ids)
            input_type_ids_list_frag.append(one_input_type_ids)
            input_mask_list_frag.append(one_input_mask)
        input_ids_list_frag = torch.tensor(input_ids_list_frag, dtype=torch.long)
        input_type_ids_list_frag = torch.tensor(input_type_ids_list_frag, dtype=torch.long)
        input_mask_list_frag = torch.tensor(input_mask_list_frag, dtype=torch.long)
        inputs_frag = [input_ids_list_frag, input_type_ids_list_frag, input_mask_list_frag]
        inputs_list.append(inputs_frag)

        return inputs_list

