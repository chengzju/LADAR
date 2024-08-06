import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class MulitAttention(nn.Module):
    def __init__(self, atten_head_num, hidden_size):
        super(MulitAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, atten_head_num)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs):
        out = inputs
        atten = self.attention(out).transpose(1,2)
        atten_sm = F.softmax(atten, -1)
        out = atten_sm @ inputs
        return out, atten


class MANetwork(nn.Module):
    def __init__(self, args, hidden_size, label_num, atten_num):
        super(MANetwork, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.droprate)
        self.feature_size = hidden_size
        self.atten_num = atten_num
        self.multi_atten = MulitAttention(self.atten_num, self.feature_size)
        self.clf = nn.Linear(self.feature_size, label_num)
        nn.init.xavier_uniform_(self.clf.weight)
        self.score_fc = nn.MaxPool1d(self.atten_num)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        out, atten = self.multi_atten(inputs)
        out = self.clf(out)
        s_out = out.transpose(1, 2)
        s_out = self.score_fc(s_out).squeeze(-1)
        out_dict = {
            'out':s_out
        }
        return out_dict

class MANetwork_PLT(nn.Module):
    def __init__(self, args, hidden_size, label_num, group_num, atten_num, group_y, label2group, device):
        super(MANetwork_PLT, self).__init__()
        self.args = args
        self.device = device
        self.group_y = group_y
        self.label2group = label2group
        self.candidates_topk = args.candidates_topk
        self.feature_size = hidden_size
        self.atten_num = atten_num
        self.group_num = group_num

        self.drop = nn.Dropout(args.droprate)
        self.multi_atten = MulitAttention(self.atten_num, self.feature_size)
        self.clf = nn.Linear(self.feature_size, group_num)
        nn.init.xavier_uniform_(self.clf.weight)
        self.score_fc = nn.MaxPool1d(self.atten_num)
        self.score_fc2 = nn.MaxPool1d(self.atten_num)

        self.encoder = nn.Linear(self.feature_size, args.hidden_size)
        self.embed = nn.Embedding(label_num, args.hidden_size)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, inputs, labels_list, is_training):
        labels, group_labels, candidates = labels_list
        inputs = self.drop(inputs)
        out, atten = self.multi_atten(inputs)

        group_logits = self.clf(out)
        c_out = group_logits.transpose(1, 2)
        c_out = self.score_fc(c_out).squeeze(-1)

        out_dict = {
            'c_out':c_out
        }

        if is_training:
            labels_bool = labels.to(dtype=torch.bool)
            target_candidates = torch.masked_select(candidates, labels_bool).detach().cpu()
            target_candidates_num = labels_bool.sum(dim=1).detach().cpu()
        _, candidates, group_candidates_scores = self.get_candidates(c_out,
                                                                     group_gd=group_labels if is_training else None,
                                                                     is_training=is_training)
        if is_training:
            bs = 0
            new_labels = []
            for i, n in enumerate(target_candidates_num.numpy()):
                be = bs + n
                c_t = set(target_candidates[bs: be].numpy())
                c_w_gd = candidates[i]
                new_labels.append(torch.tensor([1.0 if i in c_t else 0.0 for i in c_w_gd]))
                if len(c_t) != new_labels[-1].sum():
                    set_c_w_gd = set(c_w_gd)
                    for cc in list(c_t):
                        if cc in set_c_w_gd:
                            continue
                        for j in range(new_labels[-1].shape[0]):
                            if new_labels[-1][j].item() != 1:
                                c_w_gd[j] = cc
                                new_labels[-1][j] = 1.0
                                break
                bs = be
            labels = torch.stack(new_labels).to(self.device)
            out_dict['labels'] = labels
        candidates = torch.LongTensor(candidates).to(self.device)


        emb = self.encoder(out)
        embed_weights = self.embed(candidates)
        emb = emb.unsqueeze(-1)
        embed_weights = embed_weights.unsqueeze(1)
        logits = torch.matmul(embed_weights, emb).squeeze(-1)
        can_out = logits.transpose(1, 2)
        can_out = self.score_fc2(can_out).squeeze(-1)
        out_dict['l_out'] = can_out
        if not is_training:
            group_candidates_scores = torch.Tensor(group_candidates_scores).to(self.device)
            out_dict['candidates'] = candidates
            out_dict['group_candidates_scores'] = group_candidates_scores

        return out_dict

    def get_candidates(self, group_logits, group_gd=None, is_training=False, epoch=0):
        logits = torch.sigmoid(group_logits.detach())
        if group_gd is not None:
            logits += group_gd.to(logits.dtype)
        scores, indices = torch.topk(logits, k=self.candidates_topk)
        scores, indices = scores.cpu().detach().numpy(), indices.cpu().detach().numpy()
        candidates, candidates_scores = [], []
        for index, score in zip(indices, scores):
            candidates.append(self.group_y[index])
            if not is_training:
                candidates_scores.append([np.full(c.shape, s) for c, s in zip(candidates[-1], score)])
            candidates[-1] = np.concatenate(candidates[-1])
            if not is_training:
                candidates_scores[-1] = np.concatenate(candidates_scores[-1])
        max_candidates = max([i.shape[0] for i in candidates])
        candidates = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates])

        if not is_training:
            candidates_scores = np.stack(
                [np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates_scores])
        if is_training :
            return indices, candidates, candidates_scores
        return indices, candidates, candidates_scores
