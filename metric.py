from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import os



def get_precision(y_true,y_pred,classes,top=5):
    mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
    mlb.fit(y_true)
    if not isinstance(y_true, csr_matrix):
        y_true = mlb.transform(y_true)
    y_pred = mlb.transform(y_pred[:,:top])
    return y_pred.multiply(y_true).sum() / (top * y_true.shape[0])

get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)

def get_ndcg(y_true, y_pred, classes,top=5):
    mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
    mlb.fit(y_true)
    if not isinstance(y_true, csr_matrix):
        y_true = mlb.transform(y_true)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((y_true.shape[0], 1))
    for i in range(top):
        p = mlb.transform(y_pred[:, i: i + 1])
        dcg += p.multiply(y_true).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(y_true.sum(axis=-1), top) - 1])

get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)

def base1_metric(y_true, y_pred, classes):
    p1, p3, p5 = get_p_1(y_true, y_pred, classes), get_p_3(y_true, y_pred, classes), get_p_5(y_true, y_pred, classes)
    n1, n3, n5 = get_n_1(y_true, y_pred, classes), get_n_3(y_true, y_pred, classes), get_n_5(y_true,y_pred, classes)
    return p1,p3,p5,n1,n3,n5

def ps_metric(y_true, y_score, classes, train_labels):
    y_pred = np.argsort(-y_score, axis=1)
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    mlb.fit(y_true)
    inv_w = get_inv_propensity(mlb.transform(train_labels), 0.55, 1.5)
    psp1, psp3, psp5 = get_psp_1(y_true, y_pred, classes, inv_w, mlb), \
                       get_psp_3(y_true, y_pred, classes, inv_w, mlb), \
                       get_psp_5(y_true, y_pred, classes, inv_w, mlb)
    psn1, psn3, psn5 = get_psn_1(y_true, y_pred, classes, inv_w, mlb), \
                       get_psn_3(y_true, y_pred, classes, inv_w, mlb), \
                       get_psn_5(y_true, y_pred, classes, inv_w, mlb)

    return psp1,psp3,psp5,psn1,psn3,psn5

def ps_metric_pred(y_true, y_score, classes, train_labels):
    y_pred = y_score
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    mlb.fit(y_true)
    inv_w = get_inv_propensity(mlb.transform(train_labels), 0.55, 1.5)
    psp1, psp3, psp5 = get_psp_1(y_true, y_pred, classes, inv_w, mlb), \
                       get_psp_3(y_true, y_pred, classes, inv_w, mlb), \
                       get_psp_5(y_true, y_pred, classes, inv_w, mlb)
    psn1, psn3, psn5 = get_psn_1(y_true, y_pred, classes, inv_w, mlb), \
                       get_psn_3(y_true, y_pred, classes, inv_w, mlb), \
                       get_psn_5(y_true, y_pred, classes, inv_w, mlb)

    return psp1,psp3,psp5,psn1,psn3,psn5


def label2list(label):
    outputs = [[] for _ in range(label.shape[0])]
    x,y = np.where(label==1)
    for xx,yy in zip(x,y):
        outputs[xx].append(yy)
    return outputs


def get_psp(targets, prediction, classes, inv_w, mlb, top=5):
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den

get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)
get_psp_10 = partial(get_psp, top=10)


def get_psndcg(y_true, y_pred, classes, inv_w, mlb, top=5):
    log = 1.0 / np.log2(np.arange(top) + 2)
    psdcg = 0.0
    if not isinstance(y_true, csr_matrix):
        y_true = mlb.transform(y_true)
    for i in range(top):
        p = mlb.transform(y_pred[:, i: i+1]).multiply(inv_w)
        psdcg += p.multiply(y_true).sum() * log[i]
    t, den = csr_matrix(y_true.multiply(inv_w)), 0.0
    for i in range(t.shape[0]):
        num = min(top, len(t.getrow(i).data))
        den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
    return psdcg / den



get_psn_1 = partial(get_psndcg, top=1)
get_psn_3 = partial(get_psndcg, top=3)
get_psn_5 = partial(get_psndcg, top=5)
get_psn_10 = partial(get_psndcg, top=10)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


