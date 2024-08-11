from kmeans_gpu import kmeans
import torch
import random
import numpy as np
from opt import args
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return 100*acc, 100*nmi, 100*ari, 100*f1


def phi(feature, true_labels, cluster_num): # kmeans是硬聚类，所以predict_probs没法模糊表示（值都差不多）没法用
    predict_probs, predict_labels, centers = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device=args.device)
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return acc, nmi, ari, f1, predict_probs.numpy(), predict_labels.numpy(), centers

def phi_sccl(feature, true_labels, cluster_num):
    clustering_model = KMeans(n_clusters=cluster_num)
    clustering_model.fit(feature)
    predict_labels = clustering_model.labels_
    centers = clustering_model.cluster_centers_
    predict_probs = None
    acc, nmi, ari, f1 = eva(true_labels, predict_labels, show_details=False)
    return acc, nmi, ari, f1, predict_probs, predict_labels, centers
