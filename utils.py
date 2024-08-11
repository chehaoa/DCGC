import torch
import numpy as np
import random
import torch.nn.functional as F
from opt import args
from kmeans import *
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj, cluster number
    """
    load_path = "dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    cluster_num = len(np.unique(label))
    node_num = feat.shape[0]
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, torch.tensor(adj).float(), node_num, cluster_num

def normalize_tensor(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 1)
    if symmetric == 0:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)
        return mx
    
def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj

def comprehensive_similarity(Z1, Z2):
    Z1_Z2 = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
                       torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)
    S = Z1_Z2
    return S

def neighbor_similarity(A, y_pred):
    pn = compute_neighbor_dist(A, y_pred)
    K = torch.cat([torch.cat([pn @ pn.T, pn @ pn.T], dim=1),
                   torch.cat([pn @ pn.T, pn @ pn.T], dim=1)], dim=0)
    return K

def compute_neighbor_dist(A, y_p):
    p_neighbor = A @ y_p
    p_neighbor = F.normalize(p_neighbor, dim=1, p=1) # L1和L2归一化
    return p_neighbor

def compute_class_neighbor_dist(A, y_p):
    c_neighbor = y_p.T @ A @ y_p
    c_neighbor = F.normalize(c_neighbor, dim=1, p=1) # L1和L2归一化
    return c_neighbor

def save_checkpoint(z, embeds_best, epoch, loss, acc, nmi, ari, f1, save_name):
    file = open(f"./checkpoint/checkpoint_{save_name}.csv", "a+")
    print('[TEST] Epoch:{:04d} | neighbor loss:{:.4f} | ACC:{:.2f} | NMI:{:.2f} | ARI:{:.2f} | F1:{:.2f}'.format(
                                                                            epoch, loss, acc, nmi, ari, f1), file=file)
    file.close()
    if acc >= args.acc:
        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
        embeds_best = z
    return embeds_best

def get_activation(name: str):
    activations = {
        'relu': torch.nn.ReLU(), # F.relu,
        'hardtanh': torch.nn.Hardtanh(), # F.hardtanh,
        'elu': torch.nn.ELU(), # F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': torch.nn.RReLU(), # F.rrelu
    }
    return activations[name]

def train_prep():
    # fix the random seed(Training settings)
    setup_seed(args.seed)
    
    # load graph data
    features, labels, adj_low_unnormalized, node_num, cluster_num = load_graph_data(args.dataset, show_details=False)
    if min(node_num, features.shape[1]) >= args.n_input and args.n_input != -1:
        pca = PCA(n_components=args.n_input)
        features = pca.fit_transform(features)

    nnodes = labels.shape[0]
    # adj_low_unnormalized = adj_low_unnormalized - torch.diag_embed(torch.diag(adj_low_unnormalized))

    adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized, symmetric=1) # symmetric可以调整0 or 1
    adj_high = (torch.eye(nnodes) - adj_low).to(args.device)
    adj_low = adj_low.to(args.device)
    adj_low_unnormalized = adj_low_unnormalized

    return (
        adj_high,
        adj_low,
        adj_low_unnormalized,
        torch.Tensor(features),
        labels,
        node_num,
        cluster_num
    )
