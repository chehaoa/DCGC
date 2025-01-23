import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from opt import args
from utils import *
from kmeans_gpu import kmeans
from sklearn import cluster
import dgl.function as fn
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, GINConv, GATConv
from torch_geometric.utils import dense_to_sparse

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from layers import Filter_simple


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()

class DCGC(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, act, node_num, cluster_num
    ):
        super(DCGC, self).__init__()
        self.node_num = node_num
        self.cluster_num = cluster_num
        self.hdim = hidden_dim
        self.AE1 = nn.Linear(input_dim, hidden_dim)
        self.AE2 = nn.Linear(input_dim, hidden_dim)
        
        self.filter = Filter_simple(input_dim)

        # positive and negative sample pair index matrix
        self.mask = (torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)).to(args.device)
        self.pos_weight = torch.ones(node_num * 2).to(args.device)
        self.pos_neg_weight = torch.ones([node_num * 2, node_num * 2]).to(args.device)
        self.s_similarity = torch.ones([node_num, node_num]).to(args.device)

        # Clustering head
        self.cluster_centers = nn.Parameter(torch.Tensor(cluster_num, hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

        # class_neighbor_distribution head
        self.neighbor_centers = nn.Parameter(torch.Tensor(cluster_num, cluster_num))
        torch.nn.init.xavier_normal_(self.neighbor_centers.data)

        self.KL_loss = nn.KLDivLoss(reduction='batchmean')

        self._lambda = nn.Parameter(torch.tensor(0.5))
        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def forward(self, x, adj_low, adj_high):
        x = self.filter(x, adj_low, adj_high)
        Z1 = self.activate(self.AE1(x))
        Z2 = self.activate(self.AE2(x))

        Z1 = F.normalize(Z1, dim=1, p=2)
        Z2 = F.normalize(Z2, dim=1, p=2)
        Z = (Z1 + Z2)/2
        return Z1, Z2, Z
    
    def clustering_loss(self, Z1, Z2, Q):
        Z = (Z1 + Z2) / 2
        output = self.get_cluster_prob(Z)
        target = target_distribution(Q).detach()
        cluster_loss = self.KL_loss((output+1e-08).log(), target)/output.shape[0]
        return cluster_loss
    
    def neighbor_loss(self, Z1, Z2, A, Q):
        Z = (Z1 + Z2) / 2
        y_p_clu = self.get_cluster_prob(Z)
        y_p_nei = self.get_neighbor_prob(A, y_p_clu)
        
        target = target_distribution(Q).detach()
        neighbor_loss = self.KL_loss((y_p_nei+1e-08).log(), target)/y_p_nei.shape[0]
        return neighbor_loss
    
    def dual_centers_loss(self, Z1, Z2, A, Q):
        Z = (Z1 + Z2) / 2
        y_p_clu = self.get_cluster_prob(Z)
        y_p_nei = self.get_neighbor_prob(A, y_p_clu)
        
        target = target_distribution(Q).detach()
        y_p = (y_p_clu + y_p_nei) / 2
        dual_centers_loss = self.KL_loss((y_p+1e-08).log(), target)/y_p.shape[0]
        return dual_centers_loss
    
    def dual_centers_loss_ours(self, Z1, Z2, A, Q):
        Z = (Z1 + Z2) / 2
        y_p_clu = self.get_cluster_prob(Z)
        y_p_nei = self.get_neighbor_prob(A, y_p_clu)
        
        target = target_distribution(Q).detach()
        cluster_loss = self.KL_loss((y_p_clu+1e-08).log(), target)/y_p_clu.shape[0]
        neighbor_loss = self.KL_loss((y_p_nei+1e-08).log(), target)/y_p_nei.shape[0]
        dual_centers_loss = self._lambda * cluster_loss + (1 - self._lambda) * neighbor_loss
        return dual_centers_loss
    
    def infoNCE(self, Z1, Z2):
        S = comprehensive_similarity(Z1, Z2)
        pos_neg = self.mask * torch.exp(S)
        pos = torch.cat([torch.diag(S, self.node_num), torch.diag(S, -self.node_num)], dim=0)
        pos = torch.exp(pos)
        neg = (torch.sum(pos_neg, dim=1) - pos)
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * self.node_num)
        return infoNEC
    
    def neighbor_infoNCE(self, Z1, Z2, pos_neg_weight, pos_weight):
        S = comprehensive_similarity(Z1, Z2)
        pos_neg = self.mask * torch.exp(S * pos_neg_weight)
        pos = torch.cat([torch.diag(S, self.node_num), torch.diag(S, -self.node_num)], dim=0)
        pos = torch.exp(pos * pos_weight)
        neg = (torch.sum(pos_neg, dim=1) - pos)
        neighbor_infoNCE = (-torch.log(pos / (pos + neg))).sum() / (2 * self.node_num)
        return neighbor_infoNCE
    
    def pseudo_matrix(self, Z1, Z2, y_pred, A):
        S = comprehensive_similarity(Z1, Z2)
        S_norm = (S - S.min()) / (S.max() - S.min())
        y_pred = torch.tensor(y_pred)
        if y_pred.dim() == 1:
            y_hat_onehot = F.one_hot(y_pred, self.cluster_num).to(args.device)
            K_norm = neighbor_similarity(A, y_hat_onehot.float().to(args.device))
        else:
            K_norm = neighbor_similarity(A, y_pred.to(args.device))

        A = torch.cat([torch.cat([A, A], dim=1),
                       torch.cat([A, A], dim=1)], dim=0)
        K_norm[K_norm > args.tau] = 1
        K_norm[K_norm <= args.tau] = 0

        M_mat = torch.abs(K_norm - S_norm)
        M = torch.cat([torch.diag(M_mat, self.node_num), torch.diag(M_mat, -self.node_num)], dim=0)
        return M, M_mat, K_norm[:self.node_num, :self.node_num]
    
    def adjre_loss(self, Z1, Z2, A, S):
        A[S == 0] = 0
        Z = (Z1 + Z2)/2
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        re_loss = F.binary_cross_entropy(A_pred.view(-1), A.view(-1))
        return re_loss

    def assignment_hat(self, A, y_hat, centers):
        cluster_num = centers.shape[0]
        y_hat = torch.Tensor(y_hat).long()
        y_hat_onehot = F.one_hot(y_hat, cluster_num).to(args.device)
        neighbor_centers = compute_class_neighbor_dist(A, y_hat_onehot.float())
        self.neighbor_centers.data = neighbor_centers.data
        self.cluster_centers.data = centers.data
        return
    
    def assignment_prob(self, A, y_prob, centers):
        y_prob = torch.Tensor(y_prob).to(args.device)
        neighbor_centers = compute_class_neighbor_dist(A, y_prob)
        self.neighbor_centers.data = neighbor_centers.data
        self.cluster_centers.data = centers.data
        return
    
    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / args.theta))
        power = float(args.theta + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def get_neighbor_prob(self, A, y_p_clu):
        p_neighbor_clu = compute_neighbor_dist(A, y_p_clu)
        norm_squared = torch.sum((p_neighbor_clu.unsqueeze(1) - self.neighbor_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / args.theta))
        power = float(args.theta + 1) / 2
        numerator = numerator ** power
        y_p_nei = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return y_p_nei
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1
    
    def projection(self, Z1, Z2):
        Z = (Z1 + Z2)/2
        return self.fc_pipe(Z)

    def con_loss(self, Z1, Z2):
        P = F.one_hot(torch.Tensor(P).long(), center.shape[0]).float()
        P = P.to(args.device)
        center = center.to(args.device)
        Q = self.activate(P @ center)
        Q = F.normalize(Q, dim=1, p=2)
        con_loss = 0.5 * (F.kl_div(Z1, Q, reduction="batchmean") + F.kl_div(Z2, Q, reduction="batchmean"))
        return con_loss
    
    