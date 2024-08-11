import numpy as np
import torch
import torch.nn as nn
from opt import args
from utils import *
from kmeans import *
from tqdm import tqdm


class Trainer(nn.Module):
    def __init__(self, pre_model, model, optimizer):
        super(Trainer, self).__init__()
        self.pre_model = pre_model
        self.model = model
        self.optimizer = optimizer

    def train_ours(self, x, adj_low, adj_high, A, y, cluster_num):
        embeds_best = None
        for epoch in tqdm(range(args.epochs)):
            self.model.train()
            z1, z2, z = self.model(x, adj_low, adj_high)
            
            loss = args.beta * self.model.neighbor_infoNCE(z1, z2, self.model.pos_neg_weight, self.model.pos_weight) + \
                (1 - args.beta) * self.model.adjre_loss(z1, z2, A, self.model.s_similarity)

            loss.backward()
            self.optimizer.step()
                
            if epoch % args.eval_freq == 0:
                self.model.eval()
                z1, z2, z = self.model(x, adj_low, adj_high)
                acc, nmi, ari, f1, _, y_pred, centers = phi(z, y, cluster_num)

                _, M_mat, S = self.model.pseudo_matrix(z1, z2, y_pred, A)
                self.model.pos_neg_weight = M_mat.data
                self.model.s_similarity = S.data

                if acc >= args.acc:
                    args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                    y_pred_best = y_pred
                    centers_best = centers
                    torch.save(self.model.state_dict(), f"./pretrain/{args.dataset}_best.pth")

        self.model.load_state_dict(torch.load(f"./pretrain/{args.dataset}_best.pth"))
        self.model.assignment_hat(A, y_pred_best, centers_best)
        for epoch in tqdm(range(args.clu_epochs)):
            if epoch % args.update_interval == 0:
                self.model.eval()
                z1, z2, z = self.model(x, adj_low, adj_high)
                Q = self.model.get_cluster_prob(z)
                y_pred = Q.detach().data.cpu().numpy().argmax(1)

                _, M_mat, S = self.model.pseudo_matrix(z1, z2, y_pred, A)
                self.model.pos_neg_weight = M_mat.data
                self.model.s_similarity = S
                
                acc, nmi, ari, f1, _, y_pred, centers = phi(z, y, cluster_num)
                
                if acc >= args.acc:
                    args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                    embeds_best = z

            self.model.train()
            z1, z2, z = self.model(x, adj_low, adj_high)

            loss = args.beta * self.model.neighbor_infoNCE(z1, z2, self.model.pos_neg_weight, self.model.pos_weight) + \
                (1 - args.beta) * self.model.adjre_loss(z1, z2, A, self.model.s_similarity)

            loss += args.gamma * self.model.dual_centers_loss_ours(z1, z2, A, Q)

            loss.backward()
            self.optimizer.step()
        return embeds_best, z
