import sys
from tqdm import tqdm
from torch import optim
from model import DCGC
from utils import *
from kmeans import *
from visualization import *
from trainer import Trainer
from setup import setup_args

result_dir = "./result.csv"
if __name__ == '__main__':

    # setup hyper-parameter
    args = setup_args()
    
    # record results
    file = open(result_dir, "a+")
    print(args.dataset + "_DCGC", file=file)
    print("ACC,   NMI,   ARI,   F1", file=file)
    file.close()
    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    # ten runs with different random seeds
    for args.seed in range(args.runs):
        (
            adj_high,
            adj_low,
            adj_low_unnormalized,
            features,
            labels,
            node_num, 
            cluster_num
        ) = train_prep()
        
        # initialization
        args.acc, args.nmi, args.ari, args.f1, y_prob, y_hat, center = phi(features.float(), labels, cluster_num)

        # load data to device
        adj_low_unnormalized, features = map(lambda x: x.to(args.device), (adj_low_unnormalized, features))
        
        # build our network
        model = DCGC(features.shape[1], args.dims, args.activate, node_num, cluster_num).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        trainer = Trainer(None, model, optimizer)
        embeds_best, z = trainer.train_ours(features, adj_low, adj_high, adj_low_unnormalized, labels, cluster_num)
        print("Training complete")

        # indexes = np.argsort(labels)
        # sorted_embedding = embeds_best[indexes]

        # draw the clustering image or embedding heatmap
        # plot_clustering_tsne(args, embeds_best, labels, version="dccgc", img_suffix=".png",desc=f"{args.seed}", title=None, axis_show=False)
        # plot_embedding_heatmap(args, torch.matmul(sorted_embedding, sorted_embedding.t()),
        #                         version="dccgc", img_suffix=".png", desc=f"{args.seed}",
        #                         title=None, axis_show=False, color_bar_show=True)
        # t_sne(embeds_best.cpu().detach().numpy(), labels, node_num, False, args.seed)
    
        file = open(result_dir, "a+")
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
        file.close()
        acc_list.append(args.acc)
        nmi_list.append(args.nmi)
        ari_list.append(args.ari)
        f1_list.append(args.f1)

    acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
    file = open(result_dir, "a+")
    print("ACC: {:.2f} ± {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
    print("NMI: {:.2f} ± {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
    print("ARI: {:.2f} ± {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
    print("F1 : {:.2f} ± {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
    file.close()
