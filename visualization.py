import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

font = {'family': 'Times New Roman', 'size': 22}
plt.rcParams['font.family'] = font['family']
plt.rcParams['font.size'] = font['size']


def t_sne(embeds, labels, sample_num=2000, show_fig=True, desc=0):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return fig: figure
    """

    # sampling
    sample_index = np.random.randint(0, embeds.shape[0], sample_num)
    sample_embeds = embeds[sample_index]
    sample_labels = labels[sample_index]

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    # remove outlier
    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)
    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean < 3 * std).all():
            np.delete(ts_embeds, i)

    # normalization
    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    # plot
    fig = plt.figure()
    for i in range(norm_ts_embeds.shape[0]):
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], str(sample_labels[i]),
                 color=plt.cm.Set1(sample_labels[i] % 7),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE', fontsize=14)
    plt.axis('off')
    plt.savefig(f"./ours_cora_{desc}.png")
    if show_fig:
        plt.show()
    return fig

def plot_clustering_tsne(args, embedding, label, version="paper", img_suffix=".pdf", desc="default",
                         axis_show=True, title="TSNE"):
    """
    :param desc: the description of image
    :param args: the parameter settings of model
    :param embedding: the embedded representations will be drawn which were learned by model
    :param label: the groundtruth label of dataset
    :param logger: the logger to record information during the process
    :param img_suffix: the suffix of image
            '.png','.pdf','.jpg','.jpeg','.bmp','.tiff','.gif','.svg', '.eps' are available
    :param axis_show: is show the axis of image
    :param title: the title of image, default value is "TSNE", if needn't, set it to None
    :return:
    """
    clustering_tsne_filename = f"./img/clustering/{version}/{args.dataset}_{desc}{img_suffix}"
    X_tsne = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(embedding.cpu().detach().numpy())
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=2, c=plt.cm.Set1(label % args.cluster_num))
    if not axis_show:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(clustering_tsne_filename)
    plt.clf()


def plot_embedding_heatmap(args, embedding, version="paper", img_suffix=".pdf", desc="default",
                           axis_show=True, title="Heatmap", color_bar_show=True):
    """
    :param desc: the description of image
    :param args: the parameter settings of model
    :param embedding: the embedded representations will be drawn which were learned by model
    :param logger: the logger to record information during the process
    :param img_suffix: the suffix of image
            '.png','.pdf','.jpg','.jpeg','.bmp','.tiff','.gif','.svg', '.eps' are available
    :param axis_show: is show the axis of image
    :param title: the title of image, default value is "HeatMap", if needn't, set it to None
    :param color_bar_show: whether to display the color bar of the image, default value is True
    :return:
    """

    embedding_heatmap_filename = f"./img/heatmap/{version}/{args.dataset}_{desc}{img_suffix}"
    plt.imshow(embedding.cpu().detach().numpy(), cmap=plt.cm.GnBu, interpolation='nearest')
    if color_bar_show:
        plt.colorbar()
    if not axis_show:
        plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(embedding_heatmap_filename)
    plt.clf()
