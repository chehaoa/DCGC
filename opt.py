import argparse

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--gamma', type=float, default=10.0)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--device', type=str, default="cpu", help='device.')
parser.add_argument('--cluster_num', type=int, default=7, help='cluster number')

# pre-process
parser.add_argument('--n_input', type=int, default=500, help='input feature dimension')
parser.add_argument('--t', type=int, default=2, help="filtering time of Laplacian filters")

# network
parser.add_argument('--dims', type=int, default=[1500], help='hidden unit')
parser.add_argument('--activate', type=str, default='ident', help='activate function')
parser.add_argument('--theta', type=float, default=1.0, help='get_cluster_prob')
parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")

# training
parser.add_argument('--runs', type=int, default=10, help='runs')
parser.add_argument('--epochs', type=int, default=400, help='training epoch')
parser.add_argument('--clu_epochs', type=int, default=100, help='clustering loss training epoch')
parser.add_argument('--eval_freq', type=int, default=10, help='eval_freq')
parser.add_argument('--update_interval', default=1, type=int)

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--clu_lr', type=float, default=1e-4, help='clustering loss learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--acc', type=float, default=0, help='acc')
parser.add_argument('--nmi', type=float, default=0, help='nmi')
parser.add_argument('--ari', type=float, default=0, help='ari')
parser.add_argument('--f1', type=float, default=0, help='f1')

args = parser.parse_args()
