from opt import args

def setup_args():
    args.device = "cuda:1"
    args.acc = args.nmi = args.ari = args.f1 = 0

    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.clu_lr = 1e-4
        args.cluster_num = 7
        args.beta = 0.1
        args.tau = 0.5

    elif args.dataset == 'citeseer':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'sigmoid'
        args.clu_lr = 1e-4
        args.cluster_num = 6
        args.beta = 0.2
        args.tau = 0.5

    elif args.dataset == 'amap':
        args.t = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.clu_lr = 1e-5
        args.cluster_num = 8
        args.beta = 0.0
        args.tau = 0.5
#
    elif args.dataset == 'bat':
        args.t = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.clu_lr = 1e-4
        args.cluster_num = 4
        args.beta = 0.0
        args.tau = 0.2

    elif args.dataset == 'eat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.clu_lr = 1e-5
        args.cluster_num = 4
        args.beta = 0.0
        args.tau = 0.2

    elif args.dataset == 'uat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500
        args.activate = 'sigmoid'
        args.clu_lr = 1e-5
        args.cluster_num = 4
        args.beta = 0.3
        args.tau = 0.2

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-4
        args.clu_lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.beta = 0.3
        args.tau = 0.5

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("learning rate: {}".format(args.lr))
    print("clustering learning rate: {}".format(args.clu_lr))
    print("beta: {}".format(args.beta))
    print("tau: {}".format(args.tau))
    print("---------------------")

    return args
