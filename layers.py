import math
from opt import args
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class Filter_simple(Module):
    def __init__(
        self,
        in_features,
    ):
        super(Filter_simple, self).__init__()
        self.in_features = in_features

        self.att_low = nn.Parameter(torch.tensor(0.99, device=args.device))
        self.att_high = nn.Parameter(torch.tensor(0.005, device=args.device))
        self.att_mlp = nn.Parameter(torch.tensor(0.005, device=args.device))

    def forward(self, input, adj_low, adj_high):
        output_low = input
        output_high = input
        output_mlp = input
        for _ in range(args.t):
            output_low = torch.mm(adj_low, output_low)
            output_high = torch.mm(adj_high, output_high)

        return 3 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )
