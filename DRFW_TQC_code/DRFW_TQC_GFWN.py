import torch
from torch import nn

class GFWN(nn.Module):
    def __init__(self, feature_dim, groups=4):
        super(GFWN, self).__init__()
        self.groups = groups
        assert feature_dim % self.groups == 0, "feature_dim must be divisible by groups"
        self.feature_dim = feature_dim
        self.group_dim = feature_dim // groups
        self.gn = nn.GroupNorm(self.groups, self.group_dim)
        self.linear1 = nn.Linear(self.group_dim, self.group_dim)
        self.linear2 = nn.Linear(self.group_dim, self.group_dim)
        self.softmax = nn.Softmax(-1)
        self.alpha = nn.Parameter(torch.ones(1))
    def forward(self, x):
        batch_size, feature_dim = x.size()
        group_x = x.view(batch_size * self.groups, self.group_dim)
        x_norm = self.gn(group_x)
        x1 = self.linear1(x_norm)
        x2 = self.linear2(x_norm)
        x1_weights = self.softmax(x1)
        x2_weights = self.softmax(x2)
        x1_out = x_norm * x1_weights
        x2_out = x_norm * x2_weights
        out = x1_out + x2_out
        out = self.alpha * out
        out = out.view(batch_size, self.feature_dim)
        return out
