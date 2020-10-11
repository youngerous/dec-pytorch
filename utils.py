import torch
import torch.nn as nn


class SoftClusterAssignment(nn.Module):
    def __init__(
        self,
        num_cluster: int,
        hidden_dim: int,
        alpha: float = 1.0,
        centroid: torch.tensor = None,
    ):
        super(SoftClusterAssignment, self).__init__()
        self.num_cluster = num_cluster
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        if centroid is None:
            initial_centroid = torch.zeros(
                self.num_cluster, self.hidden_dim, dtype=torch.float
            )
            nn.init.normal_(initial_centroid, mean=0.0, std=0.01)
        else:
            initial_centroid = centroid
        self.centroid = initial_centroid.cuda()  ##

    def forward(self, z):
        z = z.cuda()
        diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, 2)
        numerator = 1.0 / (1.0 + (diff / self.alpha))
        power = (self.alpha + 1.0) / 2
        numerator = numerator ** power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q
