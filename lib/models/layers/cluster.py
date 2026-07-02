import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional, List


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        feature_dim: int,
        num_domains: int = 2,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ):
        """
        Domain-Invariant Alignment module inspired by DEC-style soft assignment.
        
        Args:
            num_clusters: R, number of global clusters
            feature_dim: D, dimension of features
            num_domains: M, number of domains
            mlp_hidden_dim: hidden dim for fusion MLP
            alpha: temperature (same as in DEC)
            cluster_centers: optional initial cluster centers
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.alpha = alpha

        if cluster_centers is None:
            centers = torch.zeros(num_clusters, feature_dim, dtype=torch.float)
            nn.init.xavier_uniform_(centers)
        else:
            centers = cluster_centers
        self.cluster_centers = Parameter(centers)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(num_domains * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim//3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [M*N, D] — concatenated features from all domains
        Returns:
            F_g: [N, D] — domain-invariant fused representation
            w_all: [M*N, R] — soft assignment weights
        """
        M = self.num_domains
        N = features.shape[0] // M

        norm_squared = torch.sum(
            (features.unsqueeze(1) - self.cluster_centers) ** 2, dim=2
        )  # [M*N, R]
        numerator = (1.0 + norm_squared / self.alpha) ** (-(self.alpha + 1) / 2)
        w = numerator / numerator.sum(dim=1, keepdim=True)  # soft assignment weights

        weighted_feat = torch.matmul(w, self.cluster_centers)  # [M*N, D]

        weighted_feat = weighted_feat.view(M, N, -1)  # [M, N, D]

        fused_input = torch.cat([weighted_feat[m] for m in range(M)], dim=-1)  # [N, M*D]
        F_g = self.mlp_fusion(fused_input)  # [N, D]
        F_g = F.normalize(F_g, p=2, dim=-1)

        return F_g
