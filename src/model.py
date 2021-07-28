import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.0):
        """

        Args:
            d_model:
            n_heads:
            dropout_rate:
        """
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_layer = nn.LayerNorm(d_model, eps=1e-6)

        self.q_weight = nn.Linear(d_model, d_model)
        self.k_weight = nn.Linear(d_model, d_model)
        self.v_weight = nn.Linear(d_model, d_model)

        self.fc_layer = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        """

        Args:
            q:
            k:
            v:
            mask:

        Returns:

        """
        residual, n_batch = q, q.size(0)




