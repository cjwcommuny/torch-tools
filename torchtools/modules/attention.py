import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(
            self,
            h_len,
            v_len,
            hidden_dim
    ):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(h_len, hidden_dim)
        self.U = nn.Linear(v_len, hidden_dim, bias=False)
        self.tanh = nn.Tanh()
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        #
        self.out_feature_dim = v_len


    def forward(self, h, V):
        """
        apply attention from h to V

        e = w^T * tanh(W * h + U * V^T + b)
        beta = softmax(e)
        result = weight_sum(weight=beta, V)

        :param h: shape=(batch_size, h_len)
        :param V: shape=(batch_size, seq_len, v_len)
        :return: shape=(batch_size, v_len)
        """
        batch_size, h_len = h.shape
        seq_len, v_len = V.shape[1:]
        assert batch_size == V.shape[0]
        assert h_len == self.W.in_features, f"{h.shape}, {self.W.in_features}"
        assert v_len == self.U.in_features, f"{V.shape}, {self.U.in_features}"

        h_after_projection = self.W(h).unsqueeze(dim=1) # shape=(batch_size, 1, hidden_dim)
        V_after_projection = self.U(V) # shape=(batch_size, seq_len, hidden_dim)
        e = self.w(self.tanh(h_after_projection + V_after_projection)).view(batch_size, seq_len)
        beta = self.softmax(e) # shape=(batch_size, seq_len)
        result = torch.einsum("bs,bsv->bv", beta, V)
        return result
