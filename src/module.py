import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x, mask):
        return x


class MeanPooling(nn.Module):
    def forward(self, x, mask):
        x *= mask.unsqueeze(-1).float()
        seq_len = mask.sum(dim=1, keepdim=True).float()
        return x.sum(dim=1) / seq_len


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_layers,
        activation="relu"
    ):
        super().__init__()
        trm_layer = nn.TransformerEncoderLayer(
            input_dim,
            num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.trm = nn.TransformerEncoder(trm_layer, num_layers)

    def forward(self, x, mask):
        x = self.trm(x.transpose(0, 1), src_key_padding_mask=(mask == 0))
        return x.transpose(0, 1)


class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian
    Approximation: Representing Model Uncertainty in Deep Learning"
    (https://arxiv.org/abs/1506.02142) to a 3D tensor.

    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps,
    embedding_dim)`` and samples a single dropout mask of shape ``(batch_size,
    embedding_dim)`` and applies it to every time step.
    """

    def forward(self, input_tensor):
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


class BilinearAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function. This
    function has a matrix of weights ``W`` and a bias ``b``, and the similarity between
    the two matrices ``X`` and ``Y`` is computed as ``X W Y^T + b``.

    Input: - mat1: ``(batch_size, num_rows_1, mat1_dim)`` - mat2: ``(batch_size,
        num_rows_2, mat2_dim)``

    Output: - ``(batch_size, num_rows_1, num_rows_2)``
    """

    def __init__(
        self, mat1_dim: int, mat2_dim: int, use_input_biases: bool = False,
    ) -> None:
        super().__init__()
        if use_input_biases:
            mat1_dim += 1
            mat2_dim += 1

        self.weight = nn.Parameter(torch.Tensor(1, mat1_dim, mat2_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0)

    def forward(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        if self._use_input_biases:
            bias1 = mat1.new_ones(mat1.size()[:-1] + (1,))
            bias2 = mat2.new_ones(mat2.size()[:-1] + (1,))

            mat1 = torch.cat([mat1, bias1], -1)
            mat2 = torch.cat([mat2, bias2], -1)

        intermediate = torch.matmul(mat1.unsqueeze(1), self.weight)
        final = torch.matmul(intermediate, mat2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1) + self.bias
