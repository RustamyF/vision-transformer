import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    """
    This class applies a normalization layer before applying a specified function.

    :param dim: An integer specifying the number of dimensions to normalize.
    :param fn: A function to apply to the normalized input.

    Example:
    prenorm = PreNorm(3, torch.relu)
    x = torch.randn(4, 3)
    prenorm(x)
    """

    def __init__(self, dim, fn):
        """
        Initialize the PreNorm module with a LayerNorm and specified function.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Perform the forward pass of the PreNorm module.
        Normalize the input with LayerNorm and then apply the specified function.

        :param x: Input tensor.
        :param kwargs: Additional keyword arguments passed to the specified function.
        :return: Output tensor after applying LayerNorm and the specified function.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    This class implements a feed-forward neural network with two linear layers, GELU activation and dropout layers.

    :param dim: An integer specifying the input dimensions.
    :param hidden_dim: An integer specifying the dimensions of the hidden layer.
    :param dropout: A float in the range [0, 1], specifying the dropout rate.

    Example:
    feedforward = FeedForward(3, 128, 0.1)
    x = torch.randn(4, 3)
    feedforward(x)
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        Initialize the FeedForward module with a sequential network composed of two linear layers, GELU activation and dropout layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Perform the forward pass of the FeedForward module.

        :param x: Input tensor.
        :return: Output tensor after applying the sequential network.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism described in Vaswani et al. (2017)
    This module uses the Scaled Dot-Product Attention as its core mechanism.

    Parameters:
        dim (int): the number of features in the input tensor x
        heads (int, optional): the number of heads in the Multi-Head Attention mechanism, default is 8
        dim_head (int, optional): the number of features per head, default is 64
        dropout (float, optional): the dropout rate, default is 0.0

    Returns:
        tensor: the output tensor after the Multi-Head Attention mechanism

    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Implements the Transformer architecture described in Vaswani et al. (2017)

    Parameters:
        dim (int): the number of features in the input tensor x
        depth (int): the number of layers in the Transformer architecture
        heads (int): the number of heads in the Multi-Head Attention mechanism
        dim_head (int): the number of features per head in the Multi-Head Attention mechanism
        mlp_dim (int): the hidden size in the Feed Forward layer
        dropout (float, optional): the dropout rate, default is 0.0

    Returns:
        tensor: the output tensor after passing through the Transformer architecture

    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
