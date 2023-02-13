import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from transformer import Transformer, pair


def pair(t):
    """
    This function returns a tuple with two elements, the input argument if it's a tuple or the input argument repeated twice if it's not.

    :param t: an object that could be either a tuple or any other object.
    :return: a tuple with two elements, either the input argument or the input argument repeated twice.

    Example:
    pair( (1,2) ) => (1,2)
    pair( "hello" ) => ("hello", "hello")
    """
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    """
    Class to implement the Vision Transformer (ViT) architecture

    Parameters:
    image_size (tuple): height and width of the input image.
    patch_size (tuple): height and width of the patches in which the image will be divided into.
    num_classes (int): number of classes for the classification task.
    dim (int): dimension of the input patches.
    depth (int): number of layers in the Transformer model
    heads (int): number of attention heads in the Transformer model
    mlp_dim (int): number of hidden units in the feed forward layer in the Transformer model
    pool (str): either 'cls' or 'mean', specifying how the patches will be aggregated
    channels (int): number of channels in the input image
    dim_head (int): number of hidden units in the attention layer in the Transformer model
    dropout (float): dropout rate for the feed forward layer in the Transformer model
    emb_dropout (float): dropout rate for the patch embeddings

    Returns:
    output (tensor): output logits of shape (batch_size, num_classes)
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()

        # Check if the image size is divisible by the patch size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # Compute the number of patches and the dimension of each patch
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Check if the pool argument is either 'cls' or 'mean'
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # Implement the patch to patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Initialize the position embeddings, CLS token, and dropout
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    print(preds.shape)
