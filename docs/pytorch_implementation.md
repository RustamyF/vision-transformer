# Description and Explanation of ViT (Vision Transformer) Class

The ViT (Vision Transformer) class is a custom implementation of a deep learning model for image classification tasks. It is built using the `nn.Module` from PyTorch, and uses the Transformer architecture for image processing. The class has the following attributes and methods:

## Attributes
- `image_size`: A tuple representing the height and width of the input images.
- `patch_size`: A tuple representing the height and width of the patches that the input images are divided into.
- `num_classes`: An integer representing the number of classes in the classification task.
- `dim`: The dimensionality of the hidden representation in the Transformer model.
- `depth`: The number of layers in the Transformer model.
- `heads`: The number of attention heads in the Transformer model.
- `mlp_dim`: The dimensionality of the feedforward layer in the Transformer model.
- `pool`: A string that specifies whether to use the mean pooling or the CLS token as the representation of the whole image. The possible values are 'cls' or 'mean'.
- `channels`: An integer representing the number of channels in the input images.
- `dim_head`: The dimensionality of each attention head.
- `dropout`: The dropout rate for the Transformer model.
- `emb_dropout`: The dropout rate for the image patch embedding.

## Methods

### `__init__`
The constructor initializes the different layers of the model. It performs the following steps:

1. Check if the image size is divisible by the patch size. If not, raise an error.
```python
image_height, image_width = pair(image_size)
patch_height, patch_width = pair(patch_size)

assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

```
2. Compute the number of patches and the dimension of each patch.
```python
num_patches = (image_height // patch_height) * (image_width // patch_width)
patch_dim = channels * patch_height * patch_width
```
3. Check if the `pool` argument is either 'cls' or 'mean'. If not, raise an error.
```python
assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
```
4. Initialize the patch to patch embedding, which is a sequential container consisting of:
    - Rearranging the patches to have the shape `(batch_size, num_patches, patch_height * patch_width * channels)`.
    - Applying layer normalization to the patches.
    - Applying a linear transformation to the patches to obtain a hidden representation of size `dim`.
    - Applying layer normalization to the hidden representation.

```python
self.to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
    nn.LayerNorm(patch_dim),
    nn.Linear(patch_dim, dim),
    nn.LayerNorm(dim),
)
```
5. Initialize the position embeddings, CLS token, and dropout.
```python
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
self.dropout = nn.Dropout(emb_dropout)
```
6. Initialize the Transformer model with the specified parameters `dim`, `depth`, `heads`, `dim_head`, and `mlp_dim`.
```python
self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
```
7. Initialize the `to_latent` layer, which is an identity layer.
```python
self.to_latent = nn.Identity()
```

8. Initialize the MLP head, which is a sequential container consisting of:
    - Applying layer normalization to the image representation.
    - Applying a linear transformation to the image representation to obtain the final logits of size `num_classes`.

```python
self.mlp_head = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, num_classes)
)
```
### `forward`
The forward method takes an input image and performs the following steps:

1. Pass the input image through the patch to patch embedding layer to obtain a hidden representation of each patch.
```python
x = self.to_patch_embedding(img)
b, n, _ = x.shape
```
2. Concatenate the CLS token with the hidden representation of the patches to obtain the final hidden representation.
```python
cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
x = torch.cat((cls_tokens, x), dim=1)
```
3. Add the position embeddings to the final hidden representation.
```python
x += self.pos_embedding[:, :(n + 1)]
```
4. Apply dropout to the final hidden representation.
```python
x = self.dropout(x)
```
5. Apply the transformer architecture to the final hidden representation
```python
x = self.transformer(x)
```
6. Get the mean of hiddent representation or the first row and pass it through the latent identity

```python
x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
x = self.to_latent(x)
```



## Python Source Code

::: vision_tr.simple_vit.ViT
