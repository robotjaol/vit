# Vision Transformer (ViT) Study Project

A from-scratch implementation of Vision Transformer written in a plain, readable style.
The goal is to understand how ViT works internally, not to ship a production library.

---

## Project Layout

```
vit/
  vit-pytorch/          implementation files
    __init__.py         exports ViT and SimpleViT
    vit.py              full ViT with CLS token and learnable positions
    simple_vit.py       simpler ViT with mean pooling and sincos positions

  exam/
    vit_tutorial.ipynb  step-by-step tutorial, trains on CIFAR-10

  sample/               reference code from the original vit-pytorch library
```

`vit-pytorch/` is the code we wrote. `sample/` is the upstream source we studied.
Do not edit `sample/`.

---

## What the Code Does

An image goes through these steps in order:

1. **Patch split** - the image is cut into equal-sized square patches
2. **Linear projection** - each patch is flattened and projected to a fixed embedding dimension
3. **Positional encoding** - a position signal is added so the model knows where each patch is
4. **Transformer blocks** - a stack of attention and feed-forward layers processes all patches together
5. **Pooling** - the patch outputs are collapsed into one vector (CLS token or mean)
6. **Classifier** - a linear layer maps that vector to class scores

---

## Two Models

### ViT (`vit.py`)

The original architecture from the paper. Uses a learnable CLS token prepended to the patch sequence. The CLS token output after the final Transformer block is fed to the classifier.

Also uses learnable positional embeddings: a parameter matrix of shape `(num_patches + 1, dim)` that is trained alongside the rest of the model.

```python
from vit_pytorch.vit import ViT

model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    num_heads=12,
    mlp_dim=3072,
    pool='cls',        # 'cls' or 'mean'
    channels=3,
    head_dim=64,
    dropout=0.1,
    emb_dropout=0.1
)
```

### SimpleViT (`simple_vit.py`)

A cleaner variant. No CLS token. All patch outputs are averaged after the Transformer (mean pooling). Uses fixed sinusoidal positional encoding computed from 2D grid coordinates, so no position parameters need to be trained.

```python
from vit_pytorch.simple_vit import SimpleViT

model = SimpleViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    num_heads=12,
    mlp_dim=3072,
    channels=3,
    head_dim=64
)
```

Both models accept a batch of images with shape `(batch, channels, height, width)` and return logits with shape `(batch, num_classes)`.

---

## Hyperparameter Guide

| Parameter | What it controls |
|---|---|
| `image_size` | Input image resolution. Must be divisible by `patch_size`. |
| `patch_size` | Side length of each patch in pixels. Smaller = more patches, more compute. |
| `dim` | Embedding dimension for each patch token. |
| `depth` | Number of Transformer blocks stacked. |
| `num_heads` | Number of attention heads. `dim` should be divisible by `num_heads * head_dim`. |
| `head_dim` | Dimension per attention head. Typical value is 32 or 64. |
| `mlp_dim` | Hidden size of the feed-forward network inside each block. Usually `4 * dim`. |
| `pool` | `'cls'` uses the CLS token output. `'mean'` averages all patch outputs. ViT only. |
| `dropout` | Dropout rate inside attention and feed-forward. Set to 0 to disable. |
| `emb_dropout` | Dropout applied to patch embeddings before the Transformer. ViT only. |

Patch count formula:

```
num_patches = (image_height / patch_height) * (image_width / patch_width)
```

A 224x224 image with 16x16 patches gives 196 patches. Each patch has `patch_h * patch_w * channels` raw values before projection.

---

## Notebook

`exam/vit_tutorial.ipynb` walks through the whole system from scratch:

- Why images are cut into patches and what the math looks like
- How sinusoidal positional encoding works and why it uses sin and cos at different frequencies
- How scaled dot-product attention is computed, including the Q/K/V intuition
- Why we divide scores by `sqrt(head_dim)` and what happens if we do not
- What residual connections and LayerNorm do for training stability
- A working training loop on CIFAR-10 with loss curves and prediction visualization

Run it with:

```bash
jupyter notebook exam/vit_tutorial.ipynb
```

CIFAR-10 downloads automatically through `torchvision`. No Kaggle account needed.

---

## Dependencies

```bash
pip install torch torchvision einops tqdm matplotlib
```

| Package | Purpose |
|---|---|
| `torch` | tensor operations and autograd |
| `torchvision` | dataset loading and image transforms |
| `einops` | readable tensor reshaping (`rearrange`, `repeat`) |
| `tqdm` | progress bars in the notebook |
| `matplotlib` | plotting loss curves and images |

Python 3.8 or later. Tested with PyTorch 2.x.

---

## Architecture Diagram

```
Input image (B, C, H, W)
        |
  [ Patch Split ]          cut into (H/p * W/p) patches
        |
  [ Linear Projection ]    each patch -> dim-dimensional vector
        |
  [ + Positional Encoding ] add row/col position signal
        |
  [ Transformer Block ] x depth
  |  LayerNorm
  |  Multi-Head Attention   Q, K, V projections + scaled dot product
  |  + residual
  |  LayerNorm
  |  Feed-Forward (MLP)     expand to mlp_dim, GELU, contract to dim
  |  + residual
        |
  [ Pool ]                  CLS token [0] or mean over all tokens
        |
  [ Linear Classifier ]     dim -> num_classes
        |
  Logits (B, num_classes)
```

---

## Key Design Choices

**Pre-LayerNorm.** Normalization is applied before each sub-layer, not after. This makes training more stable, especially at larger depth, because the input to each sub-layer is always well-scaled.

**Scale factor in attention.** Dot products between query and key vectors are divided by `sqrt(head_dim)` before softmax. Without this, large dot products push softmax toward near-one-hot distributions, causing gradients to vanish.

**Sinusoidal vs learnable positions.** Learnable positions are simpler to implement and work slightly better at fixed resolutions. Sinusoidal positions do not need training and generalize more naturally to resolutions not seen during training.

**Mean pooling vs CLS token.** Both work. Mean pooling is simpler because it removes one parameter (the CLS token) and does not require the model to route information to a specific position. The original paper uses a CLS token following BERT.

---

## Reference

Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929.

