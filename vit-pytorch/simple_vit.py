import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(value):
    # make it tuple
    return value if isinstance(value, tuple) else (value, value)


def build_sincos_embedding(num_rows, num_cols, dim):
    # fixed position encoding
    assert dim % 4 == 0, 'dim must be divisible by 4'

    row_ids = torch.arange(num_rows)
    col_ids = torch.arange(num_cols)

    grid_rows, grid_cols = torch.meshgrid(row_ids, col_ids, indexing='ij')

    grid_rows = grid_rows.flatten().float()
    grid_cols = grid_cols.flatten().float()

    # frequency bands for each dimension slot
    freq_count = dim // 4
    freq_bands = torch.arange(freq_count).float() / max(freq_count - 1, 1)
    freq_bands = 1.0 / (10000.0 ** freq_bands)

    row_enc = grid_rows[:, None] * freq_bands[None, :]
    col_enc = grid_cols[:, None] * freq_bands[None, :]

    pos_embed = torch.cat([
        row_enc.sin(),
        row_enc.cos(),
        col_enc.sin(),
        col_enc.cos(),
    ], dim=1)

    return pos_embed


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            qkv
        )

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = self.softmax(scores)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, head_dim, mlp_dim):
        super().__init__()
        self.final_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads, head_dim),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.final_norm(x)


class SimpleViT(nn.Module):
    # no cls token, just mean pool
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        num_heads,
        mlp_dim,
        channels=3,
        head_dim=64
    ):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)

        assert image_h % patch_h == 0 and image_w % patch_w == 0, \
            'image size must divide evenly by patch size'

        patch_dim = channels * patch_h * patch_w
        num_patch_rows = image_h // patch_h
        num_patch_cols = image_w // patch_w

        # flatten patches then project
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # fixed sincos positions
        self.register_buffer(
            'pos_embedding',
            build_sincos_embedding(num_patch_rows, num_patch_cols, dim)
        )

        self.transformer = Transformer(dim, depth, num_heads, head_dim, mlp_dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img)
        x = x + self.pos_embedding

        x = self.transformer(x)
        # average all patch tokens
        x = x.mean(dim=1)

        return self.classifier(x)
