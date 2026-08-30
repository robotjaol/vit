import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(value):
    # make it tuple
    return value if isinstance(value, tuple) else (value, value)


class FeedForward(nn.Module):
    # expand then contract
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        # scale prevents huge dot products
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

        # one linear for q, k, v
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        need_projection = not (num_heads == 1 and head_dim == dim)
        if need_projection:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # split into q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            qkv
        )

        # scaled dot product
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = self.softmax(scores)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.final_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                Attention(dim, num_heads, head_dim, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.blocks:
            # residual connections
            x = attn(x) + x
            x = ff(x) + x
        return self.final_norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        num_heads,
        mlp_dim,
        pool='cls',
        channels=3,
        head_dim=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)

        assert image_h % patch_h == 0 and image_w % patch_w == 0, \
            'image size must be divisible by patch size'

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        assert pool in ('cls', 'mean'), 'pool must be cls or mean'

        # cls token adds one extra position
        num_cls = 1 if pool == 'cls' else 0

        # cut image into flat patches then project to dim
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, num_heads, head_dim, mlp_dim, dropout)

        self.pool = pool
        self.classifier = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.patch_embed(img)

        # prepend cls token
        if self.cls_token.shape[0] > 0:
            cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=batch)
            x = torch.cat((cls_tokens, x), dim=1)

        seq_len = x.shape[1]
        x = x + self.pos_embedding[:seq_len]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        if self.classifier is None:
            return x

        # pick cls token or average all patches
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return self.classifier(x)
