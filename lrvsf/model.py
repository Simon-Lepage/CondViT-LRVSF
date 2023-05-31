# Modified from : https://github.com/openai/CLIP/blob/main/clip/model.py

import torch
from torch import nn

from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if self.weight.dtype != x.dtype:
            orig_type = x.dtype
            ret = super().forward(x.type(self.weight.dtype))
            return ret.type(orig_type)
        else:
            return super().forward(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c_fc",
                        nn.Linear(d_model, d_model * 4),
                    ),
                    ("gelu", QuickGELU()),
                    (
                        "c_proj",
                        nn.Linear(d_model * 4, d_model),
                    ),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ConditionalViT(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        n_categories: int = None,
        **kwargs,
    ):
        if kwargs:
            logger.warning(f"Got unused kwargs : {kwargs}")

        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.n_categories = n_categories
        if self.n_categories:
            self.c_embedding = nn.Embedding(self.n_categories, width)
            self.c_pos_embedding = nn.Parameter(scale * torch.randn(1, width))

        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, imgs: torch.Tensor, c: torch.Tensor = None):
        """
        imgs : Batch of images
        c : category indices. 0 = "No given category".
        """

        x = self.conv1(imgs)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # [CLS, grid] + maybe Categories.
        tokens = [self.class_embedding.tile(x.shape[0], 1, 1), x]  # NLD
        pos_embed = [self.positional_embedding]  # LD

        if self.n_categories and c is not None:  # If c is None, we don't add the token
            tokens += [self.c_embedding(c).unsqueeze(1)]  # ND -> N1D
            pos_embed += [self.c_pos_embedding]  # 1D

        x = torch.cat(
            tokens,
            dim=1,
        )  # shape = [*, grid ** 2 + 1|2, width] = N(L|L+1)D
        pos_embed = torch.cat(pos_embed, dim=0).unsqueeze(0)  # 1(L|L+1)D

        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        x = x @ self.proj

        return x


# SIZES
B32_Params = {
    "input_resolution": 224,
    "patch_size": 32,
    "width": 768,
    "layers": 12,
    "heads": 12,
    "output_dim": 512,
}

B16_Params = {
    "input_resolution": 224,
    "patch_size": 16,
    "width": 768,
    "layers": 12,
    "heads": 12,
    "output_dim": 512,
}

params = {"B32": B32_Params, "B16": B16_Params}
