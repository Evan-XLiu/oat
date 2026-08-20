from typing import List, Optional

import einops
import torch
import torch.nn as nn

from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.linear import LinearLayer
from oat.tokenizer.oat.model.pos_emb import PositionalEmbedding, PositionalEmbeddingAdder
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout


class DualBranchSinglePassDecoder(nn.Module):
    """OAT single-pass decoder with common and relative action heads."""

    def __init__(
        self,
        arm_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        token_dropout_mode: str,
        use_causal_decoder: bool,
        latent_dim: int,
        latent_horizon: int,
    ):
        super().__init__()

        self.sample_pos_emb = PositionalEmbedding(
            emb_dim,
            max_sizes=[sample_horizon],
        )
        self.latent_pos_emb = PositionalEmbeddingAdder(
            emb_dim,
            max_sizes=[latent_horizon],
        )
        self.nested_dropout = MaskedNestedDropout(
            emb_dim,
            size_sampling_mode=token_dropout_mode,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_dim,
                nhead=emb_dim // head_dim,
                dim_feedforward=4 * emb_dim,
                dropout=pdropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        self.common_head = LinearHead(emb_dim, arm_dim)
        self.relative_head = LinearHead(emb_dim, arm_dim)

        self.arm_dim = arm_dim
        self.sample_dim = 2 * arm_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.use_causal_decoder = use_causal_decoder

    def decode_modes(
        self,
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_pos_emb(shape=[self.sample_horizon]).expand(
            latents.shape[0], -1, -1
        )
        x = einops.rearrange(x, "B D T -> B T D")

        latents = self.latent_proj(latents)
        latents = self.latent_pos_emb(latents)
        latents = self.nested_dropout(latents, eval_keep_k=eval_keep_k)

        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            x = self.decoder(x, latents, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, latents)

        common_action = self.common_head(x)
        relative_action = self.relative_head(x)
        return common_action, relative_action

    def forward(
        self,
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        common_action, relative_action = self.decode_modes(
            latents, eval_keep_k=eval_keep_k
        )
        left_action = common_action + relative_action
        canonical_right_action = common_action - relative_action
        return torch.cat([left_action, canonical_right_action], dim=-1)
