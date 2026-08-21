from typing import List, Optional

import einops
import torch
import torch.nn as nn

from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.linear import LinearLayer
from oat.tokenizer.oat.model.pos_emb import PositionalEmbedding, PositionalEmbeddingAdder
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout


class PairedSinglePassDecoder(nn.Module):
    """Decode common/relative streams independently, then reconstruct both arms."""

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
        common_latent_dim: int,
        relative_latent_dim: int,
        latent_horizon: int,
    ):
        super().__init__()
        self.sample_pos_emb = PositionalEmbedding(
            emb_dim, max_sizes=[sample_horizon]
        )
        self.common_latent_proj = LinearLayer(common_latent_dim, emb_dim)
        self.relative_latent_proj = LinearLayer(relative_latent_dim, emb_dim)
        self.latent_pos_emb = PositionalEmbeddingAdder(
            emb_dim, max_sizes=[latent_horizon]
        )
        self.branch_embeddings = nn.Parameter(torch.randn(2, emb_dim))

        self.common_nested_dropout = MaskedNestedDropout(
            emb_dim, size_sampling_mode=token_dropout_mode
        )
        self.relative_nested_dropout = MaskedNestedDropout(
            emb_dim, size_sampling_mode=token_dropout_mode
        )
        self.shared_decoder = nn.TransformerDecoder(
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
        self.shared_action_head = LinearHead(emb_dim, arm_dim)

        self.arm_dim = arm_dim
        self.sample_dim = 2 * arm_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.token_dropout_mode = token_dropout_mode
        self.use_causal_decoder = use_causal_decoder

    def _synchronized_nested_dropout(
        self,
        common_latents: torch.Tensor,
        relative_latents: torch.Tensor,
        eval_keep_k: Optional[List[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_dropout_mode == "disable":
            return common_latents, relative_latents

        batch_size, horizon, _ = common_latents.shape
        if relative_latents.shape[:2] != (batch_size, horizon):
            raise ValueError("common and relative latent horizons must match")

        if self.training:
            keep_ks = self.common_nested_dropout.sample_keep_k(
                batch_size, horizon, common_latents.device
            )
        elif eval_keep_k is not None:
            if len(eval_keep_k) != batch_size:
                raise ValueError("eval_keep_k length must equal batch size")
            keep_ks = torch.as_tensor(eval_keep_k, device=common_latents.device)
        else:
            return common_latents, relative_latents

        mask = keep_ks.unsqueeze(1) <= torch.arange(
            horizon, device=common_latents.device
        ).unsqueeze(0)
        mask = mask.unsqueeze(-1)
        common_mask_token = self.common_nested_dropout.dropout_mask_token.view(
            1, 1, -1
        )
        relative_mask_token = (
            self.relative_nested_dropout.dropout_mask_token.view(1, 1, -1)
        )
        common_latents = torch.where(mask, common_mask_token, common_latents)
        relative_latents = torch.where(mask, relative_mask_token, relative_latents)
        return common_latents, relative_latents

    def _decode_stream(
        self,
        memory: torch.Tensor,
        branch_index: int,
    ) -> torch.Tensor:
        query = self.sample_pos_emb(shape=[self.sample_horizon]).expand(
            memory.shape[0], -1, -1
        )
        query = einops.rearrange(query, "B D T -> B T D")
        branch_embedding = self.branch_embeddings[branch_index].view(1, 1, -1)
        query = query + branch_embedding
        memory = memory + branch_embedding

        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                query.size(1), device=query.device
            )
            return self.shared_decoder(
                query,
                memory,
                tgt_mask=mask,
                tgt_is_causal=True,
            )
        return self.shared_decoder(query, memory)

    def decode_modes(
        self,
        common_latents: torch.Tensor,
        relative_latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_memory = self.latent_pos_emb(
            self.common_latent_proj(common_latents)
        )
        relative_memory = self.latent_pos_emb(
            self.relative_latent_proj(relative_latents)
        )
        common_memory, relative_memory = self._synchronized_nested_dropout(
            common_memory,
            relative_memory,
            eval_keep_k=eval_keep_k,
        )
        common_feature = self._decode_stream(common_memory, branch_index=0)
        relative_feature = self._decode_stream(relative_memory, branch_index=1)
        return common_feature, relative_feature

    def forward(
        self,
        common_latents: torch.Tensor,
        relative_latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        common_feature, relative_feature = self.decode_modes(
            common_latents,
            relative_latents,
            eval_keep_k=eval_keep_k,
        )
        left_feature = common_feature + relative_feature
        canonical_right_feature = common_feature - relative_feature
        left_action = self.shared_action_head(left_feature)
        canonical_right_action = self.shared_action_head(canonical_right_feature)
        return torch.cat([left_action, canonical_right_action], dim=-1)
