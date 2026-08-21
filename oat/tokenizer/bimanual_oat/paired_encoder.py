import torch
import torch.nn as nn

from oat.tokenizer.oat.encoder.register_encoder import create_causal_last_mask
from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.pos_emb import PositionalEmbeddingAdder
from oat.tokenizer.oat.model.sample_emb import SampleEmbedder
from oat.tokenizer.oat.model.transformer import Transformer


class PairedRegisterEncoder(nn.Module):
    """Independent common/relative OAT streams with shared encoder weights."""

    def __init__(
        self,
        arm_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        common_latent_dim: int,
        relative_latent_dim: int,
        num_registers: int,
    ):
        super().__init__()
        self.arm_dim = arm_dim
        self.sample_dim = 2 * arm_dim
        self.num_registers = num_registers

        self.shared_arm_emb = SampleEmbedder(arm_dim, emb_dim)
        self.temporal_pos_emb = PositionalEmbeddingAdder(
            emb_dim, max_sizes=[sample_horizon]
        )
        self.branch_embeddings = nn.Parameter(torch.randn(2, emb_dim))

        self.common_registers = nn.Parameter(torch.randn(num_registers, emb_dim))
        self.relative_registers = nn.Parameter(
            torch.randn(num_registers, emb_dim)
        )
        self.shared_transformer = Transformer(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
        )
        self.common_head = LinearHead(emb_dim, common_latent_dim)
        self.relative_head = LinearHead(emb_dim, relative_latent_dim)

        self.common_latent_dim = common_latent_dim
        self.relative_latent_dim = relative_latent_dim

    def encode_arm_features(
        self,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample.shape[-1] != self.sample_dim:
            raise ValueError(
                f"expected canonical bimanual action dimension {self.sample_dim}, "
                f"got {sample.shape[-1]}"
            )

        left, right = sample.split(self.arm_dim, dim=-1)
        left_feature = self.shared_arm_emb(left)
        right_feature = self.shared_arm_emb(right)

        # Common/relative modes are constructed before positional embeddings.
        common_feature = (left_feature + right_feature) / 2
        relative_feature = (left_feature - right_feature) / 2
        common_feature = self.temporal_pos_emb(common_feature)
        relative_feature = self.temporal_pos_emb(relative_feature)
        return common_feature, relative_feature

    def _encode_stream(
        self,
        feature: torch.Tensor,
        registers: torch.Tensor,
        branch_index: int,
        head: LinearHead,
    ) -> torch.Tensor:
        batch_size, horizon, _ = feature.shape
        branch_embedding = self.branch_embeddings[branch_index].view(1, 1, -1)
        feature = feature + branch_embedding
        expanded_registers = registers.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_registers = expanded_registers + branch_embedding

        x = torch.cat([feature, expanded_registers], dim=1)
        attention_mask = create_causal_last_mask(
            horizon, self.num_registers, str(feature.device)
        )
        x = self.shared_transformer(x, block_mask=attention_mask)
        return head(x[:, horizon:])

    def forward(
        self,
        sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_feature, relative_feature = self.encode_arm_features(sample)
        common_latents = self._encode_stream(
            common_feature,
            self.common_registers,
            branch_index=0,
            head=self.common_head,
        )
        relative_latents = self._encode_stream(
            relative_feature,
            self.relative_registers,
            branch_index=1,
            head=self.relative_head,
        )
        return common_latents, relative_latents
