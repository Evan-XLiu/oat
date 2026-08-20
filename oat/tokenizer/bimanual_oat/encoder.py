import torch
import torch.nn as nn

from oat.tokenizer.oat.encoder.register_encoder import create_causal_last_mask
from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.pos_emb import PositionalEmbeddingAdder
from oat.tokenizer.oat.model.sample_emb import SampleEmbedder
from oat.tokenizer.oat.model.transformer import Transformer


class DualBranchRegisterEncoder(nn.Module):
    """OAT register encoder with shared-arm common/relative feature fusion.

    The input must already be normalized and expressed in the canonical
    bimanual convention. The left and right arms share the same action
    embedding. Common and relative features are formed before temporal
    positional embeddings are added, preventing the relative subtraction from
    cancelling positional information.
    """

    def __init__(
        self,
        arm_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        latent_dim: int,
        num_registers: int,
    ):
        super().__init__()

        self.arm_dim = arm_dim
        self.sample_dim = 2 * arm_dim
        self.num_registers = num_registers

        self.shared_arm_emb = SampleEmbedder(arm_dim, emb_dim)
        self.common_pos_emb = PositionalEmbeddingAdder(
            emb_dim, max_sizes=[sample_horizon]
        )
        self.relative_pos_emb = PositionalEmbeddingAdder(
            emb_dim, max_sizes=[sample_horizon]
        )
        self.fusion = SampleEmbedder(2 * emb_dim, emb_dim)

        self.registers = nn.Parameter(torch.randn(num_registers, emb_dim))
        self.transformer = Transformer(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
        )
        self.head = LinearHead(emb_dim, latent_dim)

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

        # Form the two bimanual modes before adding positional embeddings.
        common_feature = (left_feature + right_feature) / 2
        relative_feature = (left_feature - right_feature) / 2

        common_feature = self.common_pos_emb(common_feature)
        relative_feature = self.relative_pos_emb(relative_feature)
        return common_feature, relative_feature

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, 2 * arm_dim)
        # return: (B, num_registers, latent_dim)
        batch_size, horizon, _ = sample.shape
        common_feature, relative_feature = self.encode_arm_features(sample)
        fused_feature = self.fusion(
            torch.cat([common_feature, relative_feature], dim=-1)
        )

        x = torch.cat(
            [
                fused_feature,
                self.registers.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        attention_mask = create_causal_last_mask(
            horizon, self.num_registers, str(sample.device)
        )
        x = self.transformer(x, block_mask=attention_mask)
        return self.head(x[:, horizon:])
