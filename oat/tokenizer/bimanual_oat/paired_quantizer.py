import torch
import torch.nn as nn

from oat.tokenizer.oat.quantizer.fsq import FSQ


class PairedFSQ(nn.Module):
    """Pair two FSQ subcodes into one mixed-radix token ID.

    With the default common vocabulary of 40 and relative vocabulary of 25,
    each paired token has 40 * 25 = 1000 values, matching standard OAT.
    """

    def __init__(
        self,
        common_quantizer: FSQ,
        relative_quantizer: FSQ,
    ):
        super().__init__()
        self.common_quantizer = common_quantizer
        self.relative_quantizer = relative_quantizer
        self.common_codebook_size = common_quantizer.codebook_size
        self.relative_codebook_size = relative_quantizer.codebook_size
        self.codebook_size = (
            self.common_codebook_size * self.relative_codebook_size
        )

    def pair_tokens(
        self,
        common_tokens: torch.Tensor,
        relative_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if common_tokens.shape != relative_tokens.shape:
            raise ValueError(
                "common and relative token tensors must have the same shape, "
                f"got {common_tokens.shape} and {relative_tokens.shape}"
            )
        return common_tokens + self.common_codebook_size * relative_tokens

    def unpair_tokens(
        self,
        paired_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_tokens = paired_tokens % self.common_codebook_size
        relative_tokens = paired_tokens // self.common_codebook_size
        return common_tokens, relative_tokens

    def forward(
        self,
        common_latents: torch.Tensor,
        relative_latents: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        common_quantized, common_tokens = self.common_quantizer(common_latents)
        relative_quantized, relative_tokens = self.relative_quantizer(
            relative_latents
        )
        paired_tokens = self.pair_tokens(common_tokens, relative_tokens)
        return (common_quantized, relative_quantized), paired_tokens

    def indices_to_embedding(
        self,
        paired_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common_tokens, relative_tokens = self.unpair_tokens(paired_tokens)
        common_latents = self.common_quantizer.indices_to_embedding(common_tokens)
        relative_latents = self.relative_quantizer.indices_to_embedding(
            relative_tokens
        )
        return common_latents, relative_latents
