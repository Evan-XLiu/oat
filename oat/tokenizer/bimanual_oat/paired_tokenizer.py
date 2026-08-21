from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from oat.tokenizer.bimanual_oat.convention import DEFAULT_RIGHT_ARM_SIGN
from oat.tokenizer.bimanual_oat.paired_decoder import PairedSinglePassDecoder
from oat.tokenizer.bimanual_oat.paired_encoder import PairedRegisterEncoder
from oat.tokenizer.bimanual_oat.paired_quantizer import PairedFSQ
from oat.tokenizer.bimanual_oat.tokenizer import BimanualOATTok
from oat.tokenizer.oat.tokenizer import pad_token_seq


PairedLatents = Tuple[torch.Tensor, torch.Tensor]


class PairedBimanualOATTok(BimanualOATTok):
    """Bimanual OAT with independent common/relative streams and paired IDs."""

    def __init__(
        self,
        encoder: PairedRegisterEncoder,
        decoder: PairedSinglePassDecoder,
        quantizer: PairedFSQ,
        right_arm_sign: Sequence[float] = DEFAULT_RIGHT_ARM_SIGN,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            right_arm_sign=right_arm_sign,
        )
        if encoder.common_latent_dim != quantizer.common_quantizer.dim:
            raise ValueError("common encoder and FSQ latent dimensions do not match")
        if encoder.relative_latent_dim != quantizer.relative_quantizer.dim:
            raise ValueError("relative encoder and FSQ latent dimensions do not match")

    def forward(self, batch) -> torch.Tensor:
        canonical_samples = self.canonicalize(batch["action"])
        normalized_samples = self.normalizer["action"].normalize(canonical_samples)
        common_latents, relative_latents = self.encoder(normalized_samples)
        (common_quantized, relative_quantized), _ = self.quantizer(
            common_latents, relative_latents
        )
        reconstructions = self.decoder(common_quantized, relative_quantized)
        return F.mse_loss(reconstructions, normalized_samples)

    def encode(
        self,
        samples: torch.Tensor,
    ) -> tuple[PairedLatents, torch.Tensor]:
        canonical_samples = self.canonicalize(samples)
        normalized_samples = self.normalizer["action"].normalize(canonical_samples)
        common_latents, relative_latents = self.encoder(normalized_samples)
        return self.quantizer(common_latents, relative_latents)

    def decode(
        self,
        latents: PairedLatents,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        common_latents, relative_latents = latents
        if eval_keep_k is None:
            eval_keep_k = [common_latents.shape[1]] * common_latents.shape[0]
        if not all(k <= self.latent_horizon for k in eval_keep_k):
            raise ValueError(f"all eval_keep_k values must be <= {self.latent_horizon}")

        normalized_canonical_samples = self.decoder(
            common_latents,
            relative_latents,
            eval_keep_k=eval_keep_k,
        )
        canonical_samples = self.normalizer["action"].unnormalize(
            normalized_canonical_samples
        )
        return self.decanonicalize(canonical_samples)

    def autoencode(
        self,
        samples: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        latents, _ = self.encode(samples)
        return self.decode(latents, eval_keep_k=eval_keep_k)

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        _, paired_tokens = self.encode(samples)
        return paired_tokens

    def detokenize(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(tokens, list):
            token_lengths = [token.shape[1] for token in tokens]
            tokens = torch.cat(
                [pad_token_seq(token, self.latent_horizon) for token in tokens],
                dim=0,
            )
        elif isinstance(tokens, torch.Tensor):
            token_lengths = [tokens.shape[1]] * tokens.shape[0]
            if tokens.shape[-1] < self.latent_horizon:
                tokens = pad_token_seq(tokens, self.latent_horizon)
        else:
            raise ValueError(f"unknown token type {type(tokens)}")

        latents = self.quantizer.indices_to_embedding(tokens)
        return self.decode(latents, eval_keep_k=token_lengths)
