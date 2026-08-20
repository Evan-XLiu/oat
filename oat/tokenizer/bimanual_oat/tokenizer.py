from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.bimanual_oat.convention import (
    DEFAULT_RIGHT_ARM_SIGN,
    apply_bimanual_convention,
    as_sign_tensor,
)
from oat.tokenizer.bimanual_oat.decoder import DualBranchSinglePassDecoder
from oat.tokenizer.bimanual_oat.encoder import DualBranchRegisterEncoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.tokenizer.oat.tokenizer import pad_token_seq


class BimanualOATTok(BaseTokenizer):
    """Ordered action tokenizer for canonicalized bimanual action chunks."""

    def __init__(
        self,
        encoder: DualBranchRegisterEncoder,
        decoder: DualBranchSinglePassDecoder,
        quantizer: FSQ,
        right_arm_sign: Sequence[float] = DEFAULT_RIGHT_ARM_SIGN,
    ):
        super().__init__()
        if encoder.arm_dim != decoder.arm_dim:
            raise ValueError(
                f"encoder arm_dim {encoder.arm_dim} != decoder arm_dim {decoder.arm_dim}"
            )

        sign = as_sign_tensor(right_arm_sign)
        if sign.numel() != encoder.arm_dim:
            raise ValueError(
                f"right-arm convention dimension {sign.numel()} != arm_dim {encoder.arm_dim}"
            )

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.register_buffer("right_arm_sign", sign, persistent=True)

        self.arm_dim = encoder.arm_dim
        self.action_dim = 2 * self.arm_dim
        self.latent_horizon = decoder.latent_horizon

    def get_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        decay_params = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.dim() >= 2
        ]
        nodecay_params = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.dim() < 2
        ]
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def canonicalize(self, samples: torch.Tensor) -> torch.Tensor:
        return apply_bimanual_convention(samples, self.right_arm_sign)

    def decanonicalize(self, samples: torch.Tensor) -> torch.Tensor:
        # The sign-only right-arm convention is self-inverse.
        return apply_bimanual_convention(samples, self.right_arm_sign)

    def forward(self, batch) -> torch.Tensor:
        canonical_samples = self.canonicalize(batch["action"])
        normalized_samples = self.normalizer["action"].normalize(canonical_samples)
        latents = self.encoder(normalized_samples)
        latents, _ = self.quantizer(latents)
        reconstructions = self.decoder(latents)
        return F.mse_loss(reconstructions, normalized_samples)

    def encode(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        canonical_samples = self.canonicalize(samples)
        normalized_samples = self.normalizer["action"].normalize(canonical_samples)
        latents = self.encoder(normalized_samples)
        return self.quantizer(latents)

    def decode(
        self,
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]
        if not all(k <= self.latent_horizon for k in eval_keep_k):
            raise ValueError(f"all eval_keep_k values must be <= {self.latent_horizon}")

        normalized_canonical_samples = self.decoder(
            latents, eval_keep_k=eval_keep_k
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
        _, tokens = self.encode(samples)
        return tokens

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
