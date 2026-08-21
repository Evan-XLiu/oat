import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.bimanual_oat.paired_decoder import PairedSinglePassDecoder
from oat.tokenizer.bimanual_oat.paired_encoder import PairedRegisterEncoder
from oat.tokenizer.bimanual_oat.paired_quantizer import PairedFSQ
from oat.tokenizer.bimanual_oat.paired_tokenizer import PairedBimanualOATTok
from oat.tokenizer.oat.quantizer.fsq import FSQ


def make_tokenizer(token_horizon: int = 4) -> PairedBimanualOATTok:
    encoder = PairedRegisterEncoder(
        arm_dim=7,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=1,
        pdropout=0.0,
        common_latent_dim=2,
        relative_latent_dim=2,
        num_registers=token_horizon,
    )
    decoder = PairedSinglePassDecoder(
        arm_dim=7,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=1,
        pdropout=0.0,
        token_dropout_mode="pow2",
        use_causal_decoder=True,
        common_latent_dim=2,
        relative_latent_dim=2,
        latent_horizon=token_horizon,
    )
    quantizer = PairedFSQ(
        common_quantizer=FSQ(levels=[8, 5]),
        relative_quantizer=FSQ(levels=[5, 5]),
    )
    tokenizer = PairedBimanualOATTok(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
    )

    samples = torch.randn(32, 8, 14)
    canonical_samples = tokenizer.canonicalize(samples)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": canonical_samples})
    tokenizer.set_normalizer(normalizer)
    return tokenizer


def test_paired_fsq_has_standard_oat_vocabulary_and_is_invertible():
    quantizer = PairedFSQ(
        common_quantizer=FSQ(levels=[8, 5]),
        relative_quantizer=FSQ(levels=[5, 5]),
    )
    assert quantizer.common_codebook_size == 40
    assert quantizer.relative_codebook_size == 25
    assert quantizer.codebook_size == 1000

    paired = torch.arange(1000)
    common, relative = quantizer.unpair_tokens(paired)
    torch.testing.assert_close(quantizer.pair_tokens(common, relative), paired)


def test_paired_tokenizer_exposes_k_tokens_for_k_plus_k_registers():
    samples = torch.randn(2, 8, 14)

    for token_horizon in (2, 4, 8):
        tokenizer = make_tokenizer(token_horizon=token_horizon)
        (common_latents, relative_latents), paired_tokens = tokenizer.encode(
            samples
        )

        assert common_latents.shape == (2, token_horizon, 2)
        assert relative_latents.shape == (2, token_horizon, 2)
        assert paired_tokens.shape == (2, token_horizon)
        assert paired_tokens.max() < 1000
        assert tokenizer.detokenize(paired_tokens).shape == samples.shape


def test_paired_tokenizer_loss_backpropagates_through_both_streams():
    tokenizer = make_tokenizer(token_horizon=4)
    samples = torch.randn(2, 8, 14)
    loss = tokenizer({"action": samples})

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert tokenizer.encoder.common_registers.grad is not None
    assert tokenizer.encoder.relative_registers.grad is not None
