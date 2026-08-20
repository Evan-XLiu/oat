import torch

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.bimanual_oat.convention import apply_bimanual_convention
from oat.tokenizer.bimanual_oat.decoder import DualBranchSinglePassDecoder
from oat.tokenizer.bimanual_oat.encoder import DualBranchRegisterEncoder
from oat.tokenizer.bimanual_oat.tokenizer import BimanualOATTok
from oat.tokenizer.oat.quantizer.fsq import FSQ


def make_tokenizer() -> BimanualOATTok:
    encoder = DualBranchRegisterEncoder(
        arm_dim=7,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=1,
        pdropout=0.0,
        latent_dim=4,
        num_registers=4,
    )
    decoder = DualBranchSinglePassDecoder(
        arm_dim=7,
        sample_horizon=8,
        emb_dim=32,
        head_dim=8,
        depth=1,
        pdropout=0.0,
        token_dropout_mode="pow2",
        use_causal_decoder=True,
        latent_dim=4,
        latent_horizon=4,
    )
    tokenizer = BimanualOATTok(
        encoder=encoder,
        decoder=decoder,
        quantizer=FSQ(levels=[8, 5, 5, 5]),
    )

    samples = torch.randn(32, 8, 14)
    canonical_samples = tokenizer.canonicalize(samples)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": canonical_samples})
    tokenizer.set_normalizer(normalizer)
    return tokenizer


def test_right_arm_convention_is_self_inverse():
    tokenizer = make_tokenizer()
    samples = torch.randn(2, 8, 14)
    transformed = apply_bimanual_convention(samples, tokenizer.right_arm_sign)
    restored = apply_bimanual_convention(transformed, tokenizer.right_arm_sign)
    torch.testing.assert_close(restored, samples)


def test_common_relative_features_are_created_before_position_embedding():
    tokenizer = make_tokenizer()
    sample = torch.randn(2, 8, 14)
    canonical_sample = tokenizer.canonicalize(sample)
    normalized_sample = tokenizer.normalizer["action"].normalize(canonical_sample)

    left, right = normalized_sample.split(7, dim=-1)
    left_feature = tokenizer.encoder.shared_arm_emb(left)
    right_feature = tokenizer.encoder.shared_arm_emb(right)
    expected_common = tokenizer.encoder.common_pos_emb(
        (left_feature + right_feature) / 2
    )
    expected_relative = tokenizer.encoder.relative_pos_emb(
        (left_feature - right_feature) / 2
    )
    common, relative = tokenizer.encoder.encode_arm_features(normalized_sample)

    torch.testing.assert_close(common, expected_common)
    torch.testing.assert_close(relative, expected_relative)


def test_bimanual_tokenizer_shapes_and_backward():
    tokenizer = make_tokenizer()
    samples = torch.randn(2, 8, 14)

    latents, tokens = tokenizer.encode(samples)
    assert latents.shape == (2, 4, 4)
    assert tokens.shape == (2, 4)
    assert tokenizer.detokenize(tokens).shape == samples.shape

    loss = tokenizer({"action": samples})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert tokenizer.encoder.fusion.patch_proj.weight.grad is not None
