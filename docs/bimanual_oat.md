# Bimanual OAT

`BimanualOATTok` adds a dual-arm common/relative representation without
changing the standard `OATTok` implementation.

## Action convention

Input actions are ordered as `[left, right]`, with seven dimensions per arm.
Before normalization, the right arm is mapped to the shared joint convention
with:

```text
[-1, 1, 1, 1, -1, -1, 1]
```

The final `1` leaves the gripper dimension unchanged. The sign transform is
self-inverse and is applied again after decoding.

## Encoder

For a normalized canonical action chunk `[B, 32, 14]`:

1. Split it into left and right chunks `[B, 32, 7]`.
2. Apply the same shared `Linear(7, 256)` to both arms.
3. Before positional embeddings, form:

   ```text
   common   = (left_feature + right_feature) / 2
   relative = (left_feature - right_feature) / 2
   ```

4. Add temporal positional embeddings to both branches.
5. Concatenate the branches and apply `Linear(512, 256)`.
6. Run the unchanged OAT register Transformer, latent head, and FSQ path.

The default configuration retains 8 registers and FSQ levels `[8, 5, 5, 5]`.

## Decoder

The OAT single-pass Transformer decoder is retained. Its output feeds two
heads that reconstruct normalized canonical common and relative actions:

```text
left            = common + relative
canonical_right = common - relative
```

After unnormalization, the right-arm convention is inverted and the original
`[B, 32, 14]` action layout is returned.

## Data and training

The Zarr action array must have shape `[N, 14]` in the original robot
convention. `BimanualZarrDataset` computes normalization statistics after
right-arm canonicalization and shares each paired joint's statistics across
the arms.

The default data path is `data/bimanual/train.zarr`. Override it as needed:

```bash
HYDRA_FULL_ERROR=1 uv run accelerate launch \
  scripts/run_workspace.py \
  --config-name=train_bimanual_oattok \
  task.tokenizer.dataset.zarr_path=/path/to/train.zarr
```

The standard OAT files and LIBERO configurations remain unchanged.
