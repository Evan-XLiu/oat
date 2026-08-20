from typing import Sequence

import numpy as np
import torch

from oat.dataset.zarr_dataset import ZarrDataset
from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.tokenizer.bimanual_oat.convention import DEFAULT_RIGHT_ARM_SIGN


class BimanualZarrDataset(ZarrDataset):
    """Zarr dataset whose action normalizer follows the bimanual convention.

    Left and canonicalized-right corresponding joints share normalization
    statistics so the shared arm embedding receives aligned coordinates.
    Returned samples remain in the original convention; BimanualOATTok applies
    and inverts the convention at its API boundary.
    """

    def __init__(
        self,
        *args,
        right_arm_sign: Sequence[float] = DEFAULT_RIGHT_ARM_SIGN,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sign = np.asarray(right_arm_sign, dtype=np.float32)
        if sign.ndim != 1 or not np.all(np.isin(sign, [-1.0, 1.0])):
            raise ValueError("right_arm_sign must be a one-dimensional sequence of -1 and 1")
        self.right_arm_sign = tuple(float(value) for value in sign)

    def get_normalizer(self, mode="limits", **kwargs):
        action = self.replay_buffer[self.action_key][:].astype(np.float32)
        arm_dim = len(self.right_arm_sign)
        if action.shape[-1] != 2 * arm_dim:
            raise ValueError(
                f"expected action dimension {2 * arm_dim}, got {action.shape[-1]}"
            )

        left = action[..., :arm_dim]
        canonical_right = action[..., arm_dim:] * np.asarray(
            self.right_arm_sign, dtype=np.float32
        )
        pooled_arm_action = np.concatenate([left, canonical_right], axis=0)

        arm_normalizer = SingleFieldLinearNormalizer()
        arm_normalizer.fit(
            pooled_arm_action,
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        params = arm_normalizer.params_dict
        bimanual_normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=torch.cat([params["scale"], params["scale"]], dim=0),
            offset=torch.cat([params["offset"], params["offset"]], dim=0),
            input_stats_dict={
                name: torch.cat([value, value], dim=0)
                for name, value in params["input_stats"].items()
            },
        )

        normalizer = LinearNormalizer()
        normalizer["action"] = bimanual_normalizer
        for key in self.numeric_obs_keys:
            field_normalizer = SingleFieldLinearNormalizer()
            field_normalizer.fit(
                self.replay_buffer[key],
                last_n_dims=1,
                mode=mode,
                **kwargs,
            )
            normalizer[key] = field_normalizer
        return normalizer
