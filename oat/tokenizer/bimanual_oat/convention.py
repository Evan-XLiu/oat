from typing import Sequence

import torch


DEFAULT_RIGHT_ARM_SIGN = (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0)


def as_sign_tensor(
    sign: Sequence[float],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    sign_tensor = torch.as_tensor(sign, device=device, dtype=dtype)
    if sign_tensor.ndim != 1:
        raise ValueError(f"right-arm convention must be one-dimensional, got {sign_tensor.shape}")
    if not torch.all((sign_tensor == 1) | (sign_tensor == -1)):
        raise ValueError("right-arm convention may only contain -1 and 1")
    return sign_tensor


def apply_right_arm_convention(
    right_action: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    """Map a right-arm action to/from the shared canonical joint convention.

    The configured sign transform is self-inverse, so the same operation is
    used both before encoding and after decoding.
    """
    if right_action.shape[-1] != sign.numel():
        raise ValueError(
            f"expected right-arm action dimension {sign.numel()}, "
            f"got {right_action.shape[-1]}"
        )
    return right_action * sign.to(device=right_action.device, dtype=right_action.dtype)


def apply_bimanual_convention(
    action: torch.Tensor,
    sign: torch.Tensor,
) -> torch.Tensor:
    """Apply the right-arm convention to a concatenated [left, right] action."""
    arm_dim = sign.numel()
    if action.shape[-1] != 2 * arm_dim:
        raise ValueError(
            f"expected bimanual action dimension {2 * arm_dim}, got {action.shape[-1]}"
        )
    left, right = action.split(arm_dim, dim=-1)
    right = apply_right_arm_convention(right, sign)
    return torch.cat([left, right], dim=-1)
