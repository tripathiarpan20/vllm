"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from vllm._C import ops


def _rms_norm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        x = x + residual
        residual = x
    orig_dtype = x.dtype
    x = x.float()

    mean = torch.mean(x * x, dim=-1, keepdim=True)
    output = x * torch.rsqrt(mean + eps)
    output = output.to(orig_dtype)
    output = output * weight
    if residual is None:
        return output
    else:
        return output, residual


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return _rms_norm(x, residual, self.weight.data, self.variance_epsilon)

    def _forward_with_custom_op(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # FIXME(woosuk): This is a hack.
        is_prompt = x.shape[1] > 1
        if is_prompt:
            return self._forward_with_custom_op(x, residual)
        else:
            return self._forward(x, residual)
