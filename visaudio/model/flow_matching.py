import logging
from typing import Callable, Optional

import torch

from torchdiffeq import odeint

log = logging.getLogger()


# Partially from https://github.com/gle-bellier/flow-matching
class FlowMatching:
    def __init__(self, min_sigma: float = 0.0, inference_mode="euler", num_steps: int = 25):
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        if not 0 <= min_sigma < 1:
            raise ValueError("min_sigma must be in [0, 1).")
        if inference_mode == "euler" and num_steps < 1:
            raise ValueError("num_steps must be positive.")

        assert self.inference_mode in ["euler", "adaptive"]
        if self.inference_mode == "adaptive" and num_steps > 0:
            log.info("The number of steps is ignored in adaptive inference mode ")

    def get_conditional_flow(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t[:, None, None].expand_as(x0)
        return (1 - (1 - self.min_sigma) * t) * x0 + t * x1

    def mse_loss(
        self, predicted_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        reduce_dim = list(range(1, len(predicted_v.shape)))
        target_v = x1 - (1 - self.min_sigma) * x0
        return target_v, (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def avg_mse_loss(self, pred_left, pred_right, target_left, target_right):
        pred_avg = (pred_left + pred_right) / 2
        target_avg = (target_left + target_right) / 2

        B = pred_avg.shape[0]
        pred_flat = pred_avg.reshape(B, -1)
        target_flat = target_avg.reshape(B, -1)

        loss = ((pred_flat - target_flat) ** 2).mean(dim=1)  # [B]
        return loss

    def latent_corr_loss(
        self,
        pred_left: torch.Tensor,
        pred_right: torch.Tensor,
        target_left: torch.Tensor,
        target_right: torch.Tensor,
        eps=1e-8,
    ) -> torch.Tensor:
        B = pred_left.shape[0]
        pred_l_flat = pred_left.reshape(B, -1)
        pred_r_flat = pred_right.reshape(B, -1)
        target_l_flat = target_left.reshape(B, -1)
        target_r_flat = target_right.reshape(B, -1)

        def corr(l, r):
            l = l - l.mean(dim=1, keepdim=True)
            r = r - r.mean(dim=1, keepdim=True)
            num = torch.sum(l * r, dim=1)
            denom = torch.sqrt(torch.sum(l**2, dim=1) * torch.sum(r**2, dim=1)) + eps
            return num / denom

        pred_corr = corr(pred_l_flat, pred_r_flat)
        target_corr = corr(target_l_flat, target_r_flat)

        loss = (pred_corr - target_corr) ** 2  # [B]
        return loss

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        Cs: list[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = torch.empty_like(x1).normal_(generator=generator)

        xt = self.get_conditional_flow(x0, x1, t)
        return x0, x1, xt, Cs

    def to_prior(self, fn: Callable, x1_left: torch.Tensor, x1_right: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1_left, x1_right, 1, 0)

    def to_data(self, fn: Callable, x0_left: torch.Tensor, x0_right: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0_left, x0_right, 0, 1)

    def run_t0_to_t1(
        self, fn: Callable, x0_left: torch.Tensor, x0_right: torch.Tensor, t0: float, t1: float
    ) -> torch.Tensor:

        if self.inference_mode == "adaptive":
            times = x0_left.new_tensor([t0, t1])
            left, right = odeint(lambda t, pair: fn(t, *pair), (x0_left, x0_right), times)
            return left[-1], right[-1]
        elif self.inference_mode == "euler":
            x_left = x0_left
            x_right = x0_right
            steps = torch.linspace(
                t0, t1, self.num_steps + 1, device=x0_left.device, dtype=x0_left.dtype
            )
            for ti, t in enumerate(steps[:-1]):
                flow_left, flow_right = fn(t, x_left, x_right)
                next_t = steps[ti + 1]
                dt = next_t - t
                x_left = x_left + dt * flow_left
                x_right = x_right + dt * flow_right

        return x_left, x_right
