import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from visaudio.ext.rotary_embeddings import compute_rope_rotations
from visaudio.model.embeddings import TimestepEmbedder
from visaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from visaudio.model.transformer_layers import FinalBlock, JointBlock, MMDitSingleBlock

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    pe_f_left: torch.Tensor
    pe_f_right: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


class PEInputProj(nn.Module):
    def __init__(self, in_channels, hidden_dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.conv1 = ChannelLastConv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.act = nn.SiLU()
        self.conv2d1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv_mlp = ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(H // 4 * W // 4 * hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        """
        x: (B, T, H, W, C)
        output: (B, T, hidden_dim)
        """
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> (b t) (h w) c")  # (B*T, H*W, C)
        x = self.act(self.conv1(x))  # (B*T, H*W, hidden_dim)
        x = rearrange(x, "bt (h w) c -> bt c h w", h=self.H, w=self.W)  # (B*T, hidden_dim, H, W)
        x = self.act(self.conv2d1(x))  # (B*T, hidden_dim, H//2, W//2)
        x = self.act(self.conv2d2(x))  # (B*T, hidden_dim, H//4, W//4)
        x = rearrange(
            x, "bt c h w -> bt (h w) c", h=self.H // 4, w=self.W // 4
        )  # (B*T, hidden_dim, H, W)
        x = self.conv_mlp(x)  # (B*T, hidden_dim, H, W)

        x = rearrange(x, "bt hw c -> bt (hw c)")
        x = F.silu(self.linear1(x))
        x = self.linear2(x)

        x = rearrange(x, "(b t) d -> b t d", b=B, t=T)
        return x


# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        clip_dim: int,
        sync_dim: int,
        pe_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int,
        fused_depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        latent_seq_len: int,
        clip_seq_len: int,
        sync_seq_len: int,
        pe_seq_len: int,
        size_len: int,
        text_seq_len: int = 77,
        left_latent_mean: Optional[torch.Tensor] = None,
        left_latent_std: Optional[torch.Tensor] = None,
        right_latent_mean: Optional[torch.Tensor] = None,
        right_latent_std: Optional[torch.Tensor] = None,
        empty_string_feat: Optional[torch.Tensor] = None,
        v2: bool = False,
    ) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._pe_seq_len = pe_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.pe_input_proj_left = PEInputProj(pe_dim, hidden_dim, size_len, size_len)
            self.pe_input_proj_right = PEInputProj(pe_dim, hidden_dim, size_len, size_len)

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)

        self.pe_cond_proj_left = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pe_cond_proj_right = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.pe_pos_emb_left = nn.Parameter(torch.randn(1, 1, size_len, size_len, pe_dim))
        self.pe_pos_emb_right = nn.Parameter(torch.randn(1, 1, size_len, size_len, pe_dim))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(
                hidden_dim, frequency_embedding_size=hidden_dim, max_period=1
            )
        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    pre_only=(i == depth - fused_depth - 1),
                )
                for i in range(depth - fused_depth)
            ]
        )

        self.fused_blocks_left = nn.ModuleList(
            [
                MMDitSingleBlock(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1
                )
                for i in range(fused_depth)
            ]
        )
        self.fused_blocks_right = nn.ModuleList(
            [
                MMDitSingleBlock(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1
                )
                for i in range(fused_depth)
            ]
        )

        if left_latent_mean is None:
            assert left_latent_std is None
            left_latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float("nan"))
            left_latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float("nan"))
            right_latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float("nan"))
            right_latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float("nan"))
        else:
            assert left_latent_std is not None and right_latent_std is not None
            assert left_latent_mean.numel() == latent_dim, (
                f"{left_latent_mean.numel()=} != {latent_dim=}"
            )
            assert right_latent_mean.numel() == latent_dim, (
                f"{right_latent_mean.numel()=} != {latent_dim=}"
            )
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.left_latent_mean = nn.Parameter(left_latent_mean.view(1, 1, -1), requires_grad=False)
        self.left_latent_std = nn.Parameter(left_latent_std.view(1, 1, -1), requires_grad=False)
        self.right_latent_mean = nn.Parameter(right_latent_mean.view(1, 1, -1), requires_grad=False)
        self.right_latent_std = nn.Parameter(right_latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)
        self.empty_pe_feat = nn.Parameter(
            torch.zeros(1, size_len, size_len, pe_dim), requires_grad=True
        )

        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(
            self._latent_seq_len,
            self.hidden_dim // self.num_heads,
            10000,
            freq_scaling=base_freq,
            device=self.device,
        )
        clip_rot = compute_rope_rotations(
            self._clip_seq_len,
            self.hidden_dim // self.num_heads,
            10000,
            freq_scaling=base_freq * self._latent_seq_len / self._clip_seq_len,
            device=self.device,
        )

        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.clip_rot = nn.Buffer(clip_rot, persistent=False)

    def update_seq_lengths(
        self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int, pe_seq_len: int
    ) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._pe_seq_len = pe_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)

        # Zero-out adaLN modulation layers in fused_blocks for left and right channels
        for block_left, block_right in zip(self.fused_blocks_left, self.fused_blocks_right):
            nn.init.constant_(block_left.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block_left.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block_right.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block_right.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.normal_(self.pe_pos_emb_left, mean=0.0, std=0.02)
        nn.init.normal_(self.pe_pos_emb_right, mean=0.0, std=0.02)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)
        nn.init.constant_(self.empty_pe_feat, 0)

    def normalize(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        return x_left.sub_(self.left_latent_mean).div_(self.left_latent_std), x_right.sub_(
            self.right_latent_mean
        ).div_(self.right_latent_std)

    def unnormalize(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        return x_left.mul_(self.left_latent_std).add_(self.left_latent_mean), x_right.mul_(
            self.right_latent_std
        ).add_(self.right_latent_mean)

    def preprocess_conditions(
        self, clip_f: torch.Tensor, sync_f: torch.Tensor, pe_f: torch.Tensor, text_f: torch.Tensor
    ) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert clip_f.shape[1] == self._clip_seq_len, f"{clip_f.shape=} {self._clip_seq_len=}"
        assert sync_f.shape[1] == self._sync_seq_len, f"{sync_f.shape=} {self._sync_seq_len=}"
        assert pe_f.shape[1] == self._pe_seq_len, f"{pe_f.shape=} {self._pe_seq_len=}"
        assert text_f.shape[1] == self._text_seq_len, f"{text_f.shape=} {self._text_seq_len=}"

        bs = clip_f.shape[0]

        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        pe_f_left = pe_f + self.pe_pos_emb_left  # (B, VN, H, W, D)
        pe_f_right = pe_f + self.pe_pos_emb_right  # (B, VN, H, W, D)

        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        pe_f_left = self.pe_input_proj_left(pe_f_left)  # (B, VN, D)
        pe_f_right = self.pe_input_proj_right(pe_f_right)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode="nearest-exact")
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # upsample the pe features to match the audio
        pe_f_left = pe_f_left.transpose(1, 2)  # (B, D, VN)
        pe_f_left = F.interpolate(pe_f_left, size=self._latent_seq_len, mode="nearest-exact")
        pe_f_left = pe_f_left.transpose(1, 2)  # (B, N, D)

        # upsample the pe features to match the audio
        pe_f_right = pe_f_right.transpose(1, 2)  # (B, D, VN)
        pe_f_right = F.interpolate(pe_f_right, size=self._latent_seq_len, mode="nearest-exact")
        pe_f_right = pe_f_right.transpose(1, 2)  # (B, N, D)

        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        return PreprocessedConditions(
            clip_f=clip_f,
            sync_f=sync_f,
            pe_f_left=pe_f_left,
            pe_f_right=pe_f_right,
            text_f=text_f,
            clip_f_c=clip_f_c,
            text_f_c=text_f_c,
        )

    def predict_flow(
        self,
        latent_left: torch.Tensor,
        latent_right: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
    ) -> torch.Tensor:
        """
        Predict flow for both left and right channels in a joint training setup.
        """
        assert latent_left.shape[1] == self._latent_seq_len, (
            f"{latent_left.shape=} {self._latent_seq_len=}"
        )
        assert latent_right.shape[1] == self._latent_seq_len, (
            f"{latent_right.shape=} {self._latent_seq_len=}"
        )

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        pe_f_left = conditions.pe_f_left
        pe_f_right = conditions.pe_f_right
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        latent_left = self.audio_input_proj(latent_left)  # (B, N, D)
        latent_right = self.audio_input_proj(latent_right)  # (B, N, D)

        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B, D)

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)

        extended_c = global_c + sync_f

        extended_c_left = torch.cat([extended_c, pe_f_left], dim=-1)  # (B, N, 2*D)
        extended_c_right = torch.cat([extended_c, pe_f_right], dim=-1)  # (B, N, 2*D)
        extended_c_left = self.pe_cond_proj_left(extended_c_left)  # nn.Linear(2*D, D)
        extended_c_right = self.pe_cond_proj_right(extended_c_right)  # nn.Linear(2*D, D)

        for block in self.joint_blocks:
            latent_left, clip_f, text_f = block(
                latent_left, clip_f, text_f, global_c, extended_c, self.latent_rot, self.clip_rot
            )  # (B, N, D)
            latent_right, clip_f, text_f = block(
                latent_right, clip_f, text_f, global_c, extended_c, self.latent_rot, self.clip_rot
            )  # (B, N, D)

        for block_left, block_right in zip(self.fused_blocks_left, self.fused_blocks_right):
            latent_left = block_left(latent_left, extended_c_left, self.latent_rot)
            latent_right = block_right(latent_right, extended_c_right, self.latent_rot)

        flow_left = self.final_layer(latent_left, global_c)  # (B, N, out_dim)
        flow_right = self.final_layer(latent_right, global_c)  # (B, N, out_dim)

        return flow_left, flow_right

    def forward(
        self,
        latent_left: torch.Tensor,
        latent_right: torch.Tensor,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        pe_f: torch.Tensor,
        text_f: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        latent: (B, N, C)
        vf: (B, T, C_V)
        t: (B,)
        """
        conditions = self.preprocess_conditions(clip_f, sync_f, pe_f, text_f)
        flow_left, flow_right = self.predict_flow(latent_left, latent_right, t, conditions)
        return flow_left, flow_right

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_pe_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_pe_feat.unsqueeze(0).expand(bs, self._pe_seq_len, -1, -1, -1)

    def get_empty_conditions(
        self, bs: int, *, negative_text_features: Optional[torch.Tensor] = None
    ) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        empty_pe = self.get_empty_pe_sequence(1)
        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_pe, empty_text)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.pe_f_left = conditions.pe_f_left.expand(bs, -1, -1)
        conditions.pe_f_right = conditions.pe_f_right.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(
        self,
        t: torch.Tensor,
        latent_left: torch.Tensor,
        latent_right: torch.Tensor,
        conditions: PreprocessedConditions,
        empty_conditions: PreprocessedConditions,
        cfg_strength: float,
    ) -> torch.Tensor:
        t = t * torch.ones(len(latent_left), device=latent_left.device, dtype=latent_left.dtype)

        if cfg_strength == 1.0:
            return self.predict_flow(latent_left, latent_right, t, conditions)
        else:
            flow_left, flow_right = self.predict_flow(latent_left, latent_right, t, conditions)
            flow_empty_left, flow_empty_right = self.predict_flow(
                latent_left, latent_right, t, empty_conditions
            )
            return cfg_strength * flow_left + (
                1 - cfg_strength
            ) * flow_empty_left, cfg_strength * flow_right + (1 - cfg_strength) * flow_empty_right

    def load_weights(self, src_dict) -> None:
        if "t_embed.freqs" in src_dict:
            del src_dict["t_embed.freqs"]
        if "latent_rot" in src_dict:
            del src_dict["latent_rot"]
        if "clip_rot" in src_dict:
            del src_dict["clip_rot"]

        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.left_latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len

    @property
    def pe_seq_len(self) -> int:
        return self._pe_seq_len


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(
        latent_dim=40,
        clip_dim=1024,
        sync_dim=768,
        pe_dim=1024,
        text_dim=1024,
        hidden_dim=64 * num_heads,
        depth=21,
        fused_depth=14,
        num_heads=num_heads,
        latent_seq_len=345,
        clip_seq_len=64,
        sync_seq_len=192,
        pe_seq_len=64,
        size_len=16,
        v2=True,
        **kwargs,
    )


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == "large_44k_v2":
        return large_44k_v2(**kwargs)
    if name == "large_44k_binaural":
        return large_44k_v2(**kwargs)
    raise ValueError(f"Unsupported ViSAudio model: {name}. Use large_44k_binaural.")


if __name__ == "__main__":
    network = get_my_mmaudio("large_44k_binaural")

    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f"Number of parameters: {num_params:.2f}M")
