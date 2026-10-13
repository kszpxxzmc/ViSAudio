"""
trainer.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import os

from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed
import torch.optim as optim
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from visaudio.model.flow_matching import FlowMatching
from visaudio.model.networks import get_my_mmaudio
from visaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from visaudio.model.utils.features_utils import FeaturesUtils
from visaudio.model.utils.parameter_groups import get_parameter_groups
from visaudio.model.utils.sample_utils import log_normal_sample
from visaudio.utils.dist_utils import (
    global_rank,
    info_if_rank_zero,
    local_rank,
    string_if_rank_zero,
)
from visaudio.utils.checkpoint import load_model_state, mmaudio_initialization
from visaudio.utils.log_integrator import Integrator
from visaudio.utils.logger import TensorboardLogger
from visaudio.utils.time_estimator import PartialTimeEstimator, TimeEstimator
from visaudio.utils.video_joiner import VideoJoiner


class Runner:
    def __init__(
        self,
        cfg: DictConfig,
        log: TensorboardLogger,
        run_path: Union[str, Path],
        for_training: bool = True,
        left_latent_mean: Optional[torch.Tensor] = None,
        left_latent_std: Optional[torch.Tensor] = None,
        right_latent_mean: Optional[torch.Tensor] = None,
        right_latent_std: Optional[torch.Tensor] = None,
    ):
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.enable_grad_scaler = cfg.enable_grad_scaler
        self.for_training = for_training
        self.cfg = cfg

        if cfg.model.endswith("16k"):
            self.seq_cfg = CONFIG_16K
            mode = "16k"
        elif cfg.model.endswith("44k"):
            self.seq_cfg = CONFIG_44K
            mode = "44k"
        elif cfg.model.endswith("44k_v2"):
            self.seq_cfg = CONFIG_44K
            mode = "44k"
        elif cfg.model.endswith("44k_binaural"):
            self.seq_cfg = CONFIG_44K
            mode = "44k"
        else:
            raise ValueError(f"Unknown model: {cfg.model}")

        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration

        empty_path = Path(cfg.empty_string_ckpt)
        empty_string_feat = None
        if empty_path.is_file():
            empty_string_feat = torch.load(empty_path, map_location="cpu", weights_only=True)
            if empty_string_feat.ndim == 3:
                empty_string_feat = empty_string_feat[0]
        elif not (
            cfg.checkpoint
            or cfg.weights
            or cfg.resume
            or (Path(run_path) / f"{cfg.exp_id}_ckpt_last.pth").is_file()
        ):
            raise FileNotFoundError(
                "Scratch training requires empty_string_ckpt from feature extraction."
            )
        self.network = DDP(
            get_my_mmaudio(
                cfg.model,
                left_latent_mean=left_latent_mean,
                left_latent_std=left_latent_std,
                right_latent_mean=right_latent_mean,
                right_latent_std=right_latent_std,
                empty_string_feat=empty_string_feat,
            ).cuda(),
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if cfg.compile:
            self.train_fn = torch.compile(self.train_fn)
            self.val_fn = torch.compile(self.val_fn)

        self.fm = FlowMatching(
            cfg.sampling.min_sigma,
            inference_mode=cfg.sampling.method,
            num_steps=cfg.sampling.num_steps,
        )

        self.ema = None
        self._pending_ema = None
        self.ema_start = cfg.ema.start

        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(cfg["seed"] + global_rank)

        if mode == "16k":
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg["vae_16k_ckpt"],
                bigvgan_vocoder_ckpt=cfg["bigvgan_vocoder_ckpt"],
                synchformer_ckpt=cfg["synchformer_ckpt"],
                enable_conditions=False,
                mode=mode,
                need_vae_encoder=False,
            )
        elif mode == "44k":
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg["vae_44k_ckpt"],
                synchformer_ckpt=cfg["synchformer_ckpt"],
                enable_conditions=False,
                mode=mode,
                need_vae_encoder=False,
            )
        self.features = self.features.cuda().eval()

        if cfg.compile:
            self.features.compile()

        self.log_normal_sampling_mean = cfg.sampling.mean
        self.log_normal_sampling_scale = cfg.sampling.scale
        self.null_condition_probability = cfg.null_condition_probability
        self.cfg_strength = cfg.cfg_strength

        self.log = log
        self.run_path = Path(run_path)

        MUSIC_cfg = cfg.data[f"{cfg.val_dataset}_test"]
        self.val_video_joiner = self.test_video_joiner = None
        if for_training and cfg.save_video_samples:
            self.val_video_joiner = VideoJoiner(
                MUSIC_cfg.root,
                self.run_path / "val-sampled-videos",
                self.sample_rate,
                self.duration_sec,
            )
        elif cfg.save_video_samples:
            self.test_video_joiner = VideoJoiner(
                MUSIC_cfg.root,
                self.run_path / "test-sampled-videos",
                self.sample_rate,
                self.duration_sec,
            )

        string_if_rank_zero(
            self.log,
            "model_size",
            f"{sum([param.nelement() for param in self.network.parameters()])}",
        )
        string_if_rank_zero(
            self.log,
            "number_of_parameters_that_require_gradient: ",
            str(
                sum(
                    [
                        param.nelement()
                        for param in filter(lambda p: p.requires_grad, self.network.parameters())
                    ]
                )
            ),
        )
        info_if_rank_zero(self.log, "torch version: " + torch.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        if for_training:
            self.enter_train()
            parameter_groups = get_parameter_groups(self.network, cfg)

            self.optimizer = optim.AdamW(
                parameter_groups,
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                betas=[0.9, 0.95],
                eps=1e-6 if self.use_amp else 1e-8,
                fused=True,
            )
            if self.enable_grad_scaler:
                self.scaler = torch.amp.GradScaler(init_scale=2048)
            self.clip_grad_norm = cfg["clip_grad_norm"]

            linear_warmup_steps = cfg["linear_warmup_steps"]

            def warmup(currrent_step: int):
                return (currrent_step + 1) / (linear_warmup_steps + 1)

            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)

            if cfg["lr_schedule"] == "constant":
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
            elif cfg["lr_schedule"] == "poly":
                total_num_iter = cfg["num_iterations"]
                next_scheduler = optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter)) ** 0.9
                )
            elif cfg["lr_schedule"] == "step":
                next_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, cfg["lr_schedule_steps"], cfg["lr_schedule_gamma"]
                )
            else:
                raise NotImplementedError

            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup_scheduler, next_scheduler], [linear_warmup_steps]
            )

            self.log_text_interval = cfg["log_text_interval"]
            self.log_extra_interval = cfg["log_extra_interval"]
            self.save_weights_interval = cfg["save_weights_interval"]
            self.save_checkpoint_interval = cfg["save_checkpoint_interval"]
            self.save_copy_iterations = cfg["save_copy_iterations"]
            self.num_iterations = cfg["num_iterations"]
            if cfg["debug"]:
                self.log_text_interval = self.log_extra_interval = 1

            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
            self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)
        else:
            self.enter_val()

    def initialize_ema(self):
        if not (self.for_training and self.cfg.ema.enable and global_rank == 0):
            return
        if self.cfg.ema.default_output_sigma not in self.cfg.ema.sigma_rels:
            raise ValueError("default_output_sigma must select one of ema.sigma_rels.")
        self.ema = PostHocEMA(
            self.network.module,
            sigma_rels=self.cfg.ema.sigma_rels,
            update_every=self.cfg.ema.update_every,
            checkpoint_every_num_steps=self.cfg.ema.checkpoint_every,
            checkpoint_folder=self.cfg.ema.checkpoint_folder,
            step_size_correction=True,
        ).cuda()
        if self._pending_ema is not None:
            self.ema.load_state_dict(self._pending_ema)
            self._pending_ema = None

    def train_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        pe_f: torch.Tensor,
        text_f: torch.Tensor,
        a_left_mean: torch.Tensor,
        a_left_std: torch.Tensor,
        a_right_mean: torch.Tensor,
        a_right_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_left_randn = torch.empty_like(a_left_mean).normal_(generator=self.rng)
        a_right_randn = torch.empty_like(a_right_mean).normal_(generator=self.rng)
        x1_left = a_left_mean + a_left_std * a_left_randn
        x1_right = a_right_mean + a_right_std * a_right_randn
        bs = x1_left.shape[0]  # batch_size * seq_len * num_channels

        x1_left, x1_right = self.network.module.normalize(x1_left, x1_right)

        t = log_normal_sample(
            x1_left,
            generator=self.rng,
            m=self.log_normal_sampling_mean,
            s=self.log_normal_sampling_scale,
        )
        x0_left, x1_left, xt_left, (clip_f, sync_f, pe_f, text_f) = self.fm.get_x0_xt_c(
            x1_left, t, Cs=[clip_f, sync_f, pe_f, text_f], generator=self.rng
        )
        x0_right, x1_right, xt_right, (clip_f, sync_f, pe_f, text_f) = self.fm.get_x0_xt_c(
            x1_right, t, Cs=[clip_f, sync_f, pe_f, text_f], generator=self.rng
        )

        # classifier-free training
        samples = torch.rand(bs, device=x1_left.device, generator=self.rng)
        null_video = samples < self.null_condition_probability
        clip_f[null_video] = self.network.module.empty_clip_feat
        sync_f[null_video] = self.network.module.empty_sync_feat
        pe_f[null_video] = self.network.module.empty_pe_feat

        samples = torch.rand(bs, device=x1_left.device, generator=self.rng)
        null_text = samples < self.null_condition_probability
        text_f[null_text] = self.network.module.empty_string_feat

        pred_v_left, pred_v_right = self.network(xt_left, xt_right, clip_f, sync_f, pe_f, text_f, t)
        target_v_left, loss_left = self.fm.mse_loss(pred_v_left, x0_left, x1_left)
        target_v_right, loss_right = self.fm.mse_loss(pred_v_right, x0_right, x1_right)

        avg_mse_loss = self.fm.avg_mse_loss(
            pred_v_left, pred_v_right, target_v_left, target_v_right
        )
        latent_corr_loss = self.fm.latent_corr_loss(
            pred_v_left, pred_v_right, target_v_left, target_v_right
        )

        loss = loss_left + loss_right + self.cfg.avg_loss_weight * avg_mse_loss

        mean_loss = loss.mean()
        return (
            x1_left,
            x1_right,
            loss,
            mean_loss,
            t,
            {
                "left_loss": loss_left.mean(),
                "right_loss": loss_right.mean(),
                "avg_mse_loss": avg_mse_loss.mean(),
                "latent_corr_loss": latent_corr_loss.mean(),
            },
        )

    def val_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        pe_f: torch.Tensor,
        text_f: torch.Tensor,
        x1_left: torch.Tensor,
        x1_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x1_left.shape[0]  # batch_size * seq_len * num_channels
        x1_left, x1_right = self.network.module.normalize(x1_left, x1_right)

        t = log_normal_sample(
            x1_left,
            generator=self.rng,
            m=self.log_normal_sampling_mean,
            s=self.log_normal_sampling_scale,
        )
        x0_left, x1_left, xt_left, (clip_f, sync_f, pe_f, text_f) = self.fm.get_x0_xt_c(
            x1_left, t, Cs=[clip_f, sync_f, pe_f, text_f], generator=self.rng
        )
        x0_right, x1_right, xt_right, (clip_f, sync_f, pe_f, text_f) = self.fm.get_x0_xt_c(
            x1_right, t, Cs=[clip_f, sync_f, pe_f, text_f], generator=self.rng
        )

        # classifier-free training
        samples = torch.rand(bs, device=x1_left.device, generator=self.rng)
        null_video = samples < self.null_condition_probability
        clip_f[null_video] = self.network.module.empty_clip_feat
        sync_f[null_video] = self.network.module.empty_sync_feat
        pe_f[null_video] = self.network.module.empty_pe_feat

        samples = torch.rand(bs, device=x1_left.device, generator=self.rng)
        null_text = samples < self.null_condition_probability
        text_f[null_text] = self.network.module.empty_string_feat

        pred_v_left, pred_v_right = self.network(xt_left, xt_right, clip_f, sync_f, pe_f, text_f, t)
        target_v_left, loss_left = self.fm.mse_loss(pred_v_left, x0_left, x1_left)
        target_v_right, loss_right = self.fm.mse_loss(pred_v_right, x0_right, x1_right)

        avg_mse_loss = self.fm.avg_mse_loss(
            pred_v_left, pred_v_right, target_v_left, target_v_right
        )
        latent_corr_loss = self.fm.latent_corr_loss(
            pred_v_left, pred_v_right, target_v_left, target_v_right
        )

        loss = loss_left + loss_right + self.cfg.avg_loss_weight * avg_mse_loss

        mean_loss = loss.mean()
        return (
            loss,
            mean_loss,
            t,
            {
                "left_loss": loss_left.mean(),
                "right_loss": loss_right.mean(),
                "avg_mse_loss": avg_mse_loss.mean(),
                "latent_corr_loss": latent_corr_loss.mean(),
            },
        )

    def train_pass(self, data, it: int = 0):

        if not self.for_training:
            raise ValueError("train_pass() should not be called when not training.")

        self.enter_train()
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data["clip_features"].cuda(non_blocking=True)
            sync_f = data["sync_features"].cuda(non_blocking=True)
            pe_f = data["pe_features"].cuda(non_blocking=True).float()
            text_f = data["text_features"].cuda(non_blocking=True)
            video_exist = data["video_exist"].cuda(non_blocking=True)
            text_exist = data["text_exist"].cuda(non_blocking=True)
            a_left_mean = data["a_left_mean"].cuda(non_blocking=True)
            a_left_std = data["a_left_std"].cuda(non_blocking=True)
            a_right_mean = data["a_right_mean"].cuda(non_blocking=True)
            a_right_std = data["a_right_std"].cuda(non_blocking=True)

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            pe_f[~video_exist] = self.network.module.empty_pe_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            self.log.data_timer.end()
            if self.log_extra_interval > 0 and it % self.log_extra_interval == 0:
                unmasked_clip_f = clip_f.clone()
                unmasked_sync_f = sync_f.clone()
                unmasked_pe_f = pe_f.clone()
                unmasked_text_f = text_f.clone()
            x1_left, x1_right, loss, mean_loss, t, loss_dict = self.train_fn(
                clip_f, sync_f, pe_f, text_f, a_left_mean, a_left_std, a_right_mean, a_right_std
            )

            self.train_integrator.add_dict({"loss": mean_loss})

            for k, v in loss_dict.items():
                self.train_integrator.add_scalar(k, v.mean() if v.numel() > 1 else v)

        if it % self.log_text_interval == 0 and it != 0:
            self.train_integrator.add_scalar("lr", self.scheduler.get_last_lr()[0])
            self.train_integrator.add_binned_tensor("binned_loss", loss, t)
            self.train_integrator.finalize("train", it)
            self.train_integrator.reset_except_hooks()

        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.clip_grad_norm
            )
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar("grad_norm", grad_norm)

        self.enter_val()
        with (
            torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16),
            torch.inference_mode(),
        ):
            try:
                if self.log_extra_interval > 0 and it % self.log_extra_interval == 0:
                    x1_left, x1_right = self.network.module.unnormalize(x1_left[0:1], x1_right[0:1])
                    mel_left = self.features.decode(x1_left)
                    mel_right = self.features.decode(x1_right)
                    audio_left = self.features.vocode(mel_left).cpu()[0]  # 1 * num_samples
                    audio_right = self.features.vocode(mel_right).cpu()[0]  # 1 * num_samples
                    audio = torch.cat([audio_left, audio_right], dim=0)  # 2 * num_samples
                    self.log.log_spectrogram(
                        "train", f"spec-gt-r{local_rank}-left", mel_left.cpu()[0], it
                    )
                    self.log.log_spectrogram(
                        "train", f"spec-gt-r{local_rank}-right", mel_right.cpu()[0], it
                    )
                    self.log.log_audio(
                        "train", f"audio-gt-r{local_rank}", audio, it, sample_rate=self.sample_rate
                    )

                    x0_left = torch.empty_like(x1_left[0:1]).normal_(generator=self.rng)
                    x0_right = torch.empty_like(x1_right[0:1]).normal_(generator=self.rng)
                    clip_f = unmasked_clip_f[0:1]
                    sync_f = unmasked_sync_f[0:1]
                    pe_f = unmasked_pe_f[0:1]
                    text_f = unmasked_text_f[0:1]
                    conditions = self.network.module.preprocess_conditions(
                        clip_f, sync_f, pe_f, text_f
                    )
                    empty_conditions = self.network.module.get_empty_conditions(x0_left.shape[0])
                    cfg_ode_wrapper = lambda t, x_left, x_right: self.network.module.ode_wrapper(
                        t, x_left, x_right, conditions, empty_conditions, self.cfg_strength
                    )
                    x1_hat_left, x1_hat_right = self.fm.to_data(cfg_ode_wrapper, x0_left, x0_right)
                    x1_hat_left, x1_hat_right = self.network.module.unnormalize(
                        x1_hat_left, x1_hat_right
                    )
                    mel_left = self.features.decode(x1_hat_left)
                    mel_right = self.features.decode(x1_hat_right)
                    audio_left = self.features.vocode(mel_left).cpu()[0]  # 1 * num_samples
                    audio_right = self.features.vocode(mel_right).cpu()[0]  # 1 * num_samples
                    audio = torch.cat([audio_left, audio_right], dim=0)  # 2 * num_samples
                    self.log.log_spectrogram(
                        "train", f"spec-r{local_rank}-left", mel_left.cpu()[0], it
                    )
                    self.log.log_spectrogram(
                        "train", f"spec-r{local_rank}-right", mel_right.cpu()[0], it
                    )
                    self.log.log_audio(
                        "train", f"audio-r{local_rank}", audio, it, sample_rate=self.sample_rate
                    )
            except Exception as e:
                self.log.warning(f"Error in extra logging: {e}")
                if self.cfg.debug:
                    raise

        completed_steps = it + 1
        save_copy = completed_steps in self.save_copy_iterations

        if (
            self.save_weights_interval > 0 and completed_steps % self.save_weights_interval == 0
        ) or save_copy:
            self.save_weights(completed_steps, save_copy=save_copy)

        if (
            self.save_checkpoint_interval > 0
            and completed_steps % self.save_checkpoint_interval == 0
        ):
            self.save_checkpoint(completed_steps)

        self.log.data_timer.start()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def validation_pass(self, data, it: int = 0):
        self.enter_val()
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data["clip_features"].cuda(non_blocking=True)
            sync_f = data["sync_features"].cuda(non_blocking=True)
            pe_f = data["pe_features"].cuda(non_blocking=True).float()
            text_f = data["text_features"].cuda(non_blocking=True)
            video_exist = data["video_exist"].cuda(non_blocking=True)
            text_exist = data["text_exist"].cuda(non_blocking=True)
            a_left_mean = data["a_left_mean"].cuda(non_blocking=True)
            a_left_std = data["a_left_std"].cuda(non_blocking=True)
            a_right_mean = data["a_right_mean"].cuda(non_blocking=True)
            a_right_std = data["a_right_std"].cuda(non_blocking=True)

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            pe_f[~video_exist] = self.network.module.empty_pe_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            a_left_randn = torch.empty_like(a_left_mean).normal_(generator=self.rng)
            a_right_randn = torch.empty_like(a_right_mean).normal_(generator=self.rng)
            x1_left = a_left_mean + a_left_std * a_left_randn
            x1_right = a_right_mean + a_right_std * a_right_randn
            self.log.data_timer.end()
            loss, mean_loss, t, loss_dict = self.val_fn(
                clip_f.clone(), sync_f.clone(), pe_f.clone(), text_f.clone(), x1_left, x1_right
            )

            self.val_integrator.add_binned_tensor("binned_loss", loss, t)
            self.val_integrator.add_dict({"loss": mean_loss})

            if loss_dict is not None:
                for k, v in loss_dict.items():
                    self.val_integrator.add_scalar(k, v)

        self.log.data_timer.start()

    @torch.inference_mode()
    def inference_pass(
        self, data, it: int, data_cfg: DictConfig, *, save_eval: bool = True
    ) -> Path:
        self.enter_val()
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data["clip_features"].cuda(non_blocking=True)
            sync_f = data["sync_features"].cuda(non_blocking=True)
            pe_f = data["pe_features"].cuda(non_blocking=True).float()
            text_f = data["text_features"].cuda(non_blocking=True)
            video_exist = data["video_exist"].cuda(non_blocking=True)
            text_exist = data["text_exist"].cuda(non_blocking=True)
            a_left_mean = data["a_left_mean"].cuda(non_blocking=True)  # for the shape only
            a_right_mean = data["a_right_mean"].cuda(non_blocking=True)  # for the shape only

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            pe_f[~video_exist] = self.network.module.empty_pe_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            x0_left = torch.empty_like(a_left_mean).normal_(generator=self.rng)
            x0_right = torch.empty_like(a_right_mean).normal_(generator=self.rng)
            conditions = self.network.module.preprocess_conditions(clip_f, sync_f, pe_f, text_f)
            empty_conditions = self.network.module.get_empty_conditions(x0_left.shape[0])
            cfg_ode_wrapper = lambda t, x_left, x_right: self.network.module.ode_wrapper(
                t, x_left, x_right, conditions, empty_conditions, self.cfg_strength
            )
            x1_hat_left, x1_hat_right = self.fm.to_data(cfg_ode_wrapper, x0_left, x0_right)
            x1_hat_left, x1_hat_right = self.network.module.unnormalize(x1_hat_left, x1_hat_right)
            mel_left = self.features.decode(x1_hat_left)
            mel_right = self.features.decode(x1_hat_right)
            audio_left = self.features.vocode(mel_left).cpu()
            audio_right = self.features.vocode(mel_right).cpu()
            audio = torch.cat([audio_left, audio_right], dim=1)  # 2 * num_samples
            for i in range(audio.shape[0]):
                video_id = data["id"][i]
                if (not self.for_training) and i == 0 and self.test_video_joiner is not None:
                    self.test_video_joiner.join(video_id, f"{video_id}", audio[i].transpose(0, 1))

                if data_cfg.output_subdir is not None:
                    if save_eval:
                        iter_naming = f"{it:09d}"
                    else:
                        iter_naming = "val-cache"
                    audio_dir = self.log.log_audio(
                        iter_naming,
                        f"{video_id}",
                        audio[i],
                        it=None,
                        sample_rate=self.sample_rate,
                        subdir=Path(data_cfg.output_subdir),
                    )
                    if save_eval and i == 0 and self.val_video_joiner is not None:
                        self.val_video_joiner.join(
                            video_id, f"{iter_naming}-{video_id}", audio[i].transpose(0, 1)
                        )
                else:
                    audio_dir = self.log.log_audio(
                        f"{data_cfg.tag}-sampled",
                        f"{video_id}",
                        audio[i],
                        it=None,
                        sample_rate=self.sample_rate,
                    )

        return Path(audio_dir)

    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        from av_bench.evaluate import evaluate
        from av_bench.extract import extract

        if global_rank == 0:
            self.log.info(f"Evaluating at iteration {it} for {data_cfg.tag}...")
        with torch.amp.autocast("cuda", enabled=False):
            if global_rank == 0:
                extract(
                    audio_path=audio_dir,
                    output_path=audio_dir / "cache",
                    device="cuda",
                    batch_size=32,
                    audio_length=8,
                )
                output_metrics = evaluate(
                    gt_audio_cache=Path(data_cfg.gt_cache), pred_audio_cache=audio_dir / "cache"
                )
                for k, v in output_metrics.items():
                    self.log.log_scalar(f"{data_cfg.tag}/{k}", v, it)
                    self.log.info(f"{data_cfg.tag}/{k:<10}: {v:.10f}")
            else:
                output_metrics = None

        return output_metrics

    def save_weights(self, it, save_copy=False):

        if global_rank != 0:
            return
        self.log.info("Saving weights...")
        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f"{self.exp_id}_{it}.pth"
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f"Network weights saved to {model_path}.")

        model_path = self.run_path / f"{self.exp_id}_last.pth"
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace("last", "shadow"))
            model_path.replace(shadow_path)
            self.log.info(f"Network weights shadowed to {shadow_path}.")

        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f"Network weights saved to {model_path}.")

    def save_checkpoint(self, it, save_copy=False):
        rng_state = {
            "generator": self.rng.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        }
        rng_states = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rng_states, rng_state)
        if global_rank != 0:
            return
        self.log.info("Saving checkpoint...")

        checkpoint = {
            "it": it,
            "weights": self.network.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "rng_states": rng_states,
            "scaler": self.scaler.state_dict() if self.enable_grad_scaler else None,
            "step_is_completed": True,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f"{self.exp_id}_ckpt_{it}.pth"
            torch.save(checkpoint, model_path)
            self.log.info(f"Checkpoint saved to {model_path}.")

        model_path = self.run_path / f"{self.exp_id}_ckpt_last.pth"
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace("last", "shadow"))
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f"Checkpoint shadowed to {shadow_path}.")

        torch.save(checkpoint, model_path)
        self.log.info(f"Checkpoint saved to {model_path}.")

        if self.ema is not None and it % self.cfg.ema.checkpoint_save == 0:
            for index, ema in enumerate(self.ema.ema_models):
                torch.save(
                    ema.state_dict(), Path(self.cfg.ema.checkpoint_folder) / f"{index}.{it}.pt"
                )
            self.log.info(f"Saving EMA checkpoint at step {it} to {self.cfg.ema.checkpoint_folder}")

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f"{self.exp_id}_ckpt_last.pth"
        if not ckpt_path.exists():
            info_if_rank_zero(self.log, f"No checkpoint found at {ckpt_path}.")
            return None
        return ckpt_path

    def get_latest_weight_path(self):
        weight_path = self.run_path / f"{self.exp_id}_last.pth"
        if not weight_path.exists():
            self.log.info(f"No weight found at {weight_path}.")
            return None
        return weight_path

    def get_final_ema_weight_path(self):
        weight_path = self.run_path / f"{self.exp_id}_ema_final.pth"
        if not weight_path.exists():
            self.log.info(f"No weight found at {weight_path}.")
            return None
        return weight_path

    def load_checkpoint_finetune(self, path):
        checkpoint = load_model_state(path)
        if "latent_mean" not in checkpoint or not any(
            key.startswith("fused_blocks.") for key in checkpoint
        ):
            raise ValueError("checkpoint= expects MMAudio weights; use weights= for ViSAudio.")
        new_checkpoint = mmaudio_initialization(checkpoint)
        if not Path(self.cfg.empty_string_ckpt).is_file():
            new_checkpoint["empty_string_feat"] = checkpoint["empty_string_feat"]
        missing_keys, unexpected_keys = self.network.module.load_state_dict(
            new_checkpoint, strict=False
        )
        allowed_missing = (
            "pe_",
            "empty_pe_feat",
            "left_latent_",
            "right_latent_",
            "empty_string_feat",
        )
        invalid_missing = [key for key in missing_keys if not key.startswith(allowed_missing)]
        if invalid_missing or unexpected_keys:
            raise ValueError(
                f"Incompatible MMAudio initialization: {invalid_missing=}, {unexpected_keys=}"
            )
        self.log.info(
            f"Initialized new spatial modules; preserved dataset statistics: {missing_keys}"
        )
        if unexpected_keys:
            self.log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        self.log.info("Network weights loaded for finetuning.")
        return 0

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        it = checkpoint["it"]
        weights = checkpoint["weights"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        if self.cfg.ema.enable and checkpoint.get("ema") is None:
            raise ValueError(
                "Checkpoint has no EMA state. Disable ema.enable or initialize from weights."
            )
        if global_rank == 0:
            self._pending_ema = checkpoint.get("ema")

        self.network.module.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)
        if self.enable_grad_scaler and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        states = checkpoint.get("rng_states")
        if states is not None:
            if len(states) != torch.distributed.get_world_size():
                raise ValueError(
                    "Exact resume requires the same WORLD_SIZE as the saved checkpoint."
                )
            state = states[global_rank]
            self.rng.set_state(state["generator"])
            torch.set_rng_state(state["torch"])
            torch.cuda.set_rng_state(state["cuda"])
        if not checkpoint.get("step_is_completed", False):
            self.log.warning(
                "Legacy checkpoint has ambiguous step numbering; using its stored iteration."
            )

        self.log.info(f"Global iteration {it} loaded.")
        self.log.info("Network weights, optimizer states, and scheduler states loaded.")

        return it

    def load_weights_in_memory(self, src_dict):
        self.network.module.load_weights(src_dict)
        self.log.info("Network weights loaded from memory.")

    def load_weights(self, path):
        src_dict = load_model_state(path)

        self.log.info(f"Importing network weights from {path}...")
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self
