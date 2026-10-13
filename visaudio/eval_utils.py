import dataclasses
import logging
from pathlib import Path
from typing import Optional

import torch
from colorlog import ColoredFormatter
from torchvision.transforms import v2

from visaudio.data.av_utils import VideoInfo, read_frames
from visaudio.model.flow_matching import FlowMatching
from visaudio.model.networks import MMAudio
from visaudio.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from visaudio.model.utils.features_utils import FeaturesUtils
from visaudio.utils.download_utils import download_model_if_needed

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    synchformer_ckpt: Path = Path("./ext_weights/synchformer_state_dict.pth")

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == "16k":
            return dataclasses.replace(CONFIG_16K)
        elif self.mode == "44k":
            return dataclasses.replace(CONFIG_44K)
        raise ValueError(f"Unknown audio mode: {self.mode}")

    def download_if_needed(self):
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)
        download_model_if_needed(self.synchformer_ckpt)


large_44k_binaural = ModelConfig(
    model_name="large_44k_binaural",
    model_path=Path("./weights/visaudio_large_44k.pth"),
    vae_path=Path("./ext_weights/v1-44.pth"),
    bigvgan_16k_path=None,
    mode="44k",
)
all_model_cfg: dict[str, ModelConfig] = {
    "large_44k_binaural": large_44k_binaural,
}


def generate(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    pe_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 64,
    sync_batch_size_multiplier: int = 64,
    pe_batch_size_multiplier: int = 64,
    image_input: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype
    batch_sizes = [
        len(value) for value in (text, clip_video, sync_video, pe_video) if value is not None
    ]
    if not batch_sizes or batch_sizes[0] == 0:
        raise ValueError("Provide a nonempty batch of text or video.")
    if len(set(batch_sizes)) != 1:
        raise ValueError("Text and video batch sizes must match.")
    bs = batch_sizes[0]
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(
            clip_video, batch_size=bs * clip_batch_size_multiplier
        )
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(
            sync_video, batch_size=bs * sync_batch_size_multiplier
        )
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if pe_video is not None and not image_input:
        pe_video = pe_video.to(device, dtype, non_blocking=True)
        pe_features = feature_utils.encode_video_with_pe(
            pe_video, batch_size=bs * pe_batch_size_multiplier
        )
    else:
        pe_features = net.get_empty_pe_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0_left = torch.randn(
        bs, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng
    )
    x0_right = torch.randn(
        bs, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng
    )
    preprocessed_conditions = net.preprocess_conditions(
        clip_features, sync_features, pe_features, text_features
    )
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None
    )

    cfg_ode_wrapper = lambda t, x_left, x_right: net.ode_wrapper(
        t, x_left, x_right, preprocessed_conditions, empty_conditions, cfg_strength
    )
    x1_left, x1_right = fm.to_data(cfg_ode_wrapper, x0_left, x0_right)
    x1_left, x1_right = net.unnormalize(x1_left, x1_right)
    spec_left = feature_utils.decode(x1_left)
    spec_right = feature_utils.decode(x1_right)
    audio_left = feature_utils.vocode(spec_left)
    audio_right = feature_utils.vocode(spec_right)
    audio = torch.cat([audio_left, audio_right], dim=1)
    return audio


LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

_PE_SIZE = 448
_PE_FPS = 8.0


def load_video(video_path: Path, duration_sec: float, load_all_frames: bool = True) -> VideoInfo:
    clip_transform = v2.Compose(
        [
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    sync_transform = v2.Compose(
        [
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    pe_transform = v2.Compose(
        [
            v2.Resize((_PE_SIZE, _PE_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    output_frames, all_frames, orig_fps = read_frames(
        video_path,
        list_of_fps=[_CLIP_FPS, _SYNC_FPS],
        start_sec=0,
        end_sec=duration_sec,
        need_all_frames=load_all_frames,
    )

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)
    pe_frames = pe_transform(clip_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS
    pe_length_sec = clip_frames.shape[0] / _PE_FPS

    if clip_length_sec < duration_sec:
        log.warning(f"Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}")
        log.warning(f"Truncating to {clip_length_sec:.2f} sec")
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f"Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}")
        log.warning(f"Truncating to {sync_length_sec:.2f} sec")
        duration_sec = sync_length_sec

    if pe_length_sec < duration_sec:
        log.warning(f"Pe video is too short: {pe_length_sec:.2f} < {duration_sec:.2f}")
        log.warning(f"Truncating to {pe_length_sec:.2f} sec")
        duration_sec = pe_length_sec

    clip_frames = clip_frames[: int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[: int(_SYNC_FPS * duration_sec)]
    pe_frames = pe_frames[: int(_PE_FPS * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=orig_fps,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        pe_frames=pe_frames,
        all_frames=all_frames if load_all_frames else None,
    )
    return video_info
