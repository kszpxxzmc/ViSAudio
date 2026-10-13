import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from visaudio.data.eval.legacy_video import load_eval_video

from visaudio.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

_PE_SIZE = 448
_PE_FPS = 8.0


class VideoDataset(Dataset):
    def __init__(
        self,
        video_root: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_root = Path(video_root)

        self.duration_sec = duration_sec

        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose(
            [
                v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.sync_transform = v2.Compose(
            [
                v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(_SYNC_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.pe_transform = v2.Compose(
            [
                v2.Resize((_PE_SIZE, _PE_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]
        frames = load_eval_video(self.video_root / (video_id + ".mp4"), self.duration_sec)
        return {"name": video_id, "caption": caption, **frames}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def __len__(self):
        return len(self.captions)


class BiAudio(VideoDataset):
    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
        self.captions = {}

        df = pd.read_csv(csv_path, sep="\t", names=["id", "label"]).to_dict("records")
        videos_no_found = []
        for row in df[1:]:
            video_name = str(row["id"])
            if video_name + ".mp4" not in videos:
                videos_no_found.append(video_name)
                continue
            if pd.isna(row["label"]):
                row["label"] = ""
            self.captions[video_name] = row["label"]

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
            log.info(f"{len(self.captions)} useable videos found")
            if videos_no_found:
                log.info(f"{len(videos_no_found)} found in {csv_path} but not in {video_root}")
                log.info(
                    "A small amount is expected, as not all videos are still available on YouTube"
                )

        self.videos = sorted(list(self.captions.keys()))
