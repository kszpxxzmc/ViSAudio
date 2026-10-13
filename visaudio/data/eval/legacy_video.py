"""Fixed-duration video conditioning matching the historical evaluation pipeline."""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2


def read_condition_frames(video: Path, duration: float, fps: int) -> torch.Tensor:
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(video)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    stream = json.loads(info.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    count = int(fps * duration)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-noautorotate",
         "-i", str(video), "-map", "0:v:0", "-an",
         "-vf", f"fps={fps},format=rgb24", "-frames:v", str(count),
         "-threads", "2", "-f", "rawvideo", "-"],
        check=True, capture_output=True, timeout=180,
    )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) == 0:
        raise ValueError(f"No decoded frames: {video}")
    chunk = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    deficit = count - len(chunk)
    tolerance = 3 if fps == 8 else 5
    if 0 < deficit < tolerance:
        chunk = torch.cat([chunk, chunk[-1:].expand(deficit, -1, -1, -1)], dim=0)
    if len(chunk) != count:
        raise ValueError(f"Video too short: {video}; expected {count} frames at {fps} fps, "
                         f"got {len(chunk)}")
    return chunk


def load_eval_video(video: Path, duration: float) -> dict[str, torch.Tensor]:
    clip = read_condition_frames(video, duration, 8)
    sync = read_condition_frames(video, duration, 25)
    clip_transform = v2.Compose([
        v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    ])
    sync_transform = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    pe_transform = v2.Compose([
        v2.Resize((448, 448), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    # The old evaluation code resized CLIP first, then derived PE from that tensor.
    clip = clip_transform(clip)
    return {"clip_video": clip, "sync_video": sync_transform(sync),
            "pe_video": pe_transform(clip)}
