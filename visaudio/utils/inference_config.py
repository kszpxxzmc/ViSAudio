"""Defaults and sidecar captions shared by video inference entry points."""

from pathlib import Path

DEFAULT_SEED = 14159265
INFERENCE_PROFILE = "legacy_eval_video_text_v1"


def add_inference_arguments(parser):
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Fixed conditioning duration in seconds.")
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt", default=None, help="Override same-stem .txt caption.")
    parser.add_argument("--video_only", "--video-only", action="store_true",
                        help="Generate from video without reading a caption or text prompt.")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full_precision", action="store_true")


def resolve_prompt(video: Path, override: str | None = None, *, video_only: bool = False) -> str:
    if video_only:
        if override is not None:
            raise ValueError("--video_only cannot be combined with --prompt.")
        return ""
    if override is not None:
        return override
    caption = video.with_suffix(".txt")
    if not caption.is_file():
        raise ValueError(f"Missing caption: {caption}; supply a same-stem .txt or --prompt.")
    text = caption.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty caption: {caption}; use --prompt '' for explicit video-only inference.")
    return text
