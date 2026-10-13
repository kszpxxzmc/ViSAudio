"""Video/text inference CLI; video jobs use the shared resumable batch engine."""

from argparse import ArgumentParser
from dataclasses import replace
import hashlib
import logging
import math
import os
from pathlib import Path
import sys

from visaudio.utils.inference_config import add_inference_arguments

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video", "--video_path", type=Path)
    inputs.add_argument("--video_dir", "--input", type=Path)
    parser.add_argument("--variant", default="large_44k_binaural",
                        choices=["large_44k_binaural"])
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--device", default=None, help="Text device or single video CUDA device.")
    add_inference_arguments(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip existing text audio; video jobs resume by default.")
    return parser


def run_video(args, parser):
    from demo_batch import main as batch_main

    gpu = args.gpus
    if args.device:
        if args.device != "cuda" and not (
                args.device.startswith("cuda:") and args.device[5:].isdigit()):
            parser.error("Video inference requires --device cuda[:index] or --gpus.")
        if ":" in args.device:
            gpu = args.device.split(":", 1)[1]
    command = [
        "--video" if args.video else "--input", str(args.video or args.video_dir),
        "--gpus", gpu, "--duration", str(args.duration), "--seed", str(args.seed),
        "--num_steps", str(args.num_steps), "--cfg_strength", str(args.cfg_strength),
        "--threads", str(args.threads),
    ]
    for flag, value in (("--output", args.output), ("--model_path", args.model_path),
                        ("--prompt", args.prompt), ("--negative_prompt", args.negative_prompt)):
        if value is not None:
            command += [flag, str(value)]
    for attribute, flag in (("full_precision", "--full_precision"), ("video_only", "--video_only"),
                            ("overwrite", "--overwrite"), ("dry_run", "--dry-run")):
        if getattr(args, attribute):
            command.append(flag)
    return batch_main(command)


def run_text(args, parser):
    checkpoint = (args.model_path or ROOT / "weights/visaudio_large_44k.pth").expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    output = (args.output or ROOT / "assets/output").expanduser().resolve()
    stem = "text_" + hashlib.sha256(args.prompt.encode()).hexdigest()[:12]
    audio_path = output / f"{stem}.flac"
    if args.dry_run:
        print(f"Text -> {audio_path}; seed={args.seed}; duration={args.duration}s")
        return 0
    if args.skip_existing and not args.overwrite and audio_path.exists():
        return 0

    import soundfile as sf
    import torch
    from visaudio.eval_utils import all_model_cfg, generate, setup_eval_logging
    from visaudio.model.flow_matching import FlowMatching
    from visaudio.model.networks import get_my_mmaudio
    from visaudio.model.utils.features_utils import FeaturesUtils
    from visaudio.utils.checkpoint import load_model_state

    setup_eval_logging()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.full_precision
    model = replace(all_model_cfg[args.variant], model_path=checkpoint)
    previous_dir = Path.cwd()
    try:
        # Auxiliary checkpoints use repository-relative paths.
        os.chdir(ROOT)
        model.download_if_needed()
        with torch.inference_mode():
            net = get_my_mmaudio(model.model_name).to(device).eval()
            net.load_weights(load_model_state(model.model_path))
            features = FeaturesUtils(
                tod_vae_ckpt=model.vae_path, synchformer_ckpt=model.synchformer_ckpt,
                mode=model.mode, need_vae_encoder=False,
            ).to(device).eval()
            seq = model.seq_cfg
            seq.duration = args.duration
            net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len,
                                   seq.sync_seq_len, seq.pe_seq_len)
            rng = torch.Generator(device=device).manual_seed(args.seed)
            with torch.amp.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
                audio = generate(
                    None, None, None, [args.prompt],
                    negative_text=[args.negative_prompt] if args.negative_prompt else None,
                    feature_utils=features, net=net,
                    fm=FlowMatching(num_steps=args.num_steps), rng=rng,
                    cfg_strength=args.cfg_strength,
                )[0].float().cpu()
            samples = round(args.duration * seq.sampling_rate)
            if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < samples:
                raise ValueError(f"Invalid generated stereo waveform: {tuple(audio.shape)}")
            if not torch.isfinite(audio).all():
                raise ValueError("Generated audio contains NaN or Inf.")
            output.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, audio[:, :samples].T.numpy(), seq.sampling_rate, subtype="PCM_24")
            logging.getLogger(__name__).info("Saved %s (seed=%s)", audio_path, args.seed)
    finally:
        os.chdir(previous_dir)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video_only and (args.prompt is not None or not (args.video or args.video_dir)):
        parser.error("--video_only requires a video input and cannot be combined with --prompt.")
    if (not math.isfinite(args.duration) or args.duration < 0.64 or args.num_steps < 1
            or not math.isfinite(args.cfg_strength) or args.cfg_strength < 0 or args.threads < 1):
        parser.error("Require finite duration >= 0.64, num_steps >= 1, CFG >= 0, threads >= 1.")
    if args.video or args.video_dir:
        return run_video(args, parser)
    if not args.prompt or not args.prompt.strip():
        parser.error("Provide --video, --video_dir, or a nonempty --prompt.")
    return run_text(args, parser)


if __name__ == "__main__":
    sys.exit(main())
