"""Multi-GPU ViSAudio inference, with resumable stereo MP4 outputs."""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from visaudio.utils.inference_config import INFERENCE_PROFILE, add_inference_arguments, resolve_prompt

DEFAULT_REPO = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    temporary.replace(path)


def probe_media(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(result.stdout)


def validate_media(path, duration, needs_video):
    media = probe_media(path)
    streams = media.get("streams", [])
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(audio) != 1 or audio[0].get("channels") != 2:
        raise ValueError(f"Expected exactly one two-channel audio stream: {path}")
    if int(audio[0].get("sample_rate", 0)) != 44100:
        raise ValueError(f"Expected 44100 Hz audio: {path}")
    actual = float(audio[0].get("duration", media.get("format", {}).get("duration", 0)))
    if not math.isfinite(actual) or abs(actual - duration) > 0.15:
        raise ValueError(f"Audio duration mismatch: {actual} vs {duration}: {path}")
    if needs_video:
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        if len(video) != 1 or video[0].get("width", 0) <= 0:
            raise ValueError(f"Missing video stream: {path}")
        video_duration = float(video[0].get("duration", media["format"].get("duration", 0)))
        if not math.isfinite(video_duration) or abs(video_duration - duration) > 0.25:
            raise ValueError(f"Video duration mismatch: {video_duration} vs {duration}: {path}")
    return media


def mux_stereo(video, audio, target, duration):
    common = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(audio), "-map", "1:a:0", "-map", "0:v:0",
    ]
    tail = [
        "-c:a", "libmp3lame", "-b:a", "128k", "-ac", "2", "-ar", "44100",
        "-t", str(duration), "-shortest", "-movflags", "+faststart",
        "-threads", "2", str(target),
    ]
    streams = probe_media(video)["streams"]
    source_video = next(stream for stream in streams if stream["codec_type"] == "video")
    try:
        if source_video.get("codec_name") != "h264" or source_video.get("pix_fmt") != "yuv420p":
            raise ValueError("Video requires H.264/yuv420p encoding.")
        subprocess.run(common + ["-c:v", "copy"] + tail, check=True, timeout=600)
        validate_media(target, duration, needs_video=True)
    except (subprocess.CalledProcessError, ValueError):
        print(f"Re-encoding video to H.264: {video}", flush=True)
        subprocess.run(
            common + ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                      "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"] + tail,
            check=True, timeout=600,
        )
        validate_media(target, duration, needs_video=True)


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def make_jobs(input_dir, output_dir, config):
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be outside the input directory.")
    videos = [input_dir] if input_dir.is_file() else sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No supported videos found: {input_dir}")
    jobs, seen = [], set()
    for video in videos:
        relative = Path(video.name) if input_dir.is_file() else video.relative_to(input_dir)
        relative = relative.with_suffix(".mp4")
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate output name (same video stem): {relative}")
        seen.add(key)
        target = output_dir / relative
        if target.resolve() == video.resolve():
            raise ValueError(f"Output would overwrite source video: {video}")
        prompt = (
            resolve_prompt(video, config.get("prompt"), video_only=config.get("video_only", False))
            if config.get("inference_profile") else config.get("prompt", "")
        )
        identity = {"source": file_identity(video), "config": config, "prompt": prompt}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        jobs.append({
            "input": str(video), "output": str(target),
            "signature": signature, "relative": relative.as_posix(), "prompt": prompt,
        })
    return jobs


def is_complete(job):
    target = Path(job["output"])
    try:
        metadata = json.loads(target.with_suffix(".json").read_text())
        if metadata.get("signature") != job["signature"]:
            return False
        duration = float(metadata["duration"])
        if not math.isfinite(duration) or duration < 0.64:
            return False
        validate_media(target, duration, needs_video=True)
        validate_media(target.with_suffix(".flac"), duration, needs_video=False)
        return True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False


class InferenceEngine:
    """One instance per worker; import CUDA only after GPU isolation."""

    def __init__(self, config):
        import torch
        from visaudio.eval_utils import all_model_cfg
        from visaudio.model.flow_matching import FlowMatching
        from visaudio.model.networks import get_my_mmaudio
        from visaudio.model.utils.features_utils import FeaturesUtils
        from visaudio.utils.checkpoint import load_model_state

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected worker environment.")
        torch.set_num_threads(config["threads"])
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.config = config
        self.device = torch.device("cuda:0")
        self.amp = torch.cuda.is_bf16_supported() and not config["full_precision"]
        self.model = replace(
            all_model_cfg["large_44k_binaural"], model_path=Path(config["model"]["path"])
        )
        self.model.download_if_needed()
        with torch.inference_mode():
            print("Loading main model (FP32)", flush=True)
            self.net = get_my_mmaudio(self.model.model_name).to(self.device).eval()
            self.net.load_weights(load_model_state(self.model.model_path))
            print("Loading CLIP, Synchformer, PE, VAE and BigVGAN (FP32)", flush=True)
            self.features = FeaturesUtils(
                tod_vae_ckpt=self.model.vae_path, synchformer_ckpt=self.model.synchformer_ckpt,
                mode=self.model.mode, need_vae_encoder=False,
            ).to(self.device).eval()
        self.fm = FlowMatching(num_steps=config["num_steps"])
        print("All models loaded", flush=True)

    def generate(self, video, audio_path, prompt=None):
        import soundfile as sf
        import torch
        from visaudio.eval_utils import generate
        from visaudio.data.eval.legacy_video import load_eval_video

        with torch.inference_mode():
            info = load_eval_video(video, self.config["duration"])
            duration = self.config["duration"]
            seq = self.model.seq_cfg
            seq.duration = duration
            self.net.update_seq_lengths(
                seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len, seq.pe_seq_len
            )
            rng = torch.Generator(device=self.device).manual_seed(self.config["seed"])
            text = prompt if prompt is not None else resolve_prompt(video, self.config.get("prompt"))
            negative = self.config.get("negative_prompt")
            with torch.amp.autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
                audio = generate(
                    info["clip_video"].unsqueeze(0), info["sync_video"].unsqueeze(0),
                    info["pe_video"].unsqueeze(0), [text],
                    negative_text=[negative] if negative else None,
                    feature_utils=self.features, net=self.net, fm=self.fm, rng=rng,
                    cfg_strength=self.config["cfg_strength"],
                )[0].float().cpu()
            samples = round(duration * seq.sampling_rate)
            if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < samples:
                raise ValueError(f"Expected generated stereo waveform; got {tuple(audio.shape)}")
            audio = audio[:, :samples]
            if not torch.isfinite(audio).all():
                raise ValueError("Generated audio contains NaN or Inf.")
            sf.write(str(audio_path), audio.T.numpy(), seq.sampling_rate, subtype="PCM_24")
        return duration

    def clear_error(self):
        import torch

        gc.collect()
        torch.cuda.empty_cache()


def worker(spec_path, rank):
    spec = json.loads(Path(spec_path).read_text())
    config = spec["config"]
    gpu = spec["gpus"][rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.chdir(config["repo"])
    sys.path.insert(0, config["repo"])
    jobs = spec["jobs"][rank::len(spec["gpus"])]
    report = Path(spec["run_dir"]) / f"worker_{rank}.json"
    results = []
    print(f"GPU {gpu}: loading model once for {len(jobs)} video(s)", flush=True)
    engine = InferenceEngine(config)
    for index, job in enumerate(jobs, 1):
        start = time.monotonic()
        result = dict(job, gpu=gpu, status="failed")
        print(f"[GPU {gpu} {index}/{len(jobs)}] {job['relative']}", flush=True)
        try:
            target = Path(job["output"])
            target.parent.mkdir(parents=True, exist_ok=True)
            # Publish the metadata last so interrupted/partial outputs never count as complete.
            target.with_suffix(".json").unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(prefix=".visaudio-", dir=target.parent) as directory:
                audio = Path(directory) / "audio.flac"
                composite = Path(directory) / "video.mp4"
                duration = engine.generate(Path(job["input"]), audio, prompt=job["prompt"])
                validate_media(audio, duration, needs_video=False)
                mux_stereo(Path(job["input"]), audio, composite, duration)
                audio.replace(target.with_suffix(".flac"))
                composite.replace(target)
            result.update(status="success", duration=duration, channels=2, sample_rate=44100)
            write_json(target.with_suffix(".json"), dict(result, config=config))
        except Exception:  # noqa: BLE001 - isolate per-video failures.
            result["error"] = traceback.format_exc()
            print(result["error"], flush=True)
        if result["status"] == "failed":
            engine.clear_error()
        result["elapsed_seconds"] = round(time.monotonic() - start, 2)
        results.append(result)
        write_json(report, results)
        print(f"{result['status'].upper()}: {job['relative']}", flush=True)
    return int(any(result["status"] != "success" for result in results))


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def launch_workers(spec, spec_path):
    processes = []
    handles = []
    interrupted = False
    try:
        for rank, gpu in enumerate(spec["gpus"]):
            log_path = Path(spec["run_dir"]) / f"gpu_{rank}.log"
            handle = log_path.open("w")
            handles.append(handle)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["OMP_NUM_THREADS"] = str(spec["config"]["threads"])
            env["MKL_NUM_THREADS"] = str(spec["config"]["threads"])
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                "--worker-spec", str(spec_path), "--worker-rank", str(rank),
            ]
            processes.append(subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            ))
            count = len(spec["jobs"][rank::len(spec["gpus"])])
            print(f"GPU {gpu}: {count} video(s); log: {log_path}", flush=True)
        pending = set(range(len(processes)))
        last_progress = None
        while pending:
            progress = []
            for rank in range(len(processes)):
                report = Path(spec["run_dir"]) / f"worker_{rank}.json"
                results = json.loads(report.read_text()) if report.exists() else []
                progress.extend(results)
                if rank in pending and processes[rank].poll() is not None:
                    pending.remove(rank)
                    print(f"GPU {spec['gpus'][rank]} exited: {processes[rank].returncode}", flush=True)
            state = (len(progress), sum(item["status"] == "failed" for item in progress))
            if state != last_progress:
                print(f"Progress: {state[0]}/{len(spec['jobs'])}; failed: {state[1]}", flush=True)
                last_progress = state
            if pending:
                time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping workers; completed files can be resumed.", flush=True)
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
    return interrupted, [process.returncode for process in processes]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--input", "--video_dir", type=Path)
    parser.add_argument("--video", "--video_path", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Default: <repo>/assets/output.")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        help="Comma-separated physical GPU IDs or GPU UUIDs, e.g. 0,1,2,3.")
    parser.add_argument("--model_path", type=Path)
    add_inference_arguments(parser)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate even if matching validated outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without starting CUDA.")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video_only and args.prompt is not None:
        parser.error("--video_only cannot be combined with --prompt.")
    if args.worker_spec is not None:
        return worker(args.worker_spec, args.worker_rank)
    if (not math.isfinite(args.duration) or args.duration < 0.64 or args.num_steps < 1
            or not math.isfinite(args.cfg_strength) or args.cfg_strength < 0 or args.threads < 1):
        parser.error("Require finite duration >= 0.64, num_steps >= 1, CFG >= 0, threads >= 1.")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if (any(not gpu or not (gpu.isdecimal() or gpu.startswith(("GPU-", "MIG-"))) for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        parser.error("--gpus must contain unique, nonempty GPU IDs or UUIDs.")
    repo = args.repo.expanduser().resolve()
    if args.input and args.video:
        parser.error("Use either --input/--video_dir or --video, not both.")
    input_dir = (args.video or args.input or repo / "assets/input").expanduser().resolve()
    output_dir = (args.output or repo / "assets/output").expanduser().resolve()
    checkpoint = args.model_path or Path("weights/visaudio_large_44k.pth")
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    checkpoint = checkpoint.resolve()
    if not (repo / "visaudio/eval_utils.py").is_file():
        parser.error(f"ViSAudio source not found: {repo}")
    if not (input_dir.is_dir() or (args.video and input_dir.is_file())) or not checkpoint.is_file():
        parser.error(f"Check input directory and checkpoint: {input_dir}, {checkpoint}")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            parser.error(f"{executable} is missing from PATH.")
    config = {
        "repo": str(repo), "model": file_identity(checkpoint),
        "duration": args.duration, "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength, "seed": args.seed, "prompt": args.prompt,
        "negative_prompt": args.negative_prompt, "full_precision": args.full_precision,
        "threads": args.threads, "output_format_version": 2,
        "inference_profile": INFERENCE_PROFILE, "seed_policy": "per_video",
        "video_only": args.video_only,
    }
    try:
        jobs = make_jobs(input_dir, output_dir, config)
    except ValueError as exc:
        parser.error(str(exc))
    skipped = [dict(job, status="skipped") for job in jobs
               if not args.overwrite and is_complete(job)]
    skipped_names = {job["relative"] for job in skipped}
    pending = [job for job in jobs if job["relative"] not in skipped_names]
    gpus = gpus[:len(pending)]
    print(f"Input: {input_dir}\nOutput: {output_dir}", flush=True)
    print(f"Videos: {len(jobs)}; validated existing: {len(skipped)}; pending: {len(pending)}")
    mode = "video only" if args.video_only else "video + caption"
    print(f"Duration: {args.duration}s; stereo 44100 Hz; seed: {args.seed}; {mode}")
    if args.duration > 8:
        print("WARNING: durations above 8 seconds are not quality-validated by this model.")
    if args.dry_run:
        for rank, gpu in enumerate(gpus):
            for job in pending[rank::len(gpus)]:
                print(f"GPU {gpu}: {job['input']} -> {job['output']}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent two launchers from assigning the same outputs concurrently.
    import fcntl

    with (output_dir / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another batch is using {output_dir}")
        run_dir = output_dir / "_runs" / (
            datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        )
        run_dir.mkdir(parents=True)
        spec = {"config": config, "gpus": gpus, "jobs": pending, "run_dir": str(run_dir)}
        spec_path = run_dir / "manifest.json"
        write_json(spec_path, spec)
        interrupted, codes = launch_workers(spec, spec_path) if pending else (False, [])
        results = list(skipped)
        for rank in range(len(gpus)):
            report = run_dir / f"worker_{rank}.json"
            if report.exists():
                results.extend(json.loads(report.read_text()))
        reported = {result["relative"] for result in results}
        results.extend(dict(job, status="failed", error="Worker stopped; see GPU log.")
                       for job in jobs if job["relative"] not in reported)
        results.sort(key=lambda item: item["relative"])
        counts = {status: sum(item["status"] == status for item in results)
                  for status in ("success", "skipped", "failed")}
        summary = {"counts": counts, "worker_exit_codes": codes, "results": results,
                   "config": config, "run_dir": str(run_dir)}
        write_json(run_dir / "summary.json", summary)
        write_json(output_dir / "summary.json", summary)
        print(f"Finished: {counts}\nSummary: {output_dir / 'summary.json'}", flush=True)
        return 130 if interrupted else int(counts["failed"] > 0 or any(codes))


def handle_sigterm(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    if "--worker-spec" not in sys.argv:
        signal.signal(signal.SIGTERM, handle_sigterm)
    raise SystemExit(main())
