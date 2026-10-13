import logging
import os
from pathlib import Path
from dataclasses import replace

import hydra
import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm

from visaudio.data.data_setup import setup_eval_dataset
from visaudio.eval_utils import ModelConfig, all_model_cfg, generate
from visaudio.model.flow_matching import FlowMatching
from visaudio.model.networks import MMAudio, get_my_mmaudio
from visaudio.model.utils.features_utils import FeaturesUtils
from visaudio.utils.checkpoint import load_model_state
from demo_batch import mux_stereo

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

local_rank = int(os.environ.get("LOCAL_RANK", 0))
log = logging.getLogger()


@torch.inference_mode()
@hydra.main(version_base="1.3", config_path="config", config_name="eval_config.yaml")
def main(cfg: DictConfig):
    if cfg.batch_size != 1:
        raise ValueError("Use batch_size=1 for the shared per-video reproducible profile.")
    device = "cuda"
    torch.cuda.set_device(local_rank)

    if cfg.model not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {cfg.model}")
    if cfg.duration_s < 0.64:
        raise ValueError("duration_s must be at least 0.64 seconds.")
    model: ModelConfig = replace(all_model_cfg[cfg.model], model_path=Path(cfg.model_path))
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    run_dir = Path(cfg.run_dir)
    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f"{cfg.dataset}-{cfg.output_name}"
    print("output_dir:", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_cfg.duration = cfg.duration_s
    net: MMAudio = get_my_mmaudio(cfg.model).to(device).eval()
    net.load_weights(load_model_state(model.model_path))
    log.info(f"Loaded weights from {model.model_path}")
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len, seq_cfg.pe_seq_len
    )
    log.info(f"Latent seq len: {seq_cfg.latent_seq_len}")
    log.info(f"Clip seq len: {seq_cfg.clip_seq_len}")
    log.info(f"Sync seq len: {seq_cfg.sync_seq_len}")
    log.info(f"Pe seq len: {seq_cfg.pe_seq_len}")

    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(
        cfg.sampling.min_sigma, inference_mode=cfg.sampling.method, num_steps=cfg.sampling.num_steps
    )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device).eval()

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    dataset, loader = setup_eval_dataset(cfg.dataset, cfg)

    with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
        for batch in tqdm(loader):
            if batch is None:
                continue
            rng.manual_seed(cfg.seed)
            audios = generate(
                batch.get("clip_video", None),
                batch.get("sync_video", None),
                batch.get("pe_video", None),
                batch.get("caption", None),
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg.cfg_strength,
                clip_batch_size_multiplier=64,
                sync_batch_size_multiplier=64,
                pe_batch_size_multiplier=64,
            )
            audios = audios.float().cpu()
            names = batch["name"]
            for audio, name in zip(audios, names):
                audio = audio[:, : round(cfg.duration_s * seq_cfg.sampling_rate)]
                import soundfile as sf

                audio_path = output_dir / f"{name}.flac"
                sf.write(audio_path, audio.T.numpy(), seq_cfg.sampling_rate, subtype="PCM_24")
                source = Path(cfg.eval_data[cfg.dataset].video_path) / f"{name}.mp4"
                target = output_dir / f"{name}.mp4"
                if source.resolve() == target.resolve():
                    raise ValueError("Output video must not overwrite the input video.")
                mux_stereo(source, audio_path, target, cfg.duration_s)


if __name__ == "__main__":
    main()
