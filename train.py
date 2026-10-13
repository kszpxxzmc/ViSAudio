"""Train ViSAudio on extracted features with torchrun."""

import logging
import random
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.distributed.elastic.multiprocessing.errors import record

from visaudio.data.data_setup import setup_training_datasets, setup_val_datasets
from visaudio.utils.dist_utils import global_rank, local_rank, world_size

log = logging.getLogger(__name__)


def resolve_start(cfg, run_dir):
    """Resume takes precedence over initialization; never silently restart."""
    if cfg.resume:
        path = Path(cfg.resume)
        if not path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {path}')
        return 'resume', path
    previous = Path(run_dir) / f'{cfg.exp_id}_ckpt_last.pth'
    if cfg.auto_resume and previous.is_file():
        return 'resume', previous
    if cfg.weights:
        return 'weights', Path(cfg.weights)
    if cfg.checkpoint:
        return 'initialize', Path(cfg.checkpoint)
    return 'scratch', None


def prepare_training_weights(cfg, start_kind, start_path):
    from visaudio.utils.download_utils import download_model_if_needed

    if start_path is not None:
        start_path = Path(start_path)
        if not start_path.is_file():
            if start_kind == 'initialize' and start_path.name == 'mmaudio_large_44k_v2.pth':
                download_model_if_needed(start_path)
            else:
                raise FileNotFoundError(f'Checkpoint not found: {start_path}')
    vae_path = Path(cfg.vae_44k_ckpt)
    if not vae_path.is_file():
        download_model_if_needed(vae_path)


@record
@hydra.main(version_base='1.3', config_path='config', config_name='train_config')
def train(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA. Launch with torchrun.')
    if cfg.batch_size < world_size or cfg.batch_size % world_size:
        raise ValueError('Global batch_size must be positive and divisible by WORLD_SIZE.')
    if cfg.model not in {'large_44k_binaural', 'large_44k_v2'}:
        raise ValueError('Only the large_44k_binaural architecture is supported.')
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed.init_process_group('nccl', timeout=timedelta(hours=2))
    try:
        run_training(cfg)
    finally:
        distributed.destroy_process_group()


def run_training(cfg):
    from visaudio.runner import Runner
    from visaudio.utils.logger import TensorboardLogger

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    start_kind, start_path = resolve_start(cfg, run_dir)
    # Only rank zero downloads into the shared checkpoint directory.
    preparation_error = [None]
    if global_rank == 0:
        try:
            prepare_training_weights(cfg, start_kind, start_path)
        except Exception as exc:
            preparation_error[0] = f'{type(exc).__name__}: {exc}'
    distributed.broadcast_object_list(preparation_error, src=0)
    if preparation_error[0] is not None:
        raise RuntimeError(f'Weight preparation failed: {preparation_error[0]}')
    logger = TensorboardLogger(cfg.exp_id, run_dir, log, is_rank0=global_rank == 0,
                               enable_email=cfg.enable_email)
    logger.info(f'Configuration: {cfg}')
    logger.info(f'Start mode: {start_kind}, checkpoint: {start_path}')
    cfg.batch_size //= world_size
    dataset, sampler, loader = setup_training_datasets(cfg)
    _, val_loader, eval_loader = setup_val_datasets(cfg)
    val_cfg = cfg.data[f'Extracted_{cfg.val_dataset}_val']

    # Resume/ViSAudio weights already contain normalization statistics.
    stats = [None] * 4
    if start_kind not in {'resume', 'weights'}:
        if global_rank == 0:
            stats = list(dataset.compute_latent_stats())
        distributed.broadcast_object_list(stats, src=0)
    trainer = Runner(cfg, log=logger, run_path=run_dir, for_training=True,
                     left_latent_mean=stats[0], left_latent_std=stats[1],
                     right_latent_mean=stats[2], right_latent_std=stats[3])
    if start_kind == 'resume':
        step = trainer.load_checkpoint(start_path)
    else:
        step = 0
        if start_kind == 'initialize':
            trainer.load_checkpoint_finetune(start_path)
        elif start_kind == 'weights':
            trainer.load_weights(start_path)
    trainer.initialize_ema()
    trainer.enter_train()

    eval_rng = torch.Generator(device='cuda').manual_seed(cfg.seed + global_rank).get_state()
    epoch = step // len(loader)
    offset = step % len(loader)
    while step < cfg.num_iterations:
        sampler.set_epoch(epoch)
        trainer.log.data_timer.start()
        for batch_index, data in enumerate(loader):
            if batch_index < offset:
                continue
            trainer.train_pass(data, step)
            step += 1
            if cfg.val_interval > 0 and step % cfg.val_interval == 0:
                train_rng = trainer.rng.get_state()
                trainer.rng.set_state(eval_rng)
                for batch in val_loader:
                    trainer.validation_pass(batch, step)
                distributed.barrier()
                trainer.val_integrator.finalize('val', step, ignore_timer=True)
                trainer.val_integrator.reset_except_hooks()
                trainer.rng.set_state(train_rng)
            if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
                train_rng = trainer.rng.get_state()
                trainer.rng.set_state(eval_rng)
                save_eval = cfg.save_eval_interval > 0 and step % cfg.save_eval_interval == 0
                audio_path = None
                for batch in eval_loader:
                    audio_path = trainer.inference_pass(batch, step, val_cfg, save_eval=save_eval)
                distributed.barrier()
                if cfg.enable_metrics and global_rank == 0 and audio_path is not None:
                    trainer.eval(audio_path, step, val_cfg)
                distributed.barrier()
                trainer.rng.set_state(train_rng)
            if step >= cfg.num_iterations:
                break
        offset = 0
        epoch += 1
    trainer.save_checkpoint(step)
    trainer.save_weights(step)
    if global_rank == 0 and trainer.ema is not None:
        index = list(cfg.ema.sigma_rels).index(cfg.ema.default_output_sigma)
        path = run_dir / f'{cfg.exp_id}_ema_final.pth'
        torch.save(trainer.ema.ema_models[index].ema_model.state_dict(), path)
        logger.info(f'Final EMA weights saved to {path}')
    distributed.barrier()
    logger.complete()


if __name__ == '__main__':
    train()
