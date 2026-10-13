"""Extract per-clip binaural, CLIP, Synchformer, PE, and text features."""

from argparse import ArgumentParser
from pathlib import Path
import sys

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['BiAudio', 'MUSIC'], required=True)
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    parser.add_argument('--video_root', type=Path, required=True)
    parser.add_argument('--tsv', type=Path, required=True)
    parser.add_argument('--feature_dir', type=Path)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--frame_batch_size', type=int, default=8)
    parser.add_argument('--full_precision', action='store_true')
    args = parser.parse_args()

    import pandas as pd
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torio.utils.ffmpeg_utils import get_versions
    from visaudio.data.data_setup import EvaluationSampler, error_avoidance_collate
    from visaudio.data.extraction.BiAudio_dataset import BiAudio
    from visaudio.data.extracted_BiAudio import ExtractedBiAudio, STAT_KEYS
    from visaudio.eval_utils import large_44k_binaural
    from visaudio.model.utils.features_utils import FeaturesUtils
    from visaudio.utils.dist_utils import local_rank, global_rank, world_size

    if not torch.cuda.is_available():
        raise RuntimeError('Feature extraction requires CUDA.')
    get_versions()
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group('nccl')
    try:
        root = args.feature_dir or Path('training') / args.dataset / 'feature'
        target = root / args.split
        target.mkdir(parents=True, exist_ok=True)
        model = large_44k_binaural
        model.download_if_needed()
        seq = model.seq_cfg
        dataset = BiAudio(args.video_root, tsv_path=args.tsv,
                          sample_rate=seq.sampling_rate, duration_sec=seq.duration,
                          audio_samples=seq.num_audio_frames)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=EvaluationSampler(dataset), num_workers=args.num_workers,
                            collate_fn=error_avoidance_collate)
        features = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                 synchformer_ckpt=model.synchformer_ckpt,
                                 mode='44k', need_vae_encoder=True).cuda().eval()
        rows = []
        with torch.inference_mode(), torch.autocast(
                'cuda', dtype=torch.bfloat16,
                enabled=not args.full_precision and torch.cuda.is_bf16_supported()):
            if global_rank == 0:
                empty_path = Path('ext_weights/empty_string.pth')
                if not empty_path.exists():
                    torch.save(features.encode_text(['']).float().cpu(), empty_path)
            for batch in tqdm(loader, disable=global_rank != 0):
                if batch is None:
                    continue
                left = features.encode_audio(batch['audio_left'].cuda())
                right = features.encode_audio(batch['audio_right'].cuda())
                output = {
                    'left_mean': left.mean.transpose(1, 2),
                    'left_std': left.std.transpose(1, 2),
                    'right_mean': right.mean.transpose(1, 2),
                    'right_std': right.std.transpose(1, 2),
                    'clip_features': features.encode_video_with_clip(
                        batch['clip_video'].cuda(), args.frame_batch_size),
                    'sync_features': features.encode_video_with_sync(
                        batch['sync_video'].cuda(), args.frame_batch_size),
                    'pe_features': features.encode_video_with_pe(
                        batch['pe_video'].cuda(), args.frame_batch_size),
                    'text_features': features.encode_text(batch['caption']),
                }
                for index, sample_id in enumerate(batch['id']):
                    sample = {key: value[index].detach().cpu().to(
                        torch.float16 if key == 'pe_features' else torch.float32)
                              for key, value in output.items()}
                    torch.save(sample, target / f'{sample_id}.pt')
                    rows.append({'id': sample_id, 'label': batch['caption'][index]})
        pd.DataFrame(rows, columns=['id', 'label']).to_csv(
            root / f'.{args.split}.rank{global_rank}.tsv', sep='\t', index=False)
        if world_size > 1:
            dist.barrier()
        if global_rank == 0:
            frames = [pd.read_csv(root / f'.{args.split}.rank{rank}.tsv', sep='\t',
                                  dtype={'id': str}).fillna({'label': ''})
                      for rank in range(world_size)]
            result = pd.concat(frames).sort_values('id')
            if result.empty:
                raise RuntimeError('No clips were extracted successfully.')
            manifest = root / f'{args.dataset}-{args.split}.tsv'
            result.to_csv(manifest, sep='\t', index=False)
            print(f'Extracted {len(result)}/{len(dataset)} clips into {target}')
            if args.split == 'train':
                extracted = ExtractedBiAudio(manifest, feature_dir=target, data_dim={})
                stats = dict(zip(STAT_KEYS, extracted.compute_latent_stats()))
                torch.save(stats, root / 'train_stats.pt')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
