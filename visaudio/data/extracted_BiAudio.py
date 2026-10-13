"""Read the per-clip feature files produced by the training extractor."""

import io
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
STAT_KEYS = ('left_mean', 'left_std', 'right_mean', 'right_std')


class ExtractedBiAudio(Dataset):
    def __init__(self, tsv_path, *, feature_dir, data_dim, stats_path=None):
        self.feature_dir = str(feature_dir).rstrip('/')
        self.data_dim = data_dim
        self.stats_path = Path(stats_path) if stats_path else None
        frame = pd.read_csv(tsv_path, sep='\t', dtype={'id': str}).fillna({'label': ''})
        if not {'id', 'label'}.issubset(frame.columns) or frame.empty:
            raise ValueError(f'Expected a nonempty TSV with id and label columns: {tsv_path}')
        self.ids = frame['id'].tolist()
        self.labels = frame['label'].tolist()
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f'Duplicate sample IDs in {tsv_path}')
        if any(Path(value).name != value or value in {'.', '..'} for value in self.ids):
            raise ValueError('Sample IDs must be basenames, not paths.')
        self.use_petrel = self.feature_dir.startswith('s3://')
        self.client = None
        log.info('Loaded %d sample IDs from %s', len(self), tsv_path)

    def _load_feature(self, sample_id):
        path = f'{self.feature_dir}/{sample_id}.pt'
        if self.use_petrel:
            if self.client is None:
                from petrel_client.client import Client
                self.client = Client(os.path.expanduser(
                    os.environ.get('PETREL_CONF', '~/.petreloss.conf')))
            source = io.BytesIO(self.client.get(path))
        else:
            source = path
        return torch.load(source, map_location='cpu', weights_only=True)

    def compute_latent_stats(self):
        if self.stats_path and self.stats_path.is_file():
            stats = torch.load(self.stats_path, map_location='cpu', weights_only=True)
            values = tuple(stats[key].float() for key in STAT_KEYS)
        else:
            # Streaming moments avoid requiring a second, hidden memmap dataset.
            counts = [0, 0]
            means = [None, None]
            m2s = [None, None]
            for sample_id in self.ids:
                feature = self._load_feature(sample_id)
                for index, key in enumerate(('left_mean', 'right_mean')):
                    x = feature[key].double()
                    n = x.shape[0]
                    mean = x.mean(0)
                    m2 = (x - mean).square().sum(0)
                    if counts[index] == 0:
                        means[index], m2s[index] = mean, m2
                    else:
                        delta = mean - means[index]
                        total = counts[index] + n
                        m2s[index] += m2 + delta.square() * counts[index] * n / total
                        means[index] += delta * n / total
                    counts[index] += n
            values = tuple(value.float() for index in range(2)
                           for value in (means[index],
                                         (m2s[index] / max(counts[index] - 1, 1)).sqrt()))
        for index, value in enumerate(values):
            if not torch.isfinite(value).all() or (index % 2 and (value <= 0).any()):
                raise ValueError('Latent statistics must be finite with positive standard deviations.')
        return values

    def __getitem__(self, index):
        feature = self._load_feature(self.ids[index])
        shape_specs = {
            'left_mean': (self.data_dim['latent_seq_len'], self.data_dim['latent_dim']),
            'left_std': (self.data_dim['latent_seq_len'], self.data_dim['latent_dim']),
            'right_mean': (self.data_dim['latent_seq_len'], self.data_dim['latent_dim']),
            'right_std': (self.data_dim['latent_seq_len'], self.data_dim['latent_dim']),
            'clip_features': (self.data_dim['clip_seq_len'], self.data_dim['clip_dim']),
            'sync_features': (self.data_dim['sync_seq_len'], self.data_dim['sync_dim']),
            'text_features': (self.data_dim['text_seq_len'], self.data_dim['text_dim']),
            'pe_features': (self.data_dim['pe_seq_len'], self.data_dim['pe_size'],
                            self.data_dim['pe_size'], self.data_dim['pe_dim']),
        }
        for key, shape in shape_specs.items():
            if key not in feature or tuple(feature[key].shape) != shape:
                raise ValueError(f'{self.ids[index]}: expected {key} shape {shape}')
        return {
            'id': self.ids[index], 'caption': self.labels[index],
            **{f'a_{key}': feature[key].float() for key in STAT_KEYS},
            **{key: feature[key] if key == 'pe_features' else feature[key].float()
               for key in shape_specs if key.endswith('_features')},
            'video_exist': torch.tensor(True), 'text_exist': torch.tensor(True),
        }

    def __len__(self):
        return len(self.ids)
