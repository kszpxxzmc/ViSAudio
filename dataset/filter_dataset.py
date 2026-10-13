import torchaudio
import librosa
import numpy as np
import os
import io
import torch
import ffmpeg
import csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def is_binaural(audio_path, threshold):
    if audio_path.as_posix().endswith(".mp4"):
        out, _ = (
        ffmpeg.input(audio_path)
        .output('pipe:1', format='wav', acodec='pcm_s16le', ac=2, ar=44100)
        .run(capture_stdout=True, capture_stderr=True)
        )
        audio_data = io.BytesIO(out)
        try:
            waveform, sample_rate = torchaudio.load(audio_data)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return False
    else:
        waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.ndim == 1:
        return False

    waveform = waveform - waveform.mean()
        
    # percent = 0.1
    # abs_wave = waveform.abs()
    # sorted_vals, _ = torch.sort(abs_wave.flatten())
    # noise_level = torch.mean(sorted_vals[:int(len(sorted_vals) * percent)]).detach().cpu().numpy()
    # if noise_level > 0.001:
    #     return False
    
    waveform = waveform / (waveform.abs().max() + 1e-9)
    left, right = waveform[0], waveform[1]
    diff = np.mean(np.abs(left.detach().cpu().numpy() - right.detach().cpu().numpy()))
    
    return diff > threshold



def filter_binaural_files(input_dir, threshold):
    audio_exts = ('.mp4')
    binaural_files = []
    
    for root, _, files in os.walk(input_dir):
        print("Scanning folder:", root)
        for f in tqdm(files):
            if f.lower().endswith(audio_exts):
                path = os.path.join(root, f)
                if is_binaural(Path(path), threshold):
                    binaural_files.append(path)

    return binaural_files

def get_binaural_list():
    DATA_PATH = <path-to-BiAudio>
    test_folder = f"{DATA_PATH}/video/test"
    test_binaural_list = filter_binaural_files(test_folder, threshold=0.01)
    print("test_binaural_list:", len(test_binaural_list))
    test_tsv = f"{DATA_PATH}/caption/test.tsv"
    filter_tsv = f"{DATA_PATH}/caption/test_valid.tsv"
    df = pd.read_csv(test_tsv, sep="\t")
    basename_set = set(os.path.basename(f)[:-4] for f in test_binaural_list)
    df_filtered = df[df['id'].isin(basename_set)]
    df_filtered.to_csv(filter_tsv, sep="\t", index=False)

    train_folder = f"{DATA_PATH}/video/train"
    train_binaural_list = filter_binaural_files(train_folder, threshold=0.01)
    print("train_binaural_list:", len(train_binaural_list))
    train_tsv = f"{DATA_PATH}/caption/train.tsv"
    filter_tsv = f"{DATA_PATH}/caption/train_valid.tsv"
    df = pd.read_csv(train_tsv, sep="\t")
    basename_set = set(os.path.basename(f)[:-4] for f in train_binaural_list)
    df_filtered = df[df['id'].isin(basename_set)]
    df_filtered.to_csv(filter_tsv, sep="\t", index=False)
        
if __name__ == "__main__":
    get_binaural_list()