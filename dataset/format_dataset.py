import os
import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import subprocess as sp
from tqdm import tqdm
import subprocess as sp
from pathlib import Path

def get_media_duration(media_path):
    """Get duration of a video or audio file using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(media_path)
    ]
    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    return float(result.stdout)

def check_video_audio_length(video_path, expected_length=8.0):
    """Check if the video and audio duration in the combined file are correct."""
    duration = get_media_duration(video_path)
    if not (expected_length - 0.1 <= duration <= expected_length + 0.1):
        print(f"Warning: '{video_path.name}' duration is {duration:.2f}s, expected 8s.")
        os.remove(video_path)
        return False
    # print(f"'{video_path.name}' is {duration:.2f}s.")
    return True

def check_all_files():
    DATA_PATH = Path(<path-to-BiAudio>)
    for video_path in tqdm(sorted((DATA_PATH / "video" / "train").glob("*.mp4"))):
        check_video_audio_length(video_path)
    for video_path in tqdm(sorted((DATA_PATH / "video" / "test").glob("*.mp4"))):
        check_video_audio_length(video_path)


def resample_audio(audio_path, output_path, target_sr=44100):
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    target_sr = 44100
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    print(f"Original audio shape: {audio.shape}, Original sample rate: {sr}")
    print(f"Resampled audio shape: {audio_resampled.shape}, Target sample rate: {target_sr}")
    sf.write(output_path, audio_resampled.T, target_sr) 

def resample_all():
    RAW_PATH = Path(<path-to-raw>)
    DATA_PATH = Path(<path-to-BiAudio>)

    (DATA_PATH / "video" / "train").mkdir(exist_ok=True, parents=True)
    (DATA_PATH / "video" / "test").mkdir(exist_ok=True, parents=True)
    train_video_list = set([p.stem for p in (RAW_PATH / "nfov" / "train").glob("*_stereo.flac")])
    test_video_list = set([p.stem for p in (RAW_PATH / "nfov" / "test").glob("*_stereo.flac")])
    
    test_video_list = sorted(list(test_video_list))
    random.shuffle(test_video_list)
    for video_id in tqdm(test_video_list):
        audio_path = RAW_PATH / "nfov" / "test" / f"{video_id}.flac"
        out_audio_path = RAW_PATH / "nfov" / "test" / f"{video_id.replace('_stereo', '_resample.flac')}"
        if not out_audio_path.exists():
            resample_audio(audio_path, out_audio_path, target_sr=44100)
        
        video_path = RAW_PATH / "nfov" / "test" / f"{video_id.replace('_stereo', '_nfov.mp4')}"
        out_av_path = DATA_PATH / "video" / "test" / f"{video_id.replace('_stereo', '.mp4')}"
        
        if out_av_path.exists():
            continue
        cmd = f"ffmpeg -y -i {video_path} -i {out_audio_path} -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 -t 8 {out_av_path}"
        _ = sp.run(cmd, shell=True, capture_output=True)

    train_video_list = sorted(list(train_video_list))
    random.shuffle(train_video_list)
    for video_id in tqdm(train_video_list):
        audio_path = RAW_PATH / "nfov" / "train" / f"{video_id}.flac"
        out_audio_path = RAW_PATH / "nfov" / "train" / f"{video_id.replace('_stereo', '_resample.flac')}"
        if not out_audio_path.exists():
            resample_audio(audio_path, out_audio_path, target_sr=44100)
        
        video_path = RAW_PATH / "nfov" / "train" / f"{video_id.replace('_stereo', '_nfov.mp4')}"
        out_av_path = DATA_PATH / "video" / "train" / f"{video_id.replace('_stereo', '.mp4')}"
        
        if out_av_path.exists():
            continue
        
        cmd = f"ffmpeg -y -i {video_path} -i {out_audio_path} -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 -t 8 {out_av_path}"
        _ = sp.run(cmd, shell=True, capture_output=True)
        
if __name__ == "__main__":
    resample_all()
    # check_all_files()