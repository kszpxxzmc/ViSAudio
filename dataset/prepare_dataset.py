import os
import sys
import json
import tempfile
import random
import base64
import io
from pathlib import Path
import subprocess as sp

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
import ffmpeg
import soundfile as sf
from scipy import signal

from equilib import equi2pers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from utils.view_video import Position, spherical_grid, spherical_harmonics_matrix

RAW_PATH = Path(<path-to-raw>)  # Change this to your desired output path
CLIP_DURATION_SEC = 8

def sec2time(sec):
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def rotx(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])

def roty(angle):
    return np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [ 0,             1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotz(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
 

def is_equirectangular(metadata):
    # Check projection field in side_data_list
    for side_data in metadata.get('side_data_list', []):
        if side_data.get('side_data_type') == 'Spherical Mapping' and side_data.get('projection') == 'equirectangular':
            return True
    return False

def clip(video_id, sec_id, split):
    input_file = RAW_PATH / "raw" / f"{video_id}.webm"
    if not input_file.exists():
        return
    if not (RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.mp4").exists():
        print(f"Clipping video {video_id} at {sec_id}s")
        metadata = ffmpeg.probe(input_file)["streams"][0]
        in_w, in_h = int(metadata["width"]), int(metadata["height"])
        print(f"Processing {input_file}, size: {in_w}x{in_h}")
        cmd = "ffmpeg -loglevel warning -y"
        cmd += f" -ss {sec2time(sec_id)} -to {sec2time(sec_id + CLIP_DURATION_SEC)}"
        cmd += " -i " + str(input_file)
        # cmd += f" -r 4 -s 896x448 "
        cmd += f" -an -s {in_w//2*2}x{in_h//2*2} "
        if not is_equirectangular(metadata):
            cmd += f"-vf v360=c3x2:e:cubic:in_forder=lfrdbu:in_frot=000313 "
        cmd += str(RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.mp4")
        try:
            print(cmd)
            _ = sp.run(cmd, shell=True, capture_output=True)
        except sp.CalledProcessError as e:
            print(e.output)
            (RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.mp4").unlink()
    
    if not (RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.opus").exists():
        print(f"Clipping audio {video_id} at {sec_id}s")
        cmd = f"ffmpeg -loglevel warning -y"
        cmd += " -i " + str(input_file)
        cmd += f" -ss {sec2time(sec_id)} -to {sec2time(sec_id + CLIP_DURATION_SEC)}"
        cmd += f" -c:a copy "
        cmd += str(RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.opus")
        try:
            print(cmd)
            _ = sp.run(cmd, shell=True, capture_output=True)
        except sp.CalledProcessError as e:
            print(e.output)
            (RAW_PATH / "clip" / split / f"{video_id}_{sec_id}.opus").unlink()

# Omnitone first-order FOA HRIR Base64 data
OmnitoneFOAHrirBase64 = [
"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",
]

def decode_hrir(base64_str):
    """Base64 → numpy array (2, N)"""
    hrir_bytes = base64.b64decode(base64_str)
    hrir_file = io.BytesIO(hrir_bytes)
    data, sr = sf.read(hrir_file)
    return data.T  # Convert to (2, N)

# Decode HRIR
hrir_wy = decode_hrir(OmnitoneFOAHrirBase64[0])
hrir_zx = decode_hrir(OmnitoneFOAHrirBase64[1])


def fuma_to_ambix(foa: np.ndarray, fmt: str) -> np.ndarray:
    """
    Convert input FOA to FuMa order/scale (W,X,Y,Z).
    fmt: 'ambix' (ACN/SN3D, W,Y,Z,X) or 'fuma' (FuMa, W,X,Y,Z).
    Returns array of shape (N, 4) in FuMa convention.
    """
    assert foa.ndim == 2 and foa.shape[1] == 4, "Input must be (samples, 4)"
    if fmt.lower() == "ambix":
        # Already in ambix format
        return foa.astype(np.float32, copy=False)

    if fmt.lower() != "fuma":
        raise ValueError("Unsupported format. Use 'ambix' or 'fuma'.")

    # fuma (FuMa) channel order = [W, X, Y, Z]
    # ambix (ACN/SN3D) channel order = [W, Y, Z, X]
    W_a = foa[:, 0]
    X_a = foa[:, 1]
    Y_a = foa[:, 2]
    Z_a = foa[:, 3]

    W_f = W_a / np.sqrt(2.0)
    X_f = X_a / np.sqrt(3.0)
    Y_f = Y_a / np.sqrt(3.0)
    Z_f = Z_a / np.sqrt(3.0)

    out = np.stack([W_f, Y_f, Z_f, X_f], axis=1).astype(np.float32)
    return out


def read_foa_opus_to_np(opus_path, sr=48000):
    """
    Read first-order FOA Opus file (Ambisonic 1 / WXYZ 4ch), return numpy array (N,4)
    and normalize sample rate to sr.
    
    Handle FFmpeg Ambisonics layout not supported issue.
    """
    if not os.path.exists(opus_path):
        raise FileNotFoundError(f"File not found: {opus_path}")

    temp_dir = tempfile.mkdtemp()
    tmp_files = [os.path.join(temp_dir, f"ch{i}.wav") for i in range(4)]

    # Map 4 channels to separate wav files using FFmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i", opus_path,
        "-ar", str(sr),
        "-map_channel", "0.0.0", tmp_files[0],
        "-map_channel", "0.0.1", tmp_files[1],
        "-map_channel", "0.0.2", tmp_files[2],
        "-map_channel", "0.0.3", tmp_files[3],
    ]

    try:
        sp.run(cmd, check=True, capture_output=True)
    except sp.CalledProcessError as e:
        print("FFmpeg output:", e.stderr.decode())
        raise RuntimeError("FFmpeg transcoding failed")

    # Read wav files and stack to numpy (N,4)
    import soundfile as sf
    channels = []
    for f in tmp_files:
        data, sr_read = sf.read(f, dtype="float32")
        if sr_read != sr:
            raise ValueError(f"Wav file sample rate {sr_read} != {sr}")
        channels.append(data)
    
    foa = np.stack(channels, axis=-1)  # shape (N,4)

    for f in tmp_files:
        os.remove(f)
    os.rmdir(temp_dir)

    return foa, sr


def get_rotation_matrix_audio(roll, pitch, yaw):
    R = rotx(roll) @ roty(pitch) @ rotz(yaw)
    return R

def rotate_foa_ambix(foa, rolls, pitches, yaws):
    N = foa.shape[0]
    M = len(rolls)
    frames_per_segment = N // M  # Frames per segment

    foa_rot = np.zeros_like(foa)

    for i in range(M):
        start = i * frames_per_segment
        end = (i+1) * frames_per_segment if i < M-1 else N  # Last segment to end
        
        R_i = get_rotation_matrix_audio(rolls[i], -pitches[i], yaws[i])
        
        # Extract XYZ for this segment
        XYZ = foa[start:end, [3,1,2]]
        
        XYZ_rot = XYZ @ R_i.T
        
        # Put back to FOA
        foa_rot[start:end, 0] = foa[start:end, 0]  # W doesn't rotate
        foa_rot[start:end, 1] = XYZ_rot[:, 1]
        foa_rot[start:end, 2] = XYZ_rot[:, 2]
        foa_rot[start:end, 3] = XYZ_rot[:, 0]
    
    return foa_rot

def convolve_stereo(input_stereo: np.ndarray, ir_stereo: np.ndarray) -> np.ndarray:
    """
    Convolve stereo input with stereo impulse response.
    
    Args:
        input_stereo: Stereo input, shape (2, N)
        ir_stereo: Stereo impulse response, shape (2, M)
        
    Returns:
        Stereo output, shape (2, L)
    """
    # Ensure input is 2D
    if input_stereo.ndim == 1:
        input_stereo = np.stack([input_stereo, input_stereo])
    
    if ir_stereo.ndim == 1:
        ir_stereo = np.stack([ir_stereo, ir_stereo])
    
    # Convolve each channel
    output_left = signal.convolve(input_stereo[0], ir_stereo[0], mode='full')
    output_right = signal.convolve(input_stereo[1], ir_stereo[1], mode='full')
    # Ensure output length matches
    min_length = min(len(output_left), len(output_right))
    output_left = output_left[:min_length]
    output_right = output_right[:min_length]
    
    return np.stack([output_left, output_right])

def foa_to_binaural(foa, hrir_wy, hrir_zx):
    W, Y, Z, X = foa.T

    wy_stereo = np.stack([W, Y])
    zx_stereo = np.stack([Z, X])
    
    wy_convolved = convolve_stereo(wy_stereo, hrir_wy)
    zx_convolved = convolve_stereo(zx_stereo, hrir_zx)
    
    wy_left = wy_convolved[0]
    wy_right = wy_convolved[1]
    
    zx_left = zx_convolved[0]
    zx_right = zx_convolved[1]
    
    left_ear = wy_left + wy_right + zx_left + zx_right
    right_ear = wy_left - wy_right + zx_left + zx_right
    
    return np.column_stack([left_ear, right_ear])


def read_video_to_tensor(video_path, device='cuda'):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    frames = frames.to(device)
    return frames

def save_video_tensor(frames, out_path, fps=30):
    frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (T,H,W,C)
    height, width = frames_np.shape[1:3]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    
    
def get_rpy(fix=False, fps=3, timesteps=100, 
            sigma_yaw=0.15, sigma_pitch=0.15, sigma_roll=0.1,
            A_yaw=1, A_pitch=0.5, A_roll=0.2,
            drift_yaw=10, drift_pitch=2, drift_roll=1): 
    """
    Generate random pitch, yaw, roll angles for camera motion simulation.
    sigma_yaw, sigma_pitch, sigma_roll: standard deviation of noise
    A_pitch, A_yaw: amplitude of pitch and yaw oscillations
    drift_yaw: drift rate for yaw
    """

    # Parameters
    t = np.linspace(0, timesteps / fps, timesteps)
    f = 1.5  # Walking frequency in Hz
    
    # # Oscillatory component
    # pitch = A_pitch * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()
    # yaw = A_yaw * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()
    # roll = A_roll * np.sin(2 * np.pi * (f * t + np.random.rand())) * np.random.rand()

    # Add drift
    rnd_pitch, rnd_yaw, rnd_roll = np.random.rand(), np.random.rand(), np.random.rand()
    if fix:
        rnd_pitch = 0
        rnd_yaw = 0
        rnd_roll = 0
    drift_yaw = random.uniform(-drift_yaw, drift_yaw) if rnd_yaw > 1/3 else 0
    drift_pitch = random.uniform(-drift_pitch, drift_pitch) if rnd_pitch > 1/3 else 0
    drift_roll = random.uniform(-drift_roll, drift_roll) if rnd_roll > 1/3 else 0
    yaw = drift_yaw * t
    pitch = drift_pitch * t
    roll = drift_roll * t

    # Convert to radians
    pitch, yaw, roll = np.radians(pitch), np.radians(yaw), np.radians(roll)

    return pitch, yaw, roll


class AmbiDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-computed spherical grid and harmonics matrix
        self.phi_mesh, self.nu_mesh = spherical_grid(2.0)
        self.mesh_p = [
            Position(phi, nu, 1.0)
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]
        matrix = torch.from_numpy(spherical_harmonics_matrix(self.mesh_p, 1).T).float()
        self.register_buffer("matrix", matrix)
        self.out_shape = (-1, self.phi_mesh.shape[0], self.phi_mesh.shape[1])

    def get_argmax_idx(self, x):
        return [int(x) for x in np.unravel_index(np.argmax(x, axis=None), x.shape)]

    def get_central_max_idx(self, x):
        """
        Return maximum value position in energy map. If multiple maxima exist, return their geometric center.
        x: 2D numpy array
        """
        # Find all maximum positions
        max_val = x.max()
        max_positions = np.argwhere(x == max_val)  # shape (N,2)

        # Calculate geometric center of these points
        center = max_positions.mean(axis=0)

        # Round to nearest integer index
        idx = [int(round(center[0])), int(round(center[1]))]
        return idx
    
    def get_angle_from_map(self, maps):
        return [self.get_argmax_idx(x) for x in maps]

    @torch.no_grad()
    def forward(self, x):
        out = torch.einsum("btc,ch->bth", x, self.matrix)
        out = (out**2).mean(1).sqrt().reshape(self.out_shape)  # BAE
        return [self.get_argmax_idx(np.flipud(x)) for x in out.cpu().numpy()]

    @torch.no_grad()
    def get_energy_map(self, x):
        out = torch.einsum("btc,ch->bth", x, self.matrix)
        out = (out**2).mean(1).sqrt().reshape(self.out_shape)  # BAE
        return [np.flipud(x) for x in out.cpu().numpy()]

def er_sphere_sample(data):
    H, W = data.shape[0], data.shape[1]
    out_data = np.zeros_like(data)
    out = list()
    for ele in range(H):
        R = np.sin(np.pi * (ele + 0.5) / H)
        S = np.round(W * R)
        sample = np.interp(
            x=np.arange(S) * (W - 0.5 * (W / S)) / S,
            xp=np.arange(W),
            fp=data[ele],
            period=360,
        )
        out_data[
            ele,
            W // 2
            - sample.shape[0] // 2 : W // 2
            - sample.shape[0] // 2
            + sample.shape[0],
        ] = sample
        out.extend(sample.tolist())

    return np.array(out)

def get_max_direction(audio_path, out_energy_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decom = AmbiDecomposition().to(device)
    gt_audio, sr = read_foa_opus_to_np(audio_path)
    gt_audio = torch.from_numpy(gt_audio).transpose(0, 1)  # (4,N) tensor

    # [B, T, C] to get energy map
    gt_audio = gt_audio.transpose(0, 1).unsqueeze(0).float().to(device)
    gt_energy_map = decom.get_energy_map(gt_audio)[0]
    idx = decom.get_central_max_idx(gt_energy_map)
    print(f"Max energy index (row, col): {idx}")
    print(f"Predicted DOA (phi, theta): {decom.phi_mesh[idx[0], idx[1]]}, {decom.nu_mesh[idx[0], idx[1]]}")
    
    
    gt_energy_map_2d = decom.get_energy_map(gt_audio)[0]  # shape (H, W)    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(decom.phi_mesh, decom.nu_mesh, gt_energy_map_2d, shading='auto', cmap='inferno')
    plt.colorbar(label='Energy')
    plt.xlabel("Phi (rad)")
    plt.ylabel("Nu (rad)")
    plt.title("2D Energy Map with Exact Angle Mapping")

    # Mark maximum energy position
    phi_max = decom.phi_mesh[idx[0], idx[1]]
    nu_max = decom.nu_mesh[idx[0], idx[1]]
    plt.scatter(phi_max, nu_max, color='cyan', s=50, marker='x', label='Max Energy')
    plt.legend()
    plt.savefig(out_energy_path)

    return decom.phi_mesh[idx[0], idx[1]], decom.nu_mesh[idx[0], idx[1]]

    
def nfov(clip_id, split):
    (RAW_PATH / "nfov" / split).mkdir(exist_ok=True, parents=True)
    
    video_path = RAW_PATH / "clip" / split / f"{clip_id}.mp4"
    audio_path = RAW_PATH / "clip" / split / f"{clip_id}.opus"

    out_traj_path = RAW_PATH / "nfov" / split / f"{clip_id}_rpy.json"
    out_video_path = RAW_PATH / "nfov" / split / f"{clip_id}_nfov.mp4"
    out_audio_path = RAW_PATH / "nfov" / split / f"{clip_id}_stereo.flac"
    out_av_path = RAW_PATH / "nfov" / split / f"{clip_id}_av.mp4"
    
    if os.path.exists(out_traj_path) and os.path.exists(out_video_path) and os.path.exists(out_audio_path) and os.path.exists(out_av_path):
        return
    
    out_energy_path = RAW_PATH / "nfov" / split / f"{clip_id}_energy.png"
    max_phi, max_theta = get_max_direction(audio_path, out_energy_path)
    
    conditional_video = read_video_to_tensor(video_path, device='cuda')
    fps = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS))
    num_frames = conditional_video.shape[0]
    print(f"FPS: {fps}, Number of frames: {num_frames}")

    pitches, yaws, rolls = get_rpy(fix=False, fps=fps, timesteps=num_frames)
    if yaws[-1] - yaws[0] > np.pi/6:
        yaw_delta = random.uniform(-np.pi/4, 0)
    elif yaws[-1] - yaws[0] < -np.pi/6:
        yaw_delta = random.uniform(0, np.pi/4)
    else:
        yaw_delta = random.uniform(-np.pi/6, np.pi/6)
    yaw_start = max_phi + yaw_delta
    
    # pitch_start = max_theta + np.random.uniform(0, max_theta) if max_theta > 0 else np.random.uniform(max_theta, 0)
    pitch_start = np.random.uniform(0, max_theta) if max_theta > 0 else np.random.uniform(max_theta, 0)
    pitch_start = np.clip(pitch_start, -np.pi/6, np.pi/6)
    yaws = yaws + yaw_start
    pitches = pitches + pitch_start
    FOV = 90
    rots = [{"roll": rolls[i], "pitch": pitches[i], "yaw": yaws[i]} for i in range(num_frames)]
    conditional_video_pers = equi2pers(
        conditional_video.to(torch.float32), fov_x=FOV,
        width=512, height=512,
        rots=rots, z_down=True
    )
    save_video_tensor(conditional_video_pers, out_video_path, fps=fps)

    data = {
        "yaws": np.degrees(yaws).tolist(),
        "pitches": np.degrees(pitches).tolist(),
        "rolls": np.degrees(rolls).tolist()
    }
    with open(out_traj_path, 'w') as f:
       json.dump(data, f, indent=4)

    print(f"Saved NFOV video to {out_video_path}")
    
    if audio_path.suffix.lower() in [".wav", ".flac", ".ogg"]:
        foa, fs = sf.read(audio_path, dtype="float32")
        foa = fuma_to_ambix(foa, fmt='ambix')  # (N,4)
    else: # opus
        foa, sr = read_foa_opus_to_np(audio_path)
    foa_rot = rotate_foa_ambix(foa, rolls, pitches, yaws)
    binaural = foa_to_binaural(foa_rot, hrir_wy, hrir_zx)
    sf.write(out_audio_path, binaural, sr)
    
    print(f"Saved binaural audio to {out_audio_path}")


if __name__ == "__main__":
    command = sys.argv[-1]
    if command == "clip":
        (RAW_PATH / "clip" / "train").mkdir(exist_ok=True, parents=True)
        (RAW_PATH / "clip" / "test").mkdir(exist_ok=True, parents=True)
        train_txt = "./split/train.txt"
        test_txt = "./split/test.txt"
        with open(train_txt, 'r') as f:
            train_list = f.read().splitlines()
            for cid in train_list:
                video_id, sec_id = cid.rsplit("_", 1)
                clip(video_id, int(sec_id), "train")
        with open(test_txt, 'r') as f:
            test_list = f.read().splitlines()
            for cid in test_list:
                video_id, sec_id = cid.rsplit("_", 1)
                clip(video_id, int(sec_id), "test")
            
    elif command == "nfov":
        (RAW_PATH / "nfov" / "train").mkdir(exist_ok=True, parents=True)
        (RAW_PATH / "nfov" / "test").mkdir(exist_ok=True, parents=True)
        train_video_list = set([p.stem for p in (RAW_PATH / "clip" / "train").glob("*.mp4")])
        test_video_list = set([p.stem for p in (RAW_PATH / "clip" / "test").glob("*.mp4")])
        for video_id in tqdm(test_video_list):
            nfov(video_id, "test")
        for video_id in tqdm(sorted(train_video_list)):
            nfov(video_id, "train")
            
    else:
        raise NotImplementedError(f"Unknown command: {command}")
        
