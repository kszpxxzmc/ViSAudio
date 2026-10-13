from math import factorial, pi, sqrt

import numpy as np
import torch


class Position:
    """
    Position in spherical coordinates
    """

    def __init__(self, phi, nu, radius=1.0):
        self.phi, self.nu, self.r = phi, nu, radius

    def _radian_to_degree(self, rad):
        return 180 / np.pi * rad

    def __str__(self):
        return "phi : {}, nu : {}, radius : {}".format(
            self._radian_to_degree(self.phi), self._radian_to_degree(self.nu), self.r
        )


def index_to_degree_order(index):
    order = int(sqrt(index))
    index -= order**2
    degree = index - order
    return order, degree


def normalization_factor(order, degree):
    return sqrt(
        (
            2.0
            - float(degree == 0)
            * float(
                factorial(order - abs(degree)) / float(factorial(order + abs(degree)))
            )
        )
    )


# single spherical harmonics functions
def spherical_harmonic_mn(order, degree, phi, nu):
    from scipy.special import lpmv

    norm = normalization_factor(order, degree)
    sph = (
        (-1) ** degree
        * norm
        * lpmv(abs(degree), order, np.sin(nu))
        * (np.cos(abs(degree) * phi) if degree >= 0 else np.sin(abs(degree) * phi))
    )
    return sph


def spherical_harmonics_matrix(positions, max_order):
    num_channels = int((max_order + 1) ** 2)
    sph_mat = np.zeros((len(positions), num_channels))
    for i, p in enumerate(positions):
        for j in range(num_channels):
            order, degree = index_to_degree_order(j)
            sph_mat[i][j] = spherical_harmonic_mn(order, degree, p.phi, p.nu)
    return sph_mat


def spherical_grid(angular_res):
    """
    Create a unit spherical grid
    """
    phi_rg = np.flip(np.arange(-180.0, 180.0, angular_res) / 180.0 * np.pi, 0)
    nu_rg = np.arange(-90.0, 90.0, angular_res) / 180.0 * np.pi
    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh


#### Audio Decoder ####


class AmbiDecoder:
    def __init__(
        self, sph_grid, ambi_order=1, use_gpu=False
    ):  # , ordering='ACN', normalization='SN3D'):
        self.use_gpu = use_gpu
        self.sph_grid = sph_grid
        self.sph_mat = spherical_harmonics_matrix(sph_grid, ambi_order).T
        if self.use_gpu:
            self.sph_mat = torch.from_numpy(self.sph_mat).cuda()

    def decode(self, samples):
        if self.use_gpu:
            assert samples.size(1) == self.sph_mat.size(0)
        else:
            assert samples.shape[1] == self.sph_mat.shape[0]
        return samples @ self.sph_mat


#### Spherical Ambisonics Visualizer ####


class SphericalAmbisonicsReader:
    def __init__(self, data, rate=48000, window=0.1, angular_res=2.0, use_gpu=False):
        self.data = data
        self.phi_mesh, self.nu_mesh = spherical_grid(angular_res)
        self.mesh_p = [
            Position(phi, nu, 1.0)
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]
        self.use_gpu = use_gpu

        # Setup decoder
        self.decoder = AmbiDecoder(self.mesh_p, ambi_order=1, use_gpu=use_gpu)

        # Compute spherical energy averaged over consecutive chunks of "window" secs
        self.window_frames = int(window * rate)
        self.n_frames = data.shape[0] / self.window_frames
        self.cur_frame = -1

    def get_next_frame(self):
        self.cur_frame += 1
        if self.cur_frame >= self.n_frames:
            return None

        # Decode ambisonics on a grid of speakers
        chunk_ambi = self.data[
            self.cur_frame
            * self.window_frames : ((self.cur_frame + 1) * self.window_frames),
            :,
        ]

        if self.use_gpu:
            chunk_ambi = torch.from_numpy(chunk_ambi).cuda()
        decoded = self.decoder.decode(chunk_ambi)

        # Compute RMS at each speaker
        if self.use_gpu:
            rms = (decoded**2).mean(0).sqrt().reshape(self.phi_mesh.shape).cpu().numpy()
        else:
            rms = np.sqrt(np.mean(decoded**2, 0)).reshape(self.phi_mesh.shape)

        return np.flipud(rms)

    def loop_frames(self):
        while True:
            rms = self.get_next_frame()
            if rms is None:
                break
            yield rms