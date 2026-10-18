"""Local tidal-tensor diagnostics for the FDM orbit integrations.

The exported Agama multipole files are read directly, so this module does not
require a working Agama installation.  Coefficients use Agama's real spherical
harmonic convention, normalized such that the l=m=0 basis function is unity.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import sph_harm


KM_S_TO_KPC_GYR = 1.022712165045695


def _real_spherical_harmonics(lmax, xyz):
    """Return Agama-normalized real spherical harmonics at Cartesian points."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    radius = np.linalg.norm(xyz, axis=1)
    polar = np.zeros(len(xyz))
    azimuth = np.zeros(len(xyz))
    nonzero = radius > 0
    polar[nonzero] = np.arccos(np.clip(xyz[nonzero, 2] / radius[nonzero], -1, 1))
    azimuth[nonzero] = np.arctan2(xyz[nonzero, 1], xyz[nonzero, 0])

    result = []
    for ell in range(lmax + 1):
        for emm in range(-ell, ell + 1):
            if emm == 0:
                value = np.sqrt(4 * np.pi) * sph_harm(0, ell, azimuth, polar).real
            elif emm > 0:
                value = np.sqrt(8 * np.pi) * sph_harm(emm, ell, azimuth, polar).real
            else:
                value = np.sqrt(8 * np.pi) * sph_harm(-emm, ell, azimuth, polar).imag
            result.append(value)
    return np.asarray(result).T


class MultipoleSnapshot:
    """Small reader/evaluator for one exported Agama Multipole potential."""

    def __init__(self, path):
        self.path = Path(path)
        lines = self.path.read_text().splitlines()
        grid_size = int(next(line.split("=", 1)[1] for line in lines if line.startswith("gridSizeR=")))
        self.lmax = int(next(line.split("=", 1)[1] for line in lines if line.startswith("lmax=")))
        rows = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped[0] in "[#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            values = np.fromstring(stripped, sep=" ")
            if values.size:
                rows.append(values)
        # Agama exports Phi_lm first and dPhi_lm/dr on a second copy of the
        # radial grid.  Potential evaluation uses only the first block.
        table = np.vstack(rows[:grid_size])
        expected = 1 + (self.lmax + 1) ** 2
        if table.shape[1] != expected:
            raise ValueError(f"{self.path}: expected {expected} columns, found {table.shape[1]}")
        self.radius = table[:, 0]
        self.coefficients = table[:, 1:]
        self._build_interpolator()

    def _build_interpolator(self):
        self._interpolator = PchipInterpolator(
            np.log(self.radius), self.coefficients, axis=0, extrapolate=False
        )

    @classmethod
    def interpolate(cls, first, second, fraction):
        """Linearly interpolate two snapshots as Agama's Evolving potential does."""
        if not np.allclose(first.radius, second.radius, rtol=0, atol=1e-13):
            raise ValueError("Cannot interpolate multipoles with different radial grids")
        fraction = float(fraction)
        result = cls.__new__(cls)
        result.path = (first.path, second.path)
        result.lmax = max(first.lmax, second.lmax)
        result.radius = first.radius.copy()
        coefficient_count = (result.lmax + 1) ** 2
        first_coefficients = np.zeros((len(result.radius), coefficient_count))
        second_coefficients = np.zeros_like(first_coefficients)
        first_coefficients[:, : first.coefficients.shape[1]] = first.coefficients
        second_coefficients[:, : second.coefficients.shape[1]] = second.coefficients
        result.coefficients = (
            (1 - fraction) * first_coefficients + fraction * second_coefficients
        )
        result._build_interpolator()
        return result

    def potential(self, xyz):
        """Evaluate Phi in (km/s)^2 at one or more Cartesian positions in kpc."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        radius = np.linalg.norm(xyz, axis=1)
        if np.any(radius > self.radius[-1]):
            raise ValueError(f"Evaluation radius exceeds multipole grid ({self.radius[-1]:g} kpc)")

        coeff = np.empty((len(xyz), self.coefficients.shape[1]))
        nonzero = radius > 0
        clipped = np.clip(radius[nonzero], self.radius[0], self.radius[-1])
        coeff[nonzero] = self._interpolator(np.log(clipped))
        coeff[~nonzero] = 0.0
        coeff[~nonzero, 0] = self.coefficients[0, 0]
        basis = _real_spherical_harmonics(self.lmax, xyz)
        return np.sum(coeff * basis, axis=1)


def tidal_tensor(snapshot, center, step_kpc=0.01):
    """Numerically evaluate T_ij=d_i d_j Phi at ``center``.

    Diagonal and mixed derivatives both use centered second-order differences.
    The returned units are (km/s)^2/kpc^2.
    """
    center = np.asarray(center, dtype=float)
    step = float(step_kpc)
    eye = np.eye(3) * step
    phi0 = snapshot.potential(center)[0]
    tensor = np.zeros((3, 3))
    for i in range(3):
        values = snapshot.potential([center + eye[i], center - eye[i]])
        tensor[i, i] = (values[0] - 2 * phi0 + values[1]) / step**2
        for j in range(i):
            values = snapshot.potential(
                [
                    center + eye[i] + eye[j],
                    center + eye[i] - eye[j],
                    center - eye[i] + eye[j],
                    center - eye[i] - eye[j],
                ]
            )
            tensor[i, j] = tensor[j, i] = (values[0] - values[1] - values[2] + values[3]) / (4 * step**2)
    return tensor


def recover_com(xvta, dt_gyr=0.1):
    """Recover R_cm by integrating the absolute mean velocity saved in xvta."""
    velocity = np.mean(xvta[:, 3:, :], axis=2)
    center = np.zeros_like(velocity)
    center[1:] = np.cumsum(
        0.5 * (velocity[1:] + velocity[:-1]) * dt_gyr * KM_S_TO_KPC_GYR,
        axis=0,
    )
    return center, velocity


def measure_tensor_series(
    case_dir,
    centers,
    step_kpc=0.01,
    nsnap=None,
    times_gyr=None,
    potential_dt_gyr=0.1,
):
    """Measure a tensor series for one or more center trajectories."""
    case_dir = Path(case_dir)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 2:
        centers = centers[None, ...]
    available = sorted(case_dir.glob("sph_harm_pot_gradual[0-9][0-9][0-9].ini"))
    if nsnap is None:
        nsnap = min(centers.shape[1], len(available))
    if nsnap > len(available):
        raise ValueError(f"Requested {nsnap} snapshots, but only {len(available)} potentials exist")
    if times_gyr is not None:
        times_gyr = np.asarray(times_gyr, dtype=float)[:nsnap]
        if len(times_gyr) != nsnap:
            raise ValueError("times_gyr is shorter than the requested tensor series")

    output = np.empty((centers.shape[0], nsnap, 3, 3))
    cache = {}
    for index in range(nsnap):
        if times_gyr is None:
            snapshot = MultipoleSnapshot(available[index])
        else:
            coordinate = times_gyr[index] / potential_dt_gyr
            lower = min(int(np.floor(coordinate)), len(available) - 1)
            upper = min(lower + 1, len(available) - 1)
            fraction = coordinate - lower if upper > lower else 0.0
            if lower not in cache:
                cache[lower] = MultipoleSnapshot(available[lower])
            if upper not in cache:
                cache[upper] = MultipoleSnapshot(available[upper])
            snapshot = (
                cache[lower]
                if upper == lower or fraction == 0
                else MultipoleSnapshot.interpolate(cache[lower], cache[upper], fraction)
            )
        for case in range(centers.shape[0]):
            output[case, index] = tidal_tensor(snapshot, centers[case, index], step_kpc)
    return output


def evolve_local_tides(x0, v0, tensors, dt_gyr=0.1, substeps=20):
    """Leapfrog integration of d2 xi / dt2 = -T(t) xi."""
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    tensors = np.asarray(tensors, dtype=float)
    position = np.empty((len(tensors), 3, len(x0)))
    velocity = np.empty_like(position)
    position[0] = x0.T
    velocity[0] = v0.T
    x = x0.copy()
    v = v0.copy()
    small_dt = dt_gyr / substeps

    for index in range(len(tensors) - 1):
        for substep in range(substeps):
            fraction0 = substep / substeps
            fraction1 = (substep + 1) / substeps
            tensor0 = (1 - fraction0) * tensors[index] + fraction0 * tensors[index + 1]
            tensor1 = (1 - fraction1) * tensors[index] + fraction1 * tensors[index + 1]
            acceleration = -(x @ tensor0.T) * KM_S_TO_KPC_GYR
            v += 0.5 * small_dt * acceleration
            x += small_dt * v * KM_S_TO_KPC_GYR
            acceleration = -(x @ tensor1.T) * KM_S_TO_KPC_GYR
            v += 0.5 * small_dt * acceleration
        position[index + 1] = x.T
        velocity[index + 1] = v.T
    return np.concatenate((position, velocity), axis=1)


def cluster_diagnostics(xvta):
    """Projected half-light radius and mean 1D velocity dispersion."""
    centered_position = xvta[:, :3] - np.mean(xvta[:, :3], axis=2, keepdims=True)
    projected_radius = np.linalg.norm(centered_position[:, :2], axis=1)
    half_radius = np.median(projected_radius, axis=1)
    sigma = np.mean(np.std(xvta[:, 3:], axis=2), axis=1)
    return half_radius, sigma
