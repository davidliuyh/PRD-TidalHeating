"""Isolated numerical controls for the FDM tidal-heating calculation."""

from pathlib import Path
import ctypes

import numpy as np


KM_S_TO_KPC_GYR = 1.022712165045695
AGAMA_GYR_CONVERSION = 0.978


def load_agama():
    """Load the local GSL libraries and return the Agama module."""
    gsl_dir = Path.home() / ".local" / "opt" / "gsl" / "lib"
    ctypes.CDLL(str(gsl_dir / "libgslcblas.so.0"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(gsl_dir / "libgsl.so.27"), mode=ctypes.RTLD_GLOBAL)
    import agama as ag

    ag.setUnits(mass=1, length=1, velocity=1)
    return ag


def derivatives_to_tidal(derivatives):
    """Convert Agama's packed acceleration derivatives into Hessian(Phi)."""
    derivatives = np.asarray(derivatives)
    tensor = np.zeros(derivatives.shape[:-1] + (3, 3))
    tensor[..., 0, 0] = -derivatives[..., 0]
    tensor[..., 1, 1] = -derivatives[..., 1]
    tensor[..., 2, 2] = -derivatives[..., 2]
    tensor[..., 0, 1] = tensor[..., 1, 0] = -derivatives[..., 3]
    tensor[..., 1, 2] = tensor[..., 2, 1] = -derivatives[..., 4]
    tensor[..., 2, 0] = tensor[..., 0, 2] = -derivatives[..., 5]
    return tensor


def tidal_series(potential, centers, times_gyr):
    derivatives = potential.eval(
        np.asarray(centers),
        t=np.asarray(times_gyr) / AGAMA_GYR_CONVERSION,
        der=True,
    )
    return derivatives_to_tidal(derivatives)


def build_static_potential(
    density_paths,
    output_path,
    lmax=20,
    grid_size_r=25,
    rmin=1e-4,
    rmax=30.0,
):
    """Build and export a static Multipole potential from one or more densities."""
    ag = load_agama()
    output_path = Path(output_path)
    if output_path.exists():
        return ag.Potential(str(output_path))
    densities = [ag.Density(str(Path(path))) for path in density_paths]
    density = densities[0] if len(densities) == 1 else ag.Density(*densities)
    potential = ag.Potential(
        type="Multipole",
        density=density,
        lmax=lmax,
        gridSizeR=grid_size_r,
        rmin=rmin,
        rmax=rmax,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    potential.export(str(output_path))
    return potential


def integrate_control(
    potential,
    initial_conditions,
    output_path,
    t_step_gyr=0.1,
    timesteps=110,
    trajsize=1000,
):
    """Integrate an orbit and save uniform, endpoint-inclusive output samples."""
    output_path = Path(output_path)
    end_time_gyr = timesteps * t_step_gyr
    if output_path.exists():
        with np.load(output_path) as cached:
            cached_time = cached.get("time_gyr")
            if (
                cached_time is not None
                and len(cached_time) == timesteps + 1
                and np.isclose(cached_time[0], 0.0)
                and np.isclose(cached_time[-1], end_time_gyr)
            ):
                return output_path
        print(f"Regenerating stale orbit output without the {end_time_gyr:g} Gyr endpoint: {output_path}")
    ag = load_agama()
    initial_conditions = np.asarray(initial_conditions)
    sample_coordinates = np.linspace(0.0, trajsize, timesteps + 1)
    lower_indices = np.floor(sample_coordinates).astype(int)
    upper_indices = np.ceil(sample_coordinates).astype(int)
    sample_fractions = sample_coordinates - lower_indices
    sample_indices = np.rint(sample_coordinates).astype(int)
    time_gyr = np.linspace(0.0, end_time_gyr, timesteps + 1)
    orbits = ag.orbit(
        potential=potential,
        ic=initial_conditions,
        time=end_time_gyr / AGAMA_GYR_CONVERSION,
        trajsize=trajsize + 1,
    )
    absolute = np.empty((timesteps + 1, 6, len(initial_conditions)))
    for particle_index, orbit in enumerate(orbits):
        trajectory = orbit[1]
        absolute[:, :, particle_index] = (
            trajectory[lower_indices] * (1.0 - sample_fractions[:, None])
            + trajectory[upper_indices] * sample_fractions[:, None]
        )
    center = np.mean(absolute[:, :3], axis=2)
    center_velocity = np.mean(absolute[:, 3:], axis=2)
    centered = absolute.copy()
    centered[:, :3] -= center[:, :, None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        xvta=centered,
        R_cm=center,
        V_cm=center_velocity,
        time_gyr=time_gyr,
        sample_indices=sample_indices,
        sample_coordinates=sample_coordinates,
        sample_lower_indices=lower_indices,
        sample_upper_indices=upper_indices,
        trajsize=np.asarray(trajsize),
    )
    return output_path


def run_agama_step_convergence(
    source_dir,
    output_dir,
    labels=("5pc", "10pc", "20pc"),
    trajsizes=(1000, 2000, 4000),
    t_step_gyr=0.1,
    timesteps=110,
    force_rerun=False,
):
    """Evolve all tracer sizes at several Agama trajectory resolutions."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    trajsizes = tuple(int(value) for value in trajsizes)
    if not trajsizes or any(value < timesteps for value in trajsizes):
        raise ValueError("each trajsize must be at least the number of output timesteps")

    potential_path = source_dir / "pot_evolving_gradual.ini"
    initial_conditions = {}
    for label in labels:
        with np.load(source_dir / f"xvta_Ric{label}.npz") as saved:
            initial_conditions[label] = saved["xvta"][0].T.copy()

    ag = load_agama()
    potential = ag.Potential(str(potential_path))
    outputs = {}
    for trajsize in trajsizes:
        for label in labels:
            output_path = output_dir / f"xvta_agama_steps{trajsize}_Ric{label}.npz"
            if force_rerun and output_path.exists():
                output_path.unlink()
            outputs[(trajsize, label)] = integrate_control(
                potential,
                initial_conditions[label],
                output_path,
                t_step_gyr=t_step_gyr,
                timesteps=timesteps,
                trajsize=trajsize,
            )
    return outputs


def run_agama_accuracy_convergence(
    source_dir,
    output_dir,
    labels=("5pc", "10pc", "20pc"),
    accuracies=(1e-8, 1e-10, 1e-12),
    t_step_gyr=0.1,
    timesteps=110,
    force_rerun=False,
):
    """Evolve all tracer sizes at several ODE accuracy tolerances."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    accuracies = tuple(float(value) for value in accuracies)
    if not accuracies or any(value <= 0 for value in accuracies):
        raise ValueError("accuracies must contain positive values")

    potential_path = source_dir / "pot_evolving_gradual.ini"
    initial_conditions = {}
    for label in labels:
        with np.load(source_dir / f"xvta_Ric{label}.npz") as saved:
            initial_conditions[label] = saved["xvta"][0].T.copy()

    ag = load_agama()
    potential = ag.Potential(str(potential_path))
    outputs = {}
    for accuracy in accuracies:
        for label in labels:
            accuracy_tag = f"{accuracy:.0e}"
            output_path = output_dir / f"xvta_agama_accuracy{accuracy_tag}_Ric{label}.npz"
            if output_path.exists() and not force_rerun:
                with np.load(output_path) as cached:
                    if np.isclose(cached["time_gyr"][-1], timesteps * t_step_gyr):
                        outputs[(accuracy, label)] = output_path
                        continue
            if output_path.exists():
                output_path.unlink()

            orbits = ag.orbit(
                potential=potential,
                ic=initial_conditions[label],
                time=timesteps * t_step_gyr / AGAMA_GYR_CONVERSION,
                trajsize=timesteps + 1,
                accuracy=accuracy,
                dtype=float,
            )
            absolute = np.empty((timesteps + 1, 6, len(initial_conditions[label])))
            for particle_index, orbit in enumerate(orbits):
                absolute[:, :, particle_index] = orbit[1]
            center = np.mean(absolute[:, :3], axis=2)
            center_velocity = np.mean(absolute[:, 3:], axis=2)
            centered = absolute.copy()
            centered[:, :3] -= center[:, :, None]
            time_gyr = np.linspace(0.0, timesteps * t_step_gyr, timesteps + 1)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                xvta=centered,
                R_cm=center,
                V_cm=center_velocity,
                time_gyr=time_gyr,
                accuracy=np.asarray(accuracy),
                output_samples=np.asarray(timesteps + 1),
            )
            outputs[(accuracy, label)] = output_path
    return outputs


def cluster_diagnostics(xvta):
    centered = xvta[:, :3] - np.mean(xvta[:, :3], axis=2, keepdims=True)
    radius = np.linalg.norm(centered[:, :2], axis=1)
    half_radius = np.median(radius, axis=1)
    sigma = np.mean(np.std(xvta[:, 3:], axis=2), axis=1)
    return half_radius, sigma


def density_paths_for_snapshot(source_dir, snapshot_index, ramp_up=10):
    source_dir = Path(source_dir)
    paths = [source_dir / f"sph_harm_gradual{snapshot_index:03d}.ini"]
    if snapshot_index < ramp_up:
        paths.append(source_dir / f"sph_harm_base_gradual{snapshot_index:03d}.ini")
    return paths


def interpolate_centers(source_time, centers, target_time):
    source_time = np.asarray(source_time)
    centers = np.asarray(centers)
    return np.column_stack(
        [np.interp(target_time, source_time, centers[:, axis]) for axis in range(3)]
    )


def run_spatial_convergence(
    source_dir,
    output_dir,
    center_time,
    centers,
    snapshot_indices=(0, 10, 20, 30, 40, 67),
    radial_grids=(25, 50, 100),
    angular_orders=(8, 12, 16, 20, 23),
):
    """Rebuild selected snapshots and compare T across radial/angular resolution."""
    ag = load_agama()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    cache_path = output_dir / "spatial_convergence.npz"
    if cache_path.exists():
        return cache_path
    snapshot_indices = np.asarray(snapshot_indices, dtype=int)
    target_time = snapshot_indices * 0.1
    target_centers = interpolate_centers(center_time, centers, target_time)
    radial_tensors = np.empty((len(radial_grids), len(snapshot_indices), 3, 3))
    angular_tensors = np.empty((len(angular_orders), len(snapshot_indices), 3, 3))

    for time_index, (snapshot, center) in enumerate(zip(snapshot_indices, target_centers)):
        density_paths = density_paths_for_snapshot(source_dir, int(snapshot))
        for config_index, grid_size in enumerate(radial_grids):
            path = output_dir / "potentials" / f"radial_r{grid_size}_t{snapshot:03d}.ini"
            potential = build_static_potential(
                density_paths, path, lmax=20, grid_size_r=int(grid_size)
            )
            radial_tensors[config_index, time_index] = derivatives_to_tidal(
                potential.eval(center, der=True)
            )
        for config_index, lmax in enumerate(angular_orders):
            path = output_dir / "potentials" / f"angular_l{lmax}_t{snapshot:03d}.ini"
            potential = build_static_potential(
                density_paths, path, lmax=int(lmax), grid_size_r=100
            )
            angular_tensors[config_index, time_index] = derivatives_to_tidal(
                potential.eval(center, der=True)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        snapshot_indices=snapshot_indices,
        times_gyr=target_time,
        centers_kpc=target_centers,
        radial_grids=np.asarray(radial_grids),
        radial_tensors=radial_tensors,
        angular_orders=np.asarray(angular_orders),
        angular_tensors=angular_tensors,
    )
    return cache_path


def run_radial_grid_orbit_convergence(
    source_dir,
    output_dir,
    radial_grids=(50, 100),
    labels=("5pc", "10pc", "20pc"),
    lmax=20,
    t_step_gyr=0.1,
    timesteps=110,
    trajsize=1000,
    force_rebuild=False,
    force_rerun=False,
):
    """Rebuild each radial representation and evolve all tracer sizes."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    radial_grids = tuple(int(value) for value in radial_grids)
    if not radial_grids or any(value <= 0 for value in radial_grids):
        raise ValueError("radial_grids must contain positive integers")

    source_snapshots = sorted(
        source_dir.glob("sph_harm_gradual[0-9][0-9][0-9].ini")
    )
    if len(source_snapshots) != timesteps:
        raise ValueError(
            f"expected {timesteps} density snapshots, found {len(source_snapshots)}"
        )

    initial_conditions = {}
    for label in labels:
        with np.load(source_dir / f"xvta_Ric{label}.npz") as saved:
            initial_conditions[label] = saved["xvta"][0].T.copy()

    ag = load_agama()
    masters = {}
    outputs = {}
    for grid_size in radial_grids:
        potential_paths = []
        for snapshot_index in range(timesteps):
            potential_path = (
                output_dir
                / "potentials"
                / f"radial_r{grid_size}_t{snapshot_index:03d}.ini"
            )
            if force_rebuild and potential_path.exists():
                potential_path.unlink()
            build_static_potential(
                density_paths_for_snapshot(source_dir, snapshot_index),
                potential_path,
                lmax=lmax,
                grid_size_r=grid_size,
            )
            potential_paths.append(potential_path.resolve())

        master_path = output_dir / f"evolving_radial_r{grid_size}.ini"
        lines = ["[Potential]", "type=Evolving", "linearInterp=True", "Timestamps"]
        for snapshot_index, potential_path in enumerate(potential_paths):
            timestamp = snapshot_index * t_step_gyr / AGAMA_GYR_CONVERSION
            lines.append(f"{timestamp:.17g} {potential_path}")
        master_path.write_text("\n".join(lines) + "\n")
        masters[grid_size] = master_path

        potential = ag.Potential(str(master_path))
        for label in labels:
            output_path = output_dir / f"xvta_radial_r{grid_size}_Ric{label}.npz"
            if force_rerun and output_path.exists():
                output_path.unlink()
            outputs[(grid_size, label)] = integrate_control(
                potential,
                initial_conditions[label],
                output_path,
                t_step_gyr=t_step_gyr,
                timesteps=timesteps,
                trajsize=trajsize,
            )

    return masters, outputs


def run_angular_order_orbit_convergence(
    source_dir,
    output_dir,
    angular_orders=(8, 12, 16, 20, 23),
    labels=("5pc", "10pc", "20pc"),
    radial_grid=100,
    t_step_gyr=0.1,
    timesteps=110,
    trajsize=1000,
    force_rebuild=False,
    force_rerun=False,
):
    """Rebuild each angular representation and evolve all tracer sizes."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    angular_orders = tuple(int(value) for value in angular_orders)
    if not angular_orders or any(value < 0 for value in angular_orders):
        raise ValueError("angular_orders must contain non-negative integers")

    source_snapshots = sorted(
        source_dir.glob("sph_harm_gradual[0-9][0-9][0-9].ini")
    )
    if len(source_snapshots) != timesteps:
        raise ValueError(
            f"expected {timesteps} density snapshots, found {len(source_snapshots)}"
        )

    initial_conditions = {}
    for label in labels:
        with np.load(source_dir / f"xvta_Ric{label}.npz") as saved:
            initial_conditions[label] = saved["xvta"][0].T.copy()

    ag = load_agama()
    masters = {}
    outputs = {}
    for lmax in angular_orders:
        potential_paths = []
        for snapshot_index in range(timesteps):
            potential_path = (
                output_dir
                / "potentials"
                / f"angular_l{lmax}_t{snapshot_index:03d}.ini"
            )
            if force_rebuild and potential_path.exists():
                potential_path.unlink()
            build_static_potential(
                density_paths_for_snapshot(source_dir, snapshot_index),
                potential_path,
                lmax=lmax,
                grid_size_r=radial_grid,
            )
            potential_paths.append(potential_path.resolve())

        master_path = output_dir / f"evolving_angular_l{lmax}.ini"
        lines = ["[Potential]", "type=Evolving", "linearInterp=True", "Timestamps"]
        for snapshot_index, potential_path in enumerate(potential_paths):
            timestamp = snapshot_index * t_step_gyr / AGAMA_GYR_CONVERSION
            lines.append(f"{timestamp:.17g} {potential_path}")
        master_path.write_text("\n".join(lines) + "\n")
        masters[lmax] = master_path

        potential = ag.Potential(str(master_path))
        for label in labels:
            output_path = output_dir / f"xvta_angular_l{lmax}_Ric{label}.npz"
            if force_rerun and output_path.exists():
                output_path.unlink()
            outputs[(lmax, label)] = integrate_control(
                potential,
                initial_conditions[label],
                output_path,
                t_step_gyr=t_step_gyr,
                timesteps=timesteps,
                trajsize=trajsize,
            )

    return masters, outputs


def force_linearity_test(
    potential,
    centers,
    times_gyr,
    output_path,
    radii_kpc=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    sample_indices=(10, 20, 30, 40, 67),
    directions=64,
    seed=12345,
):
    """Compare exact differential force with -T xi in random directions."""
    output_path = Path(output_path)
    if output_path.exists():
        return output_path
    rng = np.random.default_rng(seed)
    unit = rng.normal(size=(directions, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    radii_kpc = np.asarray(radii_kpc)
    sample_indices = np.asarray(sample_indices, dtype=int)
    median_error = np.empty((len(sample_indices), len(radii_kpc)))
    p90_error = np.empty_like(median_error)
    rms_absolute = np.empty_like(median_error)

    for row, index in enumerate(sample_indices):
        center = centers[index]
        time_internal = times_gyr[index] / AGAMA_GYR_CONVERSION
        center_force, derivatives = potential.eval(
            center, t=time_internal, acc=True, der=True
        )
        tensor = derivatives_to_tidal(derivatives)
        for column, radius in enumerate(radii_kpc):
            displacement = unit * radius
            exact = potential.eval(
                center + displacement, t=time_internal, acc=True
            ) - center_force
            linear = -(displacement @ tensor.T)
            residual = exact - linear
            relative = np.linalg.norm(residual, axis=1) / np.maximum(
                np.linalg.norm(exact, axis=1), 1e-30
            )
            median_error[row, column] = np.median(relative)
            p90_error[row, column] = np.percentile(relative, 90)
            rms_absolute[row, column] = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        radii_kpc=radii_kpc,
        sample_indices=sample_indices,
        times_gyr=np.asarray(times_gyr)[sample_indices],
        median_relative_error=median_error,
        p90_relative_error=p90_error,
        rms_absolute_error=rms_absolute,
    )
    return output_path


def force_derivative_convergence_test(
    potential,
    centers,
    times_gyr,
    output_path,
    difference_steps_kpc=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    sample_indices=(10, 20, 30, 40, 67, 90),
):
    """Compare AGAMA's force derivatives with centered force differences."""
    output_path = Path(output_path)
    if output_path.exists():
        return output_path
    difference_steps_kpc = np.asarray(difference_steps_kpc, dtype=float)
    sample_indices = np.asarray(sample_indices, dtype=int)
    direct_tensors = np.empty((len(sample_indices), 3, 3))
    finite_difference_tensors = np.empty(
        (len(difference_steps_kpc), len(sample_indices), 3, 3)
    )
    relative_error = np.empty((len(difference_steps_kpc), len(sample_indices)))

    for time_row, index in enumerate(sample_indices):
        center = np.asarray(centers[index], dtype=float)
        time_internal = times_gyr[index] / AGAMA_GYR_CONVERSION
        direct = derivatives_to_tidal(
            potential.eval(center, t=time_internal, der=True)
        )
        direct_tensors[time_row] = direct
        for step_row, step in enumerate(difference_steps_kpc):
            numerical = np.empty((3, 3))
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = step
                acceleration_plus = potential.eval(
                    center + offset, t=time_internal, acc=True
                )
                acceleration_minus = potential.eval(
                    center - offset, t=time_internal, acc=True
                )
                numerical[:, axis] = -(
                    acceleration_plus - acceleration_minus
                ) / (2.0 * step)
            finite_difference_tensors[step_row, time_row] = numerical
            relative_error[step_row, time_row] = np.linalg.norm(
                numerical - direct
            ) / max(np.linalg.norm(direct), 1e-30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        difference_steps_kpc=difference_steps_kpc,
        sample_indices=sample_indices,
        times_gyr=np.asarray(times_gyr)[sample_indices],
        direct_tensors=direct_tensors,
        finite_difference_tensors=finite_difference_tensors,
        relative_error=relative_error,
    )
    return output_path


def absolute_force_accuracy_test(
    production_potential,
    reference_paths,
    particle_positions,
    times_gyr,
    output_path,
    sample_indices=(10, 20, 30, 40, 67, 90),
):
    """Compare production accelerations with high-resolution reference forces.

    ``reference_paths`` maps snapshot indices to independently exported,
    higher-resolution static potentials built from the same density snapshot.
    The comparison is performed at the actual particle positions from the
    fiducial evolution, so it measures the force used to evolve the tracers.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return output_path
    ag = load_agama()
    particle_positions = np.asarray(particle_positions, dtype=float)
    times_gyr = np.asarray(times_gyr, dtype=float)
    sample_indices = np.asarray(sample_indices, dtype=int)
    median_error = np.empty(len(sample_indices))
    p90_error = np.empty(len(sample_indices))
    max_error = np.empty(len(sample_indices))

    for row, index in enumerate(sample_indices):
        positions = particle_positions[index]
        if positions.ndim == 2 and positions.shape[0] == 3:
            positions = positions.T
        time_internal = times_gyr[index] / AGAMA_GYR_CONVERSION
        production = production_potential.eval(
            positions, t=time_internal, acc=True
        )
        reference = ag.Potential(str(reference_paths[int(index)]))
        high_resolution = reference.eval(positions, acc=True)
        relative = np.linalg.norm(production - high_resolution, axis=1) / np.maximum(
            np.linalg.norm(high_resolution, axis=1), 1e-30
        )
        median_error[row] = np.median(relative)
        p90_error[row] = np.percentile(relative, 90)
        max_error[row] = np.max(relative)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_indices=sample_indices,
        times_gyr=times_gyr[sample_indices],
        median_relative_error=median_error,
        p90_relative_error=p90_error,
        max_relative_error=max_error,
    )
    return output_path


def write_coarse_evolving_master(source_dir, output_path, stride=2):
    """Create a coarsened Evolving master file referencing existing snapshots."""
    source_dir = Path(source_dir).resolve()
    output_path = Path(output_path)
    files = sorted(source_dir.glob("sph_harm_pot_gradual[0-9][0-9][0-9].ini"))
    indices = list(range(0, len(files), stride))
    if indices[-1] != len(files) - 1:
        indices.append(len(files) - 1)
    lines = ["[Potential]", "type=Evolving", "linearInterp=True", "Timestamps"]
    for index in indices:
        timestamp = index * 0.1 / AGAMA_GYR_CONVERSION
        lines.append(f"{timestamp:.17g} {files[index].resolve()}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def temporal_coarsening_test(
    fiducial_potential,
    source_dir,
    output_dir,
    centers,
    times_gyr,
    strides=(2, 3),
):
    """Compare time cadence and linear-versus-spline interpolation."""
    ag = load_agama()
    output_dir = Path(output_dir)
    cache_path = output_dir / "temporal_coarsening.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if "spline_tensors" in cached:
                return cache_path
    reference = tidal_series(fiducial_potential, centers, times_gyr)
    coarse = np.empty((len(strides), len(times_gyr), 3, 3))
    for row, stride in enumerate(strides):
        master = write_coarse_evolving_master(
            source_dir, output_dir / f"evolving_stride{stride}.ini", stride=int(stride)
        )
        coarse_potential = ag.Potential(str(master))
        coarse[row] = tidal_series(coarse_potential, centers, times_gyr)
    spline_master = write_coarse_evolving_master(
        source_dir, output_dir / "evolving_spline.ini", stride=1
    )
    spline_master.write_text(
        spline_master.read_text().replace("linearInterp=True", "linearInterp=False")
    )
    spline = tidal_series(ag.Potential(str(spline_master)), centers, times_gyr)
    np.savez_compressed(
        cache_path,
        times_gyr=times_gyr,
        strides=np.asarray(strides),
        reference_tensors=reference,
        coarse_tensors=coarse,
        spline_tensors=spline,
    )
    return cache_path


def solve_seed0_modes(output_path):
    """Recreate and cache the exact seed-0 eigenmode realization from fdm.ipynb."""
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    import scipy.interpolate as scinterp
    import fdmtools

    m_a = 1.0e-22
    halo_mass = 1.0e9
    scale_radius = 10.0
    setup_radius = np.logspace(-3, 3, 512)
    nfw = halo_mass / (2 * np.pi * scale_radius**3)
    nfw /= (setup_radius / scale_radius) * (1 + setup_radius / scale_radius) ** 3
    core_radius = 1.6
    soliton = 0.019 * core_radius**-4
    soliton /= (1 + 0.091 * (setup_radius / core_radius) ** 2) ** 8
    soliton *= 1.0e9
    transition = np.max(np.where(soliton > nfw)) + 1
    profile = scinterp.CubicSpline(
        setup_radius, np.concatenate((soliton[:transition], nfw[transition:]))
    )

    fdm = fdmtools.FDMTools(m_a)
    phi, density, _ = fdm.initialize_potential(profile, core=False, r_max=200.0)
    distribution = fdm.DF_invert(phi, density)
    energies, radial_functions, amplitudes, phases, _, _ = fdm.solve_amps_itr(
        200.0,
        80.0,
        phi,
        density,
        halo_mass,
        distribution,
        maxiter=1,
        method="Isotropic",
        bins=50,
        schwarz_iter=2500,
        seed=0,
    )

    radii = np.arange(0.0, 30.0, 0.1)[:251]
    radii[0] = 1.0e-5
    radial_values = np.zeros(energies.shape + (len(radii),))
    for n in range(energies.shape[0]):
        for ell in range(energies.shape[1]):
            if energies[n, ell] != 0:
                radial_values[n, ell] = radial_functions[n, ell](radii)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        energies=energies,
        radial_values=radial_values,
        amplitudes=amplitudes,
        phases=phases,
        radii=radii,
    )
    return output_path


def _write_density_coefficients(path, radii, rho_lms):
    """Write HEALPix density coefficients in AGAMA's real-harmonic format."""
    import healpy as hp

    path = Path(path)
    lmax = hp.Alm.getlmax(rho_lms.shape[1])
    labels = [f"l={ell},m={m}" for ell in range(lmax + 1) for m in range(-ell, ell + 1)]
    rows = []
    normalization = np.sqrt(4 * np.pi)
    for radius_index, radius in enumerate(radii):
        values = []
        for ell in range(lmax + 1):
            for m in range(-ell, ell + 1):
                if m >= 0:
                    index = (2 * lmax + 3 - m) * m // 2 + (ell - m)
                    value = rho_lms[radius_index, index].real / normalization
                    if m > 0:
                        value *= np.sqrt(2)
                else:
                    index = (2 * lmax + 3 + m) * (-m) // 2 + (ell + m)
                    value = -np.sqrt(2) * rho_lms[radius_index, index].imag / normalization
                values.append(value)
        rows.append(f"{radius:16.10g}" + "".join(f" {value:16.10g}" for value in values))
    header = [
        "[Density]",
        "type=DensitySphericalHarmonic",
        f"gridSizeR={len(radii)}",
        f"lmax={lmax}",
        "Coefficients",
        "#rho",
        f"{'#radius': <17}" + "".join(f"{label: <17}" for label in labels),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + rows) + "\n")
    return path


def _write_scaled_density(source_path, output_path, factor):
    """Scale every density coefficient in an AGAMA harmonic-density file."""
    source_path, output_path = Path(source_path), Path(output_path)
    output = []
    for line in source_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] in "+-."):
            values = np.fromstring(stripped, sep=" ")
            output.append(
                f"{values[0]:16.10g}"
                + "".join(f" {value * factor:16.10g}" for value in values[1:])
            )
        else:
            output.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n")
    return output_path


def generate_true_fine_potentials(
    source_dir,
    output_dir,
    resolution_multiplier=2,
    cadence_gyr=None,
    base_dt_gyr=0.1,
    end_time_gyr=11.0,
    validation_times=(0.1, 1.0, 5.0),
    principal_cutoff=None,
    force_rerun=False,
):
    """Regenerate a full seed-0 potential sequence at a requested cadence."""
    import healpy as hp

    ag = load_agama()
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if cadence_gyr is None:
        resolution_multiplier = int(resolution_multiplier)
        if resolution_multiplier < 1:
            raise ValueError("resolution_multiplier must be a positive integer")
        dt_gyr = base_dt_gyr / resolution_multiplier
    else:
        dt_gyr = float(cadence_gyr)
        if dt_gyr <= 0:
            raise ValueError("cadence_gyr must be positive")
        resolution_multiplier = base_dt_gyr / dt_gyr
    cadence_tag = f"dt{dt_gyr:g}".replace('.', 'p')
    if principal_cutoff is not None and int(principal_cutoff) < 1:
        raise ValueError("principal_cutoff must be positive")
    directory_tag = cadence_tag
    if principal_cutoff is not None:
        directory_tag += f"_npc{int(principal_cutoff)}"
    fine_dir = output_dir / f"true_time_resolution_{directory_tag}"
    master_path = fine_dir / f"pot_evolving_true_dt{dt_gyr:g}.ini"
    validation_path = fine_dir / "regeneration_validation.npz"
    if not force_rerun and master_path.exists() and validation_path.exists():
        return master_path, validation_path

    mode_path = solve_seed0_modes(output_dir / "true_time_resolution_modes_seed0.npz")
    with np.load(mode_path) as modes:
        energies = modes["energies"]
        radial_values = modes["radial_values"]
        amplitudes = modes["amplitudes"]
        phases = modes["phases"]
        radii = modes["radii"]

    hbar = 6.582119569509067e-16
    light_speed = 2.99792458e5
    km_to_kpc = 3.240779289444365e-17
    h_over_m = hbar / (1.0e-22 / light_speed**2) * km_to_kpc
    phase_rate = -energies / h_over_m * KM_S_TO_KPC_GYR
    nmax, ell_count = energies.shape
    wave_lmax = ell_count - 1
    nside = 8
    rho_lms_size = hp.Alm.getsize(3 * nside - 1)

    def build_snapshot(time_gyr, tag):
        density_path = fine_dir / "densities" / f"density_{tag}.ini"
        potential_path = fine_dir / "potentials" / f"potential_{tag}.ini"
        if potential_path.exists() and not force_rerun:
            return potential_path
        time_phase = phase_rate * time_gyr
        wave_alm_size = (wave_lmax + 2) * (wave_lmax + 1) // 2
        alm_real = np.zeros((len(radii), wave_alm_size), dtype=complex)
        alm_imag = np.zeros_like(alm_real)
        radial_indices = np.arange(nmax)
        for ell in range(ell_count):
            valid = energies[:, ell] != 0
            if principal_cutoff is not None:
                valid &= radial_indices + ell + 1 <= int(principal_cutoff)
            radial_amplitude = (
                amplitudes[valid, ell, None] * radial_values[valid, ell, :]
            )
            for m in range(ell + 1):
                index = (2 * wave_lmax + 3 - m) * m // 2 + (ell - m)
                positive = np.exp(
                    1j * (phases[valid, ell, m] + time_phase[valid, ell])
                )
                negative = (
                    np.exp(-1j * (phases[valid, ell, -m] + time_phase[valid, ell]))
                    * (-1) ** m
                )
                if m == 0:
                    value = np.sum(radial_amplitude * positive[:, None], axis=0)
                    alm_real[:, index] = value
                    alm_imag[:, index] = value / 1j
                else:
                    alm_real[:, index] = np.sum(
                        radial_amplitude * (positive + negative)[:, None] / 2, axis=0
                    )
                    alm_imag[:, index] = np.sum(
                        radial_amplitude * (positive - negative)[:, None] / (2j), axis=0
                    )
        wave = hp.alm2map(alm_real, nside=nside, pol=False)
        wave = wave + 1j * hp.alm2map(alm_imag, nside=nside, pol=False)
        ramp_fraction = min(time_gyr / 1.0, 1.0)
        rho_lms = hp.map2alm(
            ramp_fraction * np.abs(wave) ** 2, lmax=3 * nside - 1, pol=False
        )
        _write_density_coefficients(density_path, radii, rho_lms)
        densities = [ag.Density(str(density_path))]
        if time_gyr < 1.0:
            base_path = fine_dir / "densities" / f"base_{tag}.ini"
            _write_scaled_density(
                source_dir / "sph_harm_base_gradual.ini", base_path, 1 - time_gyr
            )
            densities.append(ag.Density(str(base_path)))
        total_density = densities[0] if len(densities) == 1 else ag.Density(*densities)
        potential = ag.Potential(
            type="Multipole", density=total_density, lmax=20, rmin=1.0e-4, rmax=30.0
        )
        potential_path.parent.mkdir(parents=True, exist_ok=True)
        potential.export(str(potential_path))
        return potential_path

    # Recreate a few original timestamps before accepting the sequence.
    validation_points = np.array([[1.0e-3, 0, 0], [0.01, 0.02, 0], [0.1, -0.05, 0.03]])
    validation_errors = []
    for time_gyr in validation_times:
        source_index = int(round(time_gyr / 0.1))
        regenerated = ag.Potential(str(build_snapshot(time_gyr, f"validation_{source_index:03d}")))
        original = ag.Potential(str(source_dir / f"sph_harm_pot_gradual{source_index:03d}.ini"))
        original_force = original.eval(validation_points, acc=True)
        regenerated_force = regenerated.eval(validation_points, acc=True)
        validation_errors.append(
            np.linalg.norm(regenerated_force - original_force, axis=1)
            / np.maximum(np.linalg.norm(original_force, axis=1), 1.0e-30)
        )
    validation_errors = np.asarray(validation_errors)
    np.savez_compressed(
        validation_path,
        times_gyr=np.asarray(validation_times),
        points_kpc=validation_points,
        relative_force_error=validation_errors,
    )
    source_match = np.max(validation_errors)
    if principal_cutoff is None and source_match > 5.0e-3:
        print(
            "WARNING: current fdmtools does not exactly reproduce the old unsaved "
            f"phase realization (max force difference={source_match:.3%}). "
            "Use regenerated x1 versus regenerated finer runs for convergence."
        )

    times = np.arange(0.0, end_time_gyr + 0.5 * dt_gyr, dt_gyr)
    lines = ["[Potential]", "type=Evolving", "linearInterp=True", "Timestamps"]
    for index, time_gyr in enumerate(times):
        potential_path = build_snapshot(time_gyr, f"snapshot_{index:04d}").resolve()
        lines.append(f"{time_gyr / AGAMA_GYR_CONVERSION:.17g} {potential_path}")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_text("\n".join(lines) + "\n")
    return master_path, validation_path
