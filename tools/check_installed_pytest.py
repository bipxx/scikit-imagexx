"""Run the installed CI pytest suite with the runner's required test inputs."""
import argparse
from importlib.metadata import distribution
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_INTAKES = (
    'test_regionprops_coords_scaled_benchmark_intake.py',
    'test_draw_polygon_benchmark_intake.py',
)
# These guards inspect process-wide imports. Keep each compatible group fresh.
NUMPY_ONLY_INTAKES = (
    'test_array_views_intake.py',
    'test_crop_intake.py',
    'test_slice_along_axes_intake.py',
)
FIXTURE_ONLY_CASES = (
    'test_approximate_polygon_public.py::test_approximate_polygon_fixture_without_original',
    'test_subdivide_polygon_public.py::test_subdivide_polygon_fixture_without_original',
    'test_rolling_ball_kernels_public.py::test_geometry_fixture_serializes_without_original_calls',
)
SOURCE_ONLY_CASES = (
    'test_tv_positive_zero_compare.py::test_new_controller_exact_byte_attributes_and_no_scientific_imports',
)
# The registered integration tests compare these same cases to the running
# scikit-image provider. Leave the Linux-captured callback values as history.
HISTORICAL_RDP_CALLBACK_CASES = tuple(
    f'test_approximate_polygon_public.py::test_approximate_polygon_native_public_contract[{workers}-{case}]'
    for workers in (1, 4)
    for case in ('tolerance-provider-positive', 'tolerance-mutates-coordinates')
)
# Replay the exact original traces with the frozen NumPy sort dispatch.
LINUX_ORIGINAL_TRACE_CASES = (
    'test_h_extrema_contracts.py::test_original_fixed_contract',
    'test_reconstruction_benchmark_contracts.py::test_original_frozen_contract',
    'test_reconstruction_terrain_contracts.py::test_original_frozen_parity',
)
LINUX_FENV_MODULES = (
    'test_mark_wide_outer_public.py',
    'test_draw_multiblock_lbp_public.py',
    'test_draw_multiblock_lbp_dtype_metadata.py',
    'test_draw_multiblock_lbp_batched.py',
    'test_finite_radon_saved_public.py',
    'test_finite_radon_exceptional_saved_public.py',
    'test_finite_radon_numpy_abi_public.py',
    'test_variation_of_information_public.py',
    'test_variation_of_information_current_counts_public.py',
    'test_variation_of_information_large_saved.py',
    'test_vi_public_log2_provider_large.py',
    'test_vi_public_matmul_provider_public.py',
    'test_flood_fill_public.py',
    'test_flood_assignment_protocol.py',
    'test_flood_copy_protocol.py',
    'test_flood_disjoint_assignment.py',
)
LINUX_FENV_CASES = (
    'test_regular_grid_public.py::test_saved_rows',
    'test_util_views_public.py::test_slice_saved_public',
    'test_mark_boundaries_public.py::test_saved_mark_public',
    'test_rolling_ball_public.py::test_saved_original',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('normal', 'disabled'), required=True)
    args = parser.parse_args()
    if sys.platform not in ('linux', 'win32'):
        parser.error('the installed CI suite requires Linux or Windows')
    output = ROOT / 'build' / 'ci-artifacts' / f'installed-pytest-{args.mode}'
    upstream = distribution('scikit-image').locate_file('skimage/util/_regular_grid.py').resolve()
    if not upstream.is_file():
        parser.error('the installed original regular-grid source is missing')

    env = os.environ.copy()
    env['REGULAR_GRID_UPSTREAM_SOURCE'] = str(upstream)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    switches = [f'--ignore=tests/{name}' for name in (*BENCHMARK_INTAKES, *NUMPY_ONLY_INTAKES)]
    switches.extend(f'--deselect=tests/{node}' for node in
                    (*FIXTURE_ONLY_CASES, *SOURCE_ONLY_CASES, *HISTORICAL_RDP_CALLBACK_CASES))
    if sys.platform == 'linux':
        switches.append('--ignore=tests/test_vi_public_matmul_windows_public.py')
        switches.extend(f'--deselect=tests/{node}' for node in LINUX_ORIGINAL_TRACE_CASES)
        outputs = {
            'FINITE_RADON_RECORDS': 'radon-saved-rows.json',
            'FINITE_RADON_EXCEPTIONAL_RECORDS': 'radon-exceptional-rows.json',
            'VI_PUBLIC_LOG2_LARGE_RECORD_DIR': 'vi-large-rows',
            'VI_PUBLIC_MATMUL_SMALL_RECORD_DIR': 'vi-current-rows',
        }
    else:
        switches.extend(f'--ignore=tests/{name}' for name in LINUX_FENV_MODULES)
        switches.extend(f'--deselect=tests/{node}' for node in LINUX_FENV_CASES)
        outputs = {
            'VI_PUBLIC_MATMUL_WINDOWS_RECORD_DIR': 'vi-current-rows',
            'VI_PUBLIC_MATMUL_WINDOWS_WORKER_RECORD_DIR': 'vi-window-worker-rows',
        }
    output.mkdir(parents=True, exist_ok=False)
    env.update({name: str(output / relative) for name, relative in outputs.items()})
    if args.mode == 'disabled':
        env['SKIMAGEXX_DISABLE_SIMD'] = '1'
    else:
        env.pop('SKIMAGEXX_DISABLE_SIMD', None)
    command = [sys.executable, '-m', 'pytest', '-q']
    for group in (SOURCE_ONLY_CASES, NUMPY_ONLY_INTAKES, FIXTURE_ONLY_CASES):
        result = subprocess.run([*command, *(f'tests/{node}' for node in group)],
                                cwd=ROOT, env=env, check=False)
        if result.returncode:
            return result.returncode
    if sys.platform == 'linux':
        original_env = {**env, 'NPY_DISABLE_CPU_FEATURES': 'X86_V4'}
        original_env.pop('NPY_ENABLE_CPU_FEATURES', None)
        result = subprocess.run([*command, *(f'tests/{node}' for node in LINUX_ORIGINAL_TRACE_CASES)],
                                cwd=ROOT, env=original_env, check=False)
        if result.returncode:
            return result.returncode
    return subprocess.run([*command, *switches], cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
