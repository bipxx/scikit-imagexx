"""Hosted, untimed qualification of a frozen complete public product.

Run ``python tools/public_qualification.py all --profile modern --output
build/public-qualification/modern`` on a fresh host. Each profile owns its build
and runtime venvs. ``plan`` reports source blockers without importing image code;
``inspect-wheel`` authenticates an older product without claiming qualification.
Public selectors below are deliberately bounded, not a full-library claim.
Portable/no-OpenMP/sanitizer/private-adapter evidence is not inherited implicitly.
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import inspect
import json
import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import runpy
import shutil
import subprocess
import sys
import venv
import warnings
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'public-qualification-v2'
PROFILES = {
    'windows': dict(python='3.13', numpy='2.5.4', scipy='1.18.1', headers='2.5.4', config='Release'),
    'modern': dict(python='3.13', numpy='2.5.4', scipy='1.18.1', headers='2.5.4', config='Release'),
    'minimum': dict(python='3.11', numpy='1.24.4', scipy='1.15.3', headers='2.4.6', config='Release'),
    'debug': dict(python='3.13', numpy='2.5.4', scipy='1.18.1', headers='2.5.4', config='Debug'),
}
# Exact versions read from the authenticated 40024754860 runtime wheel METADATA
# and retained whole7b build metadata, not a fresh unconstrained resolver result.
COMMON_RUNTIME = dict(imageio='2.38.1', iniconfig='2.3.1', **{'lazy-loader': '0.6'},
                      packaging='26.3', pillow='13.0.0', pluggy='1.7.0', pygments='2.21.0',
                      pytest='9.1.1', **{'scikit-image': '0.26.0'})
BUILD = {'scikit-build-core': '1.1.1', 'pybind11': '3.1.0', 'cmake': '4.4.4',
         'ninja': '1.13.2', 'packaging': '26.3', 'pathspec': '1.1.1'}
# Exact wheel bytes retained by accepted campaign40024754860; debug uses modern.
RUNTIME_WHEELS = {
    "windows": {
        "colorama": {
            "filename": "colorama-0.4.6-py2.py3-none-any.whl",
            "sha256": "4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6"
        },
        "imageio": {
            "filename": "imageio-2.38.1-py3-none-any.whl",
            "sha256": "36d23eb7423d2fb63f637098758edb3d2df3687f7125e0a5ce8596572022dcf9"
        },
        "iniconfig": {
            "filename": "iniconfig-2.3.1-py3-none-any.whl",
            "sha256": "9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7"
        },
        "lazy-loader": {
            "filename": "lazy_loader-0.6-py3-none-any.whl",
            "sha256": "77253be3391b06124a0e16105bd663b6c54470af1a9ca8e1cf026f38d58ed056"
        },
        "networkx": {
            "filename": "networkx-3.7-py3-none-any.whl",
            "sha256": "e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0"
        },
        "numpy": {
            "filename": "numpy-2.5.4-cp313-cp313-win_amd64.whl",
            "sha256": "8b4d2fd2d34e5f8c9235ee787de5631a37a28402b15cb80814df973d2be54129"
        },
        "packaging": {
            "filename": "packaging-26.3-py3-none-any.whl",
            "sha256": "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"
        },
        "pillow": {
            "filename": "pillow-13.0.0-cp313-cp313-win_amd64.whl",
            "sha256": "9dda0788a9932776a75d8bcc020076edc308ce2508b51f6739a45e3158b3fc42"
        },
        "pluggy": {
            "filename": "pluggy-1.7.0-py3-none-any.whl",
            "sha256": "7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec"
        },
        "pygments": {
            "filename": "pygments-2.21.0-py3-none-any.whl",
            "sha256": "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"
        },
        "pytest": {
            "filename": "pytest-9.1.1-py3-none-any.whl",
            "sha256": "37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c"
        },
        "scikit-image": {
            "filename": "scikit_image-0.26.0-cp313-cp313-win_amd64.whl",
            "sha256": "ac529eb9dbd5954f9aaa2e3fe9a3fd9661bfe24e134c688587d811a0233127f1"
        },
        "scipy": {
            "filename": "scipy-1.18.1-cp313-cp313-win_amd64.whl",
            "sha256": "559ed65f60c1af5a03f3912605a1b5114f522c7c32fb23c3376ae8f03219fe28"
        },
        "tifffile": {
            "filename": "tifffile-2026.10.15-py3-none-any.whl",
            "sha256": "3dcf6c56731a45c1f85c81e90dad922b0a380a2d639e77f71207e010f8ef0f7b"
        }
    },
    "modern": {
        "imageio": {
            "filename": "imageio-2.38.1-py3-none-any.whl",
            "sha256": "36d23eb7423d2fb63f637098758edb3d2df3687f7125e0a5ce8596572022dcf9"
        },
        "iniconfig": {
            "filename": "iniconfig-2.3.1-py3-none-any.whl",
            "sha256": "9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7"
        },
        "lazy-loader": {
            "filename": "lazy_loader-0.6-py3-none-any.whl",
            "sha256": "77253be3391b06124a0e16105bd663b6c54470af1a9ca8e1cf026f38d58ed056"
        },
        "networkx": {
            "filename": "networkx-3.7-py3-none-any.whl",
            "sha256": "e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0"
        },
        "numpy": {
            "filename": "numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "6ffa07666f8da0eef81d149934a626d0d95fbd6838432a33e66245423a9062c0"
        },
        "packaging": {
            "filename": "packaging-26.3-py3-none-any.whl",
            "sha256": "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"
        },
        "pillow": {
            "filename": "pillow-13.0.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "5f56499c1a74d2b862beb0c33b8016f99d095858455df9d21583bbbc1c15b582"
        },
        "pluggy": {
            "filename": "pluggy-1.7.0-py3-none-any.whl",
            "sha256": "7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec"
        },
        "pygments": {
            "filename": "pygments-2.21.0-py3-none-any.whl",
            "sha256": "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"
        },
        "pytest": {
            "filename": "pytest-9.1.1-py3-none-any.whl",
            "sha256": "37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c"
        },
        "scikit-image": {
            "filename": "scikit_image-0.26.0-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "b36ab5e778bf50af5ff386c3ac508027dc3aaeccf2161bdf96bde6848f44d21b"
        },
        "scipy": {
            "filename": "scipy-1.18.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "fdaf5ea890a6183d0565f51a61799d67081bd5b1cf03c5f4b3fd3732108625c9"
        },
        "tifffile": {
            "filename": "tifffile-2026.10.15-py3-none-any.whl",
            "sha256": "3dcf6c56731a45c1f85c81e90dad922b0a380a2d639e77f71207e010f8ef0f7b"
        }
    },
    "minimum": {
        "imageio": {
            "filename": "imageio-2.38.1-py3-none-any.whl",
            "sha256": "36d23eb7423d2fb63f637098758edb3d2df3687f7125e0a5ce8596572022dcf9"
        },
        "iniconfig": {
            "filename": "iniconfig-2.3.1-py3-none-any.whl",
            "sha256": "9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7"
        },
        "lazy-loader": {
            "filename": "lazy_loader-0.6-py3-none-any.whl",
            "sha256": "77253be3391b06124a0e16105bd663b6c54470af1a9ca8e1cf026f38d58ed056"
        },
        "networkx": {
            "filename": "networkx-3.6.1-py3-none-any.whl",
            "sha256": "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
        },
        "numpy": {
            "filename": "numpy-1.24.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "7215847ce88a85ce39baf9e89070cb860c98fdddacbaa6c0da3ffb31b3350bd5"
        },
        "packaging": {
            "filename": "packaging-26.3-py3-none-any.whl",
            "sha256": "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"
        },
        "pillow": {
            "filename": "pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c"
        },
        "pluggy": {
            "filename": "pluggy-1.7.0-py3-none-any.whl",
            "sha256": "7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec"
        },
        "pygments": {
            "filename": "pygments-2.21.0-py3-none-any.whl",
            "sha256": "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"
        },
        "pytest": {
            "filename": "pytest-9.1.1-py3-none-any.whl",
            "sha256": "37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c"
        },
        "scikit-image": {
            "filename": "scikit_image-0.26.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "74aa5518ccea28121f57a95374581d3b979839adc25bb03f289b1bc9b99c58af"
        },
        "scipy": {
            "filename": "scipy-1.15.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "39cb9c62e471b1bb3750066ecc3a3f3052b37751c7c3dfd0fd7e48900ed52982"
        },
        "tifffile": {
            "filename": "tifffile-2026.3.3-py3-none-any.whl",
            "sha256": "e8be15c94273113d31ecb7aa3a39822189dd11c4967e3cc88c178f1ad2fd1170"
        }
    }
}
# Complete relative-file-set+byte tree digests from accepted whole6 build headers.
BUILD_HEADER_TREES = {
    "windows": {
        "record_sha256": "bf735767faab4486850ae77f2724391f13c8bae4a44de02dd79f9668bc9e4eee",
        "trees": {
            "numpy": "9e6d0f823d0863c1b51deaf46050231ece4b1f439f079f6c14328bac4aa888da",
            "pybind11": "9de71b954881bbe0763c1001fb19cbe16290d7637cb0066b74812333ba2425d4"
        }
    },
    "modern": {
        "record_sha256": "a1f3f8e49311b9a896337c982f77dfbaf840a137ea0c12300c1415b676599a87",
        "trees": {
            "numpy": "4bc4bb58a3ff482b6bfc1d765fc2d43e1054da40830aa3ae2c8d05e7f8f6c4c0",
            "pybind11": "9de71b954881bbe0763c1001fb19cbe16290d7637cb0066b74812333ba2425d4"
        }
    },
    "minimum": {
        "record_sha256": "204f3627bc18f28ddee8e61a4e5e8b60dfda85b8a22417930e1dd2d3a6ccdfcf",
        "trees": {
            "numpy": "172b39c93f447c3d345f4e3efd1efd4de047c6a4512593d54bc632bb39137f9c",
            "pybind11": "9de71b954881bbe0763c1001fb19cbe16290d7637cb0066b74812333ba2425d4"
        }
    }
}

R5 = ('autolevel_percentile', 'enhance_contrast_percentile', 'gradient_percentile',
      'mean_percentile', 'pop_percentile', 'subtract_mean_percentile',
      'threshold_percentile', 'sum_percentile', 'percentile')
R6 = ('mean_bilateral', 'pop_bilateral', 'sum_bilateral')
R6_PATHS = frozenset(prefix + name for prefix in ('filters.rank.', 'filters.rank.bilateral.') for name in R6)
R6_SOURCE_PATHS = (
    'tests/test_rank_bilateral_mutation_supplement.py',
    'tests/test_rank_bilateral_masked_center_supplement.py',
    'coverage/contracts/rank-bilateral.json',
    'coverage/contracts/rank-bilateral-public.json',
    'tests/test_rank_bilateral_endian_public.py',
    'coverage/contracts/rank-bilateral-endian-public.json',
    'coverage/evidence/rank-bilateral-masked-source-pin-migration.json',
    'coverage/contracts/rank-bilateral-masked-center-public.json',
    'coverage/contracts/rank-bilateral-benchmark.json',
    'coverage/evidence/rank-bilateral-fixture-assertion-migration.json',
)
R7 = ('otsu', 'majority')
R7_PATHS = frozenset(prefix + name for prefix in ('filters.rank.', 'filters.rank.generic.') for name in R7)
R7_SOURCE_PATHS = (
    'tests/test_rank_decisions_contracts.py',
    'tests/test_rank_decisions_public.py',
    'tests/test_rank_decisions_endian_public.py',
    'tests/test_rank_decisions_native.cpp',
    'include/skimagexx/rank.hpp',
    'coverage/contracts/rank-decisions.json',
    'coverage/contracts/rank-decisions-public.json',
    'coverage/contracts/rank-decisions-endian-public.json',
    'coverage/contracts/rank-decisions-benchmark.json',
    'coverage/evidence/rank-decisions-fixture-assertion-migration.json',
)
LEGACY_SDK_FAMILIES = frozenset(('gabor', 'symmetric_eigen', 'structure_tensor', 'corner_response',
    'kitchen', 'hessian', 'shape_index', 'hessian_det', 'sato', 'meijering', 'frangi'))
FAMILIES = ('components', 'component_consumers', 'metrics', 'footprint_dense', 'rank',
            'rank_distribution', 'rank_intensity', 'rank_alias', 'projective',
            'rank_statistics', 'consumer_layout', 'rank_percentile', 'rank_bilateral', 'rank_decisions',
            'tv_chambolle', 'tv_reductions', 'tv_denoise', 'tv_bregman')
ORIGINAL_FAMILIES = FAMILIES[:14]
TV_FAMILIES = FAMILIES[:17]
PRODUCT_SCOPE = 'bregman18'
PRODUCT_MEMBERS = 110
SDK_MEMBERS = 63
TV_PATH = 'restoration.denoise_tv_chambolle'
TV_SOURCE_PATHS = ('include/skimagexx/tv_chambolle.hpp', 'include/skimagexx/tv_reductions.hpp',
    'include/skimagexx/tv_denoise.hpp', 'src/tv_chambolle.cpp', 'src/tv_reductions.cpp',
    'src/tv_denoise.cpp', 'src/bind_tv_chambolle.cpp', 'python/skimagexx/_tv_chambolle.py',
    'coverage/contracts/tv-chambolle.json',
    'coverage/evidence/tv-chambolle-public-assertion-migration.json',
    'tests/test_tv_denoise_cursor_native.cpp', 'tests/test_tv_chambolle_cursor_pipeline.py',
    'src/tv_denoise_internal.hpp', 'tests/test_tv_denoise_admission_native.cpp')
BREGMAN_PATH = 'restoration.denoise_tv_bregman'
BREGMAN_SOURCE_PATHS = ('include/skimagexx/tv_bregman.hpp', 'src/tv_bregman.cpp',
    'src/simd_tv_bregman_avx2.cpp', 'src/bind_tv_bregman.cpp', 'python/skimagexx/_tv_bregman.py',
    'tests/test_tv_bregman_native.cpp', 'tests/test_tv_bregman_native_pipeline.py',
    'tests/test_tv_bregman_public.py', 'coverage/contracts/tv-bregman.json',
    'coverage/evidence/tv-bregman-public-assertion-migration.json')
GROUPS = {
    'public-rank': ('test_rank_public.py', 'test_rank_public_admission.py',
                    'test_rank_integer_warning.py', 'test_rank_intensity_public.py',
                    'test_rank_alias_public.py', 'test_rank_statistics_public.py',
                    'test_rank_statistics_contracts.py', 'test_rank_percentile_public.py',
                    'test_rank_percentile_contracts.py'),
    'public-bilateral': ('test_rank_bilateral_contracts.py', 'test_rank_bilateral_public.py',
                         'test_rank_bilateral_masked_center_public.py', 'test_rank_bilateral_endian_public.py'),
    'public-decisions': ('test_rank_decisions_contracts.py', 'test_rank_decisions_public.py',
                         'test_rank_decisions_endian_public.py'),
    'public-TV': ('test_tv_chambolle_contracts.py', 'test_tv_chambolle_native_pipeline.py',
                  'test_tv_chambolle_cursor_pipeline.py'),
    'public-Bregman': ('test_tv_bregman_contracts.py', 'test_tv_bregman_public.py'),
    'public-consumers': ('test_component_consumers_public.py', 'test_arraymap_provider_pipeline.py'),
    'metrics-leaf-half': ('test_metrics_leaf_simd.py', 'test_metrics_half_loads.py', 'test_metrics_public.py',
                          'test_metrics_reverse_simd.py::test_reverse_original_policy_values_and_warnings'),
    'metrics-policies': ('test_metrics_reductions.py',),
    'shared': ('test_class_registry.py', 'test_api.py', 'test_inventory.py', 'test_nested_imports.py',
               'test_dtype_metadata.py', 'test_transform_classes_public.py', 'test_transform_exception_public.py'),
}
_BASE_GROUPS = GROUPS


def groups_for_source(root):
    """Counts-after has one additional genuine-public gate; before has none."""
    groups = dict(_BASE_GROUPS)
    if (Path(root) / 'tests/test_rank_windowed_counts_pipeline.py').is_file():
        groups['public-counts'] = ('test_rank_windowed_counts_pipeline.py',)
    return groups


GROUPS = groups_for_source(ROOT)
POLICIES = 'warning_hook or runtime_reduction_buffer or stateful or overflow_policies or underflow_policies or scalar_power_status'
CORPORA = {'consumers': ('component_consumers.py', 878), 'statistics': ('rank_statistics.py', 1379),
           'percentile': ('rank_percentile.py', 2552), 'bilateral': ('rank_bilateral.py', 1008),
           'decisions': ('rank_decisions.py', 694)}


class QualificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise QualificationError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def runtime_pins(profile):
    p = PROFILES[profile]
    pins = dict(COMMON_RUNTIME, numpy=p['numpy'], scipy=p['scipy'],
                networkx='3.6.1' if profile == 'minimum' else '3.7',
                tifffile='2026.3.3' if profile == 'minimum' else '2026.10.15')
    if profile == 'windows':
        pins['colorama'] = '0.4.6'
    return pins


def environment(package=None, scalar=False):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('SKIMAGEXX_', 'GIT_')) or key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX'):
            env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_WAIT_POLICY='PASSIVE', CMAKE_BUILD_PARALLEL_LEVEL='2')
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    if package is not None:
        env['PYTHONPATH'] = str(Path(package).resolve())
    if scalar:
        env['SKIMAGEXX_DISABLE_SIMD'] = '1'
    return env


def source_state(root):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], env=environment(), text=True).strip()
    require(not git('status', '--porcelain'), 'source checkout must be clean')
    return dict(revision=git('rev-parse', 'HEAD'), files_sha256={
        n: hashlib.sha256((root / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for n in git('ls-files').splitlines() if (root / n).is_file()})


def registry_expectation(root):
    # This data-only file has no package imports; never import source/python.
    namespace = runpy.run_path(str(root / 'python/skimagexx/coverage.py'))
    declared = tuple(namespace['native_functions']())
    require(len(declared) == len(set(declared)), 'duplicate source registry paths')
    names = set(declared)
    image_count = sum(namespace['execution_kind'](n) == 'native' for n in names)
    return dict(functions=sorted(names), image_count=image_count, registered_count=len(names), metadata_count=len(names)-image_count)


def sdk_family_declarations(cmake):
    """Read the repo's explicit literal/foreach ITEMS declarations, not execute CMake.

    Actual target existence and exact17 CTest discovery are independently required
    after relocation; this early source check is not native client proof.
    """
    names = list(re.findall(r'add_executable\(\s*(\w+)_consumer\b', cmake))
    for match in re.finditer(r'foreach\(\s*(\w+)\s+IN\s+ITEMS\s+([^)]*)\)(.*?)endforeach\(\s*\)', cmake, re.S):
        variable, items, body = match.groups()
        if 'add_executable(${' + variable + '}_consumer' in body:
            names.extend(re.findall(r'\b\w+\b', items))
    return names


def declared_sdk_families(cmake):
    return set(sdk_family_declarations(cmake))


def plan(root, profile, expected=None):
    registry = registry_expectation(root)
    blockers = []
    if TV_PATH not in registry['functions']:
        blockers.append('missing public TV registry: ' + TV_PATH)
    for name in TV_SOURCE_PATHS:
        if not (root / name).is_file():
            blockers.append('missing required TV source: ' + name)
    if BREGMAN_PATH not in registry['functions']:
        blockers.append('missing public Bregman registry: ' + BREGMAN_PATH)
    for name in BREGMAN_SOURCE_PATHS:
        if not (root / name).is_file():
            blockers.append('missing required Bregman source: ' + name)
    if expected is not None and registry['image_count'] != expected:
        blockers.append('explicit image count differs from source registry')
    for name in R5:
        if 'filters.rank.' + name not in registry['functions']:
            blockers.append('missing public R5 registry: ' + name)
        if 'filters.rank.generic.' + name in registry['functions']:
            blockers.append('R5 must not register generic alias: ' + name)
    for name in sorted(R6_PATHS - set(registry['functions'])):
        blockers.append('missing public R6 registry: ' + name)
    for name in R6:
        if 'filters.rank.generic.' + name in registry['functions']:
            blockers.append('R6 must not register generic alias: ' + name)
    for name in sorted(R7_PATHS - set(registry['functions'])):
        blockers.append('missing public R7 registry: ' + name)
    for name in registry['functions']:
        if name.startswith('filters.rank.') and name.rsplit('.', 1)[-1] in R7 and name not in R7_PATHS:
            blockers.append('unexpected public R7 alias: ' + name)
    for name in R7_SOURCE_PATHS:
        if not (root / name).is_file():
            blockers.append('missing required R7 source: ' + name)
    for name in R6_SOURCE_PATHS:
        if not (root / name).is_file():
            blockers.append('missing required R6 source: ' + name)
    groups = groups_for_source(root)
    for name in [n for group in groups.values() for n in group] + ['test_arraymap_dense_contracts.py']:
        if not (root / 'tests' / name.split('::')[0]).is_file():
            blockers.append('missing required test: ' + name)
    for filename, _ in CORPORA.values():
        if not (root / 'benchmarks' / filename).is_file():
            blockers.append('missing required corpus: ' + filename)
    cmake = (root / 'tests/cmake_consumer/CMakeLists.txt').read_text()
    declarations = sdk_family_declarations(cmake)
    declared = set(declarations)
    for family in sorted(declared):
        if declarations.count(family) != 1:
            blockers.append('duplicate installed SDK client: ' + family)
    for family in sorted(declared - set(FAMILIES) - LEGACY_SDK_FAMILIES):
        blockers.append('unknown installed SDK client: ' + family)
    for family in FAMILIES:
        if not (root / 'tests' / ('test_' + family + '_native.cpp')).is_file():
            blockers.append('missing required SDK client source: ' + family)
        if family not in declared:
            blockers.append('missing required installed SDK client: ' + family)
    return dict(schema=SCHEMA, profile=profile, profile_config=PROFILES[profile], registry=registry,
                runtime_pins=runtime_pins(profile), build_pins=dict(BUILD, numpy=PROFILES[profile]['headers']),
                public_groups=groups, corpora=CORPORA, SDK_families=FAMILIES, blockers=blockers,
                product_scope=PRODUCT_SCOPE, product_members=PRODUCT_MEMBERS, SDK_members=SDK_MEMBERS, SDK_checks=2*len(FAMILIES),
                R7_target_counts=dict(image_count=190, metadata_count=3, registered_count=193),
                count_semantics='image_count counts registered image paths including identity aliases, excludes metadata; registered_count adds metadata, neither is algorithm count',
                exclusions='No full-library, private-adapter, portable/no-OpenMP/sanitizer or performance claim. No cross-build NaN guarantee.')


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        require(len(entries) == 1, 'ambiguous wheel metadata')
        info = BytesParser().parsebytes(archive.read(entries[0]))
    return dict(name=canonical(info['Name']), version=info['Version'], sha256=sha(path))


def wheel_members(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        require(len(names) == len(set(names)), 'duplicate wheel members')
        for name in names:
            p = PurePosixPath(name)
            require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in p.parts[0], 'unsafe wheel path')
        records = [n for n in names if n.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'missing/ambiguous RECORD')
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        for row in rows:
            require(len(row) == 3, 'malformed RECORD row')
            p = PurePosixPath(row[0])
            require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts and '\\' not in row[0] and ':' not in p.parts[0], 'unsafe wheel path in RECORD')
        require(len(rows) == len(names) and {r[0] for r in rows} == set(names), 'RECORD member set differs')
        members = {}
        for name, digest, size in rows:
            data = archive.read(name)
            if name == records[0]:
                require(not digest and not size, 'RECORD must have empty self hash')
            else:
                expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                require(digest == expected and int(size) == len(data), 'RECORD bytes differ: ' + name)
            members[name] = hashlib.sha256(data).hexdigest()
    return members


def product_census(scope=PRODUCT_SCOPE):
    """Exact historical scopes; production qualification requires Bregman18."""
    census = {'rank14': (104, 59), 'tv17': (108, 62), 'bregman18': (110, 63)}
    require(scope in census, 'unknown authenticated product scope')
    return census[scope]


def inspect_wheel(wheel, source_archive, revision, *, scope=PRODUCT_SCOPE):
    require(wheel_metadata(wheel)['name'] == 'scikit-imagexx', 'not the product wheel')
    members = wheel_members(wheel)
    product_count, sdk_count = product_census(scope)
    require(len(members) == product_count, f'expected complete{product_count}-member product')
    sdk = {n: v for n, v in members.items() if n.startswith(('include/', 'lib/', 'share/'))}
    require(len(sdk) == sdk_count, f'expected complete{sdk_count}-member SDK')
    cores = [n for n in members if n.startswith('skimagexx/_core.') and n.endswith(('.so', '.pyd'))]
    require(len(cores) == 1, 'require exactly one production core')
    with zipfile.ZipFile(wheel) as product, zipfile.ZipFile(source_archive) as source:
        require(source.comment.decode() == revision, 'source archive revision differs')
        tv = set(TV_SOURCE_PATHS[:3]) | {'python/skimagexx/_tv_chambolle.py'}
        require((tv <= set(source.namelist())) if scope in ('tv17', 'bregman18') else not (tv & set(source.namelist())),
                'TV product scope/source headers or helper differ')
        bregman = {'include/skimagexx/tv_bregman.hpp', 'python/skimagexx/_tv_bregman.py'}
        require((bregman <= set(source.namelist())) if scope == 'bregman18' else not (bregman & set(source.namelist())),
                'Bregman product scope/source header or helper differ')
        for prefix, actual_prefix, suffix in [('python/skimagexx/', 'skimagexx/', '.py'), ('include/', 'include/', None)]:
            names = [n for n in source.namelist() if n.startswith(prefix) and not n.endswith('/') and (suffix is None or n.endswith(suffix))]
            mapping = {n: actual_prefix + n.removeprefix(prefix) for n in names}
            actual = {n for n in members if n.startswith(actual_prefix) and (suffix is None or n.endswith(suffix))}
            require(set(mapping.values()) == actual, 'complete source Python/header file set differs')
            for name, destination in mapping.items():
                require(source.read(name).replace(b'\r\n', b'\n') == product.read(destination).replace(b'\r\n', b'\n'), 'source bytes differ: ' + name)
    return dict(source_revision=revision, source_archive_sha256=sha(source_archive), wheel_sha256=sha(wheel), product_scope=scope,
                members_sha256=members, SDK_sha256=sdk, sole_core=cores[0], record_verified=True,
                scope='Metadata-only complete product authentication; no algorithm or whole8 acceptance.')


def verify_tree(directory, members):
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual == set(members), 'extracted package file set changed (including extras)')
    for name, digest in members.items():
        require(sha(directory / name) == digest, 'extracted bytes changed: ' + name)


class Journal:
    def __init__(self, out):
        self.out = out
        self.path = out / 'commands.json'
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else []

    def run(self, label, command, env=None, cwd=ROOT):
        require(label not in [row['label'] for row in self.rows], 'never overwrite/restart a recorded command')
        log = self.out / (label + '.log')
        require(not log.exists(), 'preserve earlier command log')
        row = dict(label=label, command=list(map(str, command)), cwd=str(cwd),
                   started_utc=datetime.now(timezone.utc).isoformat(), state='running', log=log.name)
        self.rows.append(row); dump(self.path, self.rows)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(row['command'], cwd=cwd, env=env or environment(), stdout=stream, stderr=subprocess.STDOUT)
            row['pid'] = process.pid; dump(self.path, self.rows)
            # Synchronous ownership: this process waits for this exact child.
            code = process.wait()
        row.update(returncode=code, state='terminal', finished_utc=datetime.now(timezone.utc).isoformat())
        dump(self.path, self.rows)
        return code


def python_in(prefix):
    return prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def tool_in(prefix, name):
    return prefix / ('Scripts' if os.name == 'nt' else 'bin') / (name + '.exe' if os.name == 'nt' else name)


def base_interpreter():
    path = Path(getattr(sys, '_base_executable', None) or sys.executable).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path), version=sys.version,
                major_minor='.'.join(map(str, sys.version_info[:2])), cache_tag=sys.implementation.cache_tag,
                os_name=os.name)


def interpreter_link_paths(base):
    # venv's POSIX aliases are the only permitted external references. Windows
    # uses copied redirectors; arbitrary Scripts/ or other links are not admitted.
    if base['os_name'] != 'posix':
        return set()
    return {env + '/bin/' + name for env in ('build-env','runtime-env')
            for name in ('python','python3','python' + base['major_minor'])}


def capture_interpreter_provenance(out, base):
    links = {}
    for name in sorted(interpreter_link_paths(base)):
        path = out / name
        if path.is_symlink():
            require(path.parent.resolve().is_relative_to(out.resolve()), 'interpreter owner directory escapes output')
            require(str(path.resolve(strict=True)) == base['path'] and sha(path) == base['sha256'], 'venv interpreter differs from declared base')
            links[name] = dict(raw_target=os.readlink(path), resolved_target=base['path'], sha256=base['sha256'])
    return dict(base=base, links=links)


def validate_link_record(name, link, actual_link, provenance):
    """Policy is independent of filesystem traversal, so all platforms test it."""
    base = provenance['base']
    require(name in interpreter_link_paths(base), 'unapproved external interpreter owner')
    require(set(link) == {'raw_target','resolved_target','sha256'}, 'invalid interpreter link record')
    require(link == actual_link, 'interpreter link target/bytes changed')
    require(link['resolved_target'] == base['path'] and link['sha256'] == base['sha256'], 'interpreter link is not the declared base')


def validate_interpreter_provenance(out, provenance):
    require(provenance['base'] == base_interpreter(), 'actual base interpreter differs from declared provenance')
    require(set(provenance['links']) <= interpreter_link_paths(provenance['base']), 'unapproved external interpreter record')
    for name, link in provenance['links'].items():
        path = out / name
        require(path.parent.resolve().is_relative_to(out.resolve()), 'interpreter owner directory escapes output')
        require(path.is_symlink() and path.is_file(), 'declared interpreter link replaced')
        raw, resolved = os.readlink(path), str(path.resolve(strict=True))
        require(raw == link['raw_target'] and resolved == link['resolved_target'] == provenance['base']['path'], 'interpreter link target changed')
        actual = dict(raw_target=raw, resolved_target=resolved, sha256=sha(path))
        validate_link_record(name, link, actual, provenance)
    for path in out.rglob('*'):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(out.resolve()):
            require(path.is_file() and path.relative_to(out).as_posix() in provenance['links'], 'unapproved outside output symlink')


def verify_record(out, name, record, provenance):
    lexical = PurePosixPath(name)
    require(bool(lexical.parts) and not lexical.is_absolute() and '..' not in lexical.parts
            and '\\' not in name and ':' not in lexical.parts[0], 'sealed record escapes output')
    path = out / name
    require(path.parent.resolve().is_relative_to(out.resolve()), 'sealed record parent escapes output')
    link = provenance['links'].get(name)
    if link is not None:
        require(path.is_symlink(), 'declared interpreter link replaced')
        raw, resolved = os.readlink(path), str(path.resolve(strict=True))
        require(raw == link['raw_target'] and resolved == link['resolved_target'] == provenance['base']['path'], 'interpreter link target changed')
        actual = dict(raw_target=raw, resolved_target=resolved, sha256=sha(path))
        validate_link_record(name, link, actual, provenance)
    else:
        require(path.resolve(strict=True).is_relative_to(out.resolve()), 'sealed record escapes output')
    require(path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'sealed artifact changed: ' + name)


def must_run(journal, label, command, env=None):
    require(journal.run(label, command, env) == 0, 'command failed; raw log retained: ' + label)


def capture_build_identity(out, profile, source, journal, buildpy, env):
    """Prospective host evidence only, before any terminal qualification seal."""
    facts = out / 'build-identity-facts.json'
    identity = out / 'build-identity.json'
    require(not facts.exists() and not identity.exists(), 'never retrofit/overwrite build identity')
    must_run(journal, 'gather-build-identity-facts', [buildpy, ROOT / 'tools/build_identity_facts.py',
             '--source-root', ROOT, '--build-root', out, '--revision', source['revision'],
             '--profile', profile, '--output', facts], env)
    require(facts.is_file(), 'actual build facts missing')
    model = json.loads(facts.read_text())
    require(model.get('schema') == 'rank-statistics-build-input-v2'
            and model.get('revision') == source['revision'] and model.get('profile') == profile
            and Path(model.get('source_root', '')).resolve() == ROOT
            and Path(model.get('build_root', '')).resolve() == out.resolve(), 'build facts source/profile/roots differ')
    must_run(journal, 'capture-build-identity', [buildpy, ROOT / 'benchmarks/rank_statistics_compare.py',
             'capture-build-identity', '--facts', facts, '--output', identity], env)
    require(identity.is_file(), 'build identity producer output missing')
    require(json.loads(identity.read_text()).get('schema') == 'rank-statistics-build-identity-v2', 'wrong build identity producer schema')
    return dict(facts_sha256=sha(facts), identity_sha256=sha(identity), schema='rank-statistics-build-identity-v2')


def validate_build_identity(out, state):
    record = state.get('build_identity')
    require(isinstance(record, dict) and record.get('schema') == 'rank-statistics-build-identity-v2', 'prospective actual build identity absent')
    for name, key in [('build-identity-facts.json','facts_sha256'),('build-identity.json','identity_sha256')]:
        path = out / name
        require(path.is_file() and sha(path) == record.get(key), 'build identity changed/absent before seal: ' + name)
    require(json.loads((out / 'build-identity.json').read_text()).get('schema') == record['schema'], 'build identity schema changed')


def prepare(out, profile, expected):
    require(not out.exists(), 'choose a fresh output; never mutate a previous trial')
    out.mkdir(parents=True)
    source = source_state(ROOT); planning = plan(ROOT, profile, expected); dump(out / 'plan.json', planning)
    require(not planning['blockers'], 'source not ready: ' + '; '.join(planning['blockers']))
    require('.'.join(map(str, sys.version_info[:2])) == PROFILES[profile]['python'], 'wrong host Python')
    require((os.name == 'nt') == (profile == 'windows'), 'wrong host platform')
    journal = Journal(out)
    base = base_interpreter()  # Pin the actual bootstrap before creating venvs/installing wheels.
    archive = out / 'source.zip'
    must_run(journal, 'source-archive', ['git', '-C', ROOT, 'archive', '--format=zip', '--output=' + str(archive), source['revision']])
    for name in ('build-env', 'runtime-env'):
        must_run(journal, 'venv-' + name, [sys.executable, '-m', 'venv', str(out / name)])
    interpreter = capture_interpreter_provenance(out, base)
    validate_interpreter_provenance(out, interpreter)
    dump(out / 'interpreter-provenance.json', interpreter)
    buildpy = python_in(out / 'build-env'); runtimepy = python_in(out / 'runtime-env')
    buildpins = planning['build_pins']; runtimepins = planning['runtime_pins']
    for label, pins, target, python in [('build', buildpins, out / 'build-wheels', buildpy), ('runtime', runtimepins, out / 'runtime-wheels', runtimepy)]:
        target.mkdir()
        specs = [name + '==' + version for name, version in sorted(pins.items())]
        if label == 'runtime':
            retained = RUNTIME_WHEELS['modern' if profile == 'debug' else profile]
            require(set(retained) == set(pins), 'retained runtime hash closure incomplete')
            lock = out / 'pinned-runtime-inputs.txt'
            lock.write_text('\n'.join(name + '==' + pins[name] + ' --hash=sha256:' + retained[name]['sha256'] for name in sorted(pins)) + '\n')
            download_args = ['--require-hashes', '-r', lock]
        else:
            download_args = specs
        must_run(journal, label + '-download', [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps', '--dest', target, *download_args])
        observed = {wheel_metadata(p)['name']: wheel_metadata(p)['version'] for p in target.glob('*.whl')}
        require(observed == pins, 'downloaded wheel versions/file set differ')
        if label == 'runtime':
            require({p.name: sha(p) for p in target.glob('*.whl')} == {v['filename']: v['sha256'] for v in retained.values()}, 'runtime wheel bytes/filenames differ from accepted reference inputs')
        must_run(journal, label + '-install', [python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--find-links', target, *specs])
    child_args = [runtimepy, Path(__file__).resolve(), '_child', '--output', out, '--profile', profile]
    buildenv = environment(); buildenv['PATH'] = str(tool_in(out / 'build-env', 'cmake').parent) + os.pathsep + buildenv.get('PATH', '')
    must_run(journal, 'build-headers', [buildpy, Path(__file__).resolve(), '_headers', '--output', out, '--profile', profile], buildenv)
    wheels = out / 'product-wheels'; wheels.mkdir()
    link_capture = []
    if os.name != 'nt':
        # This project has exactly one MODULE (_core). ld records its actual
        # reads, including implicit CRT/system inputs; unavailable LTO inputs
        # fail capture instead of being reconstructed after qualification.
        link_capture = ['--config-settings=cmake.define.CMAKE_MODULE_LINKER_FLAGS=-save-temps=obj -Wl,--dependency-file=' + str(out / 'native-build/core-link.d'),
                        '--config-settings=build.tool-args=-d;keeprsp;-d;keepdepfile']
    must_run(journal, 'build-whole-product', [buildpy, '-m', 'pip', 'wheel', ROOT, '--verbose', '--no-deps', '--no-build-isolation', '--no-index', '--wheel-dir', wheels,
              '--config-settings=cmake.build-type=' + PROFILES[profile]['config'], '--config-settings=build-dir=' + str(out / 'native-build'),
              '--config-settings=cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON', *link_capture], buildenv)
    build_identity = capture_build_identity(out, profile, source, journal, buildpy, buildenv)
    build_properties = {}
    for name in ('CMakeCache.txt', 'compile_commands.json', 'build.ninja'):
        path = out / 'native-build' / name
        if path.is_file():
            build_properties[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    require('CMakeCache.txt' in build_properties, 'missing genuine native build cache properties')
    dump(out / 'native-build-properties.json', dict(files=build_properties, config=PROFILES[profile]['config'],
         scope='Exact actual compile/cache properties retained; no comparison to unrelated private controls inferred.'))
    products = list(wheels.glob('*.whl')); require(len(products) == 1, 'require one exact complete wheel')
    package_record = inspect_wheel(products[0], archive, source['revision'])
    package = out / 'package'; package.mkdir()
    with zipfile.ZipFile(products[0]) as z:
        for name in package_record['members_sha256']:
            p = package / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(name))
    verify_tree(package, package_record['members_sha256'])
    dump(out / 'package.json', package_record)
    state = dict(schema=SCHEMA, profile=profile, source=source, expected=planning['registry'],
                 build_identity=build_identity,
                 interpreter_provenance=interpreter,
                 runtime_pins=runtimepins, product_wheel=products[0].name, profile_config=PROFILES[profile],
                 no_timing=True, public_scope=GROUPS, supplemental_scopes=['382 returned ArrayMap frozen protocols using installed public class'],
                 exclusions=planning['exclusions'])
    state['runtime_wheels'] = {p.name: wheel_metadata(p) for p in (out / 'runtime-wheels').glob('*.whl')}
    dump(out / 'state.json', state)
    must_run(journal, 'initial-public-admission', [*child_args, '--group', 'admission'], environment(package))


def headers(out, profile):
    import numpy, pybind11, scikit_build_core
    p = PROFILES[profile]
    provenance = json.loads((out / 'interpreter-provenance.json').read_text())
    require(base_interpreter() == provenance['base'], 'build venv base interpreter differs')
    require(numpy.__version__ == p['headers'] and pybind11.__version__ == BUILD['pybind11'] and scikit_build_core.__version__ == BUILD['scikit-build-core'], 'build dependency version mismatch')
    record = dict(python=sys.version, executable=sys.executable, executable_sha256=sha(sys.executable),
                  numpy_build_headers=numpy.__version__, pybind11=pybind11.__version__, scikit_build_core=scikit_build_core.__version__,
                  include_files={name: {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}
                                 for name, folder in [('numpy', Path(numpy.get_include())), ('pybind11', Path(pybind11.get_include()))]})
    retained = BUILD_HEADER_TREES['modern' if profile == 'debug' else profile]
    trees = {name: hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',',':')).encode()).hexdigest() for name, files in record['include_files'].items()}
    require(trees == retained['trees'], 'complete build header file set/bytes differ from retained accepted inputs')
    record['retained_header_provenance'] = retained
    record['base_interpreter'] = base_interpreter()
    dump(out / 'build-headers.json', record)


def bilateral_admission(core, rank, nested, generic, wrapper, registered):
    """Metadata only; fail before importing any contracts or image fixtures."""
    capability = getattr(core, 'rank_bilateral_capabilities', None)
    require(getattr(core, 'rank_bilateral_interface', None) == 'rank-bilateral-v1'
            and type(capability) is int and capability == 0x1c000000,
            'missing/wrong public R6 capability')
    entry = getattr(core, 'rank_bilateral_run', None)
    require(inspect.isbuiltin(entry) and entry.__name__ == 'rank_bilateral_run'
            and entry.__module__ == core.__name__, 'missing owned compiled R6 entry')
    require(R6_PATHS <= set(registered), 'missing/partial public R6 registry')
    require(wrapper._core is core, 'R6 alternate core worker state')
    require(core.get_num_threads() == 1, 'R6 initial global workers must be1')
    require(getattr(rank, 'bilateral', None) is nested, 'R6 nested module identity mismatch')
    require(tuple(nested.__all__) == R6, 'R6 nested exports differ')
    for name in R6:
        function = getattr(rank, name, None)
        require(callable(function) and function is getattr(nested, name, None)
                and function is getattr(wrapper, name, None), 'R6 public function identity mismatch: ' + name)
        require(inspect.unwrap(function).__module__ == 'skimagexx._rank', 'upstream R6 algorithm forwarding')
        require(name in rank.__all__ and name not in vars(generic) and name not in generic.__all__
                and 'filters.rank.generic.' + name not in registered, 'R6 generic alias/export forbidden')


def decisions_admission(core, rank, generic, wrapper, registered):
    """Only own public metadata; reject before contracts or fixture creation."""
    capability = getattr(core, 'rank_decisions_capabilities', None)
    require(getattr(core, 'rank_decisions_interface', None) == 'rank-decisions-v1'
            and type(capability) is int and capability == 0x60000000,
            'missing/wrong public R7 capability')
    entry = getattr(core, 'rank_decisions_run', None)
    require(inspect.isbuiltin(entry) and entry.__name__ == 'rank_decisions_run'
            and entry.__module__ == core.__name__, 'missing owned compiled R7 entry')
    paths = tuple(registered)
    require(all(paths.count(name) == 1 for name in R7_PATHS), 'missing/duplicate public R7 registry')
    require({name for name in paths if name.startswith('filters.rank.')
             and name.rsplit('.', 1)[-1] in R7} == R7_PATHS, 'unexpected public R7 alias')
    require(wrapper._core is core, 'R7 alternate core worker state')
    require(core.get_num_threads() == 1, 'R7 initial global workers must be1')
    for name in R7:
        function = vars(rank).get(name)
        require(callable(function) and function is vars(generic).get(name)
                and function is vars(wrapper).get(name), 'R7 own public identity mismatch: ' + name)
        require(inspect.unwrap(function).__module__ == 'skimagexx._rank'
                and function.__name__ == name, 'upstream R7 algorithm forwarding')
        require(list(rank.__all__).count(name) == 1, 'R7 canonical star exports differ')
    require(list(generic.__all__).count('otsu') == 1 and 'majority' not in generic.__all__,
            'R7 generic star exports differ')


def tv_admission(core, restoration, wrapper, registered):
    """Reject missing TV implementation before any contract fixture is built."""
    paths = tuple(registered)
    require(paths.count(TV_PATH) == 1, 'missing/duplicate public TV registry')
    require(getattr(wrapper, '_core', None) is core, 'TV alternate core worker state')
    require(core.get_num_threads() == 1, 'TV initial global workers must be1')
    for name, value in {'tv_chambolle_interface': 'tv-chambolle-stages-v1',
                        'tv_chambolle_capabilities': 63,
                        'tv_chambolle_finite_interface': 'tv-chambolle-finite-v1',
                        'tv_chambolle_finite_capabilities': 3}.items():
        actual = getattr(core, name, None)
        require(type(actual) is type(value) and actual == value, 'missing/wrong public TV capability: ' + name)
    require(getattr(core, 'tv_chambolle_finite_profiles', None) == ('1.24.4', '2.5.4'),
            'TV audited scalar/reduction profiles differ')
    for name in ('tv_chambolle_divergence_into', 'tv_chambolle_gradient_into',
                 'tv_chambolle_norm_into', 'tv_chambolle_dual_into', 'tv_chambolle_finite_run'):
        function = getattr(core, name, None)
        require(inspect.isbuiltin(function) and function.__name__ == name
                and function.__module__ == core.__name__, 'missing owned compiled TV entry: ' + name)
    function = vars(restoration).get('denoise_tv_chambolle')
    require(callable(function) and function is vars(wrapper).get('denoise_tv_chambolle'),
            'TV own public function identity mismatch')
    require(inspect.unwrap(function).__module__ == 'skimagexx._tv_chambolle'
            and function.__name__ == 'denoise_tv_chambolle', 'upstream TV algorithm forwarding')
    require(str(inspect.signature(function)) == '(image, weight=0.1, eps=0.0002, max_num_iter=200, *, channel_axis=None)',
            'TV public signature differs')


def bregman_admission(core, restoration, wrapper, registered):
    """Require the real public wrapper and builtin before fixtures or providers."""
    paths = tuple(registered)
    require(paths.count(BREGMAN_PATH) == 1, 'missing/duplicate public Bregman registry')
    require(getattr(wrapper, '_core', None) is core, 'Bregman alternate core worker state')
    require(core.get_num_threads() == 1, 'Bregman initial global workers must be1')
    for name, value in {'tv_bregman_interface': 'tv-bregman-prepared-v1',
                        'tv_bregman_capabilities': 3, 'tv_bregman_parallel_sweep': False}.items():
        actual = getattr(core, name, None)
        require(type(actual) is type(value) and actual == value, 'missing/wrong public Bregman capability: ' + name)
    entry = getattr(core, 'tv_bregman_run', None)
    require(inspect.isbuiltin(entry) and entry.__name__ == 'tv_bregman_run'
            and entry.__module__ == core.__name__, 'missing owned compiled Bregman entry')
    function = vars(restoration).get('denoise_tv_bregman')
    require(callable(function) and function is vars(wrapper).get('denoise_tv_bregman'),
            'Bregman own public function identity mismatch')
    require(inspect.unwrap(function).__module__ == 'skimagexx._tv_bregman'
            and function.__name__ == 'denoise_tv_bregman', 'upstream Bregman algorithm forwarding')
    require(str(inspect.signature(function)) == '(image, weight=5.0, max_num_iter=100, eps=0.001, isotropic=True, *, channel_axis=None)',
            'Bregman public signature differs')


def admission(out, state):
    package = out / 'package'; record = json.loads((out / 'package.json').read_text())
    verify_tree(package, record['members_sha256'])
    validate_interpreter_provenance(out, state['interpreter_provenance'])
    require(not any(k.startswith('SKIMAGEXX_') and k != 'SKIMAGEXX_DISABLE_SIMD' for k in os.environ), 'candidate/private selectors forbidden')
    require(not any(Path(p).resolve() == ROOT / 'python' for p in sys.path if p), 'source/python shadowing forbidden')
    import skimagexx as xx
    require(Path(xx.__file__).resolve().parent == package / 'skimagexx', 'public package not genuine extracted wheel')
    require(Path(xx._core.__file__).resolve() == package / record['sole_core'], 'wrong/alternate core')
    xx.set_fallback_enabled(False)
    require(xx.get_num_threads() == 1, 'initial global workers must be1')
    require(xx.set_num_threads is xx._core.set_num_threads and xx.get_num_threads is xx._core.get_num_threads, 'global worker setter not shared')
    for module in ('_rank', '_components', '_component_consumers', '_metrics', '_geometric', '_footprint_dense', '_footprint_decomposition', '_footprint_util', '_tv_chambolle', '_tv_bregman'):
        require(importlib.import_module('skimagexx.' + module)._core is xx._core, 'multiple core worker states: ' + module)
    coverage = importlib.import_module('skimagexx.coverage')
    require(sorted(coverage.native_functions()) == state['expected']['functions'], 'installed/source registry mismatch')
    require(sum(coverage.execution_kind(n) == 'native' for n in coverage.native_functions()) == state['expected']['image_count'], 'image/helper count mismatch')
    require(len(coverage.native_functions()) == state['expected']['registered_count']
            and state['expected']['metadata_count'] == state['expected']['registered_count'] - state['expected']['image_count'],
            'registered/image/metadata counts differ')
    required = dict(rank_histogram_capabilities=8191, rank_alias_interface='rank-alias-v1', rank_alias_capabilities=480,
                    rank_statistics_interface='rank-statistics-v1', rank_statistics_capabilities=0xe000,
                    rank_vector_interface='rank-vector-v1', rank_vector_capabilities=0x10000,
                    rank_percentile_interface='rank-percentile-v1', rank_percentile_capabilities=0x3fe0000)
    for name, value in required.items():
        require(getattr(xx._core, name, None) == value, 'missing/wrong public capability: ' + name)
    for name in ('rank_run', 'rank_statistics_run', 'rank_windowed_histogram_run', 'rank_noise_footprint', 'rank_percentile_run'):
        entry = getattr(xx._core, name, None)
        require(type(entry).__name__ == 'builtin_function_or_method' and entry.__module__ == xx._core.__name__, 'missing owned compiled entry: ' + name)
    rank = importlib.import_module('skimagexx.filters.rank'); generic = importlib.import_module('skimagexx.filters.rank.generic')
    for name in R5:
        function = getattr(rank, name)
        require(inspect.unwrap(function).__module__ == 'skimagexx._rank', 'upstream public algorithm forwarding')
        require(name not in vars(generic) and name not in generic.__all__, 'R5 generic path must remain absent')
    bilateral_admission(xx._core, rank, importlib.import_module('skimagexx.filters.rank.bilateral'),
                        generic, importlib.import_module('skimagexx._rank'), coverage.native_functions())
    decisions_admission(xx._core, rank, generic, importlib.import_module('skimagexx._rank'), coverage.native_functions())
    tv_admission(xx._core, importlib.import_module('skimagexx.restoration'),
                 importlib.import_module('skimagexx._tv_chambolle'), coverage.native_functions())
    bregman_admission(xx._core, importlib.import_module('skimagexx.restoration'),
                      importlib.import_module('skimagexx._tv_bregman'), coverage.native_functions())
    require(all(name not in generic.__all__ and hasattr(generic, name) for name in ('sum', 'windowed_histogram')), 'nonstar own attrs contract changed')
    if state['profile'] == 'debug':
        require(getattr(xx._core, '_consumer_gil_assertions', False) is True, 'Debug GIL assertions missing')
    versions = {}
    for name, version in state['runtime_pins'].items():
        actual = importlib.metadata.version(name); require(actual == version, 'runtime version mismatch: ' + name); versions[name] = actual
    # Authenticate installed dependency bytes to original wheel RECORDs before
    # fixtures, not merely the version string or a self-authored runtime snapshot.
    for filename, expected in state['runtime_wheels'].items():
        path = out / 'runtime-wheels' / filename
        require(wheel_metadata(path) == expected, 'runtime wheel changed')
        distribution = importlib.metadata.distribution(expected['name'])
        members = wheel_members(path)
        for name, digest in members.items():
            if name.endswith('.dist-info/RECORD'):
                continue  # pip legitimately appends generated script/installer rows.
            require('.data/' not in name, 'unassessed relocated dependency wheel data path: ' + name)
            installed = Path(distribution.locate_file(name))
            require(installed.is_file() and sha(installed) == digest, 'installed dependency differs from pinned wheel: ' + name)
    origins = {name: importlib.import_module(name).__file__ for name in ('numpy', 'scipy', 'skimage', 'pytest')}
    require(all(Path(p).resolve().is_relative_to(out / 'runtime-env') for p in origins.values()), 'runtime dependency outside isolated prefix')
    for worker in (4, 1):
        xx.set_num_threads(worker); require(xx._core.get_num_threads() == worker, 'global setter identity mismatch')
    xx.set_fallback_enabled(False)
    return dict(python=sys.version, executable=sys.executable, executable_sha256=sha(sys.executable), base_interpreter=base_interpreter(), platform=platform.platform(),
                core=xx._core.__file__, core_sha256=sha(xx._core.__file__), default_workers=1, shared_workers=[1,4],
                versions=versions, origins=origins, source_image_count=state['expected']['image_count'], source_registered_count=len(state['expected']['functions']),
                source_metadata_count=len(state['expected']['functions'])-state['expected']['image_count'],
                scalar=os.environ.get('SKIMAGEXX_DISABLE_SIMD') == '1', thread_pools={k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')},
                providers='NumPy/SciPy preserve numerical/provider semantics; this is not pure C++ fitting. BLAS/OMP pools1, native workers1/4.')


def returned_maps():
    import numpy as np, skimagexx as xx
    from skimagexx import _component_consumers as native
    spec = importlib.util.spec_from_file_location('qualification_original_arraymap_contract', ROOT / 'tests/test_arraymap_dense_contracts.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); contract._NATIVE_CLASS = type(xx.segmentation.relabel_sequential(np.array([0,1], np.int16))[1])
    require(contract._NATIVE_CLASS is native.ArrayMap, 'not the installed public returned map class')
    matrix = json.loads(contract.MATRIX.read_text()); profile = matrix['profiles'][contract.profile()]
    require(len(contract.CASES) == 382, 'frozen returned-map corpus changed')
    for worker in (1,4):
        xx.set_num_threads(worker)
        for item in contract.CASES:
            require(contract.observe(item, True) == matrix['observations'][profile['observations'][item['id']]], 'returned map observation differs: ' + item['id'])
    print(json.dumps(dict(scope='Installed public returned class vs382 frozen protocol observations; oracle type injection only, no image/source facade; not new public class registration', passed=764)))


def inventory(out, state):
    spec = importlib.util.spec_from_file_location('qualification_inventory', ROOT / 'tools/inventory.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    actual = module.inventory(); target = json.loads((ROOT / 'coverage/api.json').read_text())
    dump(out / 'actual-inventory.json', actual)
    a = {r['api']: r for r in actual['entries']}; b = {r['api']: r for r in target['entries']}
    require(a.keys() == b.keys(), 'inventory names differ')
    fields = ('kind','implementation','signature','native_default_case_dtypes','parity_test','benchmark','class_contract')
    diffs = [dict(api=n, field=k, actual=a[n].get(k), target=b[n].get(k)) for n in a for k in fields if a[n].get(k) != b[n].get(k)]
    counts = dict(collections.Counter(r['implementation'] for r in a.values()))
    dump(out / 'inventory-comparison.json', dict(differences=diffs, counts=counts, target_sha256=sha(ROOT / 'coverage/api.json')))
    require(not diffs and counts.get('native-partial') == state['expected']['image_count'], 'actual inventory differs from target')


def child(out, group):
    state = json.loads((out / 'state.json').read_text()); source_now = source_state(ROOT)
    require(source_now == state['source'], 'frozen source changed')
    facts = admission(out, state)  # BEFORE importing pytest/contracts or fixtures.
    sys.path.append(str(ROOT))  # tools/test imports only; genuine package remains first.
    if group == 'admission':
        print(json.dumps(facts, indent=2)); return 0
    if group == 'inventory':
        inventory(out, state); return 0
    if group == 'returned-maps':
        returned_maps(); return 0
    import pytest
    require(group in GROUPS, 'unknown public selector')
    mode = 'scalar' if facts['scalar'] else 'normal'
    xml = out / f'{mode}-{group}.xml'
    temporary = out / 'pytest-tmp' / (mode + '-' + group)
    require(not temporary.exists(), 'never reuse a prior pytest basetemp')
    require(temporary.parent.resolve().is_relative_to(out.resolve()), 'pytest temporary parent escapes output')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    args = ['-q', '--capture=sys', '--tb=short', '--junitxml=' + str(xml),
            '--basetemp=' + str(temporary), '-o', 'cache_dir=' + str(out / 'pytest-cache' / (mode + '-' + group))]
    args += [str(ROOT / 'tests' / n) for n in GROUPS[group]]
    if group == 'metrics-policies':
        args += ['-k', POLICIES]
    return pytest.main(args)


def create_capsule(out, family, state):
    capsule = out / ('capsule-' + family); require(not capsule.exists(), 'capsule must be fresh')
    wheels = capsule / 'wheels'; wheels.mkdir(parents=True)
    inputs = list((out / 'runtime-wheels').glob('*.whl')) + list((out / 'product-wheels').glob('*.whl'))
    rows = {}
    for path in inputs:
        record = wheel_metadata(path); require(record['name'] not in rows, 'duplicate capsule distribution')
        rows[record['name']] = record
        target = wheels / path.name
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)
        require(sha(target) == record['sha256'], 'capsule wheel copy changed')
    require(set(rows) == set(state['runtime_pins']) | {'scikit-imagexx'}, 'incomplete runtime closure')
    (capsule / 'runtime-freeze.txt').write_text('# Authenticated installed distribution metadata, not unconstrained pip output\n' + '\n'.join(n + '==' + r['version'] for n,r in sorted(rows.items())) + '\n')
    return capsule


def corpus_command(python, family, capsule, output, prepare_capsule=False):
    require(family in CORPORA, 'unknown fixed corpus')
    require(capsule.is_dir() and (capsule / 'wheels').is_dir(), 'required complete capsule absent')
    command = [str(python), str(ROOT / 'benchmarks' / CORPORA[family][0]), '--capsule', str(capsule),
               '--check-only', '--threads', '1', '4', '--output', str(output)]
    if prepare_capsule:
        command.append('--prepare-capsule')
    return command


def validate_corpus(path, family):
    report = json.loads(path.read_text())
    require(report.get('status') == 'complete-parity-check' and not report.get('failures') and not report.get('results'), 'fixed corpus incomplete/failed/or timed')
    require(len(report['checks']) == CORPORA[family][1], 'fixed corpus count changed')
    require(report['metadata']['check_only'] is True and report['metadata']['reference_only'] is False and report['metadata']['requested_workers'] == [1,4], 'corpus must be actual public untimed1/4')
    field = 'by_workers' if family == 'consumers' else 'actual_by_workers'
    require(all(row.get('status') == 'passed' and set(row.get(field, {})) == {'1','4'} for row in report['checks']), 'every fixed case must pass both worker configurations')
    if family == 'percentile':
        require([len(report[k]) for k in ('alias_checks','provider_checks','policy_checks')] == [9,54,54], 'R5 auxiliary117 contract changed')
        require(all(set(row.get('actual_by_workers',{})) == {'1','4'} for key in ('provider_checks','policy_checks') for row in report[key]), 'R5 provider/policy workers incomplete')
        require(all(row.get('reference_identity') is True and row.get('native_identity') is None for row in report['alias_checks']), 'R5 private reference identities must not count as native aliases')

    if family == 'bilateral':
        require([len(report[k]) for k in ('alias_checks','provider_checks','policy_checks')] == [3,96,18], 'R6 auxiliary117 contract changed')
        require(report['metadata'].get('validation_scope') == 'strict actual public R6', 'R6 genuine source admission absent')
        for key in ('checks','alias_checks','provider_checks','policy_checks'):
            require(all(set(row.get('actual_by_workers',{})) == {'1','4'} for row in report[key]), 'R6 auxiliary workers incomplete')
            require(all('reference' in row and all(actual == row['reference'] for actual in row['actual_by_workers'].values()) for row in report[key]), 'R6 observation differs from reference')
        require(all(row.get('reference_identity') is True and row.get('native_identity') is True for row in report['alias_checks']), 'R6 canonical/nested identities incomplete')
        require({(row.get('canonical'),row.get('alias')) for row in report['alias_checks']} ==
                {('filters.rank.'+n,'filters.rank.bilateral.'+n) for n in R6}, 'R6 alias paths differ')


    if family == 'decisions':
        require([len(report[k]) for k in ('alias_checks','provider_checks','policy_checks')] == [2,32,12], 'R7 auxiliary46 contract changed')
        meta = report['metadata']
        require(meta.get('validation_scope') == 'strict actual public R7', 'R7 genuine source admission absent')
        require(meta.get('algorithm_count') == meta.get('scalar_api_count') == 2
                and meta.get('vector_api_count') == 0 and meta.get('aliases_timed') is False
                and meta.get('fallback_enabled') is False and type(meta.get('initial_default_workers')) is int
                and meta.get('initial_default_workers') == 1,
                'R7 algorithm/worker/no-forward metadata differs')
        require(meta.get('canonical_paths') == ['filters.rank.'+name for name in R7]
                and meta.get('public_alias_paths') == ['filters.rank.generic.'+name for name in R7]
                and meta.get('operation_ids') == {'otsu':29,'majority':30}, 'R7 API/operation metadata differs')
        require(all(type(worker) is int for worker in meta['requested_workers']), 'R7 worker telemetry must be integers')
        source = json.loads((ROOT / 'coverage/contracts/rank-decisions-benchmark.json').read_text())
        inventory = source['case_inventory']
        require(len(inventory) == 694 and meta.get('case_contract_sha256') == source['case_contract_sha256'], 'R7 frozen694 inventory differs')
        require(all(type(row.get('case_index')) is int for row in report['checks'])
                and [row.get('case_index') for row in report['checks']] == list(range(694)), 'R7 missing/duplicate/reordered cases')
        for row, expected in zip(report['checks'], inventory):
            require(all(key in row and row[key] == value for key,value in expected.items()), 'R7 case specification differs')
        metadata_rows = [row for row in report['checks'] if row.get('timing_class') == 'source-metadata-layout']
        require(len(metadata_rows) == 16 and all(row.get('timing_eligible') is False for row in metadata_rows), 'R7 all16 metadata/error rows must remain untimed')
        def identical(left, right):
            try:
                return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError):
                return False
        for key in ('checks','alias_checks','provider_checks','policy_checks'):
            require(all(set(row.get('actual_by_workers',{})) == {'1','4'} for row in report[key]), 'R7 auxiliary workers incomplete')
            require(all('reference' in row and all(identical(actual,row['reference']) for actual in row['actual_by_workers'].values()) for row in report[key]), 'R7 observation differs from reference')
        require(all(row.get('timed') is False for key in ('alias_checks','provider_checks','policy_checks') for row in report[key]), 'R7 auxiliary rows must stay untimed')
        providers = {(row.get('api'),row.get('case')) for row in report['provider_checks']}
        require(providers == {('filters.rank.'+name,f'{ndim}/{stage}') for name in R7 for ndim in (2,3)
                for stage in ('success','footprint','shift_x','shift_y','shift_z','bad-out','bad-image','image-array-only')}, 'R7 provider identities differ')
        policies = {(row.get('api'),row.get('policy')) for row in report['policy_checks']}
        require(policies == {('filters.rank.'+name,policy) for name in R7 for policy in ('warn','raise','ignore','call','log','print')}, 'R7 policy identities differ')
        require(all(row.get('reference_identity') is True and row.get('native_identity') is True for row in report['alias_checks']), 'R7 canonical/generic identities incomplete')
        aliases = [(row.get('canonical'),row.get('alias')) for row in report['alias_checks']]
        require(len(set(aliases)) == 2 and set(aliases) == {('filters.rank.'+name,'filters.rank.generic.'+name) for name in R7}, 'R7 alias paths differ')


def validate_metrics(path):
    report = json.loads(path.read_text())
    require(report.get('status') == 'complete-parity-check' and not report.get('failures') and not report.get('results') and len(report.get('checks',[])) == 1231, 'metrics fixed1231 failed/not untimed')
    meta = report['metadata']
    require(meta['check_only'] is True and meta['reference_only'] is False and meta['requested_workers'] == [1,4], 'metrics must be actual public untimed1/4')
    for row in report['checks']:
        require(row.get('status') == 'passed', 'metrics case not passed')
        if row.get('expected_error'):
            require(row.get('error_parity_workers') == [1,4], 'metrics expected error not checked at both workers')
        else:
            require(set(row.get('actual_scalar_by_workers',{})) == {'1','4'}, 'metrics scalar not checked at both workers')


def windows_toolchain_model(cache_text, compiler_text, project_text, config):
    """Read actual CMake/MSVC output; do not infer a toolset from a year label."""
    cache = dict(re.findall(r'^([^#/\n:=]+):[^=\n]+=(.*)$', cache_text, re.M))
    compiler = dict(re.findall(r'^set\(\s*(\w+)\s+"?([^"\n)]*?)"?\s*\)', compiler_text, re.M))
    generator = cache.get('CMAKE_GENERATOR','')
    require(re.fullmatch(r'Visual Studio \d+ .+',generator) is not None, 'sealed Windows wheel must expose its actual Visual Studio toolchain')
    require(compiler.get('CMAKE_CXX_COMPILER_ID') == 'MSVC', 'sealed Windows compiler is not MSVC')
    architecture = compiler.get('CMAKE_CXX_COMPILER_ARCHITECTURE_ID') or compiler.get('MSVC_CXX_ARCHITECTURE_ID')
    require(architecture == cache.get('CMAKE_GENERATOR_PLATFORM') == 'x64', 'sealed Windows architecture must be x64')
    executable = compiler.get('CMAKE_CXX_COMPILER','')
    tool = re.search(r'[/\\]MSVC[/\\](\d+\.\d+\.\d+(?:\.\d+)?)[/\\]bin[/\\]Host(x64|x86)[/\\]x64[/\\]cl\.exe$',executable,re.I)
    require(tool is not None, 'cannot authenticate exact MSVC toolset/host from compiler path')
    require(bool(compiler.get('CMAKE_CXX_COMPILER_VERSION')), 'sealed compiler version absent')
    instance = cache.get('CMAKE_GENERATOR_INSTANCE','')
    require(instance and PureWindowsPath(executable).is_relative_to(PureWindowsPath(instance)), 'compiler is outside sealed Visual Studio instance')
    project = ET.fromstring(project_text)
    def tag(node):return node.tag.rsplit('}',1)[-1]
    platform_toolsets=set(); runtimes=set()
    for group in project:
        condition=group.get('Condition','').replace(' ','')
        if '==' not in condition or condition.split('==',1)[1].strip("'\"") != config+'|x64':continue
        for node in group.iter():
            if tag(node)=='PlatformToolset' and node.text:platform_toolsets.add(node.text)
            if tag(node)=='RuntimeLibrary' and node.text:runtimes.add(node.text)
    require(len(platform_toolsets)==1 and len(runtimes)==1, 'sealed platform toolset/runtime ambiguous or absent')
    platform_toolset=platform_toolsets.pop();runtime=runtimes.pop()
    require(re.fullmatch(r'v\d+',platform_toolset) is not None, 'unrecognized sealed platform toolset')
    require(runtime in ('MultiThreaded','MultiThreadedDLL','MultiThreadedDebug','MultiThreadedDebugDLL'), 'unrecognized sealed MSVC runtime')
    return dict(generator=generator,instance=instance,architecture=architecture,compiler=executable,
                compiler_version=compiler['CMAKE_CXX_COMPILER_VERSION'],toolset_version=tool.group(1),
                host_architecture=tool.group(2).lower(),platform_toolset=platform_toolset,runtime=runtime,config=config)


def read_windows_toolchain(build, config, target='skimagexx'):
    compiler_files=list((build/'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    require(len(compiler_files)==1,'successful wheel/compiler provenance missing or ambiguous')
    paths=[build/'CMakeCache.txt',compiler_files[0],build/(target+'.vcxproj')]
    model=windows_toolchain_model(*(p.read_text(encoding='utf-8-sig') for p in paths),config)
    compiler=Path(model['compiler']).resolve(strict=True)
    require(compiler.is_file(),'sealed MSVC compiler no longer available')
    model['compiler']=str(compiler);model['compiler_sha256']=sha(compiler)
    model['evidence']={p.relative_to(build).as_posix():sha(p) for p in paths}
    return model


def select_visual_studio(model, capabilities, instances):
    candidates=[g for g in capabilities.get('generators',[]) if g.get('name')==model['generator']]
    require(len(candidates)==1 and candidates[0].get('platformSupport') and candidates[0].get('toolsetSupport'), 'sealed Visual Studio generator unavailable in actual CMake')
    require(model['architecture'] in candidates[0].get('supportedPlatforms',[]), 'sealed architecture unavailable in actual CMake generator')
    matching=[i for i in instances if i.get('isComplete') is True and i.get('isLaunchable') is True
              and PureWindowsPath(i.get('installationPath',''))==PureWindowsPath(model['instance'])]
    require(len(matching)==1,'sealed Visual Studio instance unavailable or ambiguous')
    instance=matching[0]
    major=re.fullmatch(r'Visual Studio (\d+) .+',model['generator']).group(1)
    require(str(instance.get('installationVersion','')).split('.')[0]==major,'sealed generator and installed instance version mismatch')
    require(PureWindowsPath(model['compiler']).is_relative_to(PureWindowsPath(instance['installationPath'])), 'sealed compiler belongs to another instance')
    return dict(generator=model['generator'],instance=instance['installationPath'],
                installation_version=instance['installationVersion'],architecture=model['architecture'],runtime=model['runtime'],
                toolset=model['platform_toolset']+',version='+model['toolset_version']+',host='+model['host_architecture'])


def require_same_windows_toolchain(wheel, sdk_model):
    for key in ('generator','architecture','compiler_version','toolset_version','host_architecture','platform_toolset','runtime','compiler_sha256'):
        require(wheel[key]==sdk_model[key], 'SDK/wheel toolchain mismatch: '+key)
    for key in ('instance','compiler'):
        require(PureWindowsPath(wheel[key])==PureWindowsPath(sdk_model[key]), 'SDK/wheel toolchain mismatch: '+key)


def validate_sdk_listing(listing, *, scope=PRODUCT_SCOPE):
    product_census(scope)
    families = {'rank14': ORIGINAL_FAMILIES, 'tv17': TV_FAMILIES, 'bregman18': FAMILIES}[scope]
    require({row['name'] for row in listing['tests']} == {'installed_' + f + '_consumer' for f in families}
            and len(listing['tests']) == len(families), 'SDK test discovery differs/empty')


def sdk(out, state, journal):
    prefix = out / 'relocated sdk'; prefix.mkdir(); record = json.loads((out / 'package.json').read_text())
    for name, digest in record['SDK_sha256'].items():
        source = out / 'package' / name; require(sha(source) == digest, 'SDK changed before relocation')
        target = prefix / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    sdk_source = out / 'consumer source'; sdk_source.mkdir()
    with zipfile.ZipFile(out / 'source.zip') as archive:
        for name in archive.namelist():
            if not name.endswith('/') and (name.startswith('tests/cmake_consumer/') or name.startswith('tests/') and name.endswith('.cpp')):
                target = sdk_source / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(archive.read(name))
    cmake = tool_in(out / 'build-env', 'cmake'); ctest = tool_in(out / 'build-env', 'ctest'); build = out / 'sdk-build'
    config = state['profile_config']['config']; env = environment()
    configure = [cmake, '-S', sdk_source / 'tests/cmake_consumer', '-B', build, '-DCMAKE_BUILD_TYPE=' + config,
                 '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DSKIMAGEXX_EXPECT_PREFIX=' + str(prefix), '-DSKIMAGEXX_EXPECT_OPENMP=ON',
                 '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_DISABLE_FIND_PACKAGE_Python=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON']
    if os.name == 'nt':
        wheel_toolchain=read_windows_toolchain(out/'native-build',config)
        vswhere=shutil.which('vswhere')
        if vswhere is None:
            program_files=os.environ.get('ProgramFiles(x86)')
            require(program_files is not None,'vswhere location unavailable')
            vswhere=str(Path(program_files)/'Microsoft Visual Studio/Installer/vswhere.exe')
        require(Path(vswhere).is_file(),'vswhere executable unavailable')
        must_run(journal,'toolchain-cmake-capabilities',[cmake,'-E','capabilities'],env)
        must_run(journal,'toolchain-visual-studio-instances',[vswhere,'-all','-products','*','-format','json','-utf8'],env)
        capabilities=json.loads((out/'toolchain-cmake-capabilities.log').read_text(encoding='utf-8-sig'))
        instances=json.loads((out/'toolchain-visual-studio-instances.log').read_text(encoding='utf-8-sig'))
        selection=select_visual_studio(wheel_toolchain,capabilities,instances)
        dump(out/'windows-sdk-selection.json',dict(wheel=wheel_toolchain,selection=selection,
             capabilities_sha256=sha(out/'toolchain-cmake-capabilities.log'),instances_sha256=sha(out/'toolchain-visual-studio-instances.log')))
        configure += ['-G',selection['generator'],'-A',selection['architecture'],'-T',selection['toolset'],
                      '-DCMAKE_GENERATOR_INSTANCE='+selection['instance'],'-DCMAKE_MSVC_RUNTIME_LIBRARY='+selection['runtime']]
    must_run(journal, 'sdk-configure', configure, env)
    if os.name == 'nt':
        sdk_toolchain=read_windows_toolchain(build,config,FAMILIES[0]+'_consumer')
        require_same_windows_toolchain(wheel_toolchain,sdk_toolchain)
        dump(out/'windows-sdk-toolchain-verification.json',dict(wheel=wheel_toolchain,sdk=sdk_toolchain,matched=True))
    must_run(journal, 'sdk-build', [cmake, '--build', build, '--config', config, '--parallel', '2', '--target', *[f + '_consumer' for f in FAMILIES]], env)
    pattern = '^installed_(' + '|'.join(FAMILIES) + ')_consumer$'
    must_run(journal, 'sdk-list', [ctest, '--test-dir', build, '-C', config, '-R', pattern, '--show-only=json-v1'], env)
    listing = json.loads((out / 'sdk-list.log').read_text())
    validate_sdk_listing(listing)
    for mode in ('normal','scalar'):
        code = journal.run('sdk-' + mode, [ctest, '--test-dir', build, '-C', config, '-R', pattern, '--output-on-failure'], environment(scalar=mode == 'scalar'))
        require(code == 0, 'installed SDK client failed')
    verify_tree(prefix, record['SDK_sha256'])
    dump(out / 'sdk.json', dict(families=FAMILIES, checks=2*len(FAMILIES), SDK_sha256=record['SDK_sha256'], scope='Relocated complete SDK,17 independently linked installed clients normal/scalar; no source includes or Python dependency'))


def qualify(out):
    state = json.loads((out / 'state.json').read_text()); require(source_state(ROOT) == state['source'], 'source changed since build')
    journal = Journal(out); python = python_in(out / 'runtime-env'); package = out / 'package'
    base = [python, Path(__file__).resolve(), '_child', '--output', out, '--profile', state['profile']]
    # All required capsules exist before ANY corpus process can construct fixtures.
    capsules = {name: create_capsule(out, name, state) for name in CORPORA}
    for mode in ('normal','scalar'):
        env = environment(package, scalar=mode == 'scalar')
        for group in ('admission', *GROUPS, 'returned-maps'):
            journal.run(mode + '-' + group, [*base, '--group', group], env)
        if mode == 'normal':
            journal.run('actual-inventory', [*base, '--group', 'inventory'], env)
        journal.run(mode + '-metrics-corpus', [python, ROOT / 'benchmarks/metrics_reductions.py', '--check-only', '--threads', '1', '4', '--output', out / (mode + '-metrics-corpus.json')], env)
        for family, capsule in capsules.items():
            journal.run(mode + '-' + family + '-corpus', corpus_command(python, family, capsule, out / (mode + '-' + family + '-corpus.json'), mode == 'normal'), env)
    # SDK build/test occurs only in this qualification job; never measurement.
    sdk(out, state, journal)


def seal(out):
    state = json.loads((out / 'state.json').read_text()); require(source_state(ROOT) == state['source'], 'source changed')
    commands = json.loads((out / 'commands.json').read_text())
    require(commands and all(r['state'] == 'terminal' for r in commands), 'live/incomplete child cannot be sealed')
    required_labels = {'initial-public-admission','actual-inventory','sdk-configure','sdk-build','sdk-list','sdk-normal','sdk-scalar',
                       'gather-build-identity-facts','capture-build-identity'}
    for mode in ('normal','scalar'):
        required_labels.update(mode + '-' + n for n in ('admission', *GROUPS, 'returned-maps','metrics-corpus'))
        required_labels.update(mode + '-' + n + '-corpus' for n in CORPORA)
    require(required_labels <= {r['label'] for r in commands}, 'missing required qualification command')
    validate_build_identity(out, state)
    validate_interpreter_provenance(out, state['interpreter_provenance'])
    failures = [r['label'] for r in commands if r['returncode'] != 0]
    record = json.loads((out / 'package.json').read_text()); verify_tree(out / 'package', record['members_sha256'])
    require(sha(out / 'product-wheels' / state['product_wheel']) == record['wheel_sha256'], 'product wheel changed')
    counts = {}
    for mode in ('normal','scalar'):
        for group in GROUPS:
            path = out / f'{mode}-{group}.xml'
            require(path.is_file(), 'missing actual public XML: ' + group)
            xml = ET.parse(path).getroot(); suites = [xml] if xml.tag == 'testsuite' else list(xml)
            row = {k: sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            require(row['tests'] > 0, 'empty public selector: ' + group)
            if any(row[k] for k in ('failures','errors','skipped')):
                failures.append(mode + '-' + group + '-assertions-or-skips')
            counts[mode + '-' + group] = row
        for family in CORPORA:
            validate_corpus(out / (mode + '-' + family + '-corpus.json'), family)
        validate_metrics(out / (mode + '-metrics-corpus.json'))
    require(len([r for r in commands if r['label'].startswith('sdk-')]) == 5, 'SDK commands incomplete')
    records = {p.relative_to(out).as_posix(): dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    for name, row in records.items():
        verify_record(out, name, row, state['interpreter_provenance'])
    proof = dict(schema=SCHEMA, source=state['source'], profile=state['profile'], passed=not failures, failures=failures,
                 build_identity=state['build_identity'],
                 interpreter_provenance=state['interpreter_provenance'],
                 counts=counts, SDK_checks=2*len(FAMILIES), expected_registry=state['expected'], command_count=len(commands), records=records,
                 no_timing=True, inheritance='None implicit. These are bounded actual public scopes, not full-library or sanitizer/portable acceptance.',
                 providers='NumPy/SciPy numerical providers allowed explicitly; native global workers1/4 and BLAS/OMP1 recorded; pure C++ fitting not claimed.',
                 supplemental='382 returned ArrayMap protocol cases use installed public returned class; no public class-count increase. Reverse-load cross-build diagnostics NOT run without separate authenticated matched-control evidence.')
    require(not (out / 'manifest.json').exists(), 'never overwrite sealed trial')
    dump(out / 'manifest.json', proof)
    print(json.dumps(dict(passed=proof['passed'], manifest_sha256=sha(out / 'manifest.json'), files=len(records))))
    return int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan','prepare','qualify','seal','verify','all','inspect-wheel','_child','_headers'])
    parser.add_argument('--profile', choices=PROFILES, default='modern')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-native-images', type=int)
    parser.add_argument('--group', choices=['admission','inventory','returned-maps',*GROUPS])
    parser.add_argument('--wheel', type=Path); parser.add_argument('--source-archive', type=Path); parser.add_argument('--source-revision')
    args = parser.parse_args(argv); out = args.output.resolve()
    require(out.is_relative_to(ROOT / 'build'), 'outputs must be in ignored source build directory')
    if args.stage == 'plan':
        result = plan(ROOT, args.profile, args.expected_native_images); print(json.dumps(result,indent=2)); return int(bool(result['blockers']))
    if args.stage == 'inspect-wheel':
        require(args.wheel is not None and args.source_archive is not None and args.source_revision is not None, 'inspect requires exact wheel/archive/revision')
        require(not out.exists(), 'choose fresh inspection output')
        record = inspect_wheel(args.wheel, args.source_archive, args.source_revision); out.mkdir(parents=True); dump(out / 'inspection.json', record); return 0
    if args.stage == 'verify':
        proof = json.loads((out / 'manifest.json').read_text())
        require(proof['schema'] == SCHEMA and source_state(ROOT) == proof['source'], 'manifest source/schema differs')
        validate_interpreter_provenance(out, proof['interpreter_provenance'])
        for name, record in proof['records'].items():
            verify_record(out, name, record, proof['interpreter_provenance'])
        print(json.dumps(dict(passed=proof['passed'], manifest_sha256=sha(out / 'manifest.json'), records_rehashed=len(proof['records']))))
        return int(not proof['passed'])
    if args.stage == '_child':
        require(args.group is not None, 'missing child group'); return child(out, args.group)
    if args.stage == '_headers':
        headers(out, args.profile); return 0
    if args.source_revision is not None:
        require(source_state(ROOT)['revision'] == args.source_revision, 'explicit frozen source revision differs')
    if args.stage in ('prepare','all'):
        prepare(out, args.profile, args.expected_native_images)
    if args.stage in ('qualify','all'):
        qualify(out)
    if args.stage in ('seal','all'):
        return seal(out)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # This is a failure receipt, never a passing/incomplete trial seal.
        if '--output' in sys.argv:
            position = sys.argv.index('--output') + 1
            if position < len(sys.argv):
                folder = Path(sys.argv[position]).resolve()
                if folder.is_relative_to(ROOT / 'build') and folder.is_dir():
                    dump(folder / ('failure-' + str(os.getpid()) + '.json'), dict(status='failed-unsealed', exception_type=type(exc).__name__, message=str(exc), stage=sys.argv[1], no_timing=True))
        import traceback
        traceback.print_exc()
        print('QUALIFICATION BLOCKED: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
