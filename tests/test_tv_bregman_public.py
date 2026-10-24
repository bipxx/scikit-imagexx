"""Actual installed Bregman pipeline; private adapters cannot satisfy this gate."""
import importlib
import inspect
import os
from pathlib import Path

import pytest
import skimagexx as xx
from skimage.restoration import denoise_tv_bregman as reference
from skimagexx.coverage import native_functions

# Reuse the exact numerical/metadata/guard assertions. Only the fixture changes:
# it must return the installed sole production core and the public own wrapper.
from test_tv_bregman_native_pipeline import (
    workers,
    test_frozen_observations_with_actual_private_cpp,
    test_admission_before_image_access,
    test_real_trace_and_setup,
    test_prewrite_binding_guards,
    test_private_source_and_fixed_signature,
    test_actual_initialization_worker_team,
    test_binding_owners_released_on_success_and_guard_failure,
    test_scalar_hooks_cannot_invalidate_captured_input,
    test_scalar_hooks_preserve_actual_posthook_values,
    test_native_float_descriptors_accept_independent_metadata,
    test_owned_tagged_pipeline_matches_safely_certified_original,
)


@pytest.fixture(scope='module')
def implementation():
    assert not os.environ.get('SKIMAGEXX_TV_BREGMAN_CORE'), 'private adapter forbidden in public proof'
    module = importlib.import_module('skimagexx._tv_bregman')
    core = xx._core
    assert module._core is core
    assert xx.restoration.denoise_tv_bregman is module.denoise_tv_bregman
    assert module.denoise_tv_bregman is not reference
    assert inspect.unwrap(module.denoise_tv_bregman).__module__ == 'skimagexx._tv_bregman'
    assert inspect.signature(module.denoise_tv_bregman) == inspect.signature(reference)
    assert tuple(native_functions()).count('restoration.denoise_tv_bregman') == 1
    package = Path(xx.__file__).resolve().parent
    assert Path(core.__file__).resolve().parent == Path(module.__file__).resolve().parent == package
    assert xx.set_num_threads is core.set_num_threads and xx.get_num_threads is core.get_num_threads
    assert core.get_num_threads() == 1
    assert inspect.isbuiltin(module._require_kernel())
    return module, core


@pytest.fixture(autouse=True)
def strict_dispatch():
    previous = xx.set_fallback_enabled(False)
    try:
        yield
    finally:
        xx.set_fallback_enabled(previous)
