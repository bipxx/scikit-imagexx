"""Bound upstream reference pools; native workers are set explicitly per test."""
import os
for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ.setdefault(variable,'1')

# Original-only observer, no pytest tests; loaded explicitly by the fresh Li child.
collect_ignore = ["test_li_u16_bounded.py"]


# Frozen MARK/LBP guards keep their original authority and source equations.
# Current VI SDK/qualification/VI4 guards read the actual checkout separately.
import pytest


@pytest.fixture(scope='module', autouse=True)
def _frozen_mark_lbp_source_epoch(request, tmp_path_factory):
    name = request.module.__name__.rsplit('.', 1)[-1]
    frozen = {
        'test_mark_lbp_enabled_source',
        'test_mark_boundaries_benchmark_execution',
        'test_draw_multiblock_lbp_benchmark_execution',
        'test_draw_multiblock_lbp_entry_fp',
        'test_mark_lbp_controller',
    }
    if name not in frozen:
        yield
        return
    from vi_public_matmul_source_view import _historical_source_epoch, _module_source_epoch
    root, epoch = _historical_source_epoch(tmp_path_factory)
    if epoch is None:
        yield
        return
    modules = [request.module]
    if name in ('test_mark_lbp_controller', 'test_draw_multiblock_lbp_benchmark_execution'):
        import importlib
        modules.append(importlib.import_module('test_mark_boundaries_benchmark_execution'))
    with _module_source_epoch(modules, root, epoch):
        yield


@pytest.fixture(autouse=True)
def _frozen_vi_shared_source_epoch(request, tmp_path_factory):
    module = request.module.__name__.rsplit('.', 1)[-1]
    node = request.node.name
    observers = ('tests/array_views_observer.py', 'tests/slice_along_axes_observer.py')
    historical_dependencies = ()
    if module == 'test_util_views_source':
        if node == 'test_current_native_closure_and_historical_pins':
            historical_dependencies = (*observers, 'src/execution.hpp', 'tests/cmake_consumer/util_views_consumer.hpp')
        elif node == 'test_observers_body_unchanged_from_captured_dependency':
            historical_dependencies = observers
    elif module == 'test_rolling_ball_sdk_dtype_source' and node == 'test_exact_source_dependency_and_historical_projection':
        historical_dependencies = ('src/execution.hpp',)
    if historical_dependencies:
        from pathlib import Path
        from vi_public_matmul_source_view import _historical_source_epoch
        root, epoch = _historical_source_epoch(tmp_path_factory)
        if epoch is not None:
            # Each named historical assertion reads only its exact predecessor
            # inputs. All other source and artifact reads use the live checkout.
            historical = {root / path: epoch / path for path in historical_dependencies}
            read_bytes = Path.read_bytes

            def observer_bytes(path):
                return read_bytes(historical.get(path, path))

            request.getfixturevalue('monkeypatch').setattr(Path, 'read_bytes', observer_bytes)
        yield
        return
    if module != 'test_variation_of_information_integration' or node != 'test_vi_shared_source':
        yield
        return
    from vi_public_matmul_source_view import _historical_source_epoch, _module_source_epoch
    root, epoch = _historical_source_epoch(tmp_path_factory)
    if epoch is None:
        yield
        return
    with _module_source_epoch((request.module,), root, epoch):
        yield
