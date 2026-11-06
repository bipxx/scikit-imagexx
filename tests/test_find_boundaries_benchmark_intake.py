"""Installed NumPy fixture-only intake; originals and sampler never called."""
from pathlib import Path
import runpy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
MODULE=runpy.run_path(str(ROOT/'benchmarks/find_boundaries.py'))


@pytest.mark.parametrize('spec',MODULE['case_specs'](),ids=lambda s:s['id'])
def test_find_boundaries_benchmark_fixture_only(spec):
    image,options=MODULE['fixture'](spec,np)
    assert type(image)is np.ndarray and list(image.shape)==spec['shape'] and image.dtype==np.dtype(spec['dtype'])
    assert image.size==MODULE['case_bounds'](spec)['input_elements']
    assert options==dict(connectivity=spec['connectivity'],mode=spec['mode'],background=0)
    assert image.flags.writeable is not spec['readonly']
    if spec['layout']=='F':assert image.flags.f_contiguous
    if spec['layout']=='reverse':assert image.strides[-1]<0
    if spec['pattern']=='noncanonical-bool':assert set(image.view(np.uint8).ravel())=={0,2,255}
    before=MODULE['snapshot'](image)
    assert MODULE['snapshot'](image)==before
