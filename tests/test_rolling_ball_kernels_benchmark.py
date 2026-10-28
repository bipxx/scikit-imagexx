"""Inert prospective constructor-cohort checks; no scientific imports."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('rolling_constructor_benchmark',ROOT/'benchmarks/rolling_ball_kernels.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def test_frozen_parameters_and_output_bound():
    contract=json.loads(h.CONTRACT.read_bytes());cases=h.case_specs()
    assert len(cases)==len({case['case_id']for case in cases})==12
    assert (contract['threads'],contract['repeat'],contract['target_ms'],contract['raw_samples'])==(1,7,19,168)
    assert [case['parameters']['ndim']for case in cases if case['api']=='restoration.ball_kernel']==[1,2,3,4,5,8]
    for case in cases:
        shape=h.output_shape(case)
        assert list(shape)==case['expected_output_shape']and math.prod(shape)==case['expected_elements']<=contract['max_output_elements']
        assert 'dtype'not in case and 'image'not in case
        args=h.constructor_arguments(case)
        assert len(args)==2
    assert max(case['expected_elements']for case in cases)==250047


def test_ellipsoid_requested_even_shapes_are_not_output_shapes():
    for case in h.case_specs():
        if case['api']=='restoration.ellipsoid_kernel':
            requested=case['parameters']['shape'];output=h.output_shape(case)
            assert all(d%2==1 for d in output)
            assert list(output)==[value if value%2 else value+1 for value in requested]


def test_native_entry_names_and_inert_module_import():
    import ast
    assert h.ENTRIES=={'restoration.ball_kernel':'ball_kernel_apply','restoration.ellipsoid_kernel':'ellipsoid_kernel_apply'}
    tree=ast.parse((ROOT/'benchmarks/rolling_ball_kernels.py').read_text())
    top_imports=[name for node in tree.body if isinstance(node,ast.Import)for name in node.names]
    assert not any(name.name in ('numpy','scipy','skimage','skimagexx')for name in top_imports)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='measure'for n in ast.walk(main))


def test_source_reduction_retains_tree_and_positive_zero():
    import struct
    assert h.source_axis_sum([1.,4.,9.])==14.
    assert struct.pack('=d',h.source_axis_sum([0.]))==struct.pack('=d',0.)
    terms=[1e16]+[1.]*7
    assert h.source_axis_sum(terms)==1e16+6.
    assert h.source_axis_sum(terms)!=sum(terms)


def test_independent_toy_ball_oracle():
    import struct
    raw,shape=h.coordinate_oracle(dict(api='restoration.ball_kernel',parameters=dict(radius=1,ndim=1)))
    assert shape==(3,)and raw==struct.pack('=ddd',0.,1.,0.)


def test_independent_toy_ellipsoid_late_mask():
    import struct
    raw,shape=h.coordinate_oracle(dict(api='restoration.ellipsoid_kernel',parameters=dict(shape=[2,2],intensity=2.)))
    assert shape==(3,3)
    assert raw==struct.pack('=9d',math.inf,0.,math.inf,0.,2.,0.,math.inf,0.,math.inf)
