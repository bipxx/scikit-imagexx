"""Installed NumPy-only checker controls; excluded from pure prebuild selectors."""
import numpy as np
import pytest
from test_bezier_curve_benchmark import module


def snapshot(result):
    # Reuse the actual canonical helper, not a checker mirror.
    import importlib.util
    spec=importlib.util.spec_from_file_location('draw_snapshot',module.ROOT/'benchmarks/draw_coordinates.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper.snapshot(result)


@pytest.mark.parametrize('kind',('positive','positive-empty','same-object','same-empty','cross-expected',
    'view','wrong-dtype','wrong-shape','wrong-word','readonly','pair-alias','wrong-container'))
def test_bezier_benchmark_checker_faults(kind):
    n=0 if kind in ('positive-empty','same-empty') else 3
    expected=(np.arange(n,dtype=np.intp),np.arange(n,dtype=np.intp))
    actual=tuple(a.copy() for a in expected)
    if kind in ('same-object','same-empty'):actual=expected
    elif kind=='cross-expected':actual=(expected[1],actual[1])
    elif kind=='view':actual=(actual[0].view(),actual[1])
    elif kind=='wrong-dtype':actual=(actual[0].astype(np.int32),actual[1])
    elif kind=='wrong-shape':actual=(actual[0].reshape(1,3),actual[1])
    elif kind=='wrong-word':actual[0][0]=99
    elif kind=='readonly':actual[0].flags.writeable=False
    elif kind=='pair-alias':actual=(actual[0],actual[0])
    elif kind=='wrong-container':actual=list(actual)
    bound={'coordinate_pairs_upper_bound':100}
    if kind.startswith('positive'):
        module.checked_output(actual,expected,np,snapshot,bound)
    else:
        with pytest.raises(AssertionError):module.checked_output(actual,expected,np,snapshot,bound)
