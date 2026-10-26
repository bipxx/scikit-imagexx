"""Actual integer stage and unchanged provider-boundary controls; no timing."""
import pytest
from test_footprint_shapes_public import runtime


@pytest.mark.parametrize("dtype", ("uint8", "float64", "bool", "S2", "U2", "object", None))
def test_integer_radius_avoids_grid_with_source_bool_conversion(runtime, monkeypatch, dtype):
    np, xx = runtime.np, runtime.xx
    expected = runtime.original.octahedron(3, dtype=dtype)
    def denied():
        raise AssertionError("trusted integer radius must not allocate mgrid")
    class NoGrid:
        def __getitem__(self, key): return denied()
    with monkeypatch.context() as patch:
        patch.setattr(np, "mgrid", NoGrid())
        result = xx.morphology.octahedron(3, dtype=dtype)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype and result.strides == expected.strides
    assert result.flags.owndata and result.flags.writeable and result.base is None
    if result.dtype.hasobject:
        assert [type(x) for x in result.flat] == [type(x) for x in expected.flat]


@pytest.mark.parametrize("kind", ("bool", "numpy-int", "int-subclass", "float", "fractional", "complex", "negative"))
def test_generic_radius_keeps_actual_coordinate_stage(runtime, monkeypatch, kind):
    class Radius(int): pass
    np = runtime.np
    radius = {"bool": True, "numpy-int": np.int64(3), "int-subclass": Radius(3),
              "float": 3.0, "fractional": 0.5, "complex": 1+2j, "negative": -1}[kind]
    expected = runtime.original.octahedron(radius)
    original, calls = runtime.core.footprint_shapes_l1, []
    def counted(*args):
        calls.append(len(args))
        return original(*args)
    with monkeypatch.context() as patch:
        patch.setattr(runtime.core, "footprint_shapes_l1", counted)
        result = runtime.xx.morphology.octahedron(radius)
    np.testing.assert_array_equal(result, expected)
    assert calls and all(n >= 4 for n in calls)
