"""Policy regression checks use real frontend/control code and stub geometry/providers."""
import ast
import builtins
import importlib.util
from pathlib import Path
import types
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dispatch():
    spec = importlib.util.spec_from_file_location("ellipse_policy_compat", ROOT / "python/skimagexx/_compat.py")
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    events = []
    provider_result = object()

    def provider(*args):
        events.append(("provider", args))
        return provider_result

    def import_provider(name):
        events.append(("import", name))
        assert name == "skimage.draw._draw"
        return types.SimpleNamespace(_ellipse_perimeter=provider)

    compat.importlib = types.SimpleNamespace(import_module=import_provider)

    def orientation(value):
        events.append(("orientation", value))
        return float(value)

    def workers():
        events.append(("workers",))
        return 4

    def native(*args):
        events.append(("native", args))
        return core.result

    core = types.SimpleNamespace(ellipse_perimeter_orientation=orientation,
                                 get_num_threads=workers,
                                 draw_ellipse_perimeter=native, result=None)
    source = ROOT / "python/skimagexx/_draw_ellipse_perimeter.py"
    tree = ast.parse(source.read_bytes())
    # Inject only the native/scientific dependencies. Execute the actual function
    # bodies and real _compat policy, without loading geometry or NumPy.
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Import) and any(a.name == "numpy" for a in node.names))
                 and not (isinstance(node, ast.ImportFrom) and node.level == 1)]
    def guarded_import(name, *args, **kwargs):
        if name == "skimage.draw._draw":
            return import_provider(name)
        assert name.split(".")[0] not in {"numpy", "scipy", "skimage", "skimagexx"}
        return builtins.__import__(name, *args, **kwargs)

    scope = dict(_core=core, _compat=compat, np=types.SimpleNamespace(ndarray=type("Array", (), {})),
                 __builtins__={**vars(builtins), "__import__": guarded_import})
    exec(compile(tree, str(source), "exec"), scope)
    return scope["ellipse_perimeter"], compat, core, events, provider_result


@pytest.mark.parametrize("protocol_shape", (False, True))
@pytest.mark.parametrize("enabled", (False, True))
def test_both_provider_paths_obey_global_policy(dispatch, protocol_shape, enabled):
    call, compat, core, events, expected = dispatch
    shape = object() if protocol_shape else (16, 20)
    compat.set_fallback_enabled(enabled)
    if enabled:
        with pytest.warns(compat.FallbackWarning, match="skimage.draw._draw._ellipse_perimeter"):
            assert call(4, 6, 2, 3, shape=shape) is expected
        args = next(row[1] for row in events if row[0] == "provider")
        assert args[:5] == (4, 6, 2, 3, 0.0) and args[5] is shape
    else:
        with pytest.raises(NotImplementedError, match="fallback is disabled"):
            call(4, 6, 2, 3, shape=shape)
        assert not any(row[0] in ("import", "provider") for row in events)
    assert sum(row[0] == "native" for row in events) == (not protocol_shape)
    assert sum(row[0] == "workers" for row in events) == (not protocol_shape)


@pytest.mark.parametrize("protocol_shape", (False, True))
def test_fallback_warning_as_error_precedes_provider_import(dispatch, protocol_shape):
    call, compat, core, events, expected = dispatch
    shape = object() if protocol_shape else (16, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", compat.FallbackWarning)
        with pytest.raises(compat.FallbackWarning):
            call(4, 6, 2, 3, shape=shape)
    assert not any(row[0] in ("import", "provider") for row in events)


def test_native_acceptance_and_point_shape_bypass_need_no_fallback(dispatch):
    call, compat, core, events, expected = dispatch
    core.result = expected
    compat.set_fallback_enabled(False)
    assert call(4, 6, 2, 3, shape=(16, 20)) is expected
    assert call(4, 6, 0, 0, orientation=float("nan"), shape=object()) is expected
    assert not any(row[0] in ("import", "provider") for row in events)
    natives = [row[1] for row in events if row[0] == "native"]
    assert natives[0][-2:] == ((16, 20), 4)
    assert natives[1][-2:] == (None, 4)


def test_provider_receives_converted_coordinates_once(dispatch):
    call, compat, core, events, expected = dispatch
    class Index:
        def __init__(self, value):
            self.value = value
        def __index__(self):
            events.append(("index", self.value))
            return self.value
    shape = object()
    with pytest.warns(compat.FallbackWarning):
        assert call(*(Index(v) for v in (4, 6, 2, 3)), shape=shape) is expected
    assert events[:5] == [("index", 4), ("index", 6), ("index", 2), ("index", 3), ("orientation", 0)]
    assert [row[1] for row in events if row[0] == "provider"] == [(4, 6, 2, 3, 0.0, shape)]
