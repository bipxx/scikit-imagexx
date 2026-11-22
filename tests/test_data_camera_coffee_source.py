"""Verify the camera and coffee source, raw bytes, and benchmark refusal."""

import ast
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "coverage/contracts/data-camera-coffee-assets-v1.json").read_bytes())
WRAPPER = ROOT / "python/skimagexx/_data_camera_coffee.py"
RESOURCES = ROOT / "python/skimagexx/_data_assets"


def test_camera_coffee_source_members():
    tree = ast.parse(WRAPPER.read_bytes())
    assets, = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_ASSETS"]
    manifest = CONTRACT["assets"]
    assert tuple(manifest) == ("camera", "coffee")
    assert CONTRACT["performance"]["cases"] == [f"{name}-warm-local" for name in manifest]
    assert ast.literal_eval(assets) == {
        name: (asset["member"], tuple(asset["representation"]["shape"]),
               asset["representation"]["decoded_sha256"])
        for name, asset in manifest.items()
    }
    notice = (RESOURCES / CONTRACT["rights_notice"]).read_text(encoding="utf-8")
    for name, asset in manifest.items():
        raw = (RESOURCES / asset["member"]).read_bytes()
        representation = asset["representation"]
        assert sha256(raw).hexdigest() == representation["decoded_sha256"]
        assert len(raw) == prod(representation["shape"]) == representation["bytes"]
        assert asset["member"] in notice and asset["upstream_source"] in notice
        assert all(value in notice for value in asset["rights"].values())


def test_camera_coffee_benchmark_requires_authority():
    runner = ROOT / "benchmarks/data_camera_coffee.py"
    ast.parse(runner.read_bytes())
    process = subprocess.run([sys.executable, "-B", "-I", str(runner)], capture_output=True, text=True)
    assert process.returncode == 2 and "required" in process.stderr and "--go" in process.stderr
    assert "numpy" not in process.stderr and "skimage" not in process.stdout


@pytest.fixture
def private_family(monkeypatch):
    # Load only the exact source wrapper and sibling policy. The public route,
    # build products, and original public image functions are unnecessary here.
    from importlib.util import module_from_spec, spec_from_file_location

    source_init = WRAPPER.with_name("__init__.py")
    package_spec = spec_from_file_location("_camera_coffee_source", source_init)
    package = module_from_spec(package_spec)
    monkeypatch.setitem(sys.modules, package_spec.name, package)
    from _camera_coffee_source import _compat, _data_camera_coffee
    yield _data_camera_coffee, _compat
    sys.modules.pop(_data_camera_coffee.__name__)
    sys.modules.pop(_compat.__name__)


@pytest.mark.parametrize("name", ("camera", "coffee"))
def test_camera_coffee_source_returns_fresh_owned_arrays(private_family, name):
    import numpy as np

    family, policy = private_family
    policy.set_fallback_enabled(False)
    call = getattr(family, name)
    first, second = call(), call()
    expected = CONTRACT["assets"][name]["representation"]
    for result in (first, second):
        assert type(result) is np.ndarray and result.dtype.str == expected["dtype"]
        assert list(result.shape) == expected["shape"] and list(result.strides) == expected["strides"]
        assert result.base is None and result.flags.owndata and result.flags.writeable
        assert result.flags.c_contiguous and result.flags.aligned
        assert sha256(result.tobytes()).hexdigest() == expected["decoded_sha256"]
    assert not np.shares_memory(first, second)
    first.flat[0] ^= 255
    assert sha256(second.tobytes()).hexdigest() == expected["decoded_sha256"]
    assert sha256(call().tobytes()).hexdigest() == expected["decoded_sha256"]
    with pytest.raises(TypeError):
        call(None)
    with pytest.raises(TypeError):
        call(image=None)


@pytest.mark.parametrize("name", ("camera", "coffee"))
@pytest.mark.parametrize("fault", ("missing", "wrong-size", "wrong-digest"))
def test_camera_coffee_source_resource_failure_uses_policy(private_family, monkeypatch, name, fault):
    from types import SimpleNamespace
    from unittest.mock import Mock

    family, policy = private_family
    asset = CONTRACT["assets"][name]

    def read_bytes():
        if fault == "missing":
            raise FileNotFoundError(asset["member"])
        if fault == "wrong-size":
            return b"broken"
        return bytes(asset["representation"]["bytes"])

    resource = SimpleNamespace(read_bytes=read_bytes)
    joinpath = Mock(return_value=resource)
    provider = Mock(return_value=object())
    monkeypatch.setattr(family, "_PACKAGE", SimpleNamespace(joinpath=joinpath))
    monkeypatch.setattr(policy, "importlib", SimpleNamespace(
        import_module=Mock(return_value=SimpleNamespace(**{name: provider}))))
    call = getattr(family, name)
    policy.set_fallback_enabled(False)
    with pytest.raises(NotImplementedError, match="fallback is disabled"):
        call()
    provider.assert_not_called()
    policy.set_fallback_enabled(True)
    with pytest.warns(policy.FallbackWarning, match=rf"skimage\.data\.{name}"):
        assert call() is provider.return_value
    provider.assert_called_once_with()
    assert all(args == ((asset["member"],), {}) for args in joinpath.call_args_list)
