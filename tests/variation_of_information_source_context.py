"""Exact admitted VI component bytes through current shared fixture additions."""
import importlib.util
from pathlib import Path


def historical_source(root,name):
    root=Path(root);path=root/'tools/join_source_context.py'
    spec=importlib.util.spec_from_file_location('_vi_shared_source_projection',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.enabled_predecessor_bytes(root,name,(root/name).read_bytes())
