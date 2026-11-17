"""Exact current integration to immutable admitted family bytes."""
import importlib.util
from pathlib import Path


def historical_source(root,name):
    p=Path(root)/'tools/join_source_context.py'
    spec=importlib.util.spec_from_file_location('_flood_exact_source_projection',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.enabled_predecessor_bytes(root,name,(Path(root)/name).read_bytes())
