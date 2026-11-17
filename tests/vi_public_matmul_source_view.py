"""Explicit2270 source epoch for the retained log2-only source/SDK guards.

The current matmul guard reads actual current source independently. This source
view contains no build, Git, environment or retained-result copies and grants no
numerical or release evidence. Every copied byte matches its2270 Git identity.
"""
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
import pytest


def _historical_source_epoch(tmp_path_factory):
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    from tools.vi_public_matmul_context import historical_source_root
    root = Path(__file__).resolve().parents[1]
    if not (root / 'coverage/contracts/vi-public-matmul-source-history-v1.json').is_file():
        return root, None
    return root, historical_source_root(
        root, tmp_path_factory.getbasetemp() / 'vi-2270-source-inputs',
        VI_PUBLIC_MATMUL_CONTEXT_SHA256)


@contextmanager
def _module_source_epoch(modules, root, epoch):
    """Move frozen readers and their import-time Path defaults together."""
    changes = []

    def projected(value):
        if not isinstance(value, Path):
            return value
        try:
            return epoch / value.relative_to(root)
        except ValueError:
            return value

    for module in modules:
        for name, value in list(vars(module).items()):
            replacement = projected(value)
            if replacement != value:
                changes.append((module, name, value))
                setattr(module, name, replacement)
            elif isinstance(value, FunctionType) and value.__globals__ is vars(module):
                for attribute in ('__defaults__', '__kwdefaults__'):
                    previous = getattr(value, attribute)
                    if previous is None:
                        continue
                    replacement = (tuple(projected(item) for item in previous)
                                   if attribute == '__defaults__' else
                                   {key: projected(item) for key, item in previous.items()})
                    if replacement != previous:
                        changes.append((value, attribute, previous))
                        setattr(value, attribute, replacement)
    try:
        yield
    finally:
        for owner, name, value in reversed(changes):
            setattr(owner, name, value)


@pytest.fixture(scope='module', autouse=True)
def _vi_2270_source_epoch(request, tmp_path_factory):
    root, epoch = _historical_source_epoch(tmp_path_factory)
    if epoch is None:
        yield
        return
    with _module_source_epoch((request.module,), root, epoch):
        yield
