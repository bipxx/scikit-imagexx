"""Explicit2270 source epoch for the retained log2-only source/SDK guards.

The current matmul guard reads actual current source independently. This source
view contains no build, Git, environment or retained-result copies and grants no
numerical or release evidence. Every copied byte matches its2270 Git identity.
"""
from pathlib import Path
import pytest


@pytest.fixture(scope='module', autouse=True)
def _vi_2270_source_epoch(request, tmp_path_factory):
    from tools.mark_lbp_fp_context import VI_PUBLIC_MATMUL_CONTEXT_SHA256
    from tools.vi_public_matmul_context import historical_source_root
    root = Path(__file__).resolve().parents[1]
    if not (root / 'coverage/contracts/vi-public-matmul-source-history-v1.json').is_file():
        yield
        return
    epoch = historical_source_root(root, tmp_path_factory.getbasetemp() / 'vi-2270-source-inputs',
                                   VI_PUBLIC_MATMUL_CONTEXT_SHA256)
    previous = request.module.ROOT
    request.module.ROOT = epoch
    try:
        yield
    finally:
        request.module.ROOT = previous
