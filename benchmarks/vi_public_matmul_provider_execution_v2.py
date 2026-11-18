"""Versioned current VI matmul route; reuse the unchanged strict engine/guard."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi-public-matmul-provider-benchmark-ci-native-team-v3.json'
ADAPTER='benchmarks/vi_public_matmul_provider_execution.py'
ADAPTER_CANONICAL_SHA256='7a6a84c4caba8acdbc2f04850205f283a80460bd484743142229c12aa32777fa'


def executor():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    assert data['route']['contract']==CONTEXT
    assert data['route']['driver']=='benchmarks/vi_public_matmul_provider_consumer_v2.py'
    path=ROOT/ADAPTER
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==ADAPTER_CANONICAL_SHA256
    spec=importlib.util.spec_from_file_location('_vi_matmul_v2_unchanged_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    module=adapter.executor()
    module.CONTEXT=CONTEXT
    return module


def execute(module):
    return executor().execute(module)
