"""Public parity for the provider-backed data.file_hash route."""

from hashlib import sha256
import json

import skimagexx as xx
from skimage import data as original

from skimagexx import _data_file_hash
from fixtures.data_file_hash_reference_v1 import CAPTURE_LINUX, CAPTURE_WINDOWS, CONTRACT, run_contract


def test_file_hash_frozen_captures():
    manifest = json.loads(CONTRACT.read_bytes())
    assert manifest["reference"]["windows_sha256"] == sha256(CAPTURE_WINDOWS.read_bytes()).hexdigest()
    assert manifest["reference"]["linux_sha256"] == sha256(CAPTURE_LINUX.read_bytes()).hexdigest()
    assert len(json.loads(CAPTURE_WINDOWS.read_bytes())["outcomes"]) == 17
    assert len(json.loads(CAPTURE_LINUX.read_bytes())["outcomes"]) == 17


def test_file_hash_public_strict_parity(tmp_path):
    old = xx.set_fallback_enabled(False)
    try:
        assert xx.data.file_hash is _data_file_hash.file_hash
        result = run_contract(tmp_path, xx.data.file_hash, original.file_hash)
        assert len(result["results"]) == 17 and result["protocol_forms"] == 2
    finally:
        xx.set_fallback_enabled(old)
