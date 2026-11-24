"""Fixed one-worker benchmark recipe for data.checkerboard."""

from functools import partial
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def fixed_public_calls(original, candidate, native_entry):
    contract = json.loads((ROOT / 'coverage/contracts/data-checkerboard-v1.json').read_bytes())
    rep = contract['representation']
    assert str(inspect.signature(original)) == str(inspect.signature(candidate)) == contract['signature']
    assert inspect.isbuiltin(native_entry)
    entries = []

    def profile(frame, event, value):
        if event == 'call' and frame.f_code is original.__code__:
            raise AssertionError('native checkerboard called stock API')
        if event == 'c_return' and value is native_entry:
            entries.append(value)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        actual = candidate()
    finally:
        sys.setprofile(previous)
    assert entries == [native_entry]
    assert sha256(actual.tobytes()).hexdigest() == rep['decoded_sha256']
    del actual
    yield 'default', partial(original), partial(candidate)
