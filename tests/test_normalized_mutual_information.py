"""Fixed source-authenticated original corpus and private NMI parity tests."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'coverage/contracts/normalized-mutual-information.json'


def source_authentication():
    """Authenticate the audited installed Python sources before their import."""
    contract = json.loads(CONTRACT.read_text())
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    assert versions['scikit-image'] == '0.26.0'
    profile = 'windows' if sys.platform == 'win32' else 'minimum' if versions['numpy'] == '1.24.4' else 'modern'
    assert versions['numpy'] == ('1.24.4' if profile == 'minimum' else '2.5.4')
    assert versions['scipy'] == ('1.15.3' if profile == 'minimum' else '1.18.1')
    records = []
    for relative, expected in contract['source_audit'][profile].items():
        distribution = 'scikit-image' if relative.startswith('skimage/') else relative.split('/')[0]
        path = Path(importlib.metadata.distribution(distribution).locate_file(relative))
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == expected, 'Audited original source changed: ' + str(path)
        records.append(dict(path=str(path), sha256=actual, bytes=len(raw)))
    return dict(profile=profile, versions=versions, python=sys.version,
                executable=sys.executable, files=records)


def load_cases():
    spec = importlib.util.spec_from_file_location('nmi_fixed_cases', ROOT / 'tests/test_normalized_mutual_information_cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cases()


def observe(function, case):
    import numpy as np
    callbacks = []
    class Log:
        def write(self, value): callbacks.append(value)
    previous = np.geterrcall()
    trace = case['trace']
    if trace is not None: trace.clear()
    result = {}
    policy = case['policy']
    try:
        if policy == 'call': np.seterrcall(lambda text, flag: callbacks.append([text, flag]))
        elif policy == 'log': np.seterrcall(Log())
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            with np.errstate(**({'all': policy} if policy else {})):
                try:
                    value = function(case['image0'], case['image1'], bins=case['bins'])
                    scalar = np.asarray(value)
                    result['return'] = dict(type=type(value).__module__ + '.' + type(value).__name__,
                        dtype=scalar.dtype.str, shape=list(scalar.shape), bits=scalar.tobytes().hex(),
                        value=float(value) if np.isfinite(value) else 'nan' if np.isnan(value) else 'inf' if value > 0 else '-inf')
                except Exception as error:
                    result['exception'] = dict(type=type(error).__module__ + '.' + type(error).__name__, message=str(error))
        result['warnings'] = [dict(category=type(w.message).__name__, message=str(w.message)) for w in seen]
    finally:
        np.seterrcall(previous)
    result['callbacks'] = callbacks
    result['protocol'] = list(trace) if trace is not None else []
    return result


def record_reference(destination):
    auth = source_authentication()
    from skimage.metrics import normalized_mutual_information
    factory = ROOT / 'tests/test_normalized_mutual_information_cases.py'
    corpus = load_cases()
    result = dict(schema='normalized-mutual-information-original-v1', source_authentication=auth,
        contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        factory_sha256=hashlib.sha256(factory.read_bytes()).hexdigest(),
        candidate_calls=0, timings=0,
        cases=[dict(id=case['id'], expected_native=case['native'], observation=observe(normalized_mutual_information, case)) for case in corpus])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(auth['profile'], len(corpus), 'original cases; candidate calls 0; timings 0')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-reference', required=True)
    record_reference(parser.parse_args().record_reference)
