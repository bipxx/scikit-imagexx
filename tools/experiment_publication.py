"""Publish exact reviewed metadata through two fixed, existing validators.

Payloads are read from the runner event file, never step environment values or
command-line arguments, which GitHub may display in workflow logs.
This tool does not load fixtures, call image algorithms, or dispatch a workflow.
"""
from __future__ import annotations

import base64
import binascii
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
MAX_ENCODED = 40_000
MAX_COMPRESSED = 30_000
MAX_RAW = 256 * 1024
RANK_SCHEMA = 'rank-statistics-experiment-v1'
TV_SCHEMA = 'tv-chambolle-admission-experiment-v1'
RANK_SOURCE = 'benchmarks/rank_statistics_compare.py'
TV_SOURCE = 'tools/tv_chambolle_admission_artifacts.py'
CONTROLLER_INPUTS = {
    RANK_SCHEMA: (RANK_SOURCE, 'coverage/contracts/rank-statistics-comparison.json'),
    TV_SCHEMA: (TV_SOURCE, 'benchmarks/tv_chambolle_admission_compare.py',
                'tests/test_tv_chambolle_admission_compare.py',
                '.github/workflows/tv-chambolle-admission-compare.yml',
                RANK_SOURCE, 'coverage/contracts/rank-statistics-comparison.json',
                'benchmarks/tv_chambolle.py',
                'coverage/contracts/tv-chambolle-benchmark.json'),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def authenticate_payload(encoded, declared_sha):
    require(type(encoded) is str and 0 < len(encoded) <= MAX_ENCODED,
            'encoded payload exceeds bound or has invalid type')
    require(type(declared_sha) is str and re.fullmatch('[0-9a-f]{64}', declared_sha),
            'raw SHA must be lowercase hexadecimal')
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError('invalid base64 transport') from error
    require(0 < len(compressed) <= MAX_COMPRESSED, 'compressed payload exceeds bound')
    try:
        decoder = zlib.decompressobj(wbits=31)
        raw = decoder.decompress(compressed, MAX_RAW + 1)
    except zlib.error as error:
        raise RuntimeError('invalid gzip transport') from error
    require(len(raw) <= MAX_RAW and decoder.eof and not decoder.unconsumed_tail
            and not decoder.unused_data,
            'truncated, concatenated, trailing, or oversized gzip payload')
    require(hashlib.sha256(raw).hexdigest() == declared_sha, 'raw SHA differs')
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise RuntimeError('nonfinite JSON constant')


def _finite_float(value):
    result = float(value)
    require(math.isfinite(result), 'nonfinite JSON number')
    return result


def verify_controller_sources(schema, revision):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision),
            'controller revision must be lowercase hexadecimal')
    require(schema in CONTROLLER_INPUTS, 'unknown experiment schema')
    subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', revision + '^{commit}'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for relative in CONTROLLER_INPUTS[schema]:
        expected = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', revision + ':' + relative],
            stderr=subprocess.DEVNULL)
        actual = (ROOT / relative).read_bytes()
        if relative.endswith(('.py', '.yml')):
            actual = actual.replace(b'\r\n', b'\n')
        require(actual == expected, 'declared controller dependency differs')


@lru_cache(maxsize=2)
def _helper(relative):
    # Only literals from the registry below can reach this loader.
    require(relative in (RANK_SOURCE, TV_SOURCE), 'unknown validator')
    spec = importlib.util.spec_from_file_location(
        '_publication_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_rank(value, raw, revision, scratch):
    require(set(value) == {'schema', 'template_sha256', 'accepted_products'},
            'rank experiment fields differ')
    require(type(value['accepted_products']) is dict
            and set(value['accepted_products']) == {'before', 'after'},
            'rank product roles differ')
    # The established rank validator accepts a file, not a parsed dictionary.
    # Stage exact bytes privately and remove only this one owned temporary file.
    with tempfile.NamedTemporaryFile(dir=scratch, prefix='.rank-validation-',
                                     suffix='.json', delete=False) as stream:
        path = Path(stream.name)
        stream.write(raw)
    try:
        _helper(RANK_SOURCE).config(path, hashlib.sha256(raw).hexdigest())
    finally:
        path.unlink()


def _validate_tv(value, raw, revision, scratch):
    _helper(TV_SOURCE).validate_experiment(value, revision)


VALIDATORS = {RANK_SCHEMA: _validate_rank, TV_SCHEMA: _validate_tv}


def validate_certificate(raw, revision, scratch):
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    require(type(value) is dict and type(value.get('schema')) is str,
            'experiment must be a schema-bearing object')
    schema = value['schema']
    require(schema in VALIDATORS, 'unknown experiment schema')
    verify_controller_sources(schema, revision)
    VALIDATORS[schema](value, raw, revision, scratch)
    return schema


def publish(encoded, declared_sha, revision, output):
    raw = authenticate_payload(encoded, declared_sha)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'experiment.json'
    require(not target.exists(), 'experiment output already exists')
    schema = validate_certificate(raw, revision, output)
    with target.open('xb') as stream:
        stream.write(raw)
    require(target.read_bytes() == raw, 'published bytes changed')
    return dict(schema=schema, sha256=declared_sha, bytes=len(raw),
                controller_revision=revision, path=str(target))


def main():
    try:
        with Path(os.environ['GITHUB_EVENT_PATH']).open('rb') as stream:
            event_raw = stream.read(MAX_RAW + 1)
        require(len(event_raw) <= MAX_RAW, 'workflow event exceeds bound')
        event = json.loads(event_raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_float)
        inputs = event['inputs']
        require(type(inputs) is dict and set(inputs) == {
            'payload_gzip_base64', 'payload_sha256', 'controller_revision'},
            'workflow inputs differ')
        result = publish(inputs['payload_gzip_base64'],
                         inputs['payload_sha256'],
                         inputs['controller_revision'],
                         ROOT / 'build/reviewed-experiment')
    except Exception:
        # Validator exceptions may contain data-derived details. Never print
        # payloads, environment values, or their traceback in the workflow log.
        print('Experiment publication failed validation.', file=sys.stderr)
        return 1
    print('Authenticated reviewed experiment:', result['schema'],
          result['bytes'], result['sha256'], result['controller_revision'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
