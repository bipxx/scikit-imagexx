"""Prospective parity/benchmark assertions; saved results do not exist yet."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('prospective_flood_benchmark', Path(__file__).resolve().parents[1] / 'benchmarks/flood_fill.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_benchmark_method_and_frozen_cases():
    assert BENCH.METHOD['samples'] == 7
    assert BENCH.METHOD['minimum_sample_ns'] == 100_000_000
    assert BENCH.METHOD['minimum_warmup_ns'] == 250_000_000
    assert BENCH.METHOD['workers'] == 1
    rows = BENCH.case_specs()
    assert len(rows) == len({r['id'] for r in rows}) == 12
    assert {r['api'] for r in rows} == {'flood', 'flood_fill'}
    assert all(r['in_place'] is False for r in rows)
    assert max(BENCH.case_bounds(r)['logical'] for r in rows) == 24576
    assert max(BENCH.case_bounds(r)['neighbor_checks'] for r in rows) == 331776
    assert all(v is None for v in BENCH.BINDING.values())


@pytest.mark.parametrize('args', [[], ['--root-binding', 'invented.json'], ['--first-cohort']])
def test_cli_always_refuses_before_factory_or_sampler(args, monkeypatch):
    def forbidden(*unused):
        raise AssertionError('case/factory/sampler must not execute')
    monkeypatch.setattr(BENCH, 'case_specs', forbidden)
    with pytest.raises(SystemExit, match='UNBOUND'):
        BENCH.main(args)
