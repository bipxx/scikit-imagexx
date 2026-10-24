"""Inert source/metadata gates; prepared numerical replay is separate evidence."""
import ast
import ctypes
import os
import struct
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/skimagexx/_matrix_transform.py'


def functions(*names):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


@pytest.mark.parametrize('fields', (
    (512, 512, 7664, 16, 2, 32), (5, 5, 1, 16, 2, 32),
    (4096, 4096, 65536, 16, 2, 32)))
def test_only_source_bounded_whole_K_driver_and_proven_unroll(fields):
    assert functions('_skylakex_driver_fields')['_skylakex_driver_fields'](fields)


@pytest.mark.parametrize('index,value', (
    (0, 4), (1, 4), (0, 4097), (1, 4097), (2, 0), (2, 65537),
    (3, 8), (4, 4), (5, 64), (0, True), (1, 512.0)))
def test_K_split_or_unaudited_unroll_is_rejected(index, value):
    fields = [512, 512, 7664, 16, 2, 32]
    fields[index] = value
    assert not functions('_skylakex_driver_fields')['_skylakex_driver_fields'](tuple(fields))


def test_driver_metadata_schema_cannot_be_a_protocol_or_partial_tuple():
    check = functions('_skylakex_driver_fields')['_skylakex_driver_fields']
    assert not check([512, 512, 7664, 16, 2, 32])
    assert not check((512, 512))
    class Hostile:
        def __len__(self):
            raise AssertionError('user protocol invoked')
    assert not check(Hostile())


def span():
    base = 0x180000000
    address = base + (0x28216fd00 - 0x280e70000) + 0x2f0
    query = dict(allocation_base=base, state=0x1000, type=0x1000000,
                 protect=4, region_base=address, region_size=24)
    return base, address, query


def test_exact_owner_bounded_live_field_span():
    assert functions('_skylakex_table_readable')['_skylakex_table_readable'](*span())


@pytest.mark.parametrize('key,value', (
    ('allocation_base', 0), ('state', 0), ('type', 0x20000),
    ('protect', 0x104), ('protect', 1), ('protect', 0x10),
    ('region_base', 0x190000000), ('region_size', 23)))
def test_other_owner_unreadable_guard_or_cross_page_is_rejected(key, value):
    base, address, query = span()
    query[key] = value
    assert not functions('_skylakex_table_readable')['_skylakex_table_readable'](base, address, query)


@pytest.mark.parametrize('delta', (-8, 8, 0x10000000))
def test_read_is_not_a_general_pointer_primitive(delta):
    base, address, query = span()
    assert not functions('_skylakex_table_readable')['_skylakex_table_readable'](base, address + delta, query)


@pytest.mark.parametrize('core,driver,expected', (
    (b'Haswell', None, '2.5.4'), (b'SkylakeX', None, None),
    (b'SkylakeX', (512, 512, 7664, 16, 2, 32), '2.5.4'),
    (b'Unknown', (512, 512, 7664, 16, 2, 32), None)))
def test_corename_alone_never_admits_the_new_runtime(core, driver, expected):
    scope = functions('_runtime_profile')
    scope.update(_unforced=lambda: True, _authenticate_runtime=lambda: certificate,
                 _RUNTIME_CORES={'exact-pin': frozenset(('Haswell',))},
                 _windows_skylakex_driver=lambda c: driver)
    certificate = ('2.5.4', 'exact-pin', lambda: core, lambda: 1, object(), ())
    assert scope['_runtime_profile']() == expected
    assert scope['_RUNTIME_DRIVER'] == (driver if core == b'SkylakeX' and expected else None)


def test_forced_runtime_rejects_before_authentication_and_driver_reads():
    scope = functions('_runtime_profile')
    def forbidden():
        raise AssertionError('authentication attempted for forced runtime')
    scope.update(_unforced=lambda: False, _authenticate_runtime=forbidden)
    assert scope['_runtime_profile']() is None
    assert scope['_RUNTIME_DRIVER'] is None and scope['_RUNTIME_CORENAME'] is None


@pytest.mark.parametrize('fault', (None, 'query-failed', 'other-owner', 'guard-page', 'bad-fields'))
def test_private_metadata_simulation_never_reads_before_owner_and_page_checks(monkeypatch, fault):
    scope = functions('_windows_skylakex_driver', '_skylakex_table_readable', '_skylakex_driver_fields')
    scope['ctypes'] = ctypes
    base, address, query = span()
    reads = []
    def virtual_query(actual, pointer, size):
        assert actual == address
        info = pointer._obj
        info.BaseAddress = address
        info.AllocationBase = 0 if fault == 'other-owner' else base
        info.RegionSize = 24
        info.State = query['state']
        info.Protect = 0x104 if fault == 'guard-page' else query['protect']
        info.Type = query['type']
        return 0 if fault == 'query-failed' else size
    def read(actual, size):
        reads.append((actual, size))
        return struct.pack('<6i', 4 if fault == 'bad-fields' else 512, 512, 7664, 16, 2, 32)
    certificate = ('2.5.4', 'ed4f167a5330424524f45258e7ca2c8d7f03c0cba1f30ee4985553ac1f88ecf8',
                   lambda: b'SkylakeX', lambda: 1, types.SimpleNamespace(_handle=base), ())
    with monkeypatch.context() as patch:
        patch.setattr(os, 'name', 'nt')
        patch.setattr(ctypes, 'WinDLL', lambda *a, **kw: types.SimpleNamespace(VirtualQuery=virtual_query), raising=False)
        patch.setattr(ctypes, 'string_at', read)
        actual = scope['_windows_skylakex_driver'](certificate)
    assert actual == ((512, 512, 7664, 16, 2, 32) if fault is None else None)
    assert reads == ([(address, 24)] if fault in (None, 'bad-fields') else [])


@pytest.mark.parametrize('index,value', ((0, '1.24.4'), (1, 'unknown'), (2, lambda: b'Haswell'), (3, lambda: 4)))
def test_driver_read_requires_exact_binary_version_live_core_and_single_thread(monkeypatch, index, value):
    scope = functions('_windows_skylakex_driver')
    scope['ctypes'] = types.SimpleNamespace()
    certificate = ['2.5.4', 'ed4f167a5330424524f45258e7ca2c8d7f03c0cba1f30ee4985553ac1f88ecf8',
                   lambda: b'SkylakeX', lambda: 1, object(), ()]
    certificate[index] = value
    with monkeypatch.context() as patch:
        patch.setattr(os, 'name', 'nt')
        assert scope['_windows_skylakex_driver'](certificate) is None


def test_original_arithmetic_selector_keeps_source_layout_branching_and_no_calibration():
    scope = functions('_blasable', '_arithmetic_profile', '_prepared_profile')
    fake = lambda strides, aligned=True: types.SimpleNamespace(
        T=types.SimpleNamespace(strides=strides), shape=(3, 3),
        flags=types.SimpleNamespace(aligned=aligned))
    h1 = types.SimpleNamespace(shape=(1, 3))
    hn = types.SimpleNamespace(shape=(7665, 3))
    choose = scope['_prepared_profile']
    assert choose(h1, fake((8, 24)), '2.5.4', 'SkylakeX') == 3
    assert choose(h1, fake((24, 8)), '2.5.4', 'SkylakeX') == 6
    assert choose(h1, fake((8, 24), False), '2.5.4', 'SkylakeX') == 6
    assert choose(h1, fake((48, 16)), '2.5.4', 'SkylakeX') == 5
    assert choose(hn, fake((48, 16)), '2.5.4', 'SkylakeX') == 1
    assert choose(hn, fake((48, 16)), '1.24.4', 'Haswell') == 2
