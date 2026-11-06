"""Prebuild JOIN FENV source guards and actual Windows-only UCRT controls.

Linux has no UCRT route: pure source guards collect there, without skips.
No NumPy/scientific library or image/native target is imported or called.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]

def captured_input_predecessor(raw):
    link=json.loads((ROOT/'coverage/contracts/join-windows-fenv-fixture-v2.json').read_bytes())
    contexts=[]
    for key in ('captured_context','Windows_context'):
        body=(ROOT/link[key]['path']).read_bytes()
        assert hashlib.sha256(body).hexdigest()==link[key]['sha256']
        contexts.append(json.loads(body))
    phase,windows=contexts
    assert phase['public_test']==windows['public_test_path']==link['public_test']
    assert hashlib.sha256(raw).hexdigest()==phase['current_raw_sha256']==link['current_raw_sha256']
    text=raw.decode()
    for before,after in reversed(phase['inverse_replacements']):
        assert text.count(after)==1
        text=text.replace(after,before,1)
    previous=text.encode()
    assert hashlib.sha256(previous).hexdigest()==phase['predecessor_raw_sha256']==windows['current_raw_sha256']==link['Windows_fixture_raw_sha256']
    return windows,previous


def test_current_fixture_exact_predecessor_projection_and_adapter():
    c,raw=captured_input_predecessor((ROOT/'tests/test_join_segmentations_public.py').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==c['current_raw_sha256']
    text=raw.decode()
    for before,after in reversed(c['inverse_replacements']):
        assert text.count(after)==1
        text=text.replace(after,before)
    assert hashlib.sha256(text.encode()).hexdigest()==c['predecessor_raw_sha256']=='c113a3dafb68a90bccc772d44af21fd0a4ed3f638dff534487d107cee854003c'
    assert hashlib.sha256((ROOT/'tests/join_windows_fenv.py').read_bytes()).hexdigest()==c['adapter_raw_sha256']
    assert 'libc.feraiseexcept'not in raw.decode()

@pytest.mark.parametrize('tamper',('extra-byte','weakened-FENV-assertion'))
def test_captured_input_projection_refuses_unadmitted_public_bytes(tamper):
    raw=(ROOT/'tests/test_join_segmentations_public.py').read_bytes()
    if tamper=='extra-byte':bad=raw+b'\n#unadmitted'
    else:
        assertion=b'assert (bytes(before)==bytes(after)if linux else before==after)'
        assert raw.count(assertion)==1
        bad=raw.replace(assertion,b'assert True',1)
    with pytest.raises(AssertionError):captured_input_predecessor(bad)


def test_real_source_command_selects_live_export_controls_before_preparation():
    workflow=(ROOT/'.github/workflows/public-qualification.yml').read_text()
    line=next(s for s in workflow.splitlines()if 'python -m pytest'in s and 'qualification-tool-guards.log'in s)
    before,after=line.split('2>&1 |')
    assert shlex.split(before).count('tests/test_join_windows_fenv.py')==1
    assert shlex.split(after)==['tee','build/qualification-tool-guards.log']
    assert workflow.index(line)<workflow.index('Build and qualify one genuine whole product')

def test_adapter_requires_actual_opaque_capture_and_no_raiser():
    text=(ROOT/'tests/join_windows_fenv.py').read_text()
    assert 'fegetexceptflag'in text and 'fesetexceptflag'in text
    assert 'feraiseexcept'not in text and 'flags.value'not in text
    assert 'ctypes.sizeof(ctypes.c_ulong)==4'in text and '@lru_cache(maxsize=1)'in text

if sys.platform=='win32':
    from join_windows_fenv import WindowsFEnv,backend

    @pytest.mark.parametrize('rounding',[0,0x100,0x200,0x300])
    @pytest.mark.parametrize('kind,required',[('inexact',1),('overflow',5),('underflow',3),('invalid',16)])
    def test_actual_exported_flag_route_restores_real_flags_and_rounding(rounding,kind,required):
        api=backend();assert api is backend()
        outer,initial=api.capture();clear=api.library.feclearexcept
        clear.argtypes=[ctypes.c_int];clear.restype=ctypes.c_int
        try:
            assert clear(api.mask)==0 and api.fesetround(rounding)==0
            # Variable ordinary Python floating arithmetic seeds actual hardware
            # flags; opaque state is then captured by the exported CRT function.
            a,b={'inexact':(2.,3.),'overflow':(1e300,1e300),
                 'underflow':(1e-300,1e-300),'invalid':(float('inf'),float('inf'))}[kind]
            result=a/b if kind=='inexact'else a-b if kind=='invalid'else a*b
            saved,before=api.capture()
            assert before[0]&required==required and before[1]==rounding
            assert clear(api.mask)==0 and api.fesetround((rounding+0x100)&0x300)==0
            api.restore(saved,before)
            assert api.state()==before
        finally:
            api.restore(outer,initial)

    def test_actual_exception_cleanup_restores_caller_flags_and_rounding():
        api=backend();saved,before=api.capture()
        with pytest.raises(RuntimeError,match='CONTROLLED_TEST_BODY'):
            try:
                assert api.fesetround((before[1]+0x100)&0x300)==0
                raise RuntimeError('CONTROLLED_TEST_BODY')
            finally:api.restore(saved,before)
        assert api.state()==before

    def test_missing_export_is_refused_at_backend_entry():
        actual=backend().library
        class MissingFlagExport:
            def __getattr__(self,name):
                if name=='fegetexceptflag':raise AttributeError('MISSING_CONTROLLED_EXPORT')
                return getattr(actual,name)
        with pytest.raises(AttributeError,match='MISSING_CONTROLLED_EXPORT'):
            WindowsFEnv(MissingFlagExport())
