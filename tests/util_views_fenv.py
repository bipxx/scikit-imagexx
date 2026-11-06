"""Test-only Linux/glibc FEnv32 envelope; no product/scientific imports."""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'coverage/contracts/util-views-fenv-v1.json'
SOURCE_SHA256='97ff5740e8eb30c75b6e38729a6ada89ec47f8ac20fd0ceb172e06aeca9f2ae6'
LEGACY_SHA256='0c40af724c3cbca34fc6c6c027e62ddb1635609b7575a09505d578db24466220'
ABI=[32,0,4,8,12,20,28,8]
SOURCES=('tests/util_views_fenv_restore.c','tests/util_views_fenv.py','tests/util_views_fenv_build.py','tests/util_views_fenv_probe.py')


class FEnv(ctypes.Structure):
    _fields_=[('control',ctypes.c_ushort),('reserved1',ctypes.c_ushort),('status',ctypes.c_ushort),('reserved2',ctypes.c_ushort),('tags',ctypes.c_ushort),('reserved3',ctypes.c_ushort),('eip',ctypes.c_uint32),('cs',ctypes.c_ushort),('opcode',ctypes.c_ushort),('data',ctypes.c_uint32),('ds',ctypes.c_ushort),('reserved5',ctypes.c_ushort),('mxcsr',ctypes.c_uint32)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_public_source(raw):
    """Exact inverse for the existing56ec literal-hash source reader."""
    context=json.loads(CONTEXT.read_bytes());fragment=context['public_FP_inverse']
    assert raw.count(fragment['current'].encode())==1
    projected=raw.replace(fragment['current'].encode(),fragment['baseline'].encode(),1)
    assert hashlib.sha256(projected).hexdigest()==context['baseline_public_raw_sha256']
    return projected


def bundle_file(directory,name):
    path=Path(name)
    assert not path.is_absolute() and '..' not in path.parts,'FENV receipt path escapes bundle'
    out=(directory/path).resolve();assert out.is_relative_to(directory.resolve())
    return out


def validate_controls(record,library_hash):
    assert record['status']=='PASS_FULL32_CURRENT_EMITTED_HELPER' and record['library_sha256']==library_hash
    assert record['ABI']==ABI and record['original_calls']==record['scientific_imports']==0
    assert record['failure'] is None and record['initial']==record['final']
    assert len(record['positive'])==32 and len(record['negative'])==6
    assert {(r['rounding'],r['sticky'],r['C2'])for r in record['positive']}=={(r,s,c)for r in(0,0x400,0x800,0xc00)for s in(0,0x20,0x05,0x3d)for c in(0,0x400)}
    for row in record['positive']:
        assert row['expected']==row['restored'] and len(bytes.fromhex(row['restored']))==32
        value=FEnv.from_buffer_copy(bytes.fromhex(row['restored']))
        assert value.control&0xc00==row['rounding'] and value.status&0x3f==row['sticky'] and value.status&0x400==row['C2']
        assert value.mxcsr&0x3f==row['sticky'] and value.mxcsr&0x6000==row['rounding']<<3
    assert {(r['kind'],r['returncode'])for r in record['negative']}=={('null',-1),('length',-1),('nonempty-x87',-2),('unmasked-x87',-3),('unmasked-SSE',-3),('reserved-MXCSR',-4)}
    assert all(r['exact_state_unchanged']for r in record['negative'])


def validate_receipt(path,digest,*,toolchain=True):
    path=Path(path);assert path.is_file(),'FENV build receipt missing'
    assert len(digest)==64 and sha(path)==digest,'FENV build receipt digest mismatch'
    record=json.loads(path.read_bytes());directory=path.parent
    assert record['schema']=='util-views-fenv-build-v1' and record['status']=='PASS_SOURCE_BUILT_FENV32'
    assert record['ABI']==ABI and sha(CONTEXT)==record['source_context_sha256']
    assert set(record['source_pins'])==set(SOURCES)
    for name,pin in record['source_pins'].items():assert sha(ROOT/name)==pin,name
    assert record['source_pins'][SOURCES[0]]==SOURCE_SHA256
    for name,pin in record['files'].items():assert sha(bundle_file(directory,name))==pin,name
    assert 'libutil_views_fenv_restore.so'in record['files'] and 'build-command.json'in record['files']
    command=json.loads(bundle_file(directory,'build-command.json').read_bytes())
    assert command['source_pins']==record['source_pins'] and command['compiler']==record['compiler'] and command['headers']==record['headers']
    assert command['argv'][0]==record['compiler']['path'] and '-fno-fast-math'in command['argv']
    library_hash=record['files']['libutil_views_fenv_restore.so']
    assert record['controls'] and len(set(record['controls']))==len(record['controls'])
    for name in record['controls']:
        assert name in record['files'];validate_controls(json.loads(bundle_file(directory,name).read_bytes()),library_hash)
    if toolchain:
        compiler=record['compiler'];assert sha(compiler['path'])==compiler['sha256'] and sha(compiler['backend'])==compiler['backend_sha256']
        for header in record['headers']:assert sha(header['original'])==record['files'][header['captured']]
    assert any(h['original'].endswith('/bits/fenv.h')for h in record['headers'])
    return record


def configure(path):
    assert sys.platform=='linux' and platform.machine()=='x86_64' and sys.byteorder=='little','FEnv32 helper requires Linux x86_64 glibc; no Windows envelope admitted'
    assert ctypes.sizeof(FEnv)==32 and FEnv.mxcsr.offset==28 and ctypes.sizeof(ctypes.c_void_p)==8
    libc=ctypes.CDLL(None);helper=ctypes.CDLL(str(path))
    libc.gnu_get_libc_version.argtypes=[];libc.gnu_get_libc_version.restype=ctypes.c_char_p
    version=libc.gnu_get_libc_version().decode()
    for name in ('fegetenv','fesetenv'):
        getattr(libc,name).argtypes=[ctypes.c_void_p];getattr(libc,name).restype=ctypes.c_int
    for name in ('feclearexcept','feraiseexcept','fesetround','fetestexcept'):
        getattr(libc,name).argtypes=[ctypes.c_int];getattr(libc,name).restype=ctypes.c_int
    libc.fegetround.argtypes=[];libc.fegetround.restype=ctypes.c_int
    helper.reference_fenv_layout.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_size_t];helper.reference_fenv_layout.restype=ctypes.c_int
    helper.reference_fenv_restore.argtypes=[ctypes.c_void_p,ctypes.c_size_t];helper.reference_fenv_restore.restype=ctypes.c_int
    words=(ctypes.c_uint32*8)();assert helper.reference_fenv_layout(words,8)==0 and list(words)==ABI
    return libc,helper,version


@lru_cache(maxsize=2)
def source_backend(path,digest):
    record=validate_receipt(path,digest,toolchain=True)
    libc,helper,glibc=configure(bundle_file(Path(path).parent,'libutil_views_fenv_restore.so'))
    assert glibc==record['glibc']
    return libc,helper


def backend():
    mode=os.environ.get('UTIL_VIEWS_FENV_MODE','private-fixed-v1')
    if mode=='source-built-v1':
        assert not os.environ.get('UTIL_VIEWS_FENV_HELPER'),'legacy helper forbidden in genuine mode'
        path=os.environ.get('UTIL_VIEWS_FENV_RECEIPT');digest=os.environ.get('UTIL_VIEWS_FENV_RECEIPT_SHA256')
        assert path and digest,'source-built FENV receipt and digest required before target'
        # Recheck mutable artifact/source bytes at every envelope boundary;
        # expensive compiler identity is authenticated once per receipt/load.
        validate_receipt(path,digest,toolchain=False)
        return source_backend(str(Path(path).resolve()),digest)
    assert mode=='private-fixed-v1' and not os.environ.get('UTIL_VIEWS_FENV_RECEIPT'),'unknown/mixed FENV mode'
    path=Path(os.environ['UTIL_VIEWS_FENV_HELPER']);assert sha(path)==LEGACY_SHA256
    libc,helper,glibc=configure(path);assert glibc=='2.39'
    return libc,helper


def captured_call(call,expected):
    libc,helper=backend()  # Missing/tampered receipt fails before target entry.
    def run(*args,**options):
        saved=(ctypes.c_uint32*8)();assert libc.fegetenv(ctypes.byref(saved))==0
        try:
            assert libc.feclearexcept(0x3d)==0
            assert libc.feraiseexcept(expected['fenv_before']['flags'])==0
            assert libc.fesetround(expected['fenv_before']['rounding'])==0
            before=dict(flags=libc.fetestexcept(0x3d),rounding=libc.fegetround())
            try:return call(*args,**options)
            finally:
                after=dict(flags=libc.fetestexcept(0x3d),rounding=libc.fegetround())
                assert before==expected['fenv_before'] and after==expected['fenv_after']
        finally:
            assert helper.reference_fenv_restore(ctypes.byref(saved),32)==0
            actual=(ctypes.c_uint32*8)();assert libc.fegetenv(ctypes.byref(actual))==0 and bytes(actual)==bytes(saved)
    return run
