"""Actual standalone full-state controls, reused from admitted FEnv32 probe."""
import argparse
import builtins
import ctypes
import json
from pathlib import Path
import sys
import traceback
from util_views_fenv import ABI,FEnv,SOURCE_SHA256,ROOT,sha,configure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--library',type=Path,required=True);parser.add_argument('--library-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and sha(args.library)==args.library_sha256
    assert sha(ROOT/'tests/util_views_fenv_restore.c')==SOURCE_SHA256
    requests=[];old_import=builtins.__import__
    def refused(name,*a,**k):
        if name.split('.')[0]in('numpy','scipy','skimage','skimagexx'):requests.append(name);raise AssertionError('SCIENCE_TARGET_IMPORT_REFUSED')
        return old_import(name,*a,**k)
    builtins.__import__=refused
    libc,helper,glibc=configure(args.library)
    def env():
        value=FEnv();assert libc.fegetenv(ctypes.byref(value))==0;return value
    def restore(value):return helper.reference_fenv_restore(ctypes.byref(value),32)
    initial=env();assert initial.tags==0xffff and initial.control&0x3f==0x3f and initial.mxcsr&0x1f80==0x1f80
    rows=[];negative=[];failure=None
    try:
        for rounding in(0,0x400,0x800,0xc00):
            for sticky in(0,0x20,0x05,0x3d):
                for c2 in(0,0x400):
                    target=FEnv.from_buffer_copy(bytes(initial));target.control=(target.control&~0xc00)|rounding
                    target.status=(target.status&~(0x3f|0x400|0x80|0x8000))|sticky|c2
                    target.mxcsr=(target.mxcsr&~(0x3f|0x6000))|sticky|(rounding<<3)
                    assert restore(target)==0 and bytes(env())==bytes(target)
                    assert libc.fegetround()==rounding and libc.fetestexcept(0x3d)==sticky
                    assert libc.feclearexcept(0x3d)==0 and libc.feraiseexcept(0x20)==0 and libc.fesetround(rounding^0x400)==0
                    disturbed=env();assert bytes(disturbed)!=bytes(target)
                    assert restore(target)==0;actual=env();assert bytes(actual)==bytes(target)
                    rows.append(dict(rounding=rounding,sticky=sticky,C2=c2,expected=bytes(target).hex(),disturbed=bytes(disturbed).hex(),restored=bytes(actual).hex()))
        assert restore(initial)==0
        for kind in('null','length','nonempty-x87','unmasked-x87','unmasked-SSE','reserved-MXCSR'):
            bad=FEnv.from_buffer_copy(bytes(initial));before=env()
            if kind=='null':code=helper.reference_fenv_restore(None,32);wanted=-1
            elif kind=='length':code=helper.reference_fenv_restore(ctypes.byref(bad),31);wanted=-1
            else:
                if kind=='nonempty-x87':bad.tags=0xfffc;wanted=-2
                elif kind=='unmasked-x87':bad.control&=~1;wanted=-3
                elif kind=='unmasked-SSE':bad.mxcsr&=~0x80;wanted=-3
                else:bad.mxcsr|=0x10000;wanted=-4
                code=restore(bad)
            assert code==wanted and bytes(env())==bytes(before)
            negative.append(dict(kind=kind,returncode=code,exact_state_unchanged=True))
    except BaseException as error:failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        assert restore(initial)==0;final=env();assert bytes(final)==bytes(initial)
        builtins.__import__=old_import
    assert not requests and not any(n.split('.')[0]in('numpy','scipy','skimage','skimagexx')for n in sys.modules)
    record=dict(status='PASS_FULL32_CURRENT_EMITTED_HELPER'if failure is None else'FAILED_FULL32_CONTROL',library_sha256=args.library_sha256,source_sha256=SOURCE_SHA256,ABI=ABI,glibc=glibc,python=sys.version,executable=sys.executable,initial=bytes(initial).hex(),final=bytes(final).hex(),positive=rows,negative=negative,failure=failure,original_calls=0,scientific_imports=0)
    with args.output.open('x',encoding='utf-8',newline='\n')as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=record['status'],positive=len(rows),negative=len(negative),failure=failure)))
    return 1 if failure else 0


if __name__=='__main__':raise SystemExit(main())
