"""Prospective allocation-inclusive public crop workloads; no work at import."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/crop-benchmark-v1.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']


def source_safe(spec):
    assert spec['dtype']in('uint8','int16','float32','float64','object')and spec['layout']in('C','F','reverse')
    assert 1<=len(spec['shape'])<=3 and all(type(n)is int and 3<=n<=513 for n in spec['shape'])
    assert math.prod(spec['shape'])<=263000 and type(spec['width'])is int and 0<=spec['width']<=2
    assert all(n>2*spec['width']for n in spec['shape'])and type(spec['copy'])is bool
    assert spec['payload_provider']==(spec['dtype']=='object')
    if spec['dtype']=='object':assert math.prod(spec['shape'])<=272
    return dict(input_elements=math.prod(spec['shape']),max_bytes=math.prod(spec['shape'])*8)


def fixture(spec,np):
    count=source_safe(spec)['input_elements']
    if spec['dtype']=='object':a=np.array(['crop-payload-'+str(i)for i in range(count)],dtype=object)
    else:a=np.arange(count,dtype=np.int64).astype(spec['dtype'])
    a=a.reshape(spec['shape'])
    if spec['layout']=='F':a=np.asfortranarray(a)
    elif spec['layout']=='reverse':a=a[...,::-1]
    return a,dict(copy=spec['copy'],order=spec['order'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-binding',type=Path);parser.add_argument('--output',type=Path,default=Path('build/crop-performance.json'))
    args=parser.parse_args();raw=CONTRACT.read_bytes();contract=json.loads(raw)
    assert args.root_binding is not None,'PROSPECTIVE: genuine product/root measurement binding required; no target imports/calls'
    binding=json.loads(args.root_binding.read_bytes())
    assert set(binding)=={'status','source_revision','core_sha256','prepared_run','prepared_artifact','contract_sha256','root_admission_path','root_admission_sha256','deadline_utc'}
    assert binding['status']=='ROOT_AUTHORIZED_CROP_PUBLIC_MEASUREMENT'and binding['contract_sha256']==hashlib.sha256(raw).hexdigest()
    assert binding['prepared_run']and binding['prepared_artifact']and len(binding['source_revision'])==40
    admission_raw=Path(binding['root_admission_path']).read_bytes();assert hashlib.sha256(admission_raw).hexdigest()==binding['root_admission_sha256']
    admission=json.loads(admission_raw);assert admission['source']==binding['source_revision']and admission['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(binding['deadline_utc'])
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==binding['source_revision']
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    assert not args.output.exists()and not args.output.with_suffix('.batches.jsonl').exists()
    assert len(contract['cases'])==8 and all(source_safe(s)for s in contract['cases'])
    for name,pin in contract['source_pins'].items():assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==pin
    sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np,skimagexx as xx
    upstream=importlib.import_module('skimage.util.arraycrop');owned=importlib.import_module('skimagexx._crop')
    assert xx.util.crop is owned.crop and inspect.isbuiltin(xx._core.crop_view)and inspect.isbuiltin(xx._core.crop_copy_into)
    assert xx.get_num_threads()==1 and all(os.environ[n]=='1'for n in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'))
    assert hashlib.sha256(Path(xx._core.__file__).read_bytes()).hexdigest()==binding['core_sha256']
    assert hashlib.sha256(Path(upstream.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='558df9e59105586d0b0e374a33836bd447919e9e5d91f21586127b7cafabfbac'
    before=source_fingerprint(ROOT)
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    rows=[]
    for index,spec in enumerate(contract['cases']):
        a,options=fixture(spec,np);frozen=a.copy();reference=lambda:upstream.crop(a,spec['width'],**options);native=lambda:xx.util.crop(a,spec['width'],**options)
        expected=reference();calls=[];previous=sys.getprofile()
        def guard(frame,event,value):
            if event=='call'and frame.f_code is upstream.crop.__code__:raise AssertionError('candidate entered original crop')
            if event=='c_return'and value in(xx._core.crop_view,xx._core.crop_copy_into):calls.append(value.__name__)
        try:sys.setprofile(guard);actual=native()
        finally:sys.setprofile(previous)
        assert calls==(['crop_view','crop_copy_into']if spec['copy']and not spec['payload_provider']else['crop_view'])
        assert type(actual)is type(expected)is np.ndarray and actual.dtype==expected.dtype and actual.shape==expected.shape and actual.strides==expected.strides
        assert np.array_equal(actual,expected)and np.array_equal(a,frozen)
        assert bool(np.shares_memory(actual,a))==bool(np.shares_memory(expected,a))
        rows.append(dict(id=spec['id'],payload_provider=spec['payload_provider'],**measure(reference,native,7,100,journal=args.output.with_suffix('.batches.jsonl'),case_index=index,case_id=spec['id'],pass_index=0,deadline=binding['deadline_utc'])))
    assert source_fingerprint(ROOT)==before and CONTRACT.read_bytes()==raw
    args.output.write_text(json.dumps(dict(contract_sha256=hashlib.sha256(raw).hexdigest(),binding=binding,threads=1,retain_all_losses=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
