"""Prospective allocation-inclusive find_boundaries; one worker, no work at import."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import subprocess
import warnings
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'coverage/contracts/find-boundaries-benchmark-binding-v5.json'


def case_specs():return json.loads(CONTRACT.read_bytes())['cases']

def case_bounds(spec):
    assert spec['dtype']in('uint8','uint16','int32','bool')and spec['layout']in('C','F','reverse')
    assert spec['mode']in('thick','inner','outer','subpixel')and spec['pattern']in('cross','blocks','noncanonical-bool')
    shape=spec['shape'];assert 1<=len(shape)<=3 and all(type(n)is int and 2<=n<=517 for n in shape)
    assert type(spec['connectivity'])is int and 1<=spec['connectivity']<=len(shape)
    assert type(spec['background'])is int and spec['background']==0
    assert type(spec['readonly'])is bool
    pixels=math.prod(shape);output=math.prod([2*n-1 for n in shape]if spec['mode']=='subpixel'else shape)
    work=(output if spec['mode']=='subpixel'else pixels*(4 if spec['mode']=='outer'else 2))*3**len(shape)
    assert pixels<=262144 and output<=1000000 and work<=16000000
    if spec['pattern']=='cross':assert shape==[3,3]and spec['dtype']=='uint8'
    if spec['pattern']=='noncanonical-bool':assert spec['dtype']=='bool'and spec['layout']=='reverse'
    return dict(input_elements=pixels,output_elements=output,source_neighbor_work_bound=work)

def fixture(spec,np):
    bounds=case_bounds(spec)
    if spec['pattern']=='cross':image=np.array([0,0,0,0,1,2,0,2,2],dtype=np.uint8).reshape(3,3)
    else:
        values=(np.arange(bounds['input_elements'],dtype=np.uint32)//7)%4
        if spec['pattern']=='noncanonical-bool':
            values=np.where(values%3==0,0,np.where(values%3==1,2,255)).astype(np.uint8)
            image=values.reshape(spec['shape']).view(np.bool_)
        else:image=values.astype(spec['dtype']).reshape(spec['shape'])
    if spec['layout']=='F':image=np.asfortranarray(image)
    elif spec['layout']=='reverse':image=image[...,::-1]
    if spec['readonly']:image.flags.writeable=False
    return image,dict(connectivity=spec['connectivity'],mode=spec['mode'],background=spec['background'])

def snapshot(array):
    return dict(type=[type(array).__module__,type(array).__qualname__],dtype=array.dtype.str,
        metadata=None if array.dtype.metadata is None else dict(array.dtype.metadata),shape=list(array.shape),strides=list(array.strides),
        own=bool(array.flags.owndata),writeable=bool(array.flags.writeable),aligned=bool(array.flags.aligned),
        c=bool(array.flags.c_contiguous),f=bool(array.flags.f_contiguous),base_type=type(array.base).__name__,
        raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())

def load_root_binding(path, raw_contract):
    """Use the existing external first-cohort envelope without altering recipes."""
    assert path is not None, 'PROSPECTIVE: genuine input/root binding missing; no target imports or calls permitted'
    raw_binding=Path(path).read_bytes();binding=json.loads(raw_binding)
    allowed={'status','source_root','source_revision','git_argv','genuine_root_admission_path','genuine_root_admission_sha256','product_kind','interpreter','source_files_sha256','installed_site','numpy','core_path','core_sha256','output_root','routes','prepared_source','producer','runtime_path','runtime_sha256','shared_CMake_sha256','genuine_root_stager_path','genuine_root_stager_sha256','base_contract_sha256','root_go_path'}
    assert not set(binding)-allowed
    assert binding['status']=='ROOT_AUTHORIZED_GENUINE_FIRST_COHORTS' and binding['product_kind']=='GENUINE_INSTALLED_WHOLE_PRODUCT'
    assert binding['base_contract_sha256']==hashlib.sha256(raw_contract).hexdigest()
    assert Path(binding['source_root']).resolve()==ROOT
    prepared=binding['prepared_source'];producer=binding['producer']
    assert set(producer)=={'run','revision','artifact','sha256'}
    assert set(prepared)=={'revision','files_sha256'} and prepared['files_sha256']
    revision=prepared['revision'];assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
    assert revision==binding['source_revision']==producer['revision']
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==revision
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    def authenticated(field):
        raw=Path(binding[field+'_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==binding[field+'_sha256']
        return json.loads(raw)
    runtime=authenticated('runtime')
    assert runtime['source']==prepared
    assert runtime['runtime']['packages_sha256']['skimagexx'][Path(binding['core_path']).name]==binding['core_sha256']
    authenticated('genuine_root_admission')
    stager=authenticated('genuine_root_stager')
    assert str(stager['run'])==str(producer['run']) and str(stager['artifact'])==str(producer['artifact'])
    assert stager['revision']==revision and stager['artifact_sha256']==producer['sha256']
    assert len(producer['sha256'])==64 and all(c in '0123456789abcdef' for c in producer['sha256'])
    go=json.loads(Path(binding['root_go_path']).read_bytes())
    assert go['binding_sha256']==hashlib.sha256(raw_binding).hexdigest() and 'boundary' in go['routes']
    assert go['timed_batches']==0 and datetime.now(timezone.utc)<datetime.fromisoformat(go['deadline_utc'])
    cmake=(ROOT/'CMakeLists.txt').read_bytes().replace(b'\r\n',b'\n')
    assert binding['shared_CMake_sha256']==prepared['files_sha256']['CMakeLists.txt']==hashlib.sha256(cmake).hexdigest()
    contract=json.loads(raw_contract)
    # Root provenance supplies only these five runtime fields. All recipes,
    # sampler gates, method and source pins remain the tracked contract's values.
    contract.update(state='FROZEN_READY_FOR_ROOT_FIRST_COHORT',shared_CMake_sha256=binding['shared_CMake_sha256'],genuine_core_sha256=binding['core_sha256'],prepared_run=producer['run'],prepared_artifact=producer['artifact'])
    return contract,binding,hashlib.sha256(raw_binding).hexdigest()


def f_row_predecessor(name, raw):
    context_path=ROOT/'coverage/contracts/find-boundaries-f-row-source-context-v1.json'
    assert hashlib.sha256(context_path.read_bytes()).hexdigest()=='f12070193ebf7ca993f4ca04fcb5f8563d7a297a8e029386770d7ea4a7d73b29'
    context=json.loads(context_path.read_bytes());item=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==item['current_canonical_lf_sha256'],name
    for fragment in item['inverse_fragments']:
        current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1;raw=raw.replace(current,previous,1)
    assert hashlib.sha256(raw).hexdigest()==item['predecessor_canonical_lf_sha256'],name
    return raw


def layout_predecessor(name, raw):
    """Authenticate only the two independently admitted component changes."""
    raw=f_row_predecessor(name,raw)
    context_path=ROOT/'coverage/contracts/find-boundaries-layout-source-context-v1.json'
    assert hashlib.sha256(context_path.read_bytes()).hexdigest()=='63b64b7ff3b802d0433abc08d03a1ea09bf0e50f6d1391a477b9510354fe454d'
    context=json.loads(context_path.read_bytes());item=context['sources'][name]
    assert hashlib.sha256(raw).hexdigest()==item['current_sha256']
    component=context['component_contract'];component_raw=(ROOT/component['path']).read_bytes()
    assert hashlib.sha256(component_raw).hexdigest()==component['sha256']
    for fragment in item['inverse']:
        current,previous=fragment['current'].encode(),fragment['predecessor'].encode()
        assert current and raw.count(current)==1;raw=raw.replace(current,previous,1)
    assert hashlib.sha256(raw).hexdigest()==item['predecessor_sha256']
    return raw


def validate_source_closure(contract):
    successor=contract['f_row_successor'];raw=(ROOT/successor['contract']).read_bytes()
    assert set(successor)=={'contract','sha256','context','context_sha256','added_source_pins','driver_inverse','scope'}
    assert successor['context']=='coverage/contracts/find-boundaries-f-row-source-context-v1.json' and successor['context_sha256']=='f12070193ebf7ca993f4ca04fcb5f8563d7a297a8e029386770d7ea4a7d73b29'
    assert successor['contract']=='coverage/contracts/find-boundaries-benchmark-binding-v4.json'
    assert hashlib.sha256(raw).hexdigest()==successor['sha256']=='8ad09d6c73f47d017a5da741d60e18049abe6425042725e29c6e233eab2e8ff7'
    previous=json.loads(raw)
    provenance={'state','shared_CMake_sha256','genuine_core_sha256','prepared_run','prepared_artifact'}
    excluded=provenance|{'source_closure','f_row_successor'}
    assert {k:v for k,v in contract.items()if k not in excluded}=={k:v for k,v in previous.items()if k not in excluded}
    assert set(contract['source_closure'])==set(previous['source_closure'])|set(successor['added_source_pins'])
    for name,pin in contract['source_closure'].items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(raw).hexdigest()==pin,name
        if name in ('src/find_boundaries.cpp','tests/cmake_consumer/find_boundaries_consumer.hpp','tests/test_find_boundaries_layout_contract.py'):
            raw=f_row_predecessor(name,raw)
        elif name=='benchmarks/find_boundaries.py':
            for fragment in successor['driver_inverse']:
                current,predecessor=fragment['current'].encode(),fragment['predecessor'].encode()
                assert current and raw.count(current)==1;raw=raw.replace(current,predecessor,1)
        elif name in successor['added_source_pins']:
            assert pin==successor['added_source_pins'][name];continue
        else:assert pin==previous['source_closure'][name]
        assert hashlib.sha256(raw).hexdigest()==previous['source_closure'][name],name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,choices=[1],default=1);parser.add_argument('--repeat',type=int,choices=[7],default=7)
    parser.add_argument('--deadline-utc');parser.add_argument('--output',type=Path,default=Path('build/ci-find-boundaries.json'))
    parser.add_argument('--root-binding',type=Path)
    args=parser.parse_args();raw_contract=CONTRACT.read_bytes();contract,binding,binding_sha=load_root_binding(args.root_binding,raw_contract)
    assert contract['state']=='FROZEN_READY_FOR_ROOT_FIRST_COHORT','PROSPECTIVE: genuine input/root binding missing; no target imports or calls permitted'
    assert len(contract['cases'])==8 and all(case_bounds(s)for s in contract['cases'])
    validate_source_closure(contract)
    assert contract['shared_CMake_sha256']and hashlib.sha256((ROOT/'CMakeLists.txt').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['shared_CMake_sha256']
    assert contract['genuine_core_sha256']and contract['prepared_run']and contract['prepared_artifact']
    deadline=args.deadline_utc or(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat();assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    journal=args.output.with_suffix('.batches.jsonl');assert not args.output.exists()and not journal.exists()
    sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'benchmarks')]
    from montage_sustained import measure
    from provenance import source_fingerprint
    import numpy as np,scipy,skimage,skimagexx as xx
    from skimagexx import _compat
    original=importlib.import_module('skimage.segmentation.boundaries');public=importlib.import_module('skimage.segmentation')
    owned=importlib.import_module('skimagexx._find_boundaries');engine=importlib.import_module('scipy.ndimage._nd_image')
    upstream=public.find_boundaries;native=xx.segmentation.find_boundaries;entry=xx._core.find_boundaries_into
    assert native is owned.find_boundaries and upstream is original.find_boundaries
    assert str(inspect.signature(native))==str(inspect.signature(upstream))==contract['public_signature']
    assert inspect.isbuiltin(entry)and entry.__module__==xx._core.__name__=='skimagexx._core'
    core=Path(xx._core.__file__);core_sha=hashlib.sha256(core.read_bytes()).hexdigest();assert core_sha==contract['genuine_core_sha256'] and core.resolve()==Path(binding['core_path']).resolve()
    assert skimage.__version__=='0.26.0'
    assert hashlib.sha256(Path(original.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==contract['original_source_sha256']
    dependency=contract['runtime_dependencies'][np.__version__];assert scipy.__version__==dependency['scipy']
    assert hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()==dependency['ndimage_binary_sha256']
    assert hashlib.sha256(inspect.getsource(measure).encode()).hexdigest()==contract['method_sha256']
    limits={name:os.environ[name]for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')}
    assert set(limits.values())=={'1'}and xx.get_num_threads()==1;xx.set_fallback_enabled(False)
    cpu_model=platform.processor();info=Path('/proc/cpuinfo')
    if info.is_file():cpu_model=next((line.partition(':')[2].strip()for line in info.read_text().splitlines()if line.startswith('model name')),cpu_model)
    cpu_count,avx2_available=os.cpu_count(),bool(xx._core.has_avx2())
    before=source_fingerprint(ROOT);policy,handler,buffer=np.geterr(),np.geterrcall(),np.getbufsize();rows=[]
    denied_codes=set()
    for function in(original.find_boundaries,original._find_boundaries_subpixel,original.dilation,original.erosion,original.ndi.generate_binary_structure,original.ndi.minimum_filter,original.ndi.maximum_filter,original.ndi.grey_erosion,original.ndi.grey_dilation):
        while True:
            if hasattr(function,'__code__'):denied_codes.add(function.__code__)
            if not hasattr(function,'__wrapped__'):break
            function=function.__wrapped__
    for index,spec in enumerate(contract['cases']):
        image,options=fixture(spec,np);frozen=snapshot(image);address=image.ctypes.data
        with warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always');expected=upstream(image,**options)
        assert not notices and snapshot(image)==frozen and image.ctypes.data==address
        calls=[];accepted=[];previous=sys.getprofile()
        def profile(frame,event,value):
            if event=='call'and frame.f_code in denied_codes:raise AssertionError('Native find_boundaries entered original/helper code')
            if event=='c_return'and value is entry:calls.append('find_boundaries_into')
            if event=='return'and frame.f_code is owned.find_boundaries.__code__:accepted.append(frame.f_locals.get('native_result')is True)
        try:
            sys.setprofile(profile)
            with warnings.catch_warnings(record=True)as notices:
                warnings.simplefilter('always');actual=native(image,**options)
        finally:sys.setprofile(previous)
        assert not notices and calls==['find_boundaries_into']and accepted==[True]
        assert type(actual)is type(expected)is np.ndarray and actual.dtype==expected.dtype==np.dtype(bool)
        assert actual is not expected and actual is not image and not np.shares_memory(actual,expected)and not np.shares_memory(actual,image)
        assert actual.flags.owndata and actual.base is None and snapshot(actual)==snapshot(expected)
        assert actual.size==case_bounds(spec)['output_elements']and snapshot(image)==frozen and image.ctypes.data==address
        assert xx.get_num_threads()==1 and not _compat._fallback_enabled
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer
        # Allocation and destruction of each public result remain inside the
        # unchanged sustained sampler. All guards/profile instrumentation ended.
        timing=measure(lambda:upstream(image,**options),lambda:native(image,**options),args.repeat,100,
            journal=journal,case_index=index,case_id=spec['id'],pass_index=1,deadline=deadline)
        assert snapshot(image)==frozen and image.ctypes.data==address and xx.get_num_threads()==1
        assert np.geterr()==policy and np.geterrcall()is handler and np.getbufsize()==buffer and not _compat._fallback_enabled
        rows.append(dict(case_id=spec['id'],api='segmentation.find_boundaries',parameters=spec,backend='native',threads=1,
            actual_C_returns=calls,native_accepted=True,input=frozen,output=snapshot(actual),max_abs_error=0.,**timing))
        print(f"{spec['id']}: {timing['speedup']:.9f}x",flush=True)
    assert sum(sum(len(sequence)for sequence in row['samples_ms'])for row in rows)==112
    assert source_fingerprint(ROOT)==before and hashlib.sha256(core.read_bytes()).hexdigest()==core_sha and CONTRACT.read_bytes()==raw_contract and hashlib.sha256(args.root_binding.read_bytes()).hexdigest()==binding_sha
    report=dict(metadata=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,reference=skimage.__version__,cpu_model=cpu_model,logical_cpu_count=cpu_count,
        avx2_available=avx2_available,**before,core_sha256=core_sha,threads=1,repeat=7,raw_samples=112,target_ms=100,
        root_binding_sha256=binding_sha,prepared_run=contract['prepared_run'],prepared_artifact=contract['prepared_artifact'],
        warmup_calibration_min_ms=250,deadline_utc=deadline,reference_pool_limits=limits,contract_sha256=hashlib.sha256(raw_contract).hexdigest(),
        method_sha256=contract['method_sha256'],harness_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),scope=contract['scope']),results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
