"""Original-only NEW HOST SkylakeX characterization; never native admission.

No local numerical execution, native compilation, timer, CPU override or new
environment. The genuine failed product supplies import context only. Its
unchanged certificate may refuse this CPU; that refusal is preserved, not fixed.
"""
from __future__ import annotations
import argparse,ast,ctypes,gzip,hashlib,importlib,importlib.util,inspect,io,json,os,struct,subprocess,sys,traceback,warnings,zipfile
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=('tools/matrix_transform_skylakex_reference.py','tests/test_matrix_transform_skylakex_reference.py','.github/workflows/matrix-transform-skylakex-reference.yml')
PROBE_REVISION='d61b390d20d3d052e34a8282f962a42ed98d3bd4'
PROBE_SHA='c0be3406351a96d2324c07ed8292fa3ce0ae919831e8cdcc5f2f40d3953911f6'
COHORT_SHA='466d7645d841dfbbdbaf9c880d72a73f458af528299e2962493ce63e32cc2eb0'
MATRIX_SHA='6622a08fb4b420588d22325b3dc72fd053a7e4d689a4ed7ef88fe10fbc44e0f2'
BLAS_SHA='ed4f167a5330424524f45258e7ca2c8d7f03c0cba1f30ee4985553ac1f88ecf8'
UMATH_SHA='f37332834048e3aef93a33ff53f558a319cad12009defb2231ee1ba4aa6105d8'
TABLE_RVA=0x28216fd00-0x280e70000
FIELDS=('P','Q','R','unroll_m','unroll_n','unroll_mn')
MAX_ROWS=131072
MAX_ARRAY_BYTES=32*1024*1024
MAX_RETAINED=256*1024*1024
MAX_BUFFER_RETAINED=192*1024*1024 # reserve64MiB for census/journal/raw metadata
BOUNDARY_SCHEMA='matrix-skylakex-boundary-census-v2-fma'
FORBIDDEN=('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES','MKL_DEBUG_CPU_TYPE','MKL_SERVICE_FORCE_INTEL')


def validate_environment(environ):
    require(not any(k in environ for k in FORBIDDEN),'CPU forcing present, never removed')
    require(not any(environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','MATRIX_TRANSFORM_PRIVATE_BACKEND')),'interposed/private runtime refused')
    require(not any(k.startswith('SKIMAGEXX_') and k!='SKIMAGEXX_DISABLE_SIMD' for k in environ),'candidate/private selector refused')
    require(all(environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')),'thread pools must be1 before imports')


def require(ok,message):
    if not ok:raise ValueError(message)
def H(raw):return hashlib.sha256(raw).hexdigest()
def canonical(raw):return raw.replace(b'\r\n',b'\n')
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def git(*args):return subprocess.check_output(['git','-c','core.autocrlf=true','-C',str(ROOT),*args])
def probe_module():
    path=ROOT/'tools/matrix_runtime_certificate_probe.py'
    require(H(canonical(path.read_bytes()))==PROBE_SHA,'reviewed probe prerequisite differs')
    spec=importlib.util.spec_from_file_location('skylakex_reference_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def source_inputs(cpp,matrix):
    require(H(canonical(cpp))==COHORT_SHA,'old4000 source differs')
    require(H(matrix)==MATRIX_SHA,'old374 matrix observations differ')
    spec=json.loads(canonical(cpp).decode().split('// MATRIX_COHORTS_BEGIN\n/*\n')[1].split('\n*/')[0])
    require(spec['schema']=='matrix-general-finite-cohorts-v2' and spec['include_literal_base'] is True,'wrong cohort schema')
    require(len(spec['cases'])*(len(spec['rows'])+1)*len(spec['layouts'])**2==4000,'incomplete4000')
    return spec,json.loads(matrix)
def validate_fields(fields):
    require(set(fields)==set(FIELDS) and all(type(v) is int for v in fields.values()),'invalid live table schema')
    require(5<=fields['P']<=4096 and 5<=fields['Q']<=4096 and 1<=fields['R']<=65536,'unsupported live driver bounds')
    require(fields['unroll_m']==16 and fields['unroll_n']==2 and fields['unroll_mn']==32,'unassessed live kernel unroll')
def boundary_sizes(fields,k):
    validate_fields(fields);require(k in (3,4,5),'bounded K3..5 only')
    values=set(range(1,27))|{31,32,33,35,36,37,47,48,49,63,64,65}
    for center in (fields['P'],2*fields['P']):
        values.update(center+d for d in (-17,-16,-15,-8,-7,-1,0,1,7,8,15,16,17))
    for center in (fields['R'],2*fields['R']):
        values.update(center+d for d in (-13,-12,-11,-1,0,1,11,12,13))
    permit=1_000_000//(k*k);values.update((permit-1,permit,permit+1))
    values={n for n in values if n>0}
    require(max(values)<=MAX_ROWS,'complete boundary set exceeds declared workspace; no case filtering')
    return sorted(values)
def adversarial_cases():
    """Fixed binary data from association formulas, never observed/fitted words."""
    half=float.fromhex('0x1p-53');rows=[]
    for d in (2,3,4):
        patterns=[('adjacent-cancellation',[1.,-1.,half]+[half]*(d-2)),
                  ('separated-cancellation',[1.,half,-1.]+[half]*(d-2)),
                  ('signed-zero',[-0.,0.,-0.]+[0.]*(d-2))]
        for name,terms in patterns:
            matrix=[[0.]*(d+1) for _ in range(d+1)]
            for j in range(d):matrix[j]=terms[j:]+terms[:j]
            matrix[-1][-1]=1.
            rows.append(dict(id=f'boundary-D{d}-{name}',coords=[[1.]*d],matrix=matrix))
        matrix=[[float(i==j) for j in range(d+1)] for i in range(d+1)]
        # Denominator-specific lanes: a nonzero constant prevents unsafe poles.
        matrix[-1]=[1.,half,-1.]+[half]*(d-3)+[1.] if d>=3 else [half,-half,1.]
        rows.append(dict(id=f'boundary-D{d}-denominator-lanes',coords=[[1.]*d],matrix=matrix))
    # Preserve all original12 definitions above, then add independent product
    # rounding witnesses. These constants come from the exact Fraction proof,
    # never from observations or a fitted arithmetic candidate.
    a=float.fromhex('0x1.0000002000000p+0')
    b=float.fromhex('0x1.ffffffc000000p-1')
    for d in (2,3,4):
        matrix=[[float(i==j) for j in range(d+1)] for i in range(d+1)]
        matrix[0]=[-1.,b]+[0.]*(d-1)
        rows.append(dict(id=f'boundary-D{d}-fused-product',coords=[[1.,a]+[0.]*(d-2)],matrix=matrix))
    return rows
def fma_fraction_proof():
    """Exact scalar metadata only; does not execute floating multiplication."""
    a=Fraction.from_float(float.fromhex('0x1.0000002000000p+0'))
    b=Fraction.from_float(float.fromhex('0x1.ffffffc000000p-1'))
    product=a*b;previous=Fraction(1)-Fraction(1,2**53)
    require(a==1+Fraction(1,2**27) and b==1-Fraction(1,2**27),'binary witness literals differ')
    require(product==(previous+1)/2 and product-1==Fraction(-1,2**54),'exact FMA witness derivation differs')
    lower_significand=2**53-1;upper_significand=2**52
    require(lower_significand%2==1 and upper_significand%2==0,'nearest-even endpoint metadata differs')
    return dict(a_hex='0x1.0000002000000p+0',b_hex='0x1.ffffffc000000p-1',first_accumulated_term=-1,exact_product_numerator=product.numerator,exact_product_denominator=product.denominator,lower_endpoint_significand=lower_significand,upper_endpoint_significand=upper_significand,exact_fused_residual_numerator=-1,exact_fused_residual_denominator=2**54,separate_product_rounding='halfway between previous binary64 and1; ties-to-even selects1',separate_product_then_add_result=0,scope='Source-derived conditional association witness, NOT an observed original/native result.')
def case_specs(spec,fields):
    for case in spec['cases']:
        for n in [0]+spec['rows']:
            for pl in spec['layouts']:
                for ml in spec['layouts']:yield dict(family='legacy4000',case=case,n=n,pl=pl,ml=ml,extension=spec['row_extension'])
    for index,case in enumerate(adversarial_cases()):
        for n in boundary_sizes(fields,len(case['coords'][0])+1):
            for pl in spec['layouts']:
                for ml in spec['layouts']:yield dict(family='boundary-v1' if index<12 else 'boundary-fma-v2',case=case,n=n,pl=pl,ml=ml,extension=0.)
def layout(np,a,name):
    if name=='C':return np.array(a,order='C')
    if name=='F':return np.array(a,order='F')
    if name=='reverse':return a[::-1].copy()[::-1]
    if name=='step':
        b=np.zeros((a.shape[0]*2,a.shape[1]*2));b[::2,::2]=a;return b[::2,::2]
    require(name=='unaligned','unknown fixed layout')
    b=np.zeros(a.nbytes+1,np.uint8);c=np.ndarray(a.shape,dtype='f8',buffer=b,offset=1);c[:]=a;return c
def source_matmul_node(source):
    tree=ast.parse(source);function=tree.body[0]
    require(isinstance(function,ast.FunctionDef) and function.name=='_apply_homogeneous','wrong source helper')
    nodes=[n for n in function.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.MatMult)]
    require(len(nodes)==1 and ast.unparse(nodes[0])=='new_points_h = points_h @ matrix.T','original @ intake differs')
    return nodes[0]
def capture_site(helper,source):
    """Compute the authenticated post-@/pre-divs site ONCE, before fixtures."""
    require(inspect.getsource(helper)==source,'capture helper source differs')
    matmul=source_matmul_node(source);body=ast.parse(source).body[0].body
    position=next(i for i,n in enumerate(body) if n.lineno==matmul.lineno)
    following=body[position+1]
    require(ast.unparse(following)=='divs = new_points_h[:, -1]','source normalization boundary differs')
    code=helper.__code__;path=Path(code.co_filename).resolve();raw=path.read_bytes()
    provenance=dict(expected_event='line',expected_line=code.co_firstlineno+following.lineno-1,matmul_line=code.co_firstlineno+matmul.lineno-1,code_firstlineno=code.co_firstlineno,code_name=code.co_name,code_filename=str(path),source_module_sha256=H(raw),helper_source_sha256=H(source.encode()),public_co_code_sha256=H(code.co_code),co_linetable_sha256=H(code.co_linetable),scope='Public code-object metadata and authenticated source location, not a native ISA claim.')
    return dict(code=code,provenance=provenance)
def capture_original(function,helper,points,matrix,site):
    """Trace the unchanged ORIGINAL call, with no replacement prefix/math."""
    require(sys.gettrace() is None,'existing Python tracer not replaced')
    captured={};observed={};code=helper.__code__
    require(code is site['code'],'original helper code changed after site authentication')
    def trace(frame,event,arg):
        if frame.f_code is code and event=='line' and 'new_points_h' in frame.f_locals and not captured:
            require(frame.f_lineno==site['provenance']['expected_line'] and frame.f_trace_lines,'first post-@ line differs; no late/return capture')
            require('divs' not in frame.f_locals,'normalization started before source capture')
            captured.update({name:frame.f_locals[name] for name in ('matrix','points','points_h','new_points_h')})
            observed.update(site['provenance'],observed_event=event,observed_line=frame.f_lineno,f_trace_lines=frame.f_trace_lines,divs_bound_at_capture=False)
        return trace
    sys.settrace(trace)
    try:result=function(points,matrix)
    finally:sys.settrace(None)
    require(set(captured)=={'matrix','points','points_h','new_points_h'},'exact post-@ line event missing; return fallback forbidden')
    require(helper.__code__ is code,'original source replaced')
    return result,captured,observed
def points_for_case(np,case,n,extension):
    if n==0:return np.array(case['coords'],dtype=np.float64)
    return np.array([np.array(case['coords'][i%len(case['coords'])],dtype=np.float64)+i*extension for i in range(n)],dtype=np.float64)


class LastBasePoints:
    """One immutable fixture base; original calls receive fresh layout copies."""
    def __init__(self):
        self.case=None;self.n=None;self.extension=None;self.base=None
    def get(self,np,case,n,extension):
        if self.case is not case or self.n!=n or self.extension!=extension:
            base=points_for_case(np,case,n,extension)
            base.setflags(write=False)
            self.case=case;self.n=n;self.extension=extension;self.base=base
        return self.base


def readable_image_span(base,image_size,address,size,query):
    """Pure admission metadata for a prospective read; no pointer dereference."""
    require(type(base) is int and type(address) is int and type(size) is int and size>0,'invalid read request')
    require(base<=address and address+size<=base+image_size,'table outside authenticated image')
    require(query['allocation_base']==base and query['state']==0x1000,'table not committed in actual owner image')
    require(query.get('type')==0x1000000 and query['protect']&0xff in (2,4,8,0x20,0x40,0x80),'table is not readable image memory')
    require(not query['protect']&(0x100|0x01) and query['region_base']<=address and address+size<=query['region_base']+query['region_size'],'table page unreadable/guarded or span crosses region')
def runtime_scope(version,digest,corename,threads,umath_digest):
    require(version=='2.5.4' and digest==BLAS_SHA and corename==b'SkylakeX' and type(threads) is int and threads==1,'wrong actual default runtime; no forcing/substitute')
    require(umath_digest==UMATH_SHA,'loaded UMATH bytes changed')
def live_fields(wrapper,p,np):
    """Read six bounded i32 fields in the already loaded, hash-authenticated DLL."""
    certificate=wrapper._authenticate_runtime();require(certificate is not None,'actual loaded metadata authentication failed')
    version,digest,core,threads,library,identities=certificate
    runtime_scope(version,digest,core(),threads(),H(Path(identities[0][0]).read_bytes()))
    raw=Path(identities[1][0]).read_bytes();require(H(raw)==BLAS_SHA,'loaded BLAS bytes changed')
    require(H(Path(identities[0][0]).read_bytes())==UMATH_SHA,'loaded UMATH bytes changed')
    pe=struct.unpack_from('<I',raw,0x3c)[0];opt=pe+24;image_size=struct.unpack_from('<I',raw,opt+56)[0]
    section_count=struct.unpack_from('<H',raw,pe+6)[0];opt_size=struct.unpack_from('<H',raw,pe+20)[0]
    rva=TABLE_RVA+0x2f0
    sections=[]
    for i in range(section_count):
        pos=opt+opt_size+40*i;name=raw[pos:pos+8].split(b'\0')[0].decode('ascii');vs,va=struct.unpack_from('<II',raw,pos+8)
        if va<=rva and rva+24<=va+vs:sections.append(name)
    require(sections==['.data'],'audited table span not in expected image data section')
    class MBI(ctypes.Structure):
        _fields_=[('BaseAddress',ctypes.c_void_p),('AllocationBase',ctypes.c_void_p),('AllocationProtect',ctypes.c_uint32),('PartitionId',ctypes.c_uint16),('RegionSize',ctypes.c_size_t),('State',ctypes.c_uint32),('Protect',ctypes.c_uint32),('Type',ctypes.c_uint32)]
    base=library._handle;address=base+rva;info=MBI();kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.VirtualQuery.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];kernel.VirtualQuery.restype=ctypes.c_size_t
    require(kernel.VirtualQuery(address,ctypes.byref(info),ctypes.sizeof(info))==ctypes.sizeof(info),'VirtualQuery unavailable')
    query=dict(allocation_base=info.AllocationBase,region_base=info.BaseAddress,region_size=info.RegionSize,state=info.State,protect=info.Protect,type=info.Type)
    readable_image_span(base,image_size,address,24,query)
    payload=ctypes.string_at(address,24);fields=dict(zip(FIELDS,struct.unpack('<6i',payload)));validate_fields(fields)
    return dict(fields=fields,raw_hex=payload.hex(),raw_sha256=H(payload),actual_module_base=hex(base),field_rva=hex(rva),query=query,image_size=image_size,owner=p.file_record(identities[1][0]),read_only=True)
def fp_metadata():
    crt=ctypes.CDLL('ucrtbase');control=crt._controlfp;control.argtypes=[ctypes.c_uint,ctypes.c_uint];control.restype=ctypes.c_uint
    status=crt._statusfp;status.argtypes=[];status.restype=ctypes.c_uint
    word=control(0,0)
    return dict(control_word=word,status_word=status(),nearest=not bool(word&0x300),gradual=not bool(word&0x3000000),scope='Read-only CRT metadata; no flag clearing/control changes.')


class Buffers:
    def __init__(self,out):
        self.out=out;self.bytes=sum(p.stat().st_size for p in (out/'arrays').glob('*.gz'));self.records={}
    def array(self,a):
        require(a.nbytes<=MAX_ARRAY_BYTES,'array exceeds bounded reference scope')
        raw=a.tobytes(order='C');digest=H(raw)
        if digest not in self.records:
            path=self.out/'arrays'/(digest+'.gz')
            if path.exists():
                packed=path.read_bytes();require(gzip.decompress(packed)==raw,'existing raw reference buffer differs')
            else:
                packed=gzip.compress(raw,mtime=0);require(self.bytes+len(packed)<=MAX_BUFFER_RETAINED,'retention cap: preserve prior rows and fail, never omit cases')
                path.parent.mkdir(exist_ok=True);path.write_bytes(packed);self.bytes+=len(packed)
            self.records[digest]=dict(path=path.relative_to(self.out).as_posix(),raw_bytes=len(raw),raw_sha256=digest,stored_bytes=len(packed),stored_sha256=H(packed))
        return dict(buffer_sha256=digest,dtype=a.dtype.str,shape=list(a.shape),strides=list(a.strides),address_mod64=a.ctypes.data%64,aligned=bool(a.flags.aligned),C=bool(a.flags.c_contiguous),F=bool(a.flags.f_contiguous),owned=bool(a.flags.owndata),writeable=bool(a.flags.writeable),metadata=dict(a.dtype.metadata) if a.dtype.metadata else None)
def references(out,mode):
    require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_RUN_ID','').isdigit() and os.name=='nt','fresh hosted original-only execution required')
    validate_environment(os.environ)
    require(os.environ.get('SKIMAGEXX_DISABLE_SIMD')==('1' if mode=='scalar' else None),'mislabeled own-core mode')
    reports=out/'reports'/mode;reports.mkdir(parents=True,exist_ok=False)
    p=probe_module();p.probe(out) # one natural refusal retained; no production guard changed
    for name in ('observations.json','environment-before-import.json'):
        (reports/name).write_bytes((out/name).read_bytes())
    import numpy as np
    import skimagexx as xx
    from skimagexx import _matrix_transform as wrapper
    original=importlib.import_module('skimage.transform._geometric')
    spec,frozen=source_inputs((out/'source-inputs/cohorts.cpp').read_bytes(),(out/'source-inputs/matrix.json').read_bytes())
    facts=frozen['profiles']['windows-np2.5.4-py3.13']['source']['source_utf8']
    for name,text in facts.items():
        function=getattr(original.ProjectiveTransform,name.split('.')[1]) if name.startswith('ProjectiveTransform.') else getattr(original,name)
        require(inspect.getsource(function)==text,'original source mismatch: '+name)
    site=capture_site(original._apply_homogeneous,facts['_apply_homogeneous'])
    table=live_fields(wrapper,p,np);fields=table['fields'];fp=fp_metadata();require(fp['nearest'] and fp['gradual'],'unassessed FP environment')
    census=list(case_specs(spec,fields));require(sum(c['family']=='legacy4000' for c in census)==4000,'legacy cases missing')
    require(len(json.dumps(census).encode())<=32*1024*1024,'complete metadata census exceeds reserved budget')
    dump(reports/'before-fixtures.json',dict(source=facts,capture_site=site['provenance'],boundary_schema=BOUNDARY_SCHEMA,fma_witness_fraction_proof=fma_fraction_proof(),live_table=table,fp=fp,mode=mode,case_count=len(census),cases=census,original_only=True,native_acceptance=False,public_acceptance=False))
    buffers=Buffers(reports.parent);completed=0;base_points=LastBasePoints()
    before=(xx._core.matrix_transform_completed(),xx._core.matrix_transform_v2_completed());require(before==(0,0),'unexpected prior native arithmetic')
    with gzip.open(reports/'rows.jsonl.gz','wt',encoding='utf-8') as stream:
        for item in census:
            case=item['case'];n=item['n']
            points=base_points.get(np,case,n,item['extension'])
            points=layout(np,points,item['pl']);matrix=layout(np,np.array(case['matrix'],dtype='f8'),item['ml'])
            require(points.dtype.str==matrix.dtype.str=='<f8','compiled double cohort must enter original as float64')
            inputs=(points.tobytes(),matrix.tobytes());pre_fp=fp_metadata()
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');actual,operands,observed=capture_original(original.matrix_transform,original._apply_homogeneous,points,matrix,site)
            post_fp=fp_metadata()
            require(inputs==(points.tobytes(),matrix.tobytes()),'original inputs mutated')
            arrays=(base_points.base,points,matrix,actual,*operands.values());owners={}
            for a in arrays:
                require(type(a) is np.ndarray,'unassessed source array subclass')
                owner=a
                while type(owner.base) is np.ndarray:owner=owner.base
                require(owner.base is None,'unassessed source allocation owner')
                owners[id(owner)]=owner.nbytes
            # Include original division/where mask temporaries conservatively;
            # no claim about Python/library initialization's total RSS.
            workspace_bound=sum(owners.values())+4*len(points)*8
            require(workspace_bound<=MAX_ARRAY_BYTES,'reference workspace certificate exceeded')
            row=dict(family=item['family'],id=case['id'],requested_rows=n,points_layout=item['pl'],matrix_layout=item['ml'],mode=mode,capture_observed=observed,array_workspace_bound=workspace_bound,original_inputs=dict(points=buffers.array(points),matrix=buffers.array(matrix)),actual_original_intake={name:buffers.array(a) for name,a in operands.items()},original_result=buffers.array(actual),warnings=[dict(category=type(w.message).__name__,message=str(w.message)) for w in notices],FP_before=pre_fp,FP_after=post_fp,native_completed=0,native_acceptance=False)
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush();completed+=1
    after=(xx._core.matrix_transform_completed(),xx._core.matrix_transform_v2_completed());require(before==after==(0,0),'native arithmetic requested during original-only reference')
    require(table==live_fields(wrapper,p,np),'selected runtime/table changed')
    auth=json.loads((out/'input-authentication.json').read_text())
    for name,record in auth['files'].items():require(p.file_record(out/'genuine-prefix'/name)['sha256']==record['sha256'],'runtime/product input changed')
    dump(reports/'result.json',dict(completed=completed,expected=len(census),legacy_completed=4000,boundary_completed=completed-4000,boundary_schema=BOUNDARY_SCHEMA,mode=mode,buffers=buffers.records,retained_bytes=buffers.bytes,native_counters_before=before,native_counters_after=after,original_only=True,native_acceptance=False,public_acceptance=False,operation_timings=0,scope='Actual unforced NEW HOST original references, not native/profile/public acceptance. Scalar label controls own unused core only, not original BLAS ISA.'))
def prepare(out):
    require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_RUN_ID','').isdigit() and os.name=='nt','fresh hosted preparation only')
    validate_environment(os.environ)
    p=probe_module();out.mkdir(parents=True,exist_ok=False)
    revision=git('rev-parse','HEAD').decode().strip();controller={}
    require(revision==os.environ.get('GITHUB_SHA'),'host controller/source revision mismatch')
    for name in FILES:
        raw=git('show',revision+':'+name);require(canonical(raw)==canonical((ROOT/name).read_bytes()),'controller working bytes differ')
        controller[name]=dict(raw_git_sha256=H(raw),source_utf8=raw.decode())
    dump(out/'controller.json',dict(revision=revision,members=controller,probe_prerequisite_sha256=PROBE_SHA,probe_reviewed_revision=PROBE_REVISION))
    meta,raw=p.fetch_artifact(os.environ.get('GH_TOKEN'));dump(out/'donor-artifact.json',meta);p.extract_inputs(raw,out)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        source=z.read(p.PREFIX+'source.zip');require(H(source)==p.SOURCE_SHA,'wrong donor source archive')
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        cpp=z.read('tests/test_matrix_transform_native.cpp');matrix=z.read('coverage/contracts/matrix-transform.json');source_inputs(cpp,matrix)
    (out/'source-inputs').mkdir();(out/'source-inputs/cohorts.cpp').write_bytes(cpp);(out/'source-inputs/matrix.json').write_bytes(matrix)
    del raw,source
    for mode in ('normal','scalar'):
        env=dict(os.environ,PYTHONPATH=str(out/'genuine-prefix'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        if mode=='scalar':env['SKIMAGEXX_DISABLE_SIMD']='1'
        else:env.pop('SKIMAGEXX_DISABLE_SIMD',None)
        argv=[sys.executable,str(Path(__file__).resolve()),'_references','--output',str(out),'--mode',mode]
        result=subprocess.run(argv,env=env,capture_output=True)
        (out/(mode+'.stdout')).write_bytes(result.stdout);(out/(mode+'.stderr')).write_bytes(result.stderr)
        dump(out/(mode+'.command.json'),dict(argv=argv,returncode=result.returncode))
        require(result.returncode==0,'original reference child failed; preserve raw data, never retry another CPU')
    records={};retained_bytes=0
    for path in out.rglob('*'):
        if path.is_file() and 'genuine-prefix' not in path.relative_to(out).parts:
            records[path.relative_to(out).as_posix()]=dict(bytes=path.stat().st_size,sha256=H(path.read_bytes()));retained_bytes+=path.stat().st_size
    require(retained_bytes<=MAX_RETAINED,'complete scientific retention exceeds declared cap; retain everything and fail')
    dump(out/'manifest.json',dict(source=revision,records=records,original_reference=True,native_acceptance=False,public_acceptance=False,operation_timings=0))
def verify(out):
    """Standalone metadata/raw-byte verification, no array/library imports."""
    manifest=json.loads((out/'manifest.json').read_text())
    require(manifest['original_reference'] is True and manifest['native_acceptance'] is False and manifest['public_acceptance'] is False and manifest['operation_timings']==0,'reference promoted into acceptance')
    for name,row in manifest['records'].items():
        path=out/name;require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(),'unsafe retained member')
        raw=path.read_bytes();require(len(raw)==row['bytes'] and H(raw)==row['sha256'],'retained member differs')
    spec,_=source_inputs((out/'source-inputs/cohorts.cpp').read_bytes(),(out/'source-inputs/matrix.json').read_bytes())
    verified=set()
    for mode in ('normal','scalar'):
        reports=out/'reports'/mode;before=json.loads((reports/'before-fixtures.json').read_text());result=json.loads((reports/'result.json').read_text())
        fields=before['live_table']['fields'];census=list(case_specs(spec,fields))
        require(before['cases']==census and before['case_count']==len(census),'fixture census differs')
        require(before['boundary_schema']==result['boundary_schema']==BOUNDARY_SCHEMA and before['fma_witness_fraction_proof']==fma_fraction_proof(),'versioned boundary/witness metadata differs')
        expected={(r['family'],r['case']['id'],r['n'],r['pl'],r['ml']):r for r in census};keys=set(expected)
        require(len(keys)==len(census) and result['completed']==result['expected']==len(keys) and result['legacy_completed']==4000,'missing original rows')
        require(result['original_only'] is True and result['native_acceptance'] is False and result['public_acceptance'] is False and result['operation_timings']==0,'reference scope differs')
        require(result['native_counters_before']==result['native_counters_after']==[0,0],'native execution in original reference')
        seen=set()
        with gzip.open(reports/'rows.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);key=(row['family'],row['id'],row['requested_rows'],row['points_layout'],row['matrix_layout'])
                require(key in keys and key not in seen and row['mode']==mode,'unexpected/duplicate/mislabeled original row');seen.add(key)
                require(row['native_completed']==0 and row['native_acceptance'] is False,'native claim in reference row')
                observed=row['capture_observed'];site=before['capture_site']
                require(observed==dict(site,observed_event='line',observed_line=site['expected_line'],f_trace_lines=True,divs_bound_at_capture=False),'missing/wrong first post-@ line/code provenance')
                arrays=[*row['original_inputs'].values(),*row['actual_original_intake'].values(),row['original_result']]
                require(row['array_workspace_bound']<=MAX_ARRAY_BYTES and row['FP_before']['nearest'] and row['FP_before']['gradual'] and row['FP_after']['nearest'] and row['FP_after']['gradual'],'unassessed numerical environment/workspace')
                definition=expected[key];dimensions=len(definition['case']['coords'][0]);n=definition['n'] or len(definition['case']['coords'])
                require(row['original_result']['shape']==[n,dimensions] and row['original_result']['dtype']=='<f8','original result shape/type differs')
                require(row['actual_original_intake']['new_points_h']['shape']==[n,dimensions+1] and row['actual_original_intake']['matrix']['shape']==[dimensions+1,dimensions+1],'captured original @ shape differs')
                for array in arrays:
                    digest=array['buffer_sha256'];record=result['buffers'][digest]
                    elements=1
                    for size in array['shape']:
                        require(type(size) is int and size>=0,'invalid original shape');elements*=size
                    require(array['dtype']=='<f8' and record['raw_bytes']==elements*8,'raw words/float64 descriptor disagree')
                    if digest not in verified:
                        path=out/'reports'/record['path'];require(record['path']=='arrays/'+digest+'.gz','buffer path differs')
                        packed=path.read_bytes();require(H(packed)==record['stored_sha256'] and len(packed)==record['stored_bytes'],'packed buffer differs')
                        raw=gzip.decompress(packed);require(H(raw)==digest and len(raw)==record['raw_bytes'],'raw original words differ');verified.add(digest)
        require(seen==keys,'original rows omitted')
    return dict(verified=True,original_only=True,unique_raw_buffers=len(verified),native_acceptance=False,public_acceptance=False)
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','all','verify','_references'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=('normal','scalar'));args=parser.parse_args(argv)
    out=args.output.resolve();require(out.is_relative_to((ROOT/'build').resolve()),'owned workspace output required')
    if args.stage=='plan':
        p=probe_module();print(json.dumps(dict(original_only=True,legacy_cases=4000,old_protocol_observations_unchanged=374,donor_run=p.FAILED_RUN,artifact=p.ARTIFACT_ID,outer_sha256=p.ARTIFACT_SHA,BLAS_sha256=BLAS_SHA,UMATH_sha256=UMATH_SHA,profiles_modified=False,dispatch_authorized=False)));return
    try:
        if args.stage=='all':prepare(out)
        elif args.stage=='verify':print(json.dumps(verify(out)))
        else:require(args.mode is not None,'process mode required');references(out,args.mode)
    except Exception:
        out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
