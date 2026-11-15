"""Generate private standalone SDK payload fixtures from admitted saved returns.

No target resolution/call. NumPy is used only to resolve scalar dtype metadata.
All payload words come from the immutable admitted raw record, never a model.
"""
import base64,hashlib,json,math,struct,zlib
from pathlib import Path


def generate(repository,profile,np,destination):
    root=Path(repository);adpath=root/'build/draw-multiblock-lbp-raw-admission-v1/admission-v1.json'
    assert hashlib.sha256(adpath.read_bytes()).hexdigest()=='1bcfff2be1b241b989a6142a756131474e89e756dd30047b0e653dde3d822726'
    admission=json.loads(adpath.read_bytes());path=f'build/draw-multiblock-lbp-codec-capture-v1/capture-v1/{profile}/observations.json'
    raw=(root/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==admission['raw_files'][path]['sha256']
    assert np.__version__==('1.24.4'if profile=='minimum'else'2.5.4')
    rows=json.loads(raw)
    import draw_multiblock_lbp_overlap_contract as contract
    specs=contract.cases();assert [r['id']for r in rows]==[s['id']for s in specs]
    def packed(a):
        b=zlib.decompress(base64.b64decode(a['raw_zlib_base64']));assert hashlib.sha256(b).hexdigest()==a['raw_sha256'];return b
    def number(v):
        if isinstance(v,dict):
            if 'controlled_protocol'in v:return number(v['value'])
            if v.get('type')in('float32','float64'):return struct.unpack('<f'if v['type']=='float32'else'<d',bytes.fromhex(v['raw']))[0]
            if 'python_float'in v:return float('nan'if v['python_float']=='nan'else'-inf'if v['python_float']=='negative-infinity'else'inf')
        return v
    def lit(x):
        if math.isnan(x):return 'std::numeric_limits<double>::quiet_NaN()'
        if math.isinf(x):return ('-'if x<0 else'')+'std::numeric_limits<double>::infinity()'
        return float(x).hex()
    def shape(x):return '{'+','.join(map(str,x))+'}'
    def colors(x):
        if not isinstance(x,list):return [x],[]
        if x and isinstance(x[0],list):return [v for row in x for v in row],[len(x),len(x[0])]
        return x,[len(x)]
    lines=['// Derived ONLY from authenticated admitted raw arrays. No synthetic gold.', '#include <skimagexx/draw_multiblock_lbp.hpp>','#include <skimagexx/image.hpp>','#include <cassert>','#include <cstring>','#include <limits>','#include <iostream>','using namespace skimagexx;','int main(){assert(get_num_threads()==1);int count=0;']
    for spec,row in zip(specs,rows):
        if row['error']is not None:continue
        before=row['before']['items'];args=before[0]['items'];options=before[1];image=args[0]
        if 'controlled_protocol'in image:image=image['array']
        data=packed(image);gold=packed(row['result']);dtype=np.dtype(image['dtype']);outdtype=np.dtype(row['result']['dtype']);half=dtype.kind=='f'and dtype.itemsize==2
        if outdtype.byteorder=='>':gold=b''.join(gold[k:k+outdtype.itemsize][::-1]for k in range(0,len(gold),outdtype.itemsize))
        typ=('uint16'if half else'float32'if dtype.itemsize==4 else'float64')if dtype.kind=='f'else'boolean'if dtype.kind=='b'else('uint'if dtype.kind=='u'else'int')+str(dtype.itemsize*8)
        alpha=number(options['alpha']);kind=spec.get('alpha_kind')
        native_alpha=np.float32(alpha)if kind=='alpha-np-f32'else np.float64(alpha)if kind=='alpha-np-f64'else alpha
        complement=1-native_alpha
        # Exact source metadata resolution, no image/target computation.
        first=np.result_type(complement,np.empty((0,),dtype=outdtype))
        lines.extend(['{ // '+row['id'],'std::vector<unsigned char> input='+shape(data)+';','std::vector<unsigned char> expected='+shape(gold)+';','MultiblockLbpParameters p;'])
        for name in('r','c','width','height'):lines.append('p.'+name+'='+str(int(spec[name]))+';')
        lines.extend(['p.code='+str(int(spec['code']))+';p.alpha='+lit(alpha)+';p.explicit_complement=true;p.complement='+lit(float(complement))+';',f'p.first_product_float32={str(first==np.dtype("float32")).lower()};p.first_product_half={str(first==np.dtype("float16")).lower()};'])
        for dest,key in(('greater','greater'),('less','less')):
            values,dims=colors(spec[key]);values=[float(v)if isinstance(v,str)else v for v in values]
            lines.append('p.'+dest+'={{'+','.join(lit(v)for v in values)+'},'+shape(dims)+'};')
        dims=image['shape'];stride=dtype.itemsize;strides=[]
        for d in reversed(dims):strides.insert(0,stride);stride*=d
        lines.append('MultiblockLbpInput in{{input.data(),'+shape(dims)+','+shape(strides)+','+str(dtype.itemsize)+'},PixelType::'+typ+','+str(dtype.byteorder=='>').lower()+','+str(half).lower()+'};')
        lines.extend(['for(int workers:{1,4})for(bool simd:{false,true}){set_num_threads(workers);p.simd=simd;auto out=draw_multiblock_lbp(in,p);assert(out.captured_workers==workers);assert((out.shape==std::vector<std::ptrdiff_t>'+shape(row['result']['shape'])+'));','const void* data=out.is_half?static_cast<const void*>(out.float16.data()):out.is_float32?static_cast<const void*>(out.float32.data()):static_cast<const void*>(out.float64.data());','if(expected.size())assert(std::memcmp(data,expected.data(),expected.size())==0);++count;}', '}'])
    lines.extend(['set_num_threads(1);std::cout<<"PASS admitted saved SDK payload comparisons "<<count<<"\\n";}'])
    destination=Path(destination);destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(profile=profile,return_rows=sum(r['error']is None for r in rows),comparisons=sum(r['error']is None for r in rows)*4,raw_sha256=hashlib.sha256(raw).hexdigest(),generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),scope='Standalone numeric SDK logical payload/shape/budgets/scalar-SSE2 only. Public errors/protocol/representation/FP tested separately; no original calls.')
