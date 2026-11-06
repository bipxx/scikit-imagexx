"""Bounded literal recipes only; factories never call metric algorithms."""
APIS=('metrics.contingency_table','metrics.adapted_rand_error','metrics.variation_of_information')
MAX_PIXELS=64
MAX_LABEL=8
UNASSESSED=(
    'Huge labels, overflow/allocation failures and arbitrary external protocols.',
    'Malformed CSR internals, arbitrary sparse formats, complex/NaN/negative table values.',
    'Broad float/object label conversion, unbounded arrays and concurrent mutation.',
    'General native reductions/log2 and SciPy CSR class/index/storage packaging remain unclosed ownership gates.',
    'Shape-mismatch/error precedence combinations beyond this first finite recipe set.',
)


def cases():
    truth=[0,1,1,2,2,2];test=[0,2,2,1,1,2]
    def row(api,name,**kw):
        return dict(id=api+'/'+name,api='metrics.'+api,truth=truth,test=test,shape=[2,3],test_shape=[2,3],dtype='int16',layout='C',policy={},**kw)
    rows=[]
    for name,changes in (
        ('duplicates',{}),
        ('normalize-ignore',dict(normalize=True,ignore=[0],sparse_type='array',layout='F')),
        ('allignored-normalize',dict(truth=[1]*6,ignore=[1],normalize=True,policy={'all':'call'})),
        ('unequal-shapes-flat-match',dict(test_shape=[3,2])),
        ('readonly-reversed',dict(dtype='uint8',layout='reverse-readonly')),
        ('empty',dict(truth=[],test=[],shape=[0,2],test_shape=[0,2])),
        ('negative',dict(truth=[-1,0,1,1,2,2])),
        ('bad-sparse-type',dict(sparse_type='coo')),
    ):
        value=row('contingency_table',name);value.update(changes);rows.append(value)
    for name,changes in (
        ('perfect',dict(test=truth,dtype='uint16')),
        ('split-alpha0',dict(truth=[1]*6,test=[1,2,3,4,5,6],alpha=0.,ignore=[])),
        ('merge-alpha1',dict(truth=[1,2,3,4,5,6],test=[1]*6,alpha=1.,ignore=[])),
        ('quarter-alpha',dict(alpha=.25,dtype='uint64',layout='gapped')),
        ('singleton',dict(truth=[1],test=[1],shape=[1],test_shape=[1],policy={'all':'call'})),
        ('table-array',dict(table='counts-array')),
        ('table-matrix',dict(table='counts-matrix')),
        ('bad-alpha',dict(alpha=-.1)),
    ):
        value=row('adapted_rand_error',name);value.update(changes);rows.append(value)
    for name,changes in (
        ('perfect',dict(test=truth,dtype='int8')),
        ('split',dict(truth=[1]*6,test=[1,2,3,4,5,6])),
        ('merge-F',dict(truth=[1,2,3,4,5,6],test=[1]*6,layout='F')),
        ('ignore',dict(ignore=[0],layout='gapped')),
        ('allignored',dict(truth=[1]*6,ignore=[1],policy={'all':'call'})),
        ('table-normalized',dict(table='normalized-array')),
        ('table-raw-counts',dict(table='counts-array')),
        ('table-with-None-images',dict(table='normalized-array',images_none=True)),
    ):
        value=row('variation_of_information',name);value.update(changes);rows.append(value)
    return rows


def fixture(spec,np):
    assert spec in cases()
    def image(values,shape):
        value=np.array(values,dtype=spec['dtype']).reshape(shape)
        if spec['layout']=='F':value=np.array(value,order='F')
        elif spec['layout']=='gapped':
            owner=np.zeros((value.shape[0],value.shape[1]*2),dtype=value.dtype)
            owner[:,::2]=value;value=owner[:,::2]
        elif spec['layout']=='reverse-readonly':value=value[...,::-1];value.flags.writeable=False
        return value
    truth=image(spec['truth'],spec['shape']);test=image(spec['test'],spec['test_shape'])
    buffers={'truth':truth,'test':test};options={}
    name=spec['api'].split('.')[1]
    if name=='contingency_table':
        options=dict(ignore_labels=spec.get('ignore'),normalize=spec.get('normalize',False),sparse_type=spec.get('sparse_type','matrix'))
    else:
        options['ignore_labels']=tuple(spec.get('ignore',[0]if name=='adapted_rand_error'else[]))
        if name=='adapted_rand_error':options['alpha']=spec.get('alpha',.5)
    if 'table'in spec:
        import scipy.sparse as sparse
        data=np.array([1.,2.,3.,0.]);indices=np.array([1,0,1,2],dtype=np.int32);indptr=np.array([0,2,4],dtype=np.int32)
        if spec['table']=='normalized-array':data=np.array([.125,.25,.625,0.])
        constructor=sparse.csr_matrix if spec['table']=='counts-matrix'else sparse.csr_array
        table=constructor((data,indices,indptr),shape=(2,3),copy=False)
        options['table']=table;buffers.update(table_data=data,table_indices=indices,table_indptr=indptr)
    return ((None,None)if spec.get('images_none')else(truth,test)),options,buffers
