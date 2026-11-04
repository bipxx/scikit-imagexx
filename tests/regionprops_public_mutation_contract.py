"""Public returned-state mutation only; no original constructor in fixtures."""
if __package__:
    from .regionprops_state_contract import API, PROPERTIES
else:
    from regionprops_state_contract import API, PROPERTIES
UNASSESSED = (
    'Private attributes/cache mutation, custom setter/index protocols and concurrency',
    'Extent-changing, negative/stepped/out-of-bounds slices or nonbuiltin labels',
    'Other37 properties, intensity/spacing/offset/extras, objectlifecycle and equality',
    'No native/public class/performance completion credit from reference or failed diagnostics')


def cases():
    two = [0]*30
    for row,column,value in ((1,1,1),(2,1,1),(2,2,1),(1,2,2),(3,2,2)):
        two[row*6+column] = value
    three = [0]*64
    for index,value in (((0,0,0),1),((0,1,0),1),((1,1,1),1),((0,1,1),2),((2,2,2),2)):
        three[(index[0]*4+index[1])*4+index[2]] = value
    result = []
    for kind in ('label-2d','slice-2d','both-3d'):
        ndim = 3 if kind == 'both-3d' else 2
        for cache in (True,False):
            for prime in (False,True):
                mutations = []
                if kind != 'slice-2d':
                    mutations.append(['set-label',2])
                if kind != 'label-2d':
                    mutations.append(['set-slice',[[1,3]]*3 if ndim == 3 else [[2,4],[2,4]]])
                before = [['get',name] for name in ('label','slice','bbox')]
                if prime:
                    before += [['get',name] for name in ('image','coords','num_pixels','area')]
                sequence = before + mutations + [['get',name] for name in PROPERTIES]*2
                result.append(dict(id=f'regionprops-public-mutation/{kind}-cache-{cache}-prime-{prime}',api=API,
                    shape=[4,4,4] if ndim == 3 else [5,6],values=three.copy() if ndim == 3 else two.copy(),
                    dtype='int64',cache=cache,prime=prime,kind=kind,original_slice=[[0,2]]*3 if ndim == 3 else [[1,3],[1,3]],
                    sequence=sequence,holdout=kind == 'both-3d' and not prime))
    return result


def source_safe(spec):
    assert spec['api'] == API and spec['dtype'] == 'int64'
    assert spec['shape'] in ([5,6],[4,4,4])
    size = 1
    for extent in spec['shape']:
        size *= extent
    assert size == len(spec['values']) <= 64 and set(spec['values']) == {0,1,2}
    assert type(spec['cache']) is bool and type(spec['prime']) is bool
    ndim = len(spec['shape'])
    assert len(spec['original_slice']) == ndim
    for start,stop in spec['original_slice']:
        assert type(start) is int and type(stop) is int and stop-start == 2
    for action,value in spec['sequence']:
        if action == 'get':
            assert value in PROPERTIES
        elif action == 'set-label':
            assert type(value) is int and value == 2
        else:
            assert action == 'set-slice' and len(value) == ndim
            for (start,stop),(old_start,old_stop),extent in zip(value,spec['original_slice'],spec['shape']):
                assert type(start) is int and type(stop) is int
                assert 0 <= start < stop <= extent and stop-start == old_stop-old_start == 2
                assert start == old_start+1 and stop == old_stop+1
    return dict(input_elements=size,maximum_label=2,region_count_upper_bound=2,
                observed_region_index=0,mask_elements_upper_bound=2**ndim,coords_elements_upper_bound=ndim*2**ndim)


def fixture(spec,np):
    source_safe(spec)
    image = np.array(spec['values'],dtype=np.int64).reshape(spec['shape'])
    return image,dict(cache=spec['cache'])
