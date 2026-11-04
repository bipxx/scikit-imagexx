"""Observe public label/slice writes and only the existing seven properties."""
import json
import warnings
if __package__:
    from .regionprops_public_mutation_contract import API, fixture, source_safe
    from .regionprops_state_observer import encode, scientific_profile, reference_runtime, SOURCE_SHA, UTILS_SHA, SIGNATURE
else:
    from regionprops_public_mutation_contract import API, fixture, source_safe
    from regionprops_state_observer import encode, scientific_profile, reference_runtime, SOURCE_SHA, UTILS_SHA, SIGNATURE


def observe(call,spec,np):
    image,options = fixture(spec,np)
    before = encode((image,options),np)
    pointer = image.ctypes.data
    policy,handler,buffer = np.geterr(),np.geterrcall(),np.getbufsize()
    error,regions = None,None
    with warnings.catch_warnings(record=True) as constructor_notices:
        warnings.simplefilter('always')
        try:
            regions = call(image,**options)
        except Exception as exc:
            error = [type(exc).__module__,type(exc).__name__,str(exc)]
    steps,arrays = [],[]
    region_type = None
    if error is None:
        assert type(regions) is list and len(regions) in (1,2)
        region = regions[0]
        region_type = [type(region).__module__,type(region).__qualname__]
        for index,(action,argument) in enumerate(spec['sequence']):
            value,failure = None,None
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                try:
                    if action == 'get':
                        value = getattr(region,argument)
                    elif action == 'set-label':
                        region.label = argument
                        value = region.label
                    else:
                        assert action == 'set-slice'
                        region.slice = tuple(slice(start,stop) for start,stop in argument)
                        value = region.slice
                except Exception as exc:
                    failure = [type(exc).__module__,type(exc).__name__,str(exc)]
            if isinstance(value,np.ndarray):
                assert value.size <= 24 and value.dtype.kind in 'biu'
                arrays.append((index,value))
            steps.append(dict(action=action,argument=argument,result=encode(value,np),error=failure,
                warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices]))
    aliases = [dict(first=i,second=j,same_object=a is b,shares_memory=bool(np.shares_memory(a,b)))
               for i,a in arrays for j,b in arrays]
    row = dict(id=spec['id'],api=API,input_before=before,input_after=encode((image,options),np),
        constructor_error=error,constructor_warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in constructor_notices],
        returned_count=None if regions is None else len(regions),region_type=region_type,steps=steps,
        output_pair_aliases=aliases,output_input_aliases=[dict(step=i,same_object=a is image,
            shares_memory=bool(np.shares_memory(a,image))) for i,a in arrays],
        checks=dict(input_unchanged=before == encode((image,options),np),pointer_unchanged=image.ctypes.data == pointer,
            state_unchanged=np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer))
    assert all(row['checks'].values())
    json.dumps(row,allow_nan=False)
    return row
