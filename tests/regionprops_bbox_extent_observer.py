"""Only two new scalar properties/aliases; existing dependencies may be primed."""
import json
import warnings
if __package__:
    from .regionprops_bbox_extent_contract import API,fixture,source_safe
    from .regionprops_state_observer import encode,scientific_profile,reference_runtime,SOURCE_SHA,UTILS_SHA,SIGNATURE
else:
    from regionprops_bbox_extent_contract import API,fixture,source_safe
    from regionprops_state_observer import encode,scientific_profile,reference_runtime,SOURCE_SHA,UTILS_SHA,SIGNATURE


def observe(call,spec,np):
    image,options=fixture(spec,np);before=encode((image,options),np);expected_image=image.copy();pointer=image.ctypes.data
    descriptor=(image.shape,image.strides,image.dtype,image.dtype.metadata,image.flags.writeable,image.flags.owndata)
    old_policy,old_handler,old_buffer=np.geterr(),np.geterrcall(),np.getbufsize()
    phase=['constructor'];events=[];steps=[];regions=None;error=None;region_type=None
    def handler(message,flags):events.append(dict(phase=phase[0],message=message,flags=flags))
    with np.errstate(**spec['policy'],call=handler):
        policy=np.geterr().copy()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:regions=call(image,**options)
            except Exception as exc:error=[type(exc).__module__,type(exc).__name__,str(exc)]
        constructor_warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices]
        if error is None:
            assert type(regions) is list and 0<len(regions)<=2
            region=regions[0];region_type=[type(region).__module__,type(region).__qualname__]
            for index,(action,argument) in enumerate(spec['sequence']):
                phase[0]=index;result=None;failure=None
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter('always')
                    try:
                        if action=='attr':result=getattr(region,argument)
                        elif action=='item':result=region[argument]
                        elif action=='prime':
                            value=getattr(region,argument)
                            # Dependency state only: do not duplicate old image/area golds.
                            result=dict(primed=argument,type=[type(value).__module__,type(value).__qualname__])
                        elif action=='set-pixel':
                            position,pixel=argument;image[tuple(position)]=pixel;expected_image[tuple(position)]=pixel
                        else:
                            assert action=='set-slice';region.slice=tuple(slice(a,b) for a,b in argument)
                    except Exception as exc:failure=[type(exc).__module__,type(exc).__name__,str(exc)]
                assert result is None or not isinstance(result,np.ndarray)
                steps.append(dict(action=action,argument=argument,result=encode(result,np),error=failure,
                    warnings=[[type(w.message).__module__,type(w.message).__name__,str(w.message)] for w in notices]))
        stable=np.geterr()==policy and np.geterrcall() is handler and np.getbufsize()==old_buffer
    row=dict(id=spec['id'],api=API,input_before=before,input_after=encode((image,options),np),
        constructor_error=error,constructor_warnings=constructor_warnings,returned_count=None if regions is None else len(regions),
        region_type=region_type,steps=steps,FP_events=events,checks=dict(input_matches_declared_writes=np.array_equal(image,expected_image),
            pointer_unchanged=image.ctypes.data==pointer,descriptor_unchanged=descriptor==(image.shape,image.strides,image.dtype,image.dtype.metadata,image.flags.writeable,image.flags.owndata),policy_unchanged=stable,
            state_restored=np.geterr()==old_policy and np.geterrcall() is old_handler and np.getbufsize()==old_buffer))
    assert all(row['checks'].values());json.dumps(row,allow_nan=False);return row
