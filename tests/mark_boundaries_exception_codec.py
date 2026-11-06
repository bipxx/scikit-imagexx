"""Family-local typed exception-argument descriptors; never applies a ufunc."""
from crop_observer import encode as legacy_encode


def encode_exception_args(value,np):
    # Only the two newly observed descriptor classes receive local identity
    # tokens. This is not an arbitrary object-graph or pointer serialization.
    descriptors=[]
    def descriptor_type(item):
        cls=type(item)
        return dict(module=cls.__module__,name=cls.__name__,qualname=cls.__qualname__)
    def token(item):
        for index,previous in enumerate(descriptors):
            if previous is item:return index
        descriptors.append(item);return len(descriptors)-1
    def encode(item):
        if isinstance(item,np.ufunc):
            name=item.__name__
            return dict(kind='numpy.ufunc',object_index=token(item),type=descriptor_type(item),name=name,
                        numpy_namespace_identity=getattr(np,name,None)is item,nin=item.nin,nout=item.nout,
                        nargs=item.nargs,signature=item.signature,identity=encode(item.identity))
        if isinstance(item,np.dtype):
            # Frozen mark inputs/NumPy casting and no-loop errors use plain
            # scalar dtypes. Arbitrary structured/subarray/metadata domains
            # remain explicit rather than inventing a NumPy dtype engine.
            assert item.fields is None and item.subdtype is None and item.metadata is None,'UNASSESSED_EXCEPTION_DTYPE'
            return dict(kind='numpy.dtype',object_index=token(item),type=descriptor_type(item),str=item.str,
                        name=item.name,descr=legacy_encode(item.descr,np),byteorder=item.byteorder,
                        kind_code=item.kind,char=item.char,num=item.num,itemsize=item.itemsize,
                        alignment=item.alignment,hasobject=bool(item.hasobject))
        if isinstance(item,(tuple,list)):
            return dict(container=type(item).__name__,items=[encode(v)for v in item])
        if isinstance(item,dict):return {k:encode(v)for k,v in item.items()}
        return legacy_encode(item,np)
    return encode(value)
