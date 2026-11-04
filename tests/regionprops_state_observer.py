"""Observe only the frozen small returned-object sequence, never all properties."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import warnings
if __package__:
    from .regionprops_state_contract import API, PROPERTIES, fixture
    from .draw_coordinates_observer import encode as encode_leaf, scientific_profile
else:
    from regionprops_state_contract import API, PROPERTIES, fixture
    from draw_coordinates_observer import encode as encode_leaf, scientific_profile

SOURCE_SHA = 'dbf4e98dec91e319dacbd39373957d430006a1ee43f3cbb2421f2028343c3007'
UTILS_SHA = 'b659f6b7d84b8ca5df77242ad33fa1d893d7e103e182c957cd7cb3016cd14f21'
SIGNATURE = '(label_image, intensity_image=None, cache=True, *, extra_properties=None, spacing=None, offset=None)'


def encode(value, np):
    if isinstance(value, slice):
        return dict(type='slice', start=value.start, stop=value.stop, step=value.step)
    if isinstance(value, (list, tuple)):
        return dict(container=type(value).__name__, items=[encode(v, np) for v in value])
    if isinstance(value, dict):
        return {k: encode(v, np) for k, v in value.items()}
    return encode_leaf(value, np)


def reference_runtime():
    np = scientific_profile(); module = importlib.import_module('skimage.measure._regionprops')
    utils = importlib.import_module('skimage.measure._regionprops_utils')
    assert hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(Path(utils.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == UTILS_SHA
    assert str(inspect.signature(module.regionprops)) == SIGNATURE
    binary = importlib.import_module('scipy.ndimage._nd_image')
    return np, module.regionprops, dict(python=sys.version, numpy=np.__version__,
        regionprops_source_sha256=SOURCE_SHA, regionprops_utils_source_sha256=UTILS_SHA,
        scipy_nd_image_binary_sha256=hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest(),
        calls_per_spec=1, native_calls=0, timings=False)


def observe(call, spec, np):
    image, options = fixture(spec, np); before = encode((image, options), np)
    pointer = image.ctypes.data; policy = np.geterr().copy(); handler = np.geterrcall(); buffer = np.getbufsize()
    rows = []; error = None; regions = None; arrays = []; region_types = []
    with warnings.catch_warnings(record=True) as constructor_notices:
        warnings.simplefilter('always')
        try:
            regions = call(image, **options)
        except Exception as exc:
            error = [type(exc).__module__, type(exc).__name__, str(exc)]
    if error is None:
        assert type(regions) is list and len(regions) <= 3
        for index, region in enumerate(regions):
            region_types.append([type(region).__module__, type(region).__qualname__])
            steps = []
            for operation, name in spec['access_sequence']:
                value = None; failure = None
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter('always')
                    try:
                        if operation == 'attr':
                            value = getattr(region, name)
                        elif operation == 'item':
                            value = region[name]
                        elif operation == 'iter':
                            names = list(iter(region))
                            value = dict(relevant=[n for n in PROPERTIES if n in names],
                                         count=len(names), sorted=names == sorted(names))
                        else:
                            assert operation == 'mutate_input'
                            positions = np.argwhere(image == region.label)
                            if len(positions):
                                point = tuple(positions[0]); image[point] = 0
                                value = dict(index=point, assigned=0)
                    except Exception as exc:
                        failure = [type(exc).__module__, type(exc).__name__, str(exc)]
                if isinstance(value, np.ndarray):
                    assert value.size <= 192 and not value.dtype.hasobject
                    arrays.append((index, len(steps), value))
                steps.append(dict(operation=operation, name=name, result=encode(value, np), error=failure,
                                  warnings=[[type(w.message).__module__, type(w.message).__name__, str(w.message)] for w in notices]))
            rows.append(dict(index=index, steps=steps))
    input_aliases = [dict(output=[ri, si], same_object=a is image,
                          shares_memory=bool(np.shares_memory(a, image))) for ri, si, a in arrays]
    aliases = []
    for i, (ri, si, a) in enumerate(arrays):
        for rj, sj, b in arrays[i+1:]:
            aliases.append(dict(left=[ri, si], right=[rj, sj], same_object=a is b,
                                shares_memory=bool(np.shares_memory(a, b))))
    row = dict(id=spec['id'], api=API, input_before=before, input_after=encode((image, options), np),
               result_container=None if regions is None else type(regions).__name__, region_types=region_types,
               regions=rows, error=error,
               constructor_warnings=[[type(w.message).__module__, type(w.message).__name__, str(w.message)] for w in constructor_notices],
               array_aliases=aliases, array_input_aliases=input_aliases,
               checks=dict(pointer_stable=image.ctypes.data == pointer,
                           FP_state_unchanged=np.geterr() == policy and np.geterrcall() is handler and np.getbufsize() == buffer))
    assert all(row['checks'].values()); json.dumps(row, allow_nan=False)
    return row
