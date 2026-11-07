"""Public interchangeability, native dispatch, and per-function reference tests."""
import inspect
import numpy as np
import pytest
import skimage
from skimage import filters as reference_filters, color as reference_color, util as reference_util
import skimagexx as xx
from skimagexx.coverage import native_functions, native_dtypes, needs_color_input, default_call, assert_default_parity, DERIVED_MOMENTS_FUNCTIONS,COORDS_MOMENTS_FUNCTIONS, native_classes, class_metadata_providers, LEVEL_SET_FUNCTIONS
from skimagexx.coverage import set_color_default_result
from skimage import draw as reference_draw
from skimage import exposure as reference_exposure
from skimage import morphology as reference_morphology
from skimage import transform as reference_transform
from skimage import feature as reference_feature
from skimage import metrics as reference_metrics, measure as reference_measure
from skimage import segmentation as reference_segmentation
from skimage import restoration as reference_restoration
from skimage.filters import rank as reference_rank
from skimage.filters.rank import generic as reference_rank_generic
from skimage.filters.rank import bilateral as reference_rank_bilateral
from skimagexx.filters import rank as native_rank
from skimagexx.filters.rank import generic as native_rank_generic
from skimagexx.filters.rank import bilateral as native_rank_bilateral

MODULES = [(xx.filters, reference_filters), (xx.color, reference_color), (xx.util, reference_util),
           (xx.exposure, reference_exposure), (xx.morphology,reference_morphology), (xx.transform,reference_transform),
           (xx.feature,reference_feature), (xx.metrics,reference_metrics), (xx.measure,reference_measure),
           (xx.segmentation,reference_segmentation), (xx.restoration,reference_restoration),
           (xx.draw,reference_draw),
           (native_rank,reference_rank),(native_rank_generic,reference_rank_generic),
           (native_rank_bilateral,reference_rank_bilateral)]
FUNCTIONS = [(native, original, name) for native, original in MODULES for name in native.__all__
             if not inspect.isclass(getattr(native,name))]
# Pinned generic exposes these callable identity aliases outside star exports.
# Exercise signatures and values independently of the advertised __all__ list.
FUNCTIONS += [(native_rank_generic, reference_rank_generic, name)
              for name in ('sum', 'windowed_histogram', 'majority') if name not in native_rank_generic.__all__]
CLASSES = [(native,original,name) for native,original in MODULES for name in native.__all__
           if inspect.isclass(getattr(native,name))]

# The original Bregman Cython helper has unchecked halo reads for small axes.
# Use separately reviewed geometry for Bregman and max-tree APIs; their contracts retain
# unsafe geometry as explicitly blocked intake metadata, never original calls.
VALUE_CASES = [
    (native, original, name, dtype, shape)
    for native, original, name in FUNCTIONS
    # Shape constructors use one descriptor carrier, not image dtype cases.
    for dtype in ((np.float64,) if (native is xx.restoration and name in ('ball_kernel','ellipsoid_kernel'))
                  or (native is xx.measure and name in ('approximate_polygon','subdivide_polygon'))
                  else (np.uint8,) if native.__name__.split('.',1)[1]+'.'+name in (*LEVEL_SET_FUNCTIONS,'segmentation.find_boundaries')
                  else (np.uint8, np.float32, np.float64))
    for shape in (((4, 5), (7, 19), (519, 521))
                  if ((native is xx.restoration and name == 'denoise_tv_bregman')
                      or (native is xx.morphology and name in ('max_tree','max_tree_local_maxima',
                          'area_opening','area_closing','diameter_opening','diameter_closing')))
                  else ((1, 1), (3, 19), (519, 521)))
]


def test_registry_matches_per_function_tests():
    tested = {module.__name__.split('.', 1)[1]+'.'+name for module, _, name in FUNCTIONS}
    assert tested == set(native_functions())
    tested_classes={module.__name__.split('.',1)[1]+'.'+name for module,_,name in CLASSES}
    assert tested_classes==set(native_classes())
    assert set(class_metadata_providers())=={'util.FailedEstimationAccessError'}
    assert inspect.isclass(xx.util.FailedEstimationAccessError)


def test_level_set_default_shape_adapter_and_exact_checker():
    from skimagexx.coverage import level_set_default_error
    image=np.zeros((3,7),np.uint8)
    for api in LEVEL_SET_FUNCTIONS:
        args,options=default_call(api,image)
        assert args==(image.shape,) and options=={} and native_dtypes(api)==('uint8',)
    expected=np.zeros(image.shape,np.int8);actual=expected.copy()
    assert level_set_default_error(image.shape,actual,expected)==0.
    assert level_set_default_error(image,actual,expected)==0.
    with pytest.raises(AssertionError):level_set_default_error(image.shape,actual.astype(np.uint8),expected)
    with pytest.raises(AssertionError):level_set_default_error(image.shape,np.asfortranarray(actual),expected)
    actual[1,2]=1
    with pytest.raises(AssertionError):level_set_default_error(image.shape,actual,expected)


@pytest.mark.parametrize('dtype',(np.uint8,np.float32,np.float64))
def test_regular_seeds_default_geometry_dtype_and_exact_checker(dtype):
    from skimagexx.coverage import regular_seeds_default_error
    image=np.zeros((3,7),dtype)
    args,options=default_call('util.regular_seeds',image)
    assert args==(image.shape,1) and options=={'dtype':image.dtype}
    assert native_dtypes('util.regular_seeds')==('uint8','float32','float64')
    actual=xx.util.regular_seeds(*args,**options)
    expected=np.zeros(image.shape,dtype);expected[1,3]=1
    assert regular_seeds_default_error(args[0],options['dtype'],actual,expected)==0.
    wrong=actual.astype(np.int64)
    with pytest.raises(AssertionError):regular_seeds_default_error(args[0],options['dtype'],wrong,expected)
    with pytest.raises(AssertionError):regular_seeds_default_error(args[0],options['dtype'],np.asfortranarray(actual),expected)
    with pytest.raises(AssertionError):regular_seeds_default_error((1,21),options['dtype'],actual,expected)
    actual[0,0]=2
    with pytest.raises(AssertionError):regular_seeds_default_error(args[0],options['dtype'],actual,expected)


@pytest.mark.parametrize('dtype',(np.uint8,np.float32,np.float64))
def test_unique_rows_default_exact_checker(dtype):
    from skimagexx.coverage import unique_rows_default_error
    image=np.array([[2,1],[0,3],[2,1],[1,0]],dtype=dtype)
    args,options=default_call('util.unique_rows',image)
    assert len(args)==1 and args[0] is image and options=={}
    assert native_dtypes('util.unique_rows')==('uint8','float32','float64')
    actual=xx.util.unique_rows(*args,**options)
    expected=image[[1,3,0] if dtype==np.uint8 else [1,0,3]]
    assert unique_rows_default_error(image,actual,expected)==0.
    with pytest.raises(AssertionError):unique_rows_default_error(image,actual.astype(np.int64),expected)
    with pytest.raises(AssertionError):unique_rows_default_error(image,np.asfortranarray(actual),expected)
    with pytest.raises(AssertionError):unique_rows_default_error(image,actual[:,:1],expected)
    changed=actual.copy();changed.flat[0]+=1
    with pytest.raises(AssertionError):unique_rows_default_error(image,changed,expected)


@pytest.mark.parametrize('dtype', (np.uint8,np.float32,np.float64))
def test_label_points_default_geometry_and_exact_checker(dtype):
    from skimagexx.coverage import label_points_default_error
    image=np.zeros((258,260),dtype)
    image[0,0]=image[1,2]=image[255,255]=image[256,257]=1
    args,options=default_call('util.label_points',image)
    assert args[1]==image.shape and options=={}
    assert args[0].dtype==image.dtype
    assert np.array_equal(args[0],np.array([[0,0],[1,2],[255,255]],dtype))
    assert native_dtypes('util.label_points')==('uint8','float32','float64')
    actual=xx.util.label_points(*args,**options)
    expected=np.zeros(image.shape,np.uint64)
    expected[0,0]=1;expected[1,2]=2;expected[255,255]=3
    assert label_points_default_error(args[1],actual,expected)==0.
    with pytest.raises(AssertionError):label_points_default_error(args[1],actual.astype(np.float64),expected)
    with pytest.raises(AssertionError):label_points_default_error(args[1],np.asfortranarray(actual),expected)
    actual[0,0]^=np.uint64(1)
    with pytest.raises(AssertionError):label_points_default_error(args[1],actual,expected)


def test_map_array_default_mapping_and_exact_checker():
    from skimagexx.coverage import map_array_default_error
    image=np.array([[0,1,127],[2,255,3]],np.uint8)
    args,options=default_call('util.map_array',image)
    assert args[0] is image and options=={} and native_dtypes('util.map_array')==('uint8',)
    actual=xx.util.map_array(*args,**options)
    expected=np.array([[1,4,382],[7,766,10]],np.uint16)
    assert map_array_default_error(image,actual,expected)==0.
    with pytest.raises(AssertionError):map_array_default_error(image,actual.astype(np.float64),expected)
    with pytest.raises(AssertionError):map_array_default_error(image,np.asfortranarray(actual),expected)
    actual[0,1]^=np.uint16(1)
    with pytest.raises(AssertionError):map_array_default_error(image,actual,expected)


@pytest.mark.parametrize('api',('draw.line','draw.rectangle','draw.disk','draw.ellipse'))
@pytest.mark.parametrize('dtype',(np.uint8,np.float32,np.float64))
def test_draw_default_shape_geometry_and_exact_checker(api,dtype):
    from skimagexx.coverage import draw_default_error
    import sys
    # These labels carry only shapes; no input pixel or dtype becomes geometry.
    image=np.full((3,7),17,dtype)
    args,options=default_call(api,image)
    other=np.zeros(image.shape,np.dtype(dtype,metadata={'ignored':'shape-only'}))
    assert default_call(api,other)==(args,options)
    assert native_dtypes(api)==('uint8','float32','float64')
    if api=='draw.line':
        assert args==(0,0,2,6)and options=={}
        values=(np.array([0,0,1,1,1,2,2],np.intp),np.arange(7,dtype=np.intp))
        expected=tuple(np.asarray(memoryview(v))for v in values)
    elif api=='draw.rectangle':
        assert args==((0,0),)and options=={'extent':(3,7)}
        values=[np.broadcast_to(np.arange(3,dtype=np.intp),(7,3)).copy(),
                np.broadcast_to(np.arange(7,dtype=np.intp)[:,None],(7,3)).copy()]
        expected=values if np.__version__=='1.24.4'else tuple(values)
    else:
        if api=='draw.disk':
            assert args==((1.,3.),1.5)and options=={'shape':(3,7)}
            rows=np.repeat(np.arange(3,dtype=np.intp),3);columns=np.tile(np.arange(2,5,dtype=np.intp),3)
        else:
            assert args==(1.,3.,1.,7/3)and options=={'shape':(3,7)}
            rows=np.ones(5,np.intp);columns=np.arange(1,6,dtype=np.intp)
        owner=np.column_stack((rows,columns));expected=(owner[:,0],owner[:,1])
    entry=getattr(xx._core,{'draw.line':'draw_line','draw.rectangle':'draw_rectangle',
                          'draw.disk':'draw_ellipse','draw.ellipse':'draw_ellipse'}[api])
    previous=sys.getprofile();returns=[]
    def profile(frame,event,value):
        if event=='c_return'and value is entry:returns.append(entry.__name__)
    sys.setprofile(profile)
    try:actual=getattr(xx.draw,api.split('.')[1])(*args,**options)
    finally:sys.setprofile(previous)
    assert returns==[entry.__name__]
    assert draw_default_error(api,image,actual,expected)==0.
    assert_default_parity(api,image.dtype,actual,expected,image=image)


@pytest.mark.parametrize('api',('draw.line','draw.rectangle','draw.disk','draw.ellipse'))
def test_draw_exact_checker_reuses_frozen_words_and_rejects_regressions(api):
    import base64,json,zlib
    from pathlib import Path
    from skimagexx.coverage import draw_default_error,parity_tolerance
    matrix=json.loads((Path(__file__).parents[1]/'coverage/contracts/draw-coordinates.json').read_bytes())
    case={'draw.line':'line/horizontal','draw.rectangle':'rectangle/end2d',
          'draw.disk':'disk/ordinary','draw.ellipse':'ellipse/axes'}[api]
    row=next(r for r in matrix['original_observations'][np.__version__]if r['id']==case)
    def prepared():
        values=[np.frombuffer(zlib.decompress(base64.b64decode(v['raw_zlib_base64'])),dtype=np.intp).reshape(v['shape']).copy()for v in row['result']['items']]
        if api=='draw.line':return tuple(np.asarray(memoryview(v))for v in values)
        if api=='draw.rectangle':return values if row['result']['container']=='list'else tuple(values)
        owner=np.column_stack(values);return owner[:,0],owner[:,1]
    image=np.empty((9,11),np.uint8);actual,expected=prepared(),prepared()
    assert draw_default_error(api,image,actual,expected)==0.
    # Existing dtype alias/metadata tests recurse over individual coordinates.
    assert draw_default_error(api,image,actual[0],expected[0])==0.
    with pytest.raises(AssertionError):draw_default_error(api,image,actual[0].astype(np.int32),expected[0])
    with pytest.raises(AssertionError):draw_default_error(api,image,tuple(actual)if isinstance(actual,list)else list(actual),expected)
    actual[0].flat[0]+=1
    with pytest.raises(AssertionError):draw_default_error(api,image,actual,expected)
    with pytest.raises(ValueError,match='exact coordinate/weight words and storage'):parity_tolerance(api,np.uint8)


def test_unported_namespace_interchangeability():
    a = np.zeros((9, 11), dtype=bool)
    a[3:6, 4:7] = True
    with pytest.raises(NotImplementedError):
        xx.morphology.convex_hull_image(a)
    xx.set_fallback_enabled(True)
    from skimage import morphology
    with pytest.warns(xx.FallbackWarning):
        actual = xx.morphology.convex_hull_image(a)
    np.testing.assert_array_equal(actual, morphology.convex_hull_image(a))


@pytest.fixture(autouse=True, params=[1, 4])
def configuration(request):
    previous = xx.set_fallback_enabled(False)
    xx.set_num_threads(request.param)
    yield
    xx.set_fallback_enabled(previous)
    xx.set_num_threads(1)


@pytest.mark.parametrize('native,original,name', FUNCTIONS, ids=[x[2] for x in FUNCTIONS])
def test_signatures(native, original, name):
    assert inspect.signature(getattr(native, name)) == inspect.signature(getattr(original, name))


@pytest.fixture(scope='module')
def contingency_float_reference():
    # default_call uses the same bounded six-label partition for every smoke
    # shape/worker row. Keep the original CSR, including its physical storage,
    # read-only by convention and check that each comparison leaves it intact.
    from label_pair_metrics_observer import observation
    cached = {}
    def reference(dtype):
        key = np.dtype(dtype).str
        if key not in cached:
            args, options = default_call('metrics.contingency_table', np.empty((2, 3), dtype))
            try:
                value = reference_metrics.contingency_table(*args, **options)
            except TypeError as error:
                cached[key] = (None, None, (type(error), str(error)))
            else:
                cached[key] = (value, observation(value, np), None)
        value, before, error = cached[key]
        if error is None:
            assert observation(value, np) == before
        return value, before, error
    return reference


@pytest.fixture(scope='module')
def join_reference_rows():
    # No original calls: validate the actual request and the actual original
    # module before reusing admitted rows. Error equivalence is source-derived,
    # not identical-input capture: bool/float defaults keep BOTH requested dtypes.
    import ast,base64,hashlib,json,warnings,zlib
    from pathlib import Path
    from draw_coordinates_observer import encode
    root=Path(__file__).parent
    raw=(root/'fixtures/join_segmentations_originals_v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='0cb7da5d981bf95eb8895278bf9acc638ba3acd236625c48a00aa632a1b0a340'
    data=json.loads(raw);profile='minimum'if np.__version__.startswith('1.')else'modern'
    assert np.__version__==data['profiles'][profile]['runtime']['numpy']
    source_hash=data['profiles'][profile]['runtime']['source_sha256']
    assert source_hash=='8d1da1717bf35818f57156f88a94cd84a9527f48aff16d60e250b7d8045cd8be'
    rows={row['id'].split('/')[-1]:row for row in data['profiles'][profile]['rows']};cached={}
    def reference(args,options,source_path):
        source=Path(source_path).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(source).hexdigest()==source_hash
        assert type(args)is tuple and len(args)==2 and type(options)is dict and options=={}
        assert all(type(a)is np.ndarray for a in args)
        dtype=args[0].dtype
        assert dtype==args[1].dtype and dtype.metadata is None
        assert dtype.name in ('int8','uint8','int16','uint16','int32','uint32','int64','uint64','bool','float16','float32','float64')
        # Validate full physical metadata and words, not merely dtype or a
        # default_call reconstructed from the same implementation under test.
        frozen=(np.array([0,0,1,1,2,2,3,3],dtype=dtype).reshape(2,4),
                np.array([0,1,1,0,0,1,1,0],dtype=dtype).reshape(2,4))
        actual_input=encode(args,np)
        assert actual_input==encode(frozen,np)
        if dtype.kind in 'bf':
            row=rows['bool-left'if dtype.kind=='b'else'float-left']
            assert row['input_before']==row['input_after'] and not row['warnings'] and not row['FP_events']
            assert row['error']==['builtins','TypeError','label_field must have an integer dtype']
            # Exact pinned source: equal shape -> first relabel(s1). Relabel's
            # offset/min/unique/arange-or-concatenate prefix reads s1 only, then
            # rejects its non-integer kind. s2 DATA/DTYPE and return_mapping
            # are unread before this error. No map_array/native/provider call.
            tree=ast.parse(source)
            join=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='join_segmentations')
            relabel=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='relabel_sequential')
            body=join.body[1:]
            assert ast.unparse(body[0].test)=='s1.shape != s2.shape'
            assert ast.unparse(body[1].value)=='relabel_sequential(s1)'
            prefix=relabel.body[1:9]
            assert [type(n).__name__ for n in prefix]==['If','If','Assign','Assign','If','Assign','If','Assign']
            assert ast.unparse(prefix[0].test)=='offset <= 0' and ast.unparse(prefix[1].test)=='np.min(label_field) < 0'
            assert ast.unparse(prefix[3].value)=='np.unique(label_field)'
            assert ast.unparse(prefix[5].value)=='label_field.dtype'
            assert ast.unparse(prefix[6].test)=="input_type.kind not in 'iu'"
            # Bounded NumPy-only prefix intake. These are source-derived
            # intermediate checks, never claimed as captured original outputs.
            state=np.geterr(),np.geterrcall(),np.getbufsize()
            with warnings.catch_warnings(record=True)as observed:
                warnings.simplefilter('always')
                assert args[0].shape==args[1].shape and np.min(args[0])==0
                unique=np.unique(args[0]);assert unique.tolist()==([False,True]if dtype.kind=='b'else[0.,1.,2.,3.])
                out=np.concatenate([[0],np.arange(1,len(unique))])
                assert out.tolist()==list(range(len(unique))) and dtype.kind not in 'iu'
            assert not observed and state==(np.geterr(),np.geterrcall(),np.getbufsize())
            assert encode(args,np)==actual_input
            return None,row['error'],'SOURCE_DERIVED_EARLY_ERROR_EQUIVALENCE'
        name=dtype.name+'-labels';row=rows[name];gold=row['result']
        assert actual_input==row['input_before']['items'][0]
        # Omitted flag binds to the pinned signature's False default; captured
        # integer rows supplied False explicitly, except the int8 omission row.
        assert row['input_before']['items'][1]in ({},{'return_mapping':False})
        assert join_signature_default_false(source)
        if name not in cached:
            payload=zlib.decompress(base64.b64decode(gold['raw_zlib_base64']))
            assert hashlib.sha256(payload).hexdigest()==gold['raw_sha256']
            cached[name]=np.frombuffer(payload,np.dtype(gold['dtype'])).reshape(gold['shape']).copy()
        assert encode(cached[name],np)==gold
        return cached[name],None,'IDENTICAL_CAPTURED_ARRAYS_EFFECTIVE_FALSE_OPTION'
    def join_signature_default_false(source):
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='join_segmentations')
        return len(function.args.defaults)==1 and isinstance(function.args.defaults[0],ast.Constant)and function.args.defaults[0].value is False
    return reference


@pytest.mark.parametrize('native,original,name,dtype,shape', VALUE_CASES,
                         ids=[f'{n.__name__}.{name}-{np.dtype(dtype).name}-{shape}'
                              for n, _, name, dtype, shape in VALUE_CASES])
def test_every_native_function(native, original, name, dtype, shape, monkeypatch, contingency_float_reference, join_reference_rows):
    rng = np.random.default_rng(15)
    api = native.__name__.split('.', 1)[1]+'.'+name
    a = rng.integers(0, 256, (*shape, 3) if needs_color_input(api) else shape, dtype=np.uint8)
    if dtype != np.uint8:
        a = (a/255).astype(dtype)
    if np.dtype(dtype).name not in native_dtypes(api):
        if api == 'segmentation.join_segmentations':
            args,options=default_call(api,a)
            _,error,scope=join_reference_rows(args,options,inspect.getsourcefile(getattr(original,name)))
            assert scope=='SOURCE_DERIVED_EARLY_ERROR_EQUIVALENCE'
            before=np.geterr(),np.geterrcall(),np.getbufsize()
            assert error[:2]==['builtins','TypeError']
            with pytest.raises(TypeError) as actual:getattr(native,name)(*args,**options)
            assert str(actual.value)==error[2]
            assert before==(np.geterr(),np.geterrcall(),np.getbufsize())
            join_reference_rows(args,options,inspect.getsourcefile(getattr(original,name)))
            return
        if api == 'metrics.contingency_table':
            # Float labels use the declared SciPy structural CSR suffix, not
            # the integer native counting entry or whole-function fallback.
            from label_pair_metrics_observer import observation
            args, options = default_call(api, a)
            before = observation(args, np)
            expected, expected_before, error = contingency_float_reference(dtype)
            assert inspect.isbuiltin(xx._core.label_pair_counts_preweighted)
            monkeypatch.setattr(xx._core, 'label_pair_counts_preweighted',
                                lambda *a, **k: pytest.fail('float labels reached native counts'))
            if error is not None:
                with pytest.raises(error[0]) as actual_error:
                    getattr(native, name)(*args, **options)
                assert str(actual_error.value) == error[1]
            else:
                actual = getattr(native, name)(*args, **options)
                assert_default_parity(api, np.dtype(dtype), actual, expected, image=a)
                assert observation(expected, np) == expected_before
            assert observation(args, np) == before
            return
        if api=='metrics.variation_of_information':
            # Float labels use the named SciPy coordinate suffix; native VI
            # entropy remains required, without integer-count ownership credit.
            from label_pair_metrics_observer import observation
            args,options=default_call(api,a);before=observation(args,np)
            expected=getattr(original,name)(*args,**options)
            monkeypatch.setattr(xx._core,'label_pair_counts_preweighted',lambda*a,**k:pytest.fail('float VI labels reached integer native counts'))
            actual=getattr(native,name)(*args,**options)
            assert_default_parity(api,np.dtype(dtype),actual,expected,image=a)
            assert observation(args,np)==before
            return
        if api=='draw.set_color':
            args,options=default_call(api,a)
            with pytest.raises(NotImplementedError):getattr(native,name)(*args,**options)
            return
        if api == 'util.map_array':
            args,options=default_call(api,a)
            with pytest.raises(TypeError) as expected:getattr(original,name)(*args,**options)
            with pytest.raises(TypeError) as actual:getattr(native,name)(*args,**options)
            assert str(actual.value)==str(expected.value)
            return
        if api == 'measure.moments_normalized':
            # Integer output cannot store the source's required low-order NaN.
            # Safe 4x4 geometry reaches that error; it is not numeric credit.
            args, options = default_call(api, a)
            with pytest.raises(ValueError) as expected:getattr(original, name)(*args, **options)
            with pytest.raises(ValueError) as actual:getattr(native, name)(*args, **options)
            assert str(actual.value) == str(expected.value)
            return
        if api in ('morphology.remove_small_objects','morphology.remove_small_holes','segmentation.relabel_sequential','measure.regionprops'):
            # These integer-only algorithms preserve original defined dtype
            # rejection; this is separate from successful default-case dtypes.
            with pytest.raises(TypeError) as expected:getattr(original,name)(a)
            with pytest.raises(TypeError) as actual:getattr(native,name)(a)
            assert str(actual.value)==str(expected.value)
            return
        if api=='transform.ifrt2':
            args,options=default_call(api,a)
            with pytest.raises(TypeError)as expected:getattr(original,name)(*args,**options)
            with pytest.raises(type(expected.value))as actual:getattr(native,name)(*args,**options)
            assert actual.value.args==expected.value.args
            return
        if api in ('segmentation.mark_boundaries','feature.draw_multiblock_lbp'):
            args,options=default_call(api,a)
            with pytest.raises((TypeError,ValueError))as expected:getattr(original,name)(*args,**options)
            with pytest.raises(type(expected.value))as actual:getattr(native,name)(*args,**options)
            assert str(actual.value)==str(expected.value)
            return
        with pytest.raises(NotImplementedError):
            getattr(native, name)(a)
        return
    args, options = default_call(api,a)
    if api in ('segmentation.mark_boundaries','feature.draw_multiblock_lbp'):
        expected=getattr(original,name)(*args,**options)
        actual=getattr(native,name)(*args,**options)
        assert_default_parity(api,np.dtype(dtype),actual,expected,image=a)
        return
    if api in ('util.slice_along_axes','util.view_as_blocks','util.view_as_windows'):
        from util_views_reference import reference
        from crop_observer import encode
        before=encode((args,options),np)
        expected,scope,row=reference(api,args,options,inspect.getsourcefile(getattr(original,name)),np)
        actual=getattr(native,name)(*args,**options)
        assert_default_parity(api,np.dtype(dtype),actual,expected)
        assert np.shares_memory(actual,args[0])and actual is not args[0]
        assert encode((args,options),np)==before
        return
    if api in ('util.crop','util.regular_grid'):
        from util_metadata_reference import reference
        from crop_observer import encode as crop_encode
        from regular_grid_observer import encode as grid_encode
        encode=crop_encode if api=='util.crop'else grid_encode
        before=(encode(args,np),dict(options))
        expected,scope,row=reference(api,args,options,inspect.getsourcefile(getattr(original,name)),np)
        actual=getattr(native,name)(*args,**options)
        assert_default_parity(api,np.dtype(dtype),actual,expected)
        if api=='util.crop':
            assert np.shares_memory(actual,args[0]) and actual is not args[0]
            assert actual.ctypes.data==args[0].ctypes.data and actual.base is args[0].base
        assert (encode(args,np),dict(options))==before
        reference(api,args,options,inspect.getsourcefile(getattr(original,name)),np)
        return
    if api=='segmentation.join_segmentations':
        expected,error,scope=join_reference_rows(args,options,inspect.getsourcefile(getattr(original,name)));assert error is None
        assert scope=='IDENTICAL_CAPTURED_ARRAYS_EFFECTIVE_FALSE_OPTION'
        actual=getattr(native,name)(*args,**options)
        assert_default_parity(api,np.dtype(dtype),actual,expected,image=a)
        join_reference_rows(args,options,inspect.getsourcefile(getattr(original,name))) # inputs and cached storage unchanged
        return
    if api=='draw.set_color':
        expected=set_color_default_result(getattr(original,name),args,options)
        actual=set_color_default_result(getattr(native,name),args,options)
        assert_default_parity(api,np.dtype(dtype),actual,expected,image=a)
        return
    try:
        expected = getattr(original, name)(*args,**options)
    except (IndexError,ValueError,RuntimeError) as error:
        # Some valid images have no converged threshold. Match the reference's
        # algorithm failure just as we match its successful numerical output.
        with pytest.raises(type(error)) as actual_error:
            getattr(native, name)(*args,**options)
        assert str(actual_error.value)==str(error)
        return
    actual = getattr(native, name)(*args,**options)
    if api=='segmentation.find_boundaries' or api in ('metrics.adapted_rand_error','metrics.contingency_table'):
        assert_default_parity(api,np.dtype(dtype),actual,expected,image=a);return
    if api=='segmentation.relabel_sequential':
        # Compare sparse mapping vectors without dense ArrayMap conversion.
        assert type(actual) is type(expected) is tuple and len(actual)==3
        np.testing.assert_array_equal(actual[0],expected[0])
        assert actual[0].dtype==expected[0].dtype
        for value,reference in zip(actual[1:],expected[1:]):
            assert type(value).__name__==type(reference).__name__=='ArrayMap'
            assert not type(value).__module__.startswith('skimage.')
            for field in ('in_values','out_values'):
                np.testing.assert_array_equal(getattr(value,field),getattr(reference,field))
                assert getattr(value,field).dtype==getattr(reference,field).dtype
        return
    if name=='rank_order' or api=='morphology.max_tree':
        assert type(actual) is type(expected) is tuple and len(actual)==len(expected)==2
        for value,reference in zip(actual,expected):
            assert value.dtype==reference.dtype and value.shape==reference.shape
            np.testing.assert_array_equal(value,reference)
        return
    assert np.asarray(actual).dtype == np.asarray(expected).dtype
    assert_default_parity(api,np.dtype(dtype),actual,expected,
                          image=args if api == 'measure.mesh_surface_area' else args[0] if api in DERIVED_MOMENTS_FUNCTIONS or api in COORDS_MOMENTS_FUNCTIONS or api == 'measure.perimeter' or api == 'measure.euler_number' else a)


@pytest.mark.parametrize('name', ['sobel', 'prewitt', 'scharr'])
@pytest.mark.parametrize('axis', [0, 1])
def test_gradient_axes(name, axis):
    a = np.random.default_rng(0).random((17, 23), dtype=np.float32)
    np.testing.assert_allclose(getattr(xx.filters, name)(a, axis=axis),
                               getattr(reference_filters, name)(a, axis=axis), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
def test_channel_axes(axis):
    a = np.random.default_rng(0).random((13, 17))
    rgb = xx.color.gray2rgb(a, channel_axis=axis)
    np.testing.assert_array_equal(rgb, reference_color.gray2rgb(a, channel_axis=axis))
    np.testing.assert_allclose(xx.color.rgb2gray(rgb, channel_axis=axis),
                               reference_color.rgb2gray(rgb, channel_axis=axis), atol=2e-13)


@pytest.mark.parametrize('signed', [False, True])
def test_invert_signed_and_nonfinite(signed):
    a = np.array([[0.0, -0.0, np.inf, -np.inf, np.nan, -1, 2]])
    expected = reference_util.invert(a, signed)
    actual = xx.util.invert(a, signed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_pipeline_can_swap_backend():
    a = np.random.default_rng(0).integers(0, 256, (37, 41, 3), dtype=np.uint8)
    def pipeline(backend):
        gray = backend.color.rgb2gray(a)
        smooth = backend.filters.gaussian(gray, 1.2, mode='reflect')
        edges = backend.filters.scharr(smooth, axis=0)
        return backend.morphology.opening(edges > backend.filters.threshold_mean(edges))
    np.testing.assert_array_equal(pipeline(xx), pipeline(skimage))


def test_explicit_fallback_for_unsupported_options():
    a = np.arange(81, dtype=np.uint16).reshape(9, 9)
    actual = xx.filters.gaussian(a, mode='wrap')
    np.testing.assert_allclose(actual, reference_filters.gaussian(a, mode='wrap'),rtol=2e-13,atol=2e-13)
    out = np.empty(a.shape)
    result = xx.filters.gaussian(a, out=out)
    assert result is out
    np.testing.assert_allclose(out, reference_filters.gaussian(a),rtol=2e-13,atol=2e-13)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(a, nbins=32)
    assert actual == reference_filters.threshold_otsu(a, nbins=32)


def test_upstream_adapter_obeys_fallback_policy():
    from skimagexx._compat import module_attribute
    a = np.arange(9.0).reshape(3, 3)
    # Exercise the upstream adapter directly so adding native algorithms never
    # turns this policy check into an assertion that a function must stay unported.
    upstream = module_attribute('filters', 'median')
    with pytest.raises(NotImplementedError):
        upstream(a)
    xx.set_fallback_enabled(True)
    with pytest.warns(xx.FallbackWarning):
        actual = upstream(a)
    np.testing.assert_array_equal(actual, reference_filters.median(a))


def test_mask_sequence_axis_and_histogram_fallback():
    xx.set_fallback_enabled(True)
    a = np.arange(81.0).reshape(9, 9)
    mask = a > 10
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, mask, axis=[0, 1], mode='nearest')
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, mask, axis=[0, 1], mode='nearest'))
    hist = np.histogram(a, bins=16)
    hist = (hist[0], (hist[1][:-1]+hist[1][1:])/2)
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.threshold_otsu(hist=hist)
    assert actual == reference_filters.threshold_otsu(hist=hist)
    # Preserve 0.26's handling of negative axes instead of normalizing it.
    with pytest.warns(xx.FallbackWarning):
        actual = xx.filters.sobel(a, axis=-1)
    np.testing.assert_array_equal(actual, reference_filters.sobel(a, axis=-1))


def test_integral_pipeline_can_swap_backend():
    image=np.random.default_rng(545).integers(0,256,(37,41,3),dtype=np.uint8)
    def pipeline(backend):
        gray=backend.color.rgb2gray(image)
        smooth=backend.filters.gaussian(gray,.75)
        return backend.transform.integral_image(smooth)
    actual,expected=pipeline(xx),pipeline(skimage)
    assert actual.dtype==expected.dtype and actual.strides==expected.strides
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
