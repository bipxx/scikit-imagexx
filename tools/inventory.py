"""Inventory the pinned upstream API, with honest implementation/test status."""
import importlib
import functools
import inspect
import json
import re
from pathlib import Path
import skimage
import numpy as np
from skimagexx.coverage import native_functions, native_dtypes, execution_kind, native_classes, class_contract, class_metadata_providers

MODULES = ('color', 'data', 'draw', 'exposure', 'feature', 'filters', 'filters.rank', 'filters.rank.bilateral', 'filters.rank.generic', 'future',
           'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
           'restoration', 'segmentation', 'transform', 'util')

DEDICATED_EVIDENCE = {}
for _names, _test, _benchmark in (
    (('measure.inertia_tensor',),'tests/test_inertia_tensor_public.py','benchmarks/coverage.py'),
    (('measure.inertia_tensor_eigvals',),'tests/test_inertia_eigvals_public.py','benchmarks/coverage.py'),
    (('measure.moments_normalized','measure.moments_hu'),'tests/test_moments_derived_public.py','benchmarks/coverage.py'),
    (('measure.moments','measure.moments_central','measure.centroid'),'tests/test_moments_public.py','benchmarks/coverage.py'),
    (('filters.window',),'tests/test_window_public.py','benchmarks/coverage.py'),
    (('filters.filter_forward','filters.filter_inverse','filters.wiener'),'tests/test_lpi_filter_public.py','benchmarks/coverage.py'),
    (('filters.butterworth',),'tests/test_butterworth_public.py','benchmarks/coverage.py'),
    (tuple('measure.'+name for name in ('intersection_coeff','manders_coloc_coeff','manders_overlap_coeff')),
     'tests/test_colocalization_public.py','benchmarks/coverage.py'),
    (tuple('morphology.'+name for name in ('ellipse','octagon','octahedron','star','footprint_from_sequence')),
     'tests/test_footprint_shapes_public.py','benchmarks/coverage.py'),
    (('morphology.h_maxima','morphology.h_minima'),'tests/test_h_extrema_public.py','benchmarks/coverage.py'),
    (('metrics.normalized_mutual_information',),'tests/test_normalized_mutual_information_public.py','benchmarks/coverage.py'),
    (('morphology.reconstruction',),'tests/test_reconstruction_public.py','benchmarks/coverage.py'),
    (tuple('morphology.'+name for name in ('max_tree','max_tree_local_maxima','area_opening','area_closing','diameter_opening','diameter_closing')),'tests/test_max_tree_public.py','benchmarks/coverage.py'),
    (('restoration.denoise_tv_bregman',),'tests/test_tv_bregman_public.py','benchmarks/coverage.py'),
    (('restoration.denoise_tv_chambolle',),'tests/test_tv_chambolle_contracts.py','benchmarks/tv_chambolle.py'),
    (('transform.matrix_transform',),'tests/test_matrix_transform_public.py','benchmarks/coverage.py'),
    (('transform.resize','transform.rescale'),'tests/test_resize_contract.py','benchmarks/resize.py'),
    (('metrics.mean_squared_error','metrics.normalized_root_mse','metrics.peak_signal_noise_ratio'),
     'tests/test_metrics_reductions.py','benchmarks/metrics_reductions.py'),
    (tuple('morphology.'+name for name in ('footprint_rectangle','diamond','disk','ball','mirror_footprint','pad_footprint')),
     'tests/test_footprint_contracts.py','benchmarks/footprints.py'),
    (('measure.label','morphology.label'),'tests/test_connected_components_contracts.py','benchmarks/components.py'),
    (('morphology.remove_small_objects','morphology.remove_small_holes',
      'segmentation.clear_border','segmentation.relabel_sequential'),
     'tests/test_component_consumers_public.py','benchmarks/component_consumers.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.generic')
           for name in ('minimum','maximum','mean','sum','pop')),
     'tests/test_rank_public.py','benchmarks/rank.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.generic')
           for name in ('gradient','median','modal','entropy')),
     'tests/test_rank_public.py','benchmarks/rank_distribution.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.generic')
           for name in ('autolevel','equalize','enhance_contrast','threshold')),
     'tests/test_rank_intensity_public.py','benchmarks/rank_intensity.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.generic')
           for name in ('geometric_mean','noise_filter','subtract_mean','windowed_histogram')),
     'tests/test_rank_statistics_public.py','benchmarks/rank_statistics.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.bilateral')
           for name in ('mean_bilateral','pop_bilateral','sum_bilateral')),
     'tests/test_rank_bilateral_public.py','benchmarks/rank_bilateral.py'),
    (tuple(namespace+'.'+name for namespace in ('filters.rank','filters.rank.generic')
           for name in ('otsu','majority')),
     'tests/test_rank_decisions_public.py','benchmarks/rank_decisions.py'),
    (tuple('filters.rank.'+name for name in ('autolevel_percentile', 'enhance_contrast_percentile', 'gradient_percentile', 'mean_percentile', 'pop_percentile', 'subtract_mean_percentile', 'threshold_percentile', 'sum_percentile', 'percentile')),
     'tests/test_rank_percentile_public.py','benchmarks/rank_percentile.py'),
):
    for _name in _names:DEDICATED_EVIDENCE[_name]=(_test,_benchmark)


def signature_of(value):
    try:
        signature=inspect.signature(value)
        text=re.sub(r' at 0x[0-9a-fA-F]+', '', str(signature))
        # NumPy 1.24 exposes np.max as amax; later releases display max.
        # Normalize only the identical default object, not arbitrary callables.
        if any(parameter.default is np.max for parameter in signature.parameters.values()):
            text=text.replace('<function amax>', '<function max>')
        return text
    except (TypeError,ValueError):return None


def class_members(value,path):
    """Static descriptors only; never construct an instance or evaluate getters."""
    owners={}
    for base in value.__mro__:
        for name in vars(base):owners.setdefault(name,base)
    members=[]
    contract=class_contract(path) if path in (*native_classes(),*class_metadata_providers()) else None
    for name,owner in sorted(owners.items()):
        if owner is object:continue
        special=name.startswith('__') and name.endswith('__')
        if name.startswith('_') and not special:continue
        descriptor=inspect.getattr_static(value,name)
        if isinstance(descriptor,(property,functools.cached_property)):
            kind='property';signature=None
        elif isinstance(descriptor,classmethod):
            kind='classmethod';signature=signature_of(classmethod.__get__(descriptor,None,value))
        elif isinstance(descriptor,staticmethod):
            kind='staticmethod';signature=signature_of(descriptor.__func__)
        elif inspect.isroutine(descriptor):
            kind='constructor' if name=='__init__' else 'method';signature=signature_of(descriptor)
        elif not special and inspect.isdatadescriptor(descriptor):
            kind='attribute';signature=None
        else:continue
        supported=contract is not None and name in contract['members']
        members.append({'api':path+'.'+name,'kind':kind,'signature':signature,
                        'defined_by':owner.__module__+'.'+owner.__qualname__,
                        'inherited':owner is not value,
                        'implementation':('exception-metadata-member' if path in class_metadata_providers() else 'native-class-member-partial') if supported else 'upstream-only',
                        'execution_kind':contract['members'][name] if supported else 'upstream',
                        'parity_test':contract['public_parity_test'] if supported else None,
                        'benchmark':None})
    return members


def inventory():
    native = set(native_functions())
    classes=set(native_classes())
    providers=set(class_metadata_providers())
    entries = []
    for module in MODULES:
        upstream = importlib.import_module('skimage.'+module)
        names=set(getattr(upstream, '__all__', dir(upstream)))
        # Still callable in the pinned release, but removed from its export list.
        if module=='morphology':
            names.update(('binary_erosion','binary_dilation','binary_opening','binary_closing'))
        if module=='filters.rank.generic':
            # Source-backed legacy aliases of canonical rank exports, not
            # imported NumPy/SciPy/warnings dependencies or extra algorithms.
            names.update(('sum','majority','windowed_histogram'))
        for name in sorted(n for n in names if not n.startswith('_')):
            value = getattr(upstream, name)
            if not callable(value):
                continue
            path = module+'.'+name
            evidence=DEDICATED_EVIDENCE.get(path,('tests/test_api.py::test_every_native_function','benchmarks/coverage.py'))
            entries.append({'api': path, 'kind': 'class' if inspect.isclass(value) else 'callable',
                            'signature': signature_of(value),
                            'implementation': ('metadata' if execution_kind(path) == 'metadata' else 'native-partial') if path in native else 'upstream-only',
                            'native_default_case_dtypes': list(native_dtypes(path)) if path in native else [],
                            'execution_kind': execution_kind(path) if path in native else 'upstream',
                            'parity_test': evidence[0] if path in native else None,
                            'benchmark': evidence[1] if path in native else None})
            if module in ('filters.rank.generic','filters.rank.bilateral'):
                canonical=getattr(importlib.import_module('skimage.filters.rank'),name)
                if value is not canonical:raise ValueError(f'Rank alias identity changed: {path}')
                entries[-1]['alias_of']='filters.rank.'+name
            if inspect.isclass(value):
                entries[-1]['members']=class_members(value,path)
                if path in classes|providers:
                    contract=class_contract(path)
                    entries[-1].update(implementation=contract['implementation'],execution_kind='exception-metadata' if path in providers else 'stateful-class',
                                      parity_test=contract['public_parity_test'],class_contract=contract)
    inventoried = {entry['api'] for entry in entries}
    missing = (native|classes|providers)-inventoried
    if missing:
        raise ValueError(f'Native registry entries absent from upstream inventory: {sorted(missing)}')
    return {'reference_version': skimage.__version__, 'modules': list(MODULES),
            'class_member_scope':'Static public descriptors and custom data-model methods, including inherited members. Instance fields and runtime-generated attributes remain unassessed. Availability and signatures are not numerical parity.',
            'entries': entries}


def main():
    result = inventory()
    root = Path(__file__).resolve().parents[1]
    path = root/'coverage/api.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    entries = result['entries']
    implemented = [e for e in entries if e['implementation'] in ('native-partial', 'metadata','native-class-partial','exception-metadata-provider')]
    kernels = sum(e['implementation'] == 'native-partial' for e in entries)
    metadata = sum(e['implementation'] == 'metadata' for e in entries)
    class_count=sum(e['implementation']=='native-class-partial' for e in entries)
    exception_count=sum(e['implementation']=='exception-metadata-provider' for e in entries)
    lines = ['# API implementation inventory', '', f'Reference: scikit-image {skimage.__version__}.', '',
             f'{len(entries)} public callable names inventoried in {len(MODULES)} modules; '
             f'{kernels} have partially native image implementations and {metadata} '
             f'{"has" if metadata == 1 else "have"} metadata-only execution. '
             f'{class_count} have separate partial native class/state implementations and {exception_count} have exception metadata providers; these never increase image-kernel counts. '
             'Covered entries link to per-function test and benchmark files; '
             'these paths do not prove a passing contract or measured performance.', '',
             'An inventoried name is not evidence of parity. `upstream-only` entries have no native implementation, '
             'no dedicated parity test, and no native performance result. Modules outside this inventory remain unassessed. '
             'Native entries support a documented subset of input/options; unsupported calls can use explicit upstream fallback. '
             'Metadata entries return small values/arrays without running an image kernel. Identity-preserving conversion calls '
             'can return the input without processing pixels.', '',
             '| API | Implementation | Dedicated parity test file | Benchmark file |', '|---|---|---|---|']
    for entry in entries:
        lines.append(f"| `{entry['api']}` | {entry['implementation']} | {'yes (partial input contract)' if entry['parity_test'] else 'pending'} | {'yes' if entry['benchmark'] else 'pending'} |")
    classes=[entry for entry in entries if entry['kind']=='class']
    members=[member for entry in classes for member in entry['members']]
    lines+=['','## Class member inventory','',
            f'{len(classes)} exported class names expose {len(members)} statically discoverable member paths, including inherited methods and properties. '
            'Their signatures and defining classes are recorded in `api.json`. '
            'Registered class/member implementations and provider scope are separate from passing parity or measured performance. '
            'Unregistered members still require implementation, numerical/state parity and benchmarks; none increase image coverage. '
            'Constructors are not executed and property getters are not evaluated during inventory. '
            'Instance fields and runtime-generated attributes remain unassessed.','',
            '| Class | Static member paths | Native member parity |','|---|---:|---|']
    for entry in classes:lines.append(f"| `{entry['api']}` | {len(entry['members'])} | {'partial implementation; evidence separate' if entry['implementation']=='native-class-partial' else 'pending'} |")
    (path.parent/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(entries)} API names; {kernels} native-partial image, {metadata} metadata, {class_count} native-class-partial, {exception_count} exception metadata providers; {len(entries)-len(implemented)} pending')
    print(f'{len(classes)} class names; {len(members)} static member paths; execution evidence tracked separately')


if __name__ == '__main__':
    main()
