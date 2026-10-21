"""Verify report provenance/coverage and summarize original-library comparisons."""
from provenance import source_fingerprint
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    reports = [('Every covered public entry','function-coverage'),
               ('Additional integer widths','conversion-widths'),
               ('Non-default exposure parameters','exposure-parameters'),
               ('Color layouts and uint16 inputs','color-layouts'),
               ('Perceptual color parameters','perceptual-parameters'),
               ('Morphology footprints and output reuse','morphology-parameters'),
               ('Stain matrices and output widths','stain-parameters'),
               ('Lab distances and color routes','color-distance'),
               ('Histogram threshold parameters','threshold-parameters'),
               ('Local threshold windows and dtypes','local-threshold-parameters'),
               ('Alpha and grayscale layouts','alpha-parameters'),
               ('Laplace and Roberts layouts','edge-parameters'),
               ('Farid derivatives and boundaries','farid-parameters')]
    data = [(label,name,json.loads((root/'benchmarks'/f'{name}.json').read_text(encoding='utf-8')))
            for label,name in reports]
    fingerprint = source_fingerprint(root)
    source_hash = fingerprint['source_sha256']
    for _,name,report in data:
        if report['metadata'].get('source_hash_schema')!=fingerprint['source_hash_schema']:
            raise ValueError(f'{name}: provenance schema changed; rebuild and rerun this report')
        if report['metadata']['source_sha256'] != source_hash:
            raise ValueError(f'{name}: source changed; rebuild and rerun this report')
        if report['metadata']['fallback_enabled']:
            raise ValueError(f'{name}: fallback must be disabled')
        if not any(r['threads']==1 for r in report['results']):
            raise ValueError(f'{name}: one-worker comparisons are required')
    for field in ('reference','numpy','avx2'):
        if len({report['metadata'][field] for _,_,report in data}) != 1:
            raise ValueError(f'reports use different {field} settings')
    meta = data[0][2]['metadata']
    inventory = json.loads((root/'coverage/api.json').read_text(encoding='utf-8'))['entries']
    covered = {e['api'] for e in inventory if e['implementation'] in ('native-partial','metadata')}
    actual = {r['api'] for r in data[0][2]['results']}
    if covered != actual:
        raise ValueError('function report does not cover the current implementation inventory')
    for entry in inventory:
        if entry['api'] not in covered:
            continue
        serial_dtypes={r['dtype'] for r in data[0][2]['results'] if r['api']==entry['api'] and r['threads']==1}
        if serial_dtypes!=set(entry['native_default_case_dtypes']):
            raise ValueError(f'{entry["api"]}: missing one-worker default dtype comparisons')
    all_rows = [r for _,_,d in data for r in d['results']]
    serial = [r for r in all_rows if r['threads'] == 1 and r['backend'] == 'native']
    slower = [r for r in serial if r['speedup'] < 1]
    native_count = sum(e['implementation'] == 'native-partial' for e in inventory)
    metadata_count = sum(e['implementation'] == 'metadata' for e in inventory)
    lines = ['# Current performance verification','',
             f'Reference: original scikit-image {meta["reference"]}. '
             f'{len(all_rows)} comparisons on {meta["platform"]}; CPU: {meta["cpu"]}. '
             'Individual reports record timestamps, CPU, '
             'versions, raw samples and options. All reports match the current source/build SHA-256 '
             '(POSIX relative paths and LF line endings):',
             f'`{source_hash}`.','',
             '| Report | Cases | Worker counts |','|---|---:|---|']
    for label,name,d in data:
        threads = ', '.join(str(t) for t in sorted({r['threads'] for r in d['results']}))
        lines.append(f'| [{label}]({name}.md) | {len(d["results"])} | {threads} |')
    lines += ['',
              'Every workload checks values and dtype against original scikit-image before '
              'timing with fallback disabled. Integer-width conversion cases also require exact '
              'values and matching identity behavior. Timings include allocation, Python dispatch, '
              'integer normalization and any needed input copying. Inputs/imports/I/O are excluded. '
              'Reference thread pools use one thread. Native worker counts are explicit.','',
              'Examples from the 1024² one-worker default-parameter comparisons '
              '(color-space routing is RGB to HSV; CIE76 compares raw Lab values '
              'with a channel-reversed second input):','',
              '| Operation | Input | Original ms | C++ ms | Reference / C++ |',
              '|---|---|---:|---:|---:|']
    examples = {('filters.gaussian','uint8'),('filters.sobel','uint8'),('filters.threshold_otsu','uint8'),
                ('color.rgb2yuv','uint8'),('color.rgb2xyz','uint8'),('color.rgb2xyz','float32'),
                ('color.xyz2rgb','float32'),('color.rgb2ycbcr','float64'),
                ('color.rgb2hsv','float32'),('color.hsv2rgb','float32'),
                ('color.rgb2lab','float64'),('color.lab2lch','float64'),
                ('color.rgb2luv','float32'),('color.luv2xyz','float64'),('color.luv2rgb','float32'),
                ('color.rgb2hed','uint8'),('color.separate_stains','float32'),('color.combine_stains','float64'),
                ('color.deltaE_cie76','float32'),('color.deltaE_cie76','float64'),('color.convert_colorspace','float32'),
                ('color.deltaE_ciede94','float32'),('color.deltaE_ciede2000','float32'),('color.deltaE_cmc','float32'),
                ('morphology.erosion','uint8'),('morphology.opening','float32'),
                ('morphology.binary_dilation','float64')}
    for r in data[0][2]['results']:
        if r['size'] == 1024 and r['threads'] == 1 and (r['api'],r['dtype']) in examples:
            lines.append(f'| {r["api"]} | {r["dtype"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    morphology_report=next(report for _,name,report in data if name=='morphology-parameters')
    boolean_rows=[r for r in morphology_report['results'] if r['threads']==1 and r['dtype']=='bool' and r['workload']=='rectangle-31' and not r['api'].startswith('morphology.binary_')]
    lines+=['','Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):','',
            '| Operation | Image | Original ms | C++ ms | Reference / C++ |','|---|---|---:|---:|---:|']
    for r in boolean_rows:
        lines.append(f'| {r["api"]} | {r["size"]}² | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    lines+=['','The [fixed-workload tuning experiment](morphology-tuning.md) preserves '
            'separate historical before/after C++ snapshots. Its cases are not added to this current-source total.']
    lines += ['',f'{len(serial)-len(slower)} of {len(serial)} one-worker native processing workloads '
              'measured faster than the reference. These are workload cases, including repeated '
              'functions and parameter variants. Identity returns, copies and dtype metadata are '
              'excluded from this count and labeled separately in the full reports.','',
              f'{len(slower)} one-worker processing workloads remained slower:','',
              '| Operation | Input | Shape | Layout | Workload | Reference / C++ |',
              '|---|---|---:|---|---|---:|']
    for r in slower:
        dtype = r.get('dtype',r.get('input_dtype',''))
        shape="×".join(map(str,r["shape"])) if "shape" in r else f'{r["size"]}²'
        lines.append(f'| {r["api"]} | {dtype} | {shape or "scalar"} | {r.get("layout","standard")} | {r.get("workload","default")} | {r["speedup"]:.4f}x |')
    lines += ['',
              'The color port fuses integer normalization, matrix arithmetic and sRGB transfer. '
              'Uniform floating layouts now avoid a full contiguous copy: planar input uses '
              'contiguous channel loads, other uniform strides use gathers, and scalar tails '
              'preserve the same arithmetic. General nonuniform layouts and unaligned arrays '
              'can still require copying. Simple matrix transforms remain sensitive to dispatch, '
              'shuffling and memory traffic; this run has no hardware-counter profiling to '
              'quantify their individual contribution. Further optimization remains necessary.','',
              'Morphology reuses row mappings across border pixels and stops binary reductions '
              'when the result is known for every SIMD lane. Large rectangular footprints use '
              'linear-work prefix/suffix scans; small input types use narrower working buffers '
              'when conversion preserves extrema, while 64-bit integer scans retain double '
              'buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction '
              'use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD '
              'packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan '
              'passes. Other storage and boundary constants retain the general native path.','',
              'Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. '
              'Cached three-value white metadata preserves NumPy dot rounding at singular '
              'denominators. Float32 inverse cube powers retain libm rounding because sRGB '
              'amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. '
              'Cross-platform real-color stress tests separately bound the one-ULP luminance '
              'difference and its propagation through XYZ/sRGB; see the numerical contract.','',
              'Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. '
              'The [fixed linear-color experiment](linear-color-tuning.md) retains historical '
              'before/after timings separately from this aggregate. Stain kernels fuse input '
              'normalization, logarithmic scaling, matrix arithmetic and reconstruction. '
              'Uint8 separation uses a log lookup table; floating layouts use direct loads '
              'or gathers. Float32 log values retain float32 precision before NumPy-compatible '
              'promotion, so output float64 does not imply a float64 input error budget.','',
              'CIE76 fuses raw Lab conversion, subtraction, squared components and square root '
              'without temporary image arrays. Uniform floating layouts use direct loads or gathers; '
              'integer/mixed blocks convert raw values before vector arithmetic. Aligned nonuniform '
              'CIE76 views now use per-row offsets and signed strides instead of full image copies. '
              'Weighted Lab metrics preserve dtype-specific rounding, double hue cancellation '
              'and promoted trigonometric stages. Float32 CIEDE2000 uses the documented '
              'hue-roundoff budget, while float64 retains strict comparisons. Color-space routing '
              'preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter '
              'report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.','',
              'The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves '
              'historical before/after snapshots separately from these current-source totals.','',
              'The [fixed nonuniform-layout experiment](distance-layout-tuning.md) records '
              'the separate one-worker copy-removal comparison, including all raw samples.','',
              'Yen, Triangle and ISODATA use native histograms and selectors. Floating bin '
              'edges and centers are small NumPy metadata arrays, while image scans, bin '
              'assignment, accumulation and threshold selection run in C++. AVX2 computes '
              'floating bin indices; exact edge corrections preserve histogram parity. '
              'Yen retains the installed NumPy float32 log rounding because near ties can '
              'change the selected bin. Supplied-histogram cases measure selection without '
              'an image scan; degenerate warning and error contracts are parity-tested.','',
              'Niblack and Sauvola share reflected input preparation, ordered float64 '
              'integral sums and squared sums. Independent prefix columns and output pixels '
              'use AVX2 while preserving each cumulative sequence. Float32 window sums round '
              'before division; strong window/weight/range metadata retains upstream promotion. '
              'The parameter report includes all eleven input types, noncontiguous layouts, '
              'large/anisotropic windows and singleton-depth N-D volumes.','',
              'RGBA compositing fuses integer normalization, blending and clipping with strict SIMD arithmetic. '
              'Grayscale RGB/RGBA expansion replicates raw storage and normalizes endian components directly. '
              'Both use physical stride order and partition pixels across the global worker limit; '
              'the alpha report includes planar, Fortran, negative and stepped layouts plus background/alpha options. '
              'The [fixed grayscale expansion experiment](alpha-tuning.md) retains initial, optimized and '
              'confirmation observations separately from this aggregate.','',
              'Laplace uses sparse neighbor loads in scipy coefficient order, retaining singleton-axis terms. '
              'Roberts fuses cross differences and rounded magnitude stages. Half and endian conversion, '
              'mask erosion/broadcast multiplication, layout preparation and allocation are included in the edge report. '
              'The [fixed column experiment](edge-tuning.md) retains separate baseline, optimized and '
              'confirmation snapshots. The [metadata experiment](edge-metadata-tuning.md) '
              'retains the separate validation-overhead comparison.','',
              'Measurements use ordinary system load without CPU affinity or frequency locking. '
              'Small differences near 1x can change between runs. These are measured comparisons '
              'on this machine, not universal speed guarantees. CI runs parity and benchmark smoke '
              'checks on Windows/Linux without asserting speed ratios on shared runners.','',
              f'Coverage remains partial: {native_count} native image entries, {metadata_count} '
              f'metadata helpers, {len(inventory)-len(covered)} inventoried entries still pending. '
              'See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and '
              '[documented input contracts](../README.md). Run `python benchmarks/verify.py` '
              'to rebuild and reproduce all thirteen reports, including this summary. '
              'Use `--threads 1` for serial-only verification. The summary rejects mixed source '
              'hashes, missing APIs and absent one-worker comparisons.']
    (root/'benchmarks/STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(all_rows)} comparisons verified; {len(serial)-len(slower)}/{len(serial)} serial workloads faster; {source_hash}')


if __name__ == '__main__':
    main()
