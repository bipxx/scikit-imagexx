"""Verify report provenance/coverage and summarize original-library comparisons."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    reports = [('Every covered public entry','function-coverage'),
               ('Additional integer widths','conversion-widths'),
               ('Non-default exposure parameters','exposure-parameters'),
               ('Color layouts and uint16 inputs','color-layouts'),
               ('Perceptual color parameters','perceptual-parameters'),
               ('Morphology footprints and output reuse','morphology-parameters')]
    data = [(label,name,json.loads((root/'benchmarks'/f'{name}.json').read_text(encoding='utf-8')))
            for label,name in reports]
    digest = hashlib.sha256()
    for p in sorted(list((root/'src').glob('*'))+list((root/'include/skimagexx').glob('*'))+
                    list((root/'python/skimagexx').glob('*.py'))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    source_hash = digest.hexdigest()
    for _,name,report in data:
        if report['metadata']['source_sha256'] != source_hash:
            raise ValueError(f'{name}: source changed; rebuild and rerun this report')
        if report['metadata']['fallback_enabled']:
            raise ValueError(f'{name}: fallback must be disabled')
    for field in ('reference','numpy','avx2'):
        if len({report['metadata'][field] for _,_,report in data}) != 1:
            raise ValueError(f'reports use different {field} settings')
    meta = data[0][2]['metadata']
    inventory = json.loads((root/'coverage/api.json').read_text(encoding='utf-8'))['entries']
    covered = {e['api'] for e in inventory if e['implementation'] in ('native-partial','metadata')}
    actual = {r['api'] for r in data[0][2]['results']}
    if covered != actual:
        raise ValueError('function report does not cover the current implementation inventory')
    all_rows = [r for _,_,d in data for r in d['results']]
    serial = [r for r in all_rows if r['threads'] == 1 and r['backend'] == 'native']
    slower = [r for r in serial if r['speedup'] < 1]
    native_count = sum(e['implementation'] == 'native-partial' for e in inventory)
    metadata_count = sum(e['implementation'] == 'metadata' for e in inventory)
    lines = ['# Current performance verification','',
             f'Reference: original scikit-image {meta["reference"]}. '
             f'{len(all_rows)} comparisons on {meta["platform"]}; CPU: {meta["cpu"]}. '
             'Individual reports record timestamps, CPU, '
             'versions, raw samples and options. All reports match the current source SHA-256:',
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
              'Examples from the 1024² one-worker default-parameter comparisons:','',
              '| Operation | Input | Original ms | C++ ms | Reference / C++ |',
              '|---|---|---:|---:|---:|']
    examples = {('filters.gaussian','uint8'),('filters.sobel','uint8'),('filters.threshold_otsu','uint8'),
                ('color.rgb2yuv','uint8'),('color.rgb2xyz','uint8'),('color.rgb2xyz','float32'),
                ('color.xyz2rgb','float32'),('color.rgb2ycbcr','float64'),
                ('color.rgb2hsv','float32'),('color.hsv2rgb','float32'),
                ('color.rgb2lab','float64'),('color.lab2lch','float64'),
                ('morphology.erosion','uint8'),('morphology.opening','float32'),
                ('morphology.binary_dilation','float64')}
    for r in data[0][2]['results']:
        if r['size'] == 1024 and r['threads'] == 1 and (r['api'],r['dtype']) in examples:
            lines.append(f'| {r["api"]} | {r["dtype"]} | {r["reference_ms"]:.4f} | {r["cpp_ms"]:.4f} | {r["speedup"]:.2f}x |')
    lines += ['',f'{len(serial)-len(slower)} of {len(serial)} one-worker native processing workloads '
              'measured faster than the reference. These are workload cases, including repeated '
              'functions and parameter variants. Identity returns, copies and dtype metadata are '
              'excluded from this count and labeled separately in the full reports.','',
              f'{len(slower)} one-worker processing workloads remained slower:','',
              '| Operation | Input | Size | Layout | Reference / C++ |',
              '|---|---|---:|---|---:|']
    for r in slower:
        dtype = r.get('dtype',r.get('input_dtype',''))
        lines.append(f'| {r["api"]} | {dtype} | {r["size"]}² | {r.get("layout","standard")} | {r["speedup"]:.2f}x |')
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
              'use SIMD. Boolean grayscale rectangles remain a bottleneck: their full scan '
              'passes can cost more than the reference on nearly uniform intermediate images.','',
              'Measurements use ordinary system load without CPU affinity or frequency locking. '
              'Small differences near 1x can change between runs. These are measured comparisons '
              'on this machine, not universal speed guarantees. CI runs parity and benchmark smoke '
              'checks on Windows/Linux without asserting speed ratios on shared runners.','',
              f'Coverage remains partial: {native_count} native image entries, {metadata_count} '
              f'metadata helpers, {len(inventory)-len(covered)} inventoried entries still pending. '
              'See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and '
              '[documented input contracts](../README.md). Reproduce all six reports, then run '
              '`python benchmarks/summarize.py` to check their source hashes and API coverage '
              'before regenerating this summary.']
    (root/'benchmarks/STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(all_rows)} comparisons verified; {len(serial)-len(slower)}/{len(serial)} serial workloads faster; {source_hash}')


if __name__ == '__main__':
    main()
