"""Run small per-feature development gates using the existing parity assertions.

This is a development selector, not a replacement for product qualification or
an assertion that an upstream-only feature has numerical parity.
"""
import argparse
from collections import Counter
import fnmatch
import importlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def select_entries(entries, patterns):
    selected = []
    for pattern in patterns:
        matches = [entry for entry in entries if fnmatch.fnmatchcase(entry['api'], pattern)]
        if not matches:
            raise ValueError(f'No inventoried API matches {pattern!r}')
        selected.extend(matches)
    names = {entry['api'] for entry in selected}
    return [entry for entry in entries if entry['api'] in names]


def smoke_case(parameters, supported_dtypes):
    """Select existing nondegenerate fixtures; never synthesize smaller inputs."""
    if parameters.get('configuration') != 1:
        return False
    if 'shape' not in parameters:  # signature and registry checks
        return True
    dtype = parameters['dtype'].__name__
    return (parameters['shape'] in ((3, 19), (7, 19))
            and dtype in supported_dtypes)


class FeatureSelection:
    def __init__(self, entries, phase):
        self.entries = {entry['api']: entry for entry in entries}
        self.phase = phase
        self.results = []
        self.item_apis = {}
        self.collected = Counter()

    def pytest_collection_modifyitems(self, session, config, items):
        keep, deselected = [], []
        for item in items:
            parameters = getattr(getattr(item, 'callspec', None), 'params', {})
            api = (parameters['entry']['api']
                   if item.path.name == 'test_feature_api_surface.py' else None)
            if item.path.name == 'test_api.py' and 'native' in parameters and 'name' in parameters:
                api = parameters['native'].__name__.split('.', 1)[1] + '.' + parameters['name']
            elif item.path.name in ('test_dtype_aliases.py', 'test_dtype_metadata.py'):
                api = parameters.get('api')
            selected = api is None or api in self.entries
            if self.phase == 'smoke' and item.path.name == 'test_api.py':
                selected = (item.originalname == 'test_registry_matches_per_function_tests'
                            and parameters.get('configuration') == 1) or (
                    api in self.entries and smoke_case(parameters,
                        self.entries[api]['native_default_case_dtypes']))
            if selected:
                keep.append(item)
                self.item_apis[item.nodeid] = api
                if api:
                    self.collected[api] += 1
            else:
                deselected.append(item)
        items[:] = keep
        config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.outcome != 'passed':
            self.results.append(dict(nodeid=report.nodeid,
                                     api=self.item_apis.get(report.nodeid),
                                     stage=report.when, outcome=report.outcome))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('surface', 'smoke', 'contract'), default='smoke')
    parser.add_argument('--api', action='append', help='API name or glob; repeatable (default: all)')
    parser.add_argument('--package-dir', type=Path, help='Parent of an existing skimagexx package; no build')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/feature-tests/report.json')
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        parser.error('Run without -O: parity helpers require Python assertions')
    inventory = json.loads((ROOT / 'coverage/api.json').read_text(encoding='utf-8'))
    try:
        selected = select_entries(inventory['entries'], args.api or ['*'])
    except ValueError as error:
        parser.error(str(error))

    # Set external pool limits before NumPy or the original library is imported.
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.pop('PYTEST_ADDOPTS', None)
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    if args.package_dir:
        package_dir = args.package_dir.resolve(strict=True)
        if not (package_dir / 'skimagexx/__init__.py').is_file():
            parser.error('--package-dir must contain skimagexx/__init__.py')
        sys.path.insert(0, str(package_dir))
    original = importlib.import_module('skimage')
    candidate = importlib.import_module('skimagexx')
    registry = importlib.import_module('skimagexx.coverage')
    expected = {entry['api'] for entry in inventory['entries']
                if entry['implementation'] in ('native-partial', 'metadata')}
    observed = set(registry.native_functions())
    report = dict(phase=args.phase, reference_version=original.__version__,
                  package=str(Path(candidate.__file__).resolve()),
                  simd_disabled=os.environ.get('SKIMAGEXX_DISABLE_SIMD'),
                  inventory_entries=len(inventory['entries']), selected_entries=len(selected),
                  implementations=dict(Counter(entry['implementation'] for entry in selected)),
                  numerical_gaps=[entry['api'] for entry in selected if not entry['parity_test']],
                  status='preflight-failed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if original.__version__ != inventory['reference_version'] or observed != expected:
        report.update(missing_registered=sorted(expected-observed),
                      unexpected_registered=sorted(observed-expected))
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        parser.exit(2, f'Package/reference does not match inventory. See {args.report}\n')

    import pytest
    plugin = FeatureSelection(selected, args.phase)
    targets = [str(ROOT / 'tests/test_feature_api_surface.py')]
    if args.phase == 'smoke':
        targets.extend(str(ROOT / 'tests/test_api.py') + '::' + name for name in
                       ('test_registry_matches_per_function_tests', 'test_signatures', 'test_every_native_function'))
    elif args.phase == 'contract':
        targets.extend(str(ROOT / path) for path in sorted({
            entry['parity_test'] for entry in selected if entry['parity_test']}))
    os.chdir(ROOT)
    exit_code = pytest.main(['-q', '--tb=short', *targets], plugins=[plugin])
    report.update(status='passed' if exit_code == 0 else 'failed', exit_code=int(exit_code),
                  collected_by_api=dict(plugin.collected), results=plugin.results,
                  outcomes=dict(Counter(result['outcome'] for result in plugin.results)),
                  note='Surface checks are not numerical parity. Missing parity routes remain gaps. '
                       'Smoke reuses bounded existing cases; full contracts and benchmarks are separate.')
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Feature report: {args.report}')
    print(f'Selected APIs: {len(selected)}; missing parity routes: {len(report["numerical_gaps"])}')
    return int(exit_code)


if __name__ == '__main__':
    raise SystemExit(main())
