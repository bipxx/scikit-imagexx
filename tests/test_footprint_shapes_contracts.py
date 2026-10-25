"""Stdlib-only prospective contract checks; no original or candidate calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_factory():
    spec = importlib.util.spec_from_file_location('footprint_shapes_cases',
                                                 ROOT / 'tests/footprint_shapes_cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProspectiveContracts(unittest.TestCase):
    def test_passive_recursive_entry_counters(self):
        path = ROOT / 'tests/footprint_shapes_reference.py'
        spec = importlib.util.spec_from_file_location('prospective_reference', path)
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        def recursive(n):
            if n: return recursive(n-1)
            return 17
        counts, profile = reference.entry_profiler(recursive.__code__)
        self.assertIsNone(sys.getprofile())
        sys.setprofile(profile)
        try: result = recursive(2)
        finally: sys.setprofile(None)
        self.assertEqual(result, 17)
        self.assertEqual(counts, dict(outer_call=1, outer_return=1, nested_call=2,
                                     nested_return=2, total_call=3, total_return=3))
        # A later ordinary call is another outer entry, not a lingering child.
        sys.setprofile(profile)
        try: recursive(0)
        finally: sys.setprofile(None)
        self.assertEqual(counts['outer_call'], 2)
        self.assertEqual(counts['outer_return'], 2)

    def test_source_declared_all_generator_recursion(self):
        rows = load_factory().cases()
        def expected(row):
            api, mode = row['api'], row['kwargs'].get('decomposition')
            return int((api == 'ellipse' and mode == 'crosses') or
                       (api == 'octahedron' and mode == 'sequence') or
                       (api == 'octagon' and mode == 'sequence' and
                        row['args'][0] <= 2 and row['args'][1] <= 2))
        self.assertEqual({api:sum(expected(x) for x in rows if x['api'] == api)
                          for api in load_factory().APIS},
                         dict(ellipse=20, octagon=6, octahedron=17, star=0,
                              footprint_from_sequence=0))
        tree = ast.parse((ROOT / 'tests/footprint_shapes_reference.py').read_bytes())
        observe = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'observe')
        assigned = next(x.value for x in observe.body if isinstance(x, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'nested' for t in x.targets))
        expression = ast.Expression(assigned)
        for row in rows:
            self.assertEqual(eval(compile(expression, '<source-counter-rule>', 'eval'),
                                  {'int': int}, {'spec': row, 'api':row['api'],
                                                'mode':row['kwargs'].get('decomposition')}), expected(row))

    def test_unique_partition(self):
        factory = load_factory()
        rows = factory.cases()
        self.assertEqual(len(rows), 226)
        self.assertEqual(len({x['id'] for x in rows}), len(rows))
        self.assertEqual({x['api'] for x in rows}, set(factory.APIS))
        self.assertTrue(all(x['policy'] == 'default' for x in rows))
        self.assertEqual(len(factory.future_cases()), 20)
        self.assertFalse(set(x['id'] for x in rows) &
                         set(x['id'] for x in factory.future_cases()))

    def test_specs_are_copy_isolated_and_json(self):
        factory = load_factory()
        rows = factory.cases(); rows[0]['args'][0] = 1000000
        self.assertEqual(factory.cases()[0]['args'], [0, 0])
        json.dumps(factory.cases(), allow_nan=False)

    def test_sequence_frontier_is_explicit(self):
        rows = load_factory().cases()
        self.assertEqual(sum(x['api'] == 'footprint_from_sequence' for x in rows), 67)
        for pattern in ('first-zero', 'later-zero', 'even', 'cube', 'asymmetric'):
            self.assertEqual(sum(x['id'].startswith('footprint_from_sequence/' + pattern + '-')
                                 for x in rows), 5)
        self.assertEqual(sum('/invalid-' in x['id'] and x['api'] == 'footprint_from_sequence'
                             for x in rows), 7)

    def test_generator_extent_and_future_bound(self):
        factory = load_factory()
        for row in factory.future_cases():
            args, api = row['args'], row['api']
            if api == 'ellipse': n = (2*args[0]+1)*(2*args[1]+1)
            elif api == 'octagon': n = (args[0]+2*args[1])**2
            elif api == 'octahedron': n = (2*args[0]+1)**3
            elif api == 'star': n = (2*args[0]+1+2*(args[0]//2))**2
            else: n = (1+2*row['kwargs']['future_repetitions'])**2
            self.assertLessEqual(n, 262144)
            self.assertLessEqual(192*n + 16*1024**2, 256*1024**2)

    def test_exact_spec_digest_rejects_mutation(self):
        rows = load_factory().cases()
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
        original = digest(rows)
        rows[0]['args'][0] = 1
        self.assertNotEqual(original, digest(rows))
        rows = load_factory().cases(); rows.pop()
        self.assertNotEqual(original, digest(rows))

    def test_no_scientific_import_or_timer(self):
        for name in ('footprint_shapes_cases.py', 'footprint_shapes_reference.py'):
            tree = ast.parse((ROOT / 'tests' / name).read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(x.name.split('.')[0] in
                        {'numpy', 'scipy', 'skimage', 'skimagexx', 'time'} for x in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module.split('.')[0],
                                     {'numpy', 'scipy', 'skimage', 'skimagexx', 'time'})

    def test_prospective_plan_pins(self):
        plan = json.loads((ROOT / 'coverage/contracts/footprint-shapes-plan.json').read_bytes())
        self.assertEqual(plan['native_admission'], 'pending')
        self.assertEqual(plan['original_calls'], 0)
        for path, digest in plan['owned_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        payload = json.dumps(load_factory().cases(), sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), plan['cases_sha256'])

    def test_journal_precedes_postchecks_and_no_policy_setter(self):
        tree = ast.parse((ROOT / 'tests/footprint_shapes_reference.py').read_bytes())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'observe')
        journal = next(x for x in fn.body if isinstance(x, ast.Expr) and
                       isinstance(x.value, ast.Call) and isinstance(x.value.func, ast.Name)
                       and x.value.func.id == 'journal')
        self.assertTrue(all(x.lineno > journal.lineno for x in fn.body if isinstance(x, ast.Assert)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute): self.assertNotIn(node.attr, ('seterr', 'seterrcall'))


if __name__ == '__main__': unittest.main()
