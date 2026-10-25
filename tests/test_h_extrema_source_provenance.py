"""Stdlib metadata controls only: no H module/test body or image execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h_closed_metadata_reader', ROOT / 'tests/h_extrema_source_provenance.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


class HSourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / H.RECORD).read_bytes())
        self.current = {name: (ROOT / name).read_bytes() for name in H.ALLOWED}
        self.reader = (ROOT / 'tests/h_extrema_source_provenance.py').read_bytes()

    def validate(self):
        return H.verify_record(self.record, self.current, self.reader)

    def test_exact_full_four_file_inverse(self):
        self.assertTrue(self.validate())

    def test_compile_all_modules_without_execution(self):
        for name, raw in self.current.items():
            compile(ast.parse(raw), '<current-noexec>/' + name, 'exec')
            old = self.record['files'][name]['original_source_utf8'].encode()
            compile(ast.parse(old), '<historical-noexec>/' + name, 'exec')

    def test_six_actual_test_lineage_checkpoints(self):
        self.assertEqual(len(self.record['historical_test_lineage']), 6)
        self.assertEqual(tuple(x['revision'] for x in self.record['historical_test_lineage']), H.LINEAGE)
        self.assertEqual(self.record['historical_test_lineage'][0]['raw_sha256'],
                         json.loads((ROOT / 'coverage/contracts/h-extrema.json').read_bytes())['test_canonical_lf_sha256'])

    def test_strict_tail_preserved_verbatim(self):
        name = 'tests/test_h_extrema_contracts.py'
        old = self.record['files'][name]['original_source_utf8'].encode()
        old_body = H._body(old, ast.parse(old), 'test_strict_native_pending')
        new_body = H._body(self.current[name], ast.parse(self.current[name]), 'test_strict_native_pending')
        tail = old_body[old_body.index(b'    old=xx.get_num_threads()'):]
        self.assertTrue(new_body.endswith(tail))
        for required in (b'if path in declared:', b'else:\n        assert path not in declared',
                         b'inspect.isbuiltin(entry)', b'public is vars(owned)[name]', b'owned._core is core'):
            self.assertIn(required, new_body)

    def test_actual_reader_returns_current_and_predecessor_separately(self):
        for name, (expected, _) in H.ALLOWED.items():
            result = H.authenticate_source(ROOT, name, expected)
            self.assertTrue(result['migrated'])
            self.assertEqual(result['current_raw_sha256'], H.sha(self.current[name]))
            self.assertEqual(result['historical_raw_sha256'], expected)
            self.assertEqual(result['record_sha256'], H.sha((ROOT / H.RECORD).read_bytes()))

    def test_unknown_path_has_no_mismatch_escape(self):
        with self.assertRaises(AssertionError):
            H.authenticate_source(ROOT, 'tests/arbitrary.py', '0' * 64)

    def test_wrong_historical_pin_has_no_escape(self):
        name = next(iter(H.ALLOWED))
        with self.assertRaises(AssertionError):
            H.authenticate_source(ROOT, name, H.sha(self.current[name]))

    def test_counterfeit_original_capture_rejected(self):
        name = next(iter(H.ALLOWED))
        self.record['files'][name]['original_source_utf8'] = self.current[name].decode()
        self.record['files'][name]['original_raw_sha256'] = H.sha(self.current[name])
        with self.assertRaises(AssertionError):
            self.validate()

    def test_missing_record_rejected(self):
        with self.assertRaises(AssertionError):
            H.verify_record({}, self.current, self.reader)

    def test_missing_current_member_rejected(self):
        self.current.pop(next(iter(H.ALLOWED)))
        with self.assertRaises(AssertionError):
            self.validate()

    def test_extra_source_or_body_rejected(self):
        self.current['tests/unlisted.py'] = b'pass\n'
        with self.assertRaises(AssertionError):
            self.validate()
        self.current.pop('tests/unlisted.py')
        entry = self.record['files'][next(iter(H.ALLOWED))]
        entry['body_replacements'].append(copy.deepcopy(entry['body_replacements'][0]))
        with self.assertRaises(AssertionError):
            self.validate()

    def test_arbitrary_replacement_segment_is_not_a_body(self):
        name = next(iter(H.ALLOWED))
        entry = self.record['files'][name]
        entry['body_replacements'][0]['original_body_utf8'] = entry['original_source_utf8']
        entry['body_replacements'][0]['current_body_utf8'] = self.current[name].decode()
        with self.assertRaises(AssertionError):
            self.validate()

    def test_numerical_helper_changes_rejected_even_with_new_current_hash(self):
        targets = {'tests/test_h_extrema_contracts.py': (b'assert len(rows)==468', b'assert len(rows)==467'),
                   'tests/h_extrema_reference.py': (b'def load(name, filename):', b'def load(other, filename):'),
                   'tests/h_extrema_floating_reference.py': (b'def main():', b'def changed_main():'),
                   'tests/test_h_extrema_floating_cases.py': (b'image.size <= 35', b'image.size <= 34')}
        for name, (before, after) in targets.items():
            with self.subTest(name=name):
                record, current = copy.deepcopy(self.record), dict(self.current)
                self.assertIn(before, current[name])
                current[name] = current[name].replace(before, after, 1)
                record['files'][name]['current_raw_sha256'] = H.sha(current[name])
                record['files'][name]['current_canonical_lf_sha256'] = H.sha(current[name].replace(b'\r\n', b'\n'))
                with self.assertRaises(AssertionError):
                    H.verify_record(record, current, self.reader)

    def test_decorator_signature_or_outside_byte_changes_rejected(self):
        name = next(iter(H.ALLOWED))
        before = self.current[name]
        self.current[name] = before.replace(b'def test_strict_native_pending(name,workers):',
                                            b'def test_strict_native_pending(name,workers=1):', 1)
        self.record['files'][name]['current_raw_sha256'] = H.sha(self.current[name])
        self.record['files'][name]['current_canonical_lf_sha256'] = H.sha(self.current[name])
        with self.assertRaises(AssertionError):
            self.validate()

    def test_lineage_or_reader_tamper_rejected(self):
        record = copy.deepcopy(self.record)
        record['historical_test_lineage'][0]['revision'] = H.LINEAGE[-1]
        with self.assertRaises(AssertionError):
            H.verify_record(record, self.current, self.reader)
        with self.assertRaises(AssertionError):
            H.verify_record(self.record, self.current, self.reader + b'\n')


if __name__ == '__main__':
    unittest.main()
