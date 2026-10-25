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
        predecessor_raw = (ROOT / 'coverage/evidence/h-extrema-reference-assertion-migration-9b.json').read_bytes()
        self.assertEqual(H.sha(predecessor_raw), 'bdf3c49437f9b2a33486efc41f3a237c4a39a0b2c66884636a880311ae261754')
        predecessor = json.loads(predecessor_raw)['files'][name]
        old = predecessor['original_source_utf8'].encode()
        for replacement in predecessor['body_replacements']:
            old = old.replace(replacement['original_body_utf8'].encode(), replacement['current_body_utf8'].encode(), 1)
        self.assertEqual(H.sha(old), predecessor['current_raw_sha256'])
        old_body = H._body(old, ast.parse(old), 'test_strict_native_pending')
        new_body = H._body(self.current[name], ast.parse(self.current[name]), 'test_strict_native_pending')
        tail = old_body[old_body.index(b'    old=xx.get_num_threads()'):]
        old_call = b'        with pytest.raises(NotImplementedError):getattr(xx.morphology,name)(object(),1)\n'
        new_call = (b'        with pytest.raises(expected_exception) as caught:getattr(xx.morphology,name)(object(),1)\n'
                    b'        if expected_source_error is not None:\n'
                    b'            assert dict(module=type(caught.value).__module__,type=type(caught.value).__name__,message=str(caught.value))==expected_source_error\n')
        self.assertEqual(tail.count(old_call), 1)
        self.assertTrue(new_body.endswith(tail.replace(old_call, new_call, 1)))
        self.assertEqual(self.current[name].replace(new_body, old_body, 1), old)
        for required in (b'if path in declared:', b'else:\n        assert path not in declared',
                         b'inspect.isbuiltin(entry)', b'public is vars(owned)[name]', b'owned._core is core',
                         b"core.h_extrema_real_interface=='h-extrema-real-v1'"):
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

    def _object_record(self, raw):
        self.assertEqual(H.sha(raw), '10383844bf54762cf09293f5270cd6d49713371bba5d558b172f0293e512dff8')
        contract = json.loads(raw)
        self.assertEqual(contract['schema'], 'h-exact-object-original-preentry-contract-v1')
        self.assertEqual(set(contract['profiles']), {'windows', 'modern', 'minimum'})
        self.assertEqual(contract['original_H_calls'], 6)
        self.assertEqual(contract['flood_or_candidate_calls'], 0)
        self.assertIs(contract['native_or_public_acceptance'], False)
        for profile, entry in contract['profiles'].items():
            report_raw = entry['raw_report_utf8'].encode('utf-8')
            self.assertEqual(H.sha(report_raw), entry['raw_report_sha256'])
            self.assertEqual(len(report_raw), entry['raw_report_bytes'])
            report = json.loads(report_raw)
            self.assertEqual(report['profile'], profile)
            self.assertEqual(report['observer_sha256'], contract['observer_sha256'])
            self.assertEqual(report['plan_sha256'], contract['plan_sha256'])
            self.assertEqual(set(report['observations']), {c['id'] for c in contract['cases']})
            self.assertFalse(report['failures'])
            for row in report['observations'].values():
                self.assertEqual(row['error'], dict(module='builtins', type='TypeError',
                    message="unsupported operand type(s) for -: 'object' and 'object'"))
                self.assertEqual(row['warnings'], [])
                self.assertEqual(row['original_H_entry_count'], 1)
                self.assertEqual(row['reconstruction_denial_attempts'], [])
                for field in ('state_identity_restored', 'provider_identities_restored',
                              'source_entry_identities_restored', 'reconstruction_entry_identities_restored'):
                    self.assertTrue(all(row[field].values()))
        return contract

    def test_registered_object_record_lossless_projection(self):
        raw = (ROOT / 'coverage/contracts/h-extrema-object-preentry.json').read_bytes()
        self._object_record(raw)
        body = H._body(self.current['tests/test_h_extrema_contracts.py'],
                       ast.parse(self.current['tests/test_h_extrema_contracts.py']), 'test_strict_native_pending')
        self.assertIn(H.sha(raw).encode(), body)
        self.assertIn(b'expected_exception=NotImplementedError;expected_source_error=None', body)
        self.assertIn(b'expected_exception=TypeError', body)
        self.assertIn(b"real_entry.__name__=='h_extrema_real_mask'", body)

    def test_object_record_missing_wrong_profile_error_or_hash_rejected(self):
        raw = (ROOT / 'coverage/contracts/h-extrema-object-preentry.json').read_bytes()
        for mode in ('missing', 'profile', 'class', 'message'):
            with self.subTest(mode=mode):
                data = json.loads(raw)
                if mode == 'missing':
                    data = {}
                elif mode == 'profile':
                    data['profiles']['other'] = data['profiles'].pop('minimum')
                else:
                    profile = data['profiles']['windows']
                    report = json.loads(profile['raw_report_utf8'])
                    row = next(iter(report['observations'].values()))
                    row['error']['type' if mode == 'class' else 'message'] = 'NotImplementedError' if mode == 'class' else 'changed'
                    profile['raw_report_utf8'] = json.dumps(report)
                with self.assertRaises(AssertionError):
                    self._object_record(json.dumps(data).encode())

    def test_registered_old_NI_selection_rejected_preregistered_NI_preserved(self):
        def selections(raw):
            fn = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'test_strict_native_pending')
            branch = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'path in declared')
            assignments = [n for n in ast.walk(branch) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'expected_exception' for t in n.targets)]
            assert len(assignments) == 1 and ast.unparse(assignments[0].value) == 'TypeError'
            assert ast.unparse(branch.orelse[0]) == 'assert path not in declared'
            initial = [n for n in fn.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'expected_exception' for t in n.targets)]
            assert len(initial) == 1 and ast.unparse(initial[0].value) == 'NotImplementedError'
        raw = self.current['tests/test_h_extrema_contracts.py']
        selections(raw)
        with self.assertRaises(AssertionError):
            selections(raw.replace(b'expected_exception=TypeError', b'expected_exception=NotImplementedError', 1))


if __name__ == '__main__':
    unittest.main()
