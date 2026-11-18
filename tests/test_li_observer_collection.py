"""Guard the exact zero-test original observer exclusion without importing it."""
import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'test_li_u16_bounded.py'
RAW = (ROOT / 'tests' / NAME).read_bytes().replace(b'\r\n', b'\n')
EXPECTED = '2057a6a3aad4679c84c339f2012d06d019ebe01e3173b3503c9a4754e95ca5f0'
OLD = b'"""Bound upstream reference pools; native workers are set explicitly per test."""\nimport os\nfor variable in (\'OPENBLAS_NUM_THREADS\',\'MKL_NUM_THREADS\',\'OMP_NUM_THREADS\',\'BLIS_NUM_THREADS\'):\n    os.environ.setdefault(variable,\'1\')\n'
APPEND = b'\n# Original-only observer, no pytest tests; loaded explicitly by the fresh Li child.\ncollect_ignore = ["test_li_u16_bounded.py"]\n'
CURRENT = (ROOT / "tests/conftest.py").read_bytes().replace(b"\r\n", b"\n")
# The source-only conftest changes through 6457fdc1 preserve the Li prefix.
SOURCE_EPOCH_SUFFIX = 'f20ffbf74319a2a818b1662f97e8cb8bdd8d30ae017160dc0e1eeab27db4e994'

def no_tests(raw):
    tree=ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            assert not node.name.startswith(('test','Test','pytest_')), 'new pytest candidate/hook requires policy review'
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.NamedExpr)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                for n in ast.walk(target):
                    if isinstance(n,ast.Name):assert not n.id.startswith(('test','Test','pytest_'))
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            for alias in node.names:assert not (alias.asname or alias.name.split('.')[0]).startswith(('test','Test','pytest_'))
    return True

def exact_policy(raw):
    tree=ast.parse(raw);matches=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='collect_ignore' for t in n.targets)]
    assert len(matches)==1 and ast.literal_eval(matches[0].value)==[NAME]
    assert not any(isinstance(n,ast.Name) and n.id=='collect_ignore_glob' for n in ast.walk(tree))
    prefix=OLD+APPEND
    raw=raw.replace(b'\r\n', b'\n')
    assert raw.startswith(prefix)
    assert hashlib.sha256(raw[len(prefix):]).hexdigest()==SOURCE_EPOCH_SUFFIX
    return True

class LiObserverCollectionTests(unittest.TestCase):
    def test_actual_observer_canonical_pin_and_no_tests(self):
        self.assertEqual(hashlib.sha256(RAW).hexdigest(),EXPECTED);self.assertTrue(no_tests(RAW))
    def test_exact_single_relative_observer_policy(self):self.assertTrue(exact_policy(CURRENT))
    def test_preserve_original_thread_pool_setup(self):
        compile(ast.parse(CURRENT),'<conftest-noexec>','exec')
        self.assertTrue(exact_policy(CURRENT))
        self.assertEqual(CURRENT[:len(OLD)],OLD)
    def test_new_test_function_rejected(self):
        with self.assertRaises(AssertionError):no_tests(RAW+b'\ndef test_added(): pass\n')
    def test_new_test_class_rejected(self):
        with self.assertRaises(AssertionError):no_tests(RAW+b'\nclass TestAdded: pass\n')
    def test_imported_test_alias_rejected(self):
        with self.assertRaises(AssertionError):no_tests(RAW+b'\nfrom elsewhere import helper as test_added\n')
    def test_extra_or_pattern_ignore_rejected(self):
        for text in (CURRENT.replace(b'"test_li_u16_bounded.py"',b'"test_li_u16_bounded.py", "test_api.py"'),CURRENT.replace(b'collect_ignore =',b'collect_ignore_glob ='),CURRENT.replace(b'test_li_u16_bounded.py',b'test_li_*.py'), CURRENT+b'\ncollect_ignore.append("test_api.py")\n'):
            with self.assertRaises(AssertionError):exact_policy(text)
    def test_unreviewed_fixture_suffix_rejected(self):
        with self.assertRaises(AssertionError):
            exact_policy(CURRENT + b'\n# Unreviewed change outside the Li prefix.\n')
    def test_fresh_child_gate_path_and_source_unchanged(self):
        raw=(ROOT/'tests/test_li_u16_pipeline.py').read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),'91587055f960211a2cf7d86e04d644dd13308acc7805316f55724e57de97eaf0')
        tree=ast.parse(raw);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_installed_product');body=ast.unparse(fn)
        self.assertIn("with_name('test_li_u16_bounded.py')",body)
        self.assertLess(body.index('spec.loader.exec_module(observer)'),body.index('import skimagexx as xx'))
        self.assertLess(body.index('expected ='),body.index('import skimagexx as xx'))
        self.assertIn("'--audit-installed-product'",ast.unparse(tree))
