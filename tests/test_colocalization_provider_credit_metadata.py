"""Stdlib-only checks of the declared provider-credit partition, no pixels."""
import ast
import hashlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
PIPELINE=ROOT/'tests/test_colocalization_native_pipeline.py'


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('numpy','scipy','skimage','skimagexx'):
            raise AssertionError('metadata check attempted scientific import: '+fullname)


def cases():
    path=ROOT/'tests/colocalization_cases.py'
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='adbee581546e2031377853393b354be0dff41731eac920eb839c37b6c9f9db59'
    guard=NoScience();sys.meta_path.insert(0,guard)
    try:
        spec=importlib.util.spec_from_file_location('credit_partition_frozen_factory',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.cases()
    finally:sys.meta_path.remove(guard)


def predicate():
    tree=ast.parse(PIPELINE.read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_frozen_public_semantics')
    nodes=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id=='nonfinite_mcc' for t in n.targets)]
    assert len(nodes)==1
    return compile(ast.Expression(nodes[0].value),'<actual-credit-predicate>','eval')


class ProviderCreditMetadata(unittest.TestCase):
    def test_exact_source_partition_includes_ten_not_nine_outcomes(self):
        code=predicate();rows=cases()
        actual={r['id'] for r in rows if eval(code,{},dict(spec=r))}
        base='manders_coloc_coeff/'
        expected={base+'special/'+value+suffix for value in ('nan0','inf0')
                  for suffix in ('','/policy-default')}
        expected|={base+'fp/'+value+'-'+policy for value in ('nan0','inf0')
                   for policy in ('ignore','raise','call')}
        self.assertEqual(len(rows),536);self.assertEqual(len(actual),10)
        self.assertEqual(actual,expected)
        self.assertIn(base+'fp/inf0-raise',actual) # source-error outcome also classified

    def test_every_ordinary_native_case_remains_outside_provider_predicate(self):
        rows=cases();code=predicate()
        ordinary=[r for r in rows if r['group']=='dtype' or(r['group']=='layout' and
                  not(r['api']=='manders_coloc_coeff' and r['name']=='reverse'))]
        self.assertEqual(len(ordinary),118)
        self.assertTrue(all(not eval(code,{},dict(spec=r)) for r in ordinary))

    def test_unrelated_specials_and_other_functions_rejected(self):
        code=predicate()
        for api in ('intersection_coeff','manders_overlap_coeff'):
            self.assertFalse(eval(code,{},dict(spec=dict(api=api,group='special',name='nan0'))))
        for group in ('dtype','layout','protocol','binary_control'):
            self.assertFalse(eval(code,{},dict(spec=dict(api='manders_coloc_coeff',group=group,name='nan0'))))
        for value in ('zero','negative0','negative1','tiny','overflow0','overflow1','mismatch'):
            self.assertFalse(eval(code,{},dict(spec=dict(api='manders_coloc_coeff',group='special',name=value))))

    def test_no_values_or_expected_outcomes_in_predicate(self):
        tree=ast.parse(PIPELINE.read_bytes())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='nonfinite_mcc' for t in n.targets))
        self.assertEqual({n.id for n in ast.walk(assignment.value) if isinstance(n,ast.Name)},{'spec'})
        self.assertNotIn('id', {n.value for n in ast.walk(assignment.value) if isinstance(n,ast.Constant)})
        calls=[n for n in ast.walk(assignment.value) if isinstance(n,ast.Call)]
        self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0].func),'spec.get')


if __name__=='__main__':unittest.main()
