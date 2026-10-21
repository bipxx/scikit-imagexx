"""Inventory mechanics only: these checks establish no native numerical parity."""
import functools
import inspect
from tools.inventory import class_members,signature_of


def test_static_class_inventory_does_not_run_constructors_or_getters():
    class Parent:
        def __init__(self):raise AssertionError('constructor must not run')
        def __call__(self,points):raise AssertionError('method must not run')
        @property
        def expensive(self):raise AssertionError('property must not run')
        @functools.cached_property
        def cached(self):raise AssertionError('cached property must not run')
        def _internal(self):pass
    class Child(Parent):
        def estimate(self,data,*,weight=1):pass
        @classmethod
        def named(cls,label):pass
        @staticmethod
        def validate(data):pass
    rows=class_members(Child,'example.Child')
    records={row['api'].rsplit('.',1)[1]:row for row in rows}
    assert set(records)=={'__init__','__call__','expensive','cached','estimate','named','validate'}
    assert records['__init__']['kind']=='constructor'
    assert records['__call__']['inherited']
    assert records['expensive']['kind']==records['cached']['kind']=='property'
    assert records['named']['signature']==str(inspect.signature(Child.named))
    assert records['validate']['signature']==str(inspect.signature(Child.validate))
    assert records['estimate']['signature']==str(inspect.signature(Child.estimate))
    assert all(row['implementation']=='upstream-only' and row['parity_test'] is None and row['benchmark'] is None for row in rows)


def test_opaque_signature_is_explicitly_unknown():
    assert signature_of(object()) is None
