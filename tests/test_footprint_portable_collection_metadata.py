"""Stdlib-only portable collection policy controls; no scientific imports."""
import ast
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
PIPELINE=ROOT/'tests/test_footprint_shapes_native_pipeline.py'


def collection_policy(config):
    tree=ast.parse(PIPELINE.read_bytes())
    first=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
               isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_collection_path')
    last=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
              isinstance(n.targets[0],ast.Name) and n.targets[0].id=='DECLARED_WORKERS')
    parent=ROOT/'build/portable-collection-metadata';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        file=Path(tmp)/'config.json';file.write_text(json.dumps(config))
        env={} if config is None else {'FOOTPRINT_SHAPES_TEST_CONFIG':str(file)}
        scope=dict(os=types.SimpleNamespace(environ=env),json=json,Path=Path)
        exec(compile(ast.Module(body=tree.body[first:last+1],type_ignores=[]),'<actual-collection-policy>','exec'),scope)
        return scope['DECLARED_WORKERS']


class PortableCollectionMetadata(unittest.TestCase):
    def test_portable_keeps_every_scientific_id_once(self):
        workers=collection_policy(dict(profile='portable',portable=True))
        self.assertEqual(workers,(1,))
        pairs={(i,w) for i in range(226) for w in workers}
        self.assertEqual(len(pairs),226)
        self.assertEqual(len(pairs)+3,229)
        self.assertTrue(all(w==1 for _,w in pairs))

    def test_ordinary_profiles_keep_original_workers(self):
        for profile in ('windows','modern','minimum','sanitizer'):
            self.assertEqual(collection_policy(dict(profile=profile,portable=False)),(1,4))
        self.assertEqual(collection_policy(None),(1,4))

    def test_mismatched_or_untyped_declaration_refused(self):
        for config in (dict(profile='portable',portable=False),dict(profile='modern',portable=True),
                       dict(profile='portable',portable=1),dict(profile='other',portable=False)):
            with self.assertRaises(AssertionError):collection_policy(config)

    def test_exact_id_factory_and_old_scientific_body_still_present(self):
        tree=ast.parse(PIPELINE.read_bytes())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_exact_frozen_shape_sequence')
        decorator=next(n for n in fn.decorator_list if isinstance(n,ast.Call) and n.args and
                       isinstance(n.args[0],ast.Constant) and n.args[0].value=='workers')
        self.assertEqual(ast.dump(decorator.args[1]),ast.dump(ast.Name(id='DECLARED_WORKERS',ctx=ast.Load())))
        names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'test_declared_global_worker_budget','test_cross_plan_rejects_owned_unaligned_before_typed_read',
                         'test_metadata_check_accepts_owned_roots_views_and_empty_arrays'}<=names)
        self.assertIn('range(226)',ast.unparse(fn))


if __name__=='__main__':unittest.main()
