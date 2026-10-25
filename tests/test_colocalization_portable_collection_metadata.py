"""Stdlib-only collection controls; no scientific or compiled imports."""
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "tests/test_colocalization_native_pipeline.py"


def collection_policy(config):
    tree = ast.parse(PIPELINE.read_bytes())
    first = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_collection_path")
    last = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "DECLARED_WORKERS")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "config.json"
        file.write_text(json.dumps(config))
        env = {} if config is None else {"COLOCALIZATION_TEST_CONFIG": str(file)}
        scope = dict(os=types.SimpleNamespace(environ=env), json=json, Path=Path)
        exec(compile(ast.Module(body=tree.body[first:last + 1], type_ignores=[]),
                     "<actual-collection-policy>", "exec"), scope)
        return scope["DECLARED_WORKERS"]


class PortableCollectionMetadata(unittest.TestCase):
    def test_portable_keeps_all_536_ids_once(self):
        workers = collection_policy(dict(profile="portable", portable=True))
        self.assertEqual(workers, (1,))
        pairs = {(i, w) for i in range(536) for w in workers}
        self.assertEqual(len(pairs), 536)
        self.assertEqual(len(pairs) + 2, 538)
        self.assertTrue(all(w == 1 for _, w in pairs))

    def test_ordinary_profiles_keep_original_workers(self):
        for profile in ("windows", "modern", "minimum", "sanitizer"):
            self.assertEqual(collection_policy(dict(profile=profile, portable=False)), (1, 4))
        self.assertEqual(collection_policy(None), (1, 4))

    def test_mismatched_or_untyped_declarations_rejected(self):
        for config in (dict(profile="portable", portable=False), dict(profile="modern", portable=True),
                       dict(profile="portable", portable=1), dict(profile="other", portable=False)):
            with self.assertRaises(AssertionError):
                collection_policy(config)

    def test_closed_scientific_and_metadata_collection(self):
        tree = ast.parse(PIPELINE.read_bytes())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual({name for name in functions if name.startswith("test_")},
                         {"test_frozen_public_semantics", "test_owner_metadata_only", "test_declared_global_worker_budget"})
        decorators = functions["test_frozen_public_semantics"].decorator_list
        params = {n.args[0].value: n.args[1] for n in decorators}
        self.assertEqual(ast.dump(params["workers"]), ast.dump(ast.Name(id="DECLARED_WORKERS", ctx=ast.Load())))
        self.assertEqual(ast.dump(params["index"]), ast.dump(ast.parse("range(536)", mode="eval").body))


if __name__ == "__main__":
    unittest.main()
