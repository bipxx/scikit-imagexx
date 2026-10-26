"""Metadata-only controls for the frozen/live colocalization oracle policy."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = dict(api="manders_overlap_coeff", id="metadata-policy-control", group="dtype", name="float64")
path = Path(__file__).with_name("test_colocalization_public.py")
module_spec = importlib.util.spec_from_file_location("colocalization_public_policy", path)
PUBLIC = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(PUBLIC)


def record():
    return dict(result=dict(kind="numpy-scalar", module="numpy", type="float64", dtype="<f8",
                            raw="000000000000f03f"), error=None, warnings=[], events=[],
                setup_events=[], input_before=[], input_after=[], result_aliases=[], state={})


class ReferenceExpectationTests(unittest.TestCase):
    def test_unchanged_contract(self):
        frozen = record()
        self.assertEqual(PUBLIC._reference_expectation(SPEC, copy.deepcopy(frozen), frozen), frozen)

    def test_only_supported_moc_float_raw_can_follow_fresh_original(self):
        for name, width in (("float16", 2), ("float32", 4), ("float64", 8)):
            with self.subTest(dtype=name):
                frozen = record()
                frozen["result"].update(type=name, dtype=f"<f{width}", raw="00" * width)
                fresh = copy.deepcopy(frozen)
                fresh["result"]["raw"] = "01" * width
                before = copy.deepcopy((fresh, frozen))
                self.assertEqual(PUBLIC._reference_expectation(SPEC, fresh, frozen), fresh)
                self.assertEqual((fresh, frozen), before)

    def test_all_other_semantic_fields_remain_frozen(self):
        for key in PUBLIC.SEMANTIC_FIELDS:
            if key == "result":
                continue
            with self.subTest(field=key):
                frozen = record(); fresh = copy.deepcopy(frozen)
                fresh[key] = {"changed": True}
                with self.assertRaises(AssertionError):
                    PUBLIC._reference_expectation(SPEC, fresh, frozen)

    def test_all_nonraw_result_metadata_remains_frozen(self):
        changes = dict(kind="Python-float", module="foreign", type="float32", dtype=">f8", extra=True)
        for key, value in changes.items():
            with self.subTest(field=key):
                frozen = record(); fresh = copy.deepcopy(frozen)
                fresh["result"][key] = value
                with self.assertRaises(AssertionError):
                    PUBLIC._reference_expectation(SPEC, fresh, frozen)

    def test_other_apis_and_error_results_have_no_exception(self):
        for api, error in (("intersection_coeff", None), ("manders_coloc_coeff", None),
                           ("manders_overlap_coeff", {"type": "ValueError"})):
            with self.subTest(api=api, error=error):
                frozen = record(); frozen["error"] = error
                fresh = copy.deepcopy(frozen); fresh["result"]["raw"] = "01" * 8
                with self.assertRaises(AssertionError):
                    PUBLIC._reference_expectation(dict(SPEC, api=api), fresh, frozen)

    def test_unsupported_results_stay_exact(self):
        for result in (dict(kind="Python-float", raw_f64="00" * 8),
                       dict(kind="ndarray", dtype="<f8", raw="00" * 8),
                       dict(kind="numpy-scalar", module="numpy", type="int64", dtype="<i8", raw="00" * 8),
                       dict(kind="numpy-scalar", module="numpy", type="longdouble", dtype="<f16", raw="00" * 16)):
            with self.subTest(result=result["kind"], dtype=result.get("dtype")):
                frozen = record(); frozen["result"] = result
                fresh = copy.deepcopy(frozen)
                key = "raw_f64" if "raw_f64" in result else "raw"
                fresh["result"][key] = "01" * (len(result[key]) // 2)
                with self.assertRaises(AssertionError):
                    PUBLIC._reference_expectation(SPEC, fresh, frozen)

    def test_malformed_scalar_words_are_rejected(self):
        for raw in (None, 123, "0", "00" * 4, "gg" * 8):
            with self.subTest(raw=raw):
                frozen = record(); fresh = copy.deepcopy(frozen); fresh["result"]["raw"] = raw
                with self.assertRaises(AssertionError):
                    PUBLIC._reference_expectation(SPEC, fresh, frozen)

    def test_case_captures_original_first_and_requires_fresh_native_bits(self):
        for stale in (False, True):
            with self.subTest(stale_native=stale):
                frozen = record(); fresh = copy.deepcopy(frozen)
                fresh["result"]["raw"] = "0000000000000040"
                events, journal, core = [], Mock(), Mock()
                core.get_num_threads.return_value = 1
                def original(*args):
                    events.append("original")
                    journal.append(fresh)
                    return {"observation": fresh}
                observer = SimpleNamespace(RawJournal=lambda path: journal, RawAssets=lambda path: object(),
                                           observe=original, semantic_projection=PUBLIC.OBSERVER.semantic_projection)
                runtime = dict(core=core, observer=observer, factory=None, original=None, np=None,
                               expected={SPEC["id"]: frozen})
                def native(*args):
                    events.append("native")
                    journal.close.assert_called_once()
                    return dict(copy.deepcopy(frozen if stale else fresh), observer_error=None,
                                phase="complete", restored=True, forbidden_calls=[], array_providers=[],
                                c_events=[dict(event="c_return", name="coloc_sum")],
                                stage_returns=[dict(name="coloc_sum", admitted=True)])
                with patch.object(PUBLIC, "_observe", side_effect=native):
                    if stale:
                        with self.assertRaises(AssertionError):
                            PUBLIC._check_case(runtime, SPEC, Path("unused-metadata-only"), 4)
                    else:
                        PUBLIC._check_case(runtime, SPEC, Path("unused-metadata-only"), 4)
                self.assertEqual(events, ["original", "native"])
                self.assertEqual([call.args for call in core.set_num_threads.call_args_list], [(4,), (1,)])


if __name__ == "__main__":
    unittest.main()
