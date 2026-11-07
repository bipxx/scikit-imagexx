"""Allocated whole-public VI comparison; no authority without root binding."""
import json
from pathlib import Path
from variation_of_information_cases_v1 import case_specs,case_bounds,fixture,native_requirements
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/variation-of-information-benchmark-execution-v5.json'


def plan():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    return dict(state=data['state'],cases=case_specs(),bounds=[case_bounds(s)for s in case_specs()],
                method=data['method'],unbound=data['unbound'],providers=data['providers'],
                numerical_parity='Saved two-profile references and corrected private parity admitted; exact actual genuine preflight still required.',
                original_calls=0,timed_batches=0)


def main():
    from variation_of_information_execution import execute
    return execute(globals())


if __name__=='__main__':main()
