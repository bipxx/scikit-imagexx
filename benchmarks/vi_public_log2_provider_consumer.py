"""New VI provider source/measurement plan; no runtime authority at import."""
import json
from pathlib import Path
from variation_of_information_cases_v1 import case_specs,case_bounds,fixture,native_requirements
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/vi-public-log2-provider-benchmark-v1.json'


def plan():
    data=json.loads((ROOT/CONTEXT).read_bytes())
    return dict(state=data['state'],cases=case_specs(),bounds=[case_bounds(s)for s in case_specs()],
        method=data['method'],unbound=data['unbound'],providers=data['providers'],
        numerical_parity='Pending compiled exact-source public/FP/SDK/four-profile evidence; no old source authority reused.',
        original_calls=0,timed_batches=0)


def main():
    from vi_public_log2_provider_execution import execute
    return execute(globals())


if __name__=='__main__':main()
