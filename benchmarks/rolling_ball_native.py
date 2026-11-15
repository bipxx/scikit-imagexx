"""Allocation-inclusive rolling-ball public benchmark; authority stays UNBOUND."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/rolling-ball-benchmark-execution-v1.json'


def case_specs():
    return json.loads((ROOT/'coverage/contracts/rolling-ball-v1.json').read_bytes())['benchmark_cases']


def case_bounds(spec):
    import math
    assert spec in case_specs()
    n=math.prod(spec['shape']);k=(2*math.ceil(spec['radius'])+1)**len(spec['shape'])
    assert n==spec['input_elements']and k==spec['kernel_elements']and n*k==spec['addition_upper']
    assert spec['workers']==1
    return dict(input_elements=n,kernel_elements=k,addition_upper=n*k)


def plan():
    context=json.loads((ROOT/CONTEXT).read_bytes())
    return dict(api='restoration.rolling_ball',state='PROSPECTIVE_UNBOUND_NO_MEASUREMENT',
                cases=case_specs(),bounds=[case_bounds(s)for s in case_specs()],
                method=context['method'],unbound=context['unbound'],
                source=None,producer=None,runtime=None,core=None,SDK=None,root_GO=None,
                next_gate='Genuine root product/core/SDK/F1 binding, actual outer first-cohort admission, separate measurement GO; no current execution authority.')


def main():
    import sys
    sys.path.insert(0,str(ROOT/'benchmarks'))
    from rolling_ball_execution import execute
    return execute(globals())


if __name__=='__main__':
    main()
