"""Fixed IGG allocation-inclusive public-call consumer, currently UNBOUND."""
import json
from math import prod
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTEXT='coverage/contracts/inverse-gaussian-gradient-benchmark-execution-v4.json'


def case_specs():return json.loads((ROOT/'coverage/contracts/inverse-gaussian-gradient-v1.json').read_bytes())['benchmark_cases']


def case_bounds(spec):
    assert spec in case_specs()
    assert prod(spec['shape'])==spec['input_elements']
    return {k:spec[k]for k in ('input_elements','output_elements','correlate_multiply_add_terms_upper','named_numerical_bytes_upper')}


def fixture(spec,np):
    case_bounds(spec)
    value=(np.arange(prod(spec['shape']),dtype=np.int64)%251).reshape(spec['shape']).astype(spec['dtype'])
    if spec['layout']=='F':value=np.asfortranarray(value)
    elif spec['layout']=='reverse':value=value[(slice(None,None,-1),)*value.ndim]
    return (value,),dict(alpha=spec['alpha'],sigma=spec['sigma'])


def native_requirements(spec):
    case_bounds(spec)
    return dict(get_num_threads=1,inverse_gaussian_magnitude=1,inverse_gaussian_finish=4)


def plan():
    c=json.loads((ROOT/CONTEXT).read_bytes())
    return dict(state='PROSPECTIVE_UNBOUND_NO_MEASUREMENT',cases=case_specs(),method=c['method'],unbound=c['unbound'])


def main():
    import sys
    oldpath=list(sys.path)
    try:
        sys.path.insert(0,str(ROOT/'benchmarks'))
        from inverse_gaussian_gradient_execution import execute
        return execute(globals())
    finally:sys.path[:]=oldpath


if __name__=='__main__':main()
