"""The four frozen VI workloads, with source bounds but no accuracy authority."""
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def prior():return runpy.run_path(str(ROOT/'benchmarks/variation_of_information.py'))
def case_specs():return prior()['case_specs']()


def case_bounds(spec):
    bound=prior()['source_safe'](spec)
    n=bound['pixels'];labels=spec['labels']
    assert n in(16384,65536)and labels in(8,64)and spec['ignore']==[0]
    # Plain nonnegative labels, both sparse axes<=64, logicalNNZ<=4096,
    # physical source storage<=N. At least N-ceil(N/labels)>0 retained pixels.
    # Positive conditional weights/log arguments, no zero-division case here.
    return dict(bound,physical_CSR_slots=n,entropy_rows=labels,entropy_columns=labels,
                maximum_logical_NNZ=labels*labels,output_words=2,
                numerical_parity='PENDING actual exact original/candidate preflight before sampler')


def fixture(spec,np):return prior()['fixture'](spec,np)


def native_requirements(spec):
    case_bounds(spec)
    # Two existing phase budgets; not a whole-public-call single snapshot claim.
    # Prefix storage4, VI CSR boundary normalization/storage3.
    return dict(get_num_threads=2,label_pair_check_storage=7,
                label_pair_counts_preweighted=1,variation_of_information_csr=1)
