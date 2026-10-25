"""Bound upstream reference pools; native workers are set explicitly per test."""
import os
for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ.setdefault(variable,'1')

# Original-only observer, no pytest tests; loaded explicitly by the fresh Li child.
collect_ignore = ["test_li_u16_bounded.py"]
