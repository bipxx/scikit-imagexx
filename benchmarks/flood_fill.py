"""Prospective allocated public-call cases; no executable timing authority."""
from math import prod

METHOD = dict(samples=7, minimum_sample_ns=100_000_000,
              minimum_warmup_ns=250_000_000, workers=1,
              call='allocated public call; copy/fill included',
              classification='retain every slower case; no blanket speed claim')
BINDING = dict(source=None, producer=None, runtime=None, core=None, SDK=None,
               root_admission=None, first_cohort_GO=None, measurement_GO=None)


def case_specs():
    out = []
    for api in ('flood', 'flood_fill'):
        for name, shape, layout, dtype, tolerance in (
            ('tiny', [5, 7], 'C', 'uint8', None),
            ('dense-C', [128, 192], 'C', 'uint8', None),
            ('dense-F', [128, 192], 'F', 'int16', None),
            ('reverse', [128, 192], 'reverse', 'int16', None),
            ('volume', [16, 24, 32], 'C', 'uint16', None),
            ('tolerance', [128, 192], 'C', 'float32', 1.0),
        ):
            out.append(dict(id=api + ':' + name, api=api, shape=shape,
                            layout=layout, dtype=dtype, tolerance=tolerance,
                            pattern='bounded 0/1 plateau with barrier2',
                            seed=[n // 2 for n in shape], connectivity=1,
                            new_value=7, in_place=False))
    return out


def case_bounds(row):
    shape = row['shape']
    assert 1 <= len(shape) <= 3 and all(0 < n <= 192 for n in shape)
    logical = prod(shape)
    padded = prod(n + 2 for n in shape)
    return dict(logical=logical, padded=padded, queue_entries=logical,
                neighbor_checks=logical * (3 ** len(shape)),
                storage_bytes=logical * 32 + padded * 17 +
                max(64, logical * 2) * 8 + (3 ** len(shape)) * 40)


def main(argv=None):
    # Deliberate prospective refusal BEFORE any scientific import, case factory,
    # API resolution or sampler. Runtime args cannot confer authority in v1.
    raise SystemExit('UNBOUND: safety/reference/native/genuine/first-cohort/measurement GO pending')


if __name__ == '__main__':
    main()
