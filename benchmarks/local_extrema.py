"""Allocation-inclusive one-worker case definitions for the local extrema pair."""

from math import prod

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public inputs already prepared; include result allocation')


def case_specs():
    for api in ('local_maxima', 'local_minima'):
        for name, shape, dtype, layout, connectivity, indices, borders, levels in (
            ('tiny', (5, 7), 'uint8', 'C', 1, False, True, 4),
            ('dense', (128, 192), 'uint16', 'C', 2, False, True, 4096),
            ('plateaus', (256, 320), 'int16', 'F', 1, False, False, 4),
            ('indices', (128, 192), 'int32', 'reverse', 2, True, True, 16),
        ):
            yield dict(id=api + ':' + name, api='morphology.' + api, shape=shape, dtype=dtype,
                       layout=layout, connectivity=connectivity, indices=indices,
                       allow_borders=borders, levels=levels, seed=19)


def make_case(row):
    import numpy as np

    dtype = np.dtype(row['dtype'])
    image = np.random.default_rng(row['seed']).integers(row['levels'], size=row['shape'], dtype=dtype)
    if row['layout'] == 'F':
        image = np.asfortranarray(image)
    elif row['layout'] == 'reverse':
        image = image[:, ::-1]
    assert image.size == prod(row['shape'])
    options = {key: row[key] for key in ('connectivity', 'indices', 'allow_borders')}
    return (image,), options
