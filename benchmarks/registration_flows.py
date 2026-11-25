"""Fixed allocation-inclusive optical-flow calls for paired one-worker timing."""


def case_specs():
    yield "ilk-default-f32", "optical_flow_ilk", (96, 128), "float32", {}
    yield (
        "ilk-gaussian-prefilter-f64", "optical_flow_ilk", (65, 83), "float64",
        {"radius": 5, "num_warp": 3, "gaussian": True, "prefilter": True, "dtype": "float64"},
    )
    yield "tvl1-default-f32", "optical_flow_tvl1", (96, 128), "float32", {}
    yield (
        "tvl1-prefilter-f64", "optical_flow_tvl1", (67, 91), "float64",
        {"attachment": 10, "tightness": 0.22, "num_warp": 3, "num_iter": 7,
         "prefilter": True, "dtype": "float64"},
    )


def make_cases():
    import numpy as np
    from scipy.ndimage import gaussian_filter, shift

    for name, api, shape, dtype, kwargs in case_specs():
        rng = np.random.default_rng(9141)
        reference = gaussian_filter(rng.random(shape).astype(dtype), 0.9)
        moving = shift(reference, (1.25, -1.5), order=1, mode="nearest")
        yield name, api, (reference, moving), kwargs
