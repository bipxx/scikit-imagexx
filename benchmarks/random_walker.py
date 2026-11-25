"""Fixed one-worker public random walker comparisons."""

METHOD = {
    "workers": 1,
    "samples": 7,
    "minimum_sample_ns": 100_000_000,
    "minimum_warmup_ns": 250_000_000,
    "call": "public graph construction, solve, and result allocation",
}


def case_specs():
    return (
        {"id": "gray32x48-u8-default", "shape": (32, 48), "dtype": "uint8", "options": {}},
        {
            "id": "gray48x64-f64-spaced-probabilities",
            "shape": (48, 64),
            "dtype": "float64",
            "options": {"beta": 50, "spacing": (1.0, 1.5), "return_full_prob": True},
        },
        {
            "id": "volume12x16x20-f64-three-classes",
            "shape": (12, 16, 20),
            "dtype": "float64",
            "options": {"beta": 70, "spacing": (2.0, 1.0, 1.5), "mode": None},
        },
    )


def fixture(spec, np):
    shape = spec["shape"]
    rng = np.random.default_rng(42032)
    grid = np.indices(shape)
    width = shape[-1]
    ends = grid[-1] + (grid[0] % 3 - 1)
    image = 0.17 * rng.random(shape) + 0.63 * (ends >= width // 2)
    labels = np.zeros(shape, dtype=np.int32)
    labels[..., :2] = 1
    labels[..., -2:] = 5
    if len(shape) == 3:
        image += 0.18 * (grid[0] > shape[0] // 2)
        labels[-2:, :, width // 3:width // 2] = 9
    if spec["dtype"] == "uint8":
        image = (175 * image).astype(np.uint8)
    return (image, labels), dict(spec["options"])


def make_cases():
    import numpy as np

    for spec in case_specs():
        args, options = fixture(spec, np)
        yield spec["id"], args, options
