"""Fixed one-worker tomography calls, including public preparation and allocation."""

import numpy as np


def fixed_public_calls():
    rng = np.random.default_rng(20261124)
    rows, columns = np.ogrid[:128, :128]
    phantom = (((rows - 60) / 30) ** 2 + ((columns - 69) / 22) ** 2 < 1).astype(np.float64)
    rectangle = rng.integers(0, 256, size=(87, 128), dtype=np.uint8)
    angles = np.linspace(0, 180, 75, endpoint=False)
    sinogram = rng.random((128, 90))
    single = sinogram.astype(np.float32)
    shifts = np.linspace(-0.3, 0.3, 90)

    yield "radon_circle_f64", "radon", (phantom,), {}
    yield "radon_padded_u8", "radon", (rectangle, angles), {"circle": False}
    yield "iradon_ramp_linear_f64", "iradon", (sinogram,), {}
    yield "iradon_hamming_nearest_f32", "iradon", (single,), {
        "filter_name": "hamming", "interpolation": "nearest", "circle": False, "output_size": 110,
    }
    yield "iradon_hann_cubic_f64", "iradon", (sinogram,), {
        "filter_name": "hann", "interpolation": "cubic",
    }
    yield "sart_default_f64", "iradon_sart", (sinogram,), {}
    yield "sart_shift_clip_f32", "iradon_sart", (single,), {
        "projection_shifts": shifts, "clip": (-0.1, 0.2), "relaxation": 0.2,
    }
