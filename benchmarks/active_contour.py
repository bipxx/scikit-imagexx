"""Fixed allocation-inclusive active contour comparisons at one worker."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public prepared input; include result allocation and disposal')


def case_specs():
    for label, shape, points, dtype, channels, boundary, iterations, w_line, w_edge in (
        ('tiny-quadratic-grid', (3, 3), 5, 'float64', 1, 'periodic', 3, 1.0, 0.0),
        ('gray-periodic', (80, 100), 121, 'float64', 1, 'periodic', 70, 0.2, 1.0),
        ('gray-float32', (112, 150), 161, 'float32', 1, 'periodic', 50, 0.1, 1.0),
        ('rgb-byte-fixed-free', (120, 160), 143, 'uint8', 3, 'fixed-free', 40, 0.1, 1.0),
        ('long-gray-snake', (128, 192), 800, 'float64', 1, 'periodic', 40, 0.2, 1.0),
    ):
        yield dict(id=label, api='segmentation.active_contour', shape=shape, points=points,
                   dtype=dtype, channels=channels, boundary=boundary, iterations=iterations,
                   w_line=w_line, w_edge=w_edge)


def make_case(spec):
    import numpy as np

    rows, columns = spec['shape']
    y, x = np.ogrid[:rows, :columns]
    y = y / (rows - 1) - 0.5
    x = x / (columns - 1) - 0.5
    radial = np.exp(-18.0 * (x**2 + y**2))
    image = 0.8 * radial + 0.1 + 0.05 * (x + y)
    if spec['channels'] == 3:
        image = np.stack([image, image * 0.8, 0.3 + 0.4 * radial - 0.1 * x], axis=-1)
    dtype = spec['dtype']
    image = (image * 255 if dtype == 'uint8' else image).astype(dtype)

    closed = spec['boundary'] == 'periodic'
    angle = np.linspace(0, 2 * np.pi if closed else 1.5 * np.pi, spec['points'], endpoint=not closed)
    snake = np.stack([(rows - 1) * (0.5 + 0.34 * np.sin(angle)),
                      (columns - 1) * (0.5 + 0.34 * np.cos(angle))], axis=1)
    options = dict(w_line=spec['w_line'], w_edge=spec['w_edge'], convergence=0.0,
                   max_num_iter=spec['iterations'], boundary_condition=spec['boundary'])
    return (image, snake), options
