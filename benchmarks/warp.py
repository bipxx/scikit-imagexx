"""Fixed, warmed public warp and warp_coords workloads."""


def fixed_cases(np):
    rng = np.random.default_rng(88)
    image = rng.uniform(.1, .95, (512, 576))
    affine = np.array([[1.01, -.04, 3.2], [.06, .95, -7.3], [0., 0., 1.]])
    projective = np.array([[1.01, -.04, 3.2], [.06, .95, -7.3], [.0002, -.0003, 1.]])
    return (
        ('affine-bilinear', 'warp', (image, affine), {}),
        ('projective-bicubic', 'warp', (image, projective), dict(order=3, mode='reflect')),
        ('affine-nearest', 'warp', (image, affine), dict(order=0, mode='edge')),
        ('projective-bspline', 'warp', (image, projective), dict(order=2, mode='symmetric')),
        ('projective-coordinates', 'warp_coords', (projective, (512, 576)), {}),
    )


def fixed_public_calls(namespace, np):
    for label, name, operands, options in fixed_cases(np):
        if name == 'warp_coords':
            matrix, shape = operands
            operands = (namespace.ProjectiveTransform(matrix=matrix), shape)
        yield label, getattr(namespace, name), operands, options
