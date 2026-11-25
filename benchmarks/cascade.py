"""Fixed, conversion- and allocation-inclusive Cascade detection fixture."""


def fixed_cases(data):
    return [
        {
            "id": "astronaut-gray-512",
            "model": data.lbp_frontal_face_cascade_filename(),
            "args": (data.astronaut()[:, :, 1].copy(), 1.2, 1.0, (24, 24), (180, 180)),
            "kwargs": {"min_neighbor_number": 2, "intersection_score_threshold": 0.5},
        },
    ]


def public_call(cascade, args, kwargs):
    return cascade.detect_multi_scale(*args, **kwargs)
