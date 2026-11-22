"""Fixed one-worker allocation-inclusive synthetic blob workloads."""

METHOD = dict(samples=7, minimum_sample_ns=100_000_000, minimum_warmup_ns=250_000_000,
              workers=1, call='public synthetic image generation including allocation and RNG')


def case_specs():
    for name, options in (
        ('line-nearest', dict(length=129, n_dim=1, volume_fraction=0.15)),
        ('line-wrap', dict(length=257, n_dim=1, blob_size_fraction=0.09, boundary_mode='wrap')),
        ('default-plane', {}),
        ('plane-wrap', dict(length=168, blob_size_fraction=0.13, boundary_mode='wrap', volume_fraction=0.82)),
        ('plane-zero-volume', dict(length=192, volume_fraction=0.)),
        ('volume-nearest', dict(length=39, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.1)),
        ('volume-wrap', dict(length=37, n_dim=3, blob_size_fraction=0.19, boundary_mode='wrap', volume_fraction=0.9)),
        ('volume-full', dict(length=34, n_dim=3, blob_size_fraction=0.25, volume_fraction=1.)),
    ):
        yield dict(id=name, api='data.binary_blobs', kwargs=dict(rng=84207, **options))


def make_case(spec):
    return (), spec['kwargs'].copy()
