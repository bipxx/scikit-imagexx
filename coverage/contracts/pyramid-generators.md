# Gaussian and Laplacian pyramid generators

The native route covers ordinary owning, aligned, C-order arrays of uint8, float32 or float64. It accepts a nonempty two-dimensional grayscale image or a three-dimensional image with an integer channel axis. Axes are capped at 512, total size at 262,144, and float magnitude at 65,536. It accepts plain integer max_layer from -1 to 8, downscale from 1.5 to 4, order 0 or 1, sigma from 0 to 4 or None, and reflect, constant or wrap. Initially unsupported calls lazily delegate through the standard compatibility policy.

Gaussian first yields the converted image. For supported floating input that is the caller's exact array. The next reduction uses that live object. If an edit makes the next input incompatible with the spatial route, that reduction can delegate on the edited object. Gaussian still computes and suppresses the first shape-preserving reduction at exhaustion.

Laplacian keeps each smoothed image for the next resize, and returns the current image minus its smoothing in the same dtype. Editing a yielded residual leaves the continuation independent. The explicit layer limit or the spatial log-based default governs exhaustion, including repeats of an unchanged shape.

Python owns the generator protocol, shape math and typed subtraction. The compiled Gaussian and resize stages come from the existing spatial-pyramid dependency. There is no added C++ or SDK symbol. The initial output/storage, evolving state and stage order use the original scikit-image 0.26.0 captures in the Windows and minimum profiles. The fixed public comparison fully consumes each fresh iterator, including all allocations and yields. Shared integration, live installed coverage, speed and release qualification are separate gates.
