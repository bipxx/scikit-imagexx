# Ordered Hough ellipse pilot

The private C++ search accepts ordinary C-contiguous, two-dimensional Boolean and uint8 masks with each extent at most 64 and at most 128 nonzero pixels. Threshold ranges from 0 to 128, minimum major size from 1 to 64, and the optional maximum minor size from 1 to 32. The four supported accuracy settings are 0.75, 1, 1.5 and 2. Mask bytes with any nonzero value mark an edge. Calls outside the bounded domain use the existing compatibility policy.

A mask with no edge pixels returns an owned float64 array of shape (0, 6). A mask with edge pixels returns an owned one-dimensional structured array, including when no detections survive. Its fields are a native NumPy index integer and five float64 values, in the original order. The input stays read-only to this call. Detections follow the original order of endpoint pairs, including the original first-bin tie, strict threshold, orientation convention and height/width dependent round-to-even default radius.

The native search runs serially under round-to-nearest. Disable FP contraction and use strict floating-point semantics when building this component. Histograms use the same exclusive range stop, exact dyadic edges, final-edge inclusion and first maximum as the reference within scope. The SDK accepts a complete C-order mask byte span and returns an owning vector; a disengaged optional reports an unsupported request.

The fixed parity fixture contains 14 stock observations from both retained scikit-image 0.26.0 profiles, including exact per-field and ordered output hashes, a valid strided compatibility case and the original rank error. The fixed warm comparison uses one worker and the four prepared masks from that fixture. Timed entry includes the public call and output allocation. Genuine normal/forced-scalar native ownership, installed SDK qualification and per-case performance are separate release gates.

[Original scikit-image 0.26.0 ellipse search](https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/transform/_hough_transform.pyx#L94)
