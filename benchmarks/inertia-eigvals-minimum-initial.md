# Inertia eigenvalues: first minimum-Linux result

[Release40383375851](https://github.com/bipxx/scikit-imagexx/actions/runs/40383375851) measures exact source `812a87003898cbe41706933ac9399648d0604316`, merged as `d54f664`. This source **predates the K=3 fusion**. Only **one of six cases beats scikit-image**, with geometric mean original/current **0.393826159x. All five losses and all **84 raw latency samples** are preserved in the [exact raw JSON](inertia-eigvals-minimum-initial.json).

## All six cases

| Image size | Input dtype | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| 256x256 | uint8 | 0.176983 | 0.236260 | 0.749103379x | +0.059277 |
| 256x256 | float32 | 0.149264 | 0.120053 | 1.243321954x | -0.029211 |
| 256x256 | float64 | 0.167388 | 0.227704 | 0.735113750x | +0.060316 |
| 1024x1024 | uint8 | 0.801227 | 4.722088 | 0.169676405x | +3.920861 |
| 1024x1024 | float32 | 0.680703 | 3.068773 | 0.221816048x | +2.388070 |
| 1024x1024 | float64 | 0.624967 | 4.316426 | 0.144788156x | +3.691458 |

## Scope and validation

These are automatic `measure.inertia_tensor_eigvals(image)` calls at sizes 256/1024 and actual image dtypes uint8/float32/float64, using one global worker and seven warmed paired rounds. Timing includes moments, tensor assembly, eigenvalue solving, clipping, allocation and list wrapping. Genuine original central moments/tensors prepare the frozen solver-plus-Weyl checker outside timing. Both implementations run in the same fresh offline process; source/wheel/runtime/package/SDK identity checks agree before and after. A detailed CPU model was not captured. This first original-library comparison is not a prior-C++ gain.

C++ owns image projections, ordinary finite real tensor assembly and the existing symmetric eigenvalue solver. NumPy retains preparation, clipping, scalar-list construction and unsupported solver domains. Supplied T precedence and lower-triangle interpretation are tested. Existing solver tolerances are unchanged, with the frozen tensor uncertainty propagated using the maximum absolute row sum.

The genuine installed product passes all nine public contract groups in normal/scalar modes: 2,426 checks, including 69 eigenvalue tests per mode, plus 1,365 smoke checks. Separate two-profile component evidence passes 1,548 checks and 88 shared checks. Prior tensor/moments records and arrays match exactly. Accepted full supported-profile coverage stays 203 image APIs; source coverage is 223 native-partial image APIs plus three metadata functions.

The [kept K=3 tensor experiment](inertia-k3-same-host.md) measured only the tensor API. Its gain must not be inferred for this eigenvalue API. A fresh combined product and separate 12-case controlled comparison are being prepared for both APIs; this initial raw report remains unchanged.

Raw JSON SHA-256: `accd393189c8049fb7265b6e11c28c32749ec5d4f741d87af9642bdd3ce731c5`. Parent artifact12679444364 SHA-256: `cccfc10f578ac33edc4707e52992c8f70e49ed6d20ad5c1b132cfa5107d84282`.
