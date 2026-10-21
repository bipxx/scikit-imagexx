"""C++ image kernels with an explicitly limited scikit-image parity surface."""
from ._core import gaussian, rgb2gray, sobel, threshold_otsu, set_num_threads, get_num_threads, has_avx2

__version__ = "0.1.0"
__all__ = ["gaussian", "rgb2gray", "sobel", "threshold_otsu", "set_num_threads", "get_num_threads", "has_avx2"]
