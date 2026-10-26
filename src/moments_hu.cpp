#include "moments_hu.hpp"

namespace skimagexx::detail {
namespace {

template<class T>
void hu(const T* data, std::ptrdiff_t row_stride,
        std::ptrdiff_t col_stride, T* out7) {
    const auto nu = [=](std::ptrdiff_t row, std::ptrdiff_t col) {
        return data[row * row_stride + col * col_stride];
    };

    // Literal typed temporary and assignment order from scikit-image 0.26.0
    // measure/_moments_cy.pyx. Compile with strict FP and contraction disabled.
    T t0 = nu(3, 0) + nu(1, 2);
    T t1 = nu(2, 1) + nu(0, 3);
    T q0 = t0 * t0;
    T q1 = t1 * t1;
    T n4 = 4 * nu(1, 1);
    T s = nu(2, 0) + nu(0, 2);
    T d = nu(2, 0) - nu(0, 2);
    out7[0] = s;
    out7[1] = d * d + n4 * nu(1, 1);
    out7[3] = q0 + q1;
    out7[5] = d * (q0 - q1) + n4 * t0 * t1;
    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;
    q0 = nu(3, 0) - 3 * nu(1, 2);
    q1 = 3 * nu(2, 1) - nu(0, 3);
    out7[2] = q0 * q0 + q1 * q1;
    out7[4] = q0 * t0 + q1 * t1;
    out7[6] = q1 * t0 - q0 * t1;
}

}  // namespace

void moments_hu(const float* data, std::ptrdiff_t row_stride,
                std::ptrdiff_t col_stride, float* out7) {
    hu(data, row_stride, col_stride, out7);
}

void moments_hu(const double* data, std::ptrdiff_t row_stride,
                std::ptrdiff_t col_stride, double* out7) {
    hu(data, row_stride, col_stride, out7);
}

}  // namespace skimagexx::detail
