#include "skimagexx/structural_similarity.hpp"
#include "skimagexx/metrics.hpp"
#include "skimagexx/uniform_filter.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::tuple<double, double, double, double, bool> metric_fast_reduce(
    const MetricImageView&, const MetricImageView&, std::size_t, MetricNormalization, bool, std::size_t);
template<class T>
std::size_t ssim_vertical_row(const T*, const T*, const T*, const T*, T*, const T*,
                              double*, T*, std::size_t, double);
template<class T>
std::size_t ssim_score_row(const double*, T*, std::size_t, double, T, T, T);
}
#endif

namespace skimagexx {
namespace {

class ScoreEnvironment {
public:
    ScoreEnvironment() : active_(std::feholdexcept(&saved_) == 0) {}
    ~ScoreEnvironment() { if (active_) std::fesetenv(&saved_); }
    bool ready() const {
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
        if (_mm_getcsr() & 0x8040) return false;
#endif
        return active_ && std::fegetround() == FE_TONEAREST;
    }
    bool clean() const {
        return std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) == 0;
    }

private:
    std::fenv_t saved_;
    bool active_;
};

template<class T>
T score_pixel(T a, T b, T aa, T bb, T ab, T norm, T c1, T c2) {
    const T vx = norm * (aa - a * a);
    const T vy = norm * (bb - b * b);
    const T vxy = norm * (ab - a * b);
    const T a1 = (T(2) * a) * b + c1;
    const T a2 = T(2) * vxy + c2;
    const T b1 = (a * a + b * b) + c1;
    const T b2 = (vx + vy) + c2;
    return (a1 * a2) / (b1 * b2);
}

template<class T>
bool filtered_score(const T* left, const T* right, std::size_t rows, std::size_t cols,
                    std::size_t window, T norm, T c1, T c2, double& sum, PixelType type) {
    const auto size = rows * cols;
    const int workers = detail::active_workers(size);
    const std::vector<std::size_t> shape{rows, cols};
    std::vector<T> scratch(size);
    const auto filter = [&](const T* input, T* output) {
        return detail::uniform_filter1d_into(input, scratch.data(), type, shape, 0, window, workers) &&
               detail::uniform_filter1d_into(scratch.data(), output, type, shape, 1, window, workers);
    };

    std::vector<T> ux(size), uy(size), uxx(size), uyy(size), uxy(size);
    if (!filter(left, ux.data()) || !filter(right, uy.data())) return false;
    const auto filter_product = [&](const T* a, const T* b, std::vector<T>& output) {
        for (std::size_t i = 0; i < size; ++i) output[i] = a[i] * b[i];
        return filter(output.data(), output.data());
    };
    if (!filter_product(left, left, uxx) || !filter_product(right, right, uyy) ||
        !filter_product(left, right, uxy)) return false;

    const auto pad = (window - 1) / 2;
    for (std::size_t y = pad; y < rows - pad; ++y) {
        double row_sum = 0.0;
        for (std::size_t x = pad; x < cols - pad; ++x) {
            const auto i = y * cols + x;
            row_sum += score_pixel(ux[i], uy[i], uxx[i], uyy[i], uxy[i], norm, c1, c2);
        }
        sum += row_sum;
    }
    return true;
}

template<class T>
double rolling_score(const T* left, const T* right, std::size_t rows, std::size_t cols,
                     std::size_t window, T norm, T c1, T c2) {
    const auto pad = window / 2;
    const auto width = cols - 2 * pad;
    const double divisor = static_cast<double>(window);
    // The image owns x and y. Cache only their three rounded products for
    // subtraction when a row leaves the window.
    const auto pitch = 3 * cols;
    const std::unique_ptr<T[]> ring(new T[window * pitch]);
    std::vector<double> vertical(5 * cols);
    const std::unique_ptr<T[]> means(new T[5 * cols]);
    const std::unique_ptr<double[]> horizontal(new double[5 * width]);
    const std::unique_ptr<T[]> values(new T[width]);
#ifdef SKIMAGEXX_AVX2
    const bool vectorized = has_avx2();
#endif
    const auto advance = [&](std::size_t incoming, std::size_t outgoing, T* output) {
        const T* a = left + incoming * cols;
        const T* b = right + incoming * cols;
        auto* products = ring.get() + (incoming % window) * pitch;
        const bool rolling = outgoing != rows;
        const T* old_a = rolling ? left + outgoing * cols : nullptr;
        const T* old_b = rolling ? right + outgoing * cols : nullptr;
        const auto* old_products = rolling ? ring.get() + (outgoing % window) * pitch : nullptr;
        std::size_t x = 0;
#ifdef SKIMAGEXX_AVX2
        if (vectorized)
            x = simd::ssim_vertical_row(a, b, old_a, old_b, products, old_products,
                                       vertical.data(), output, cols, divisor);
#endif
        for (; x < cols; ++x) {
            const T aa = a[x] * a[x], bb = b[x] * b[x], ab = a[x] * b[x];
            const T next[] = {a[x], b[x], aa, bb, ab};
            if (rolling) {
                const T previous[] = {old_a[x], old_b[x], old_products[x],
                                      old_products[cols + x], old_products[2 * cols + x]};
                for (std::size_t m = 0; m < 5; ++m)
                    vertical[m * cols + x] += double(next[m]) - double(previous[m]);
            } else {
                for (std::size_t m = 0; m < 5; ++m)
                    vertical[m * cols + x] += double(next[m]);
            }
            products[x] = aa;
            products[cols + x] = bb;
            products[2 * cols + x] = ab;
            if (output)
                for (std::size_t m = 0; m < 5; ++m) output[m * cols + x] = T(vertical[m * cols + x] / divisor);
        }
    };

    // Match the two uniform passes, including reflected leading-edge sums
    // and their subtract-then-add updates before the first interior pixel.
    for (std::size_t k = pad; k > 0; --k) advance(k - 1, rows, nullptr);
    for (std::size_t k = 0; k <= pad; ++k) advance(k, rows, nullptr);
    for (std::size_t y = 1; y < pad; ++y) advance(y + pad, pad - y, nullptr);

    double sum = 0.0;
    for (std::size_t y = pad; y < rows - pad; ++y) {
        const auto outgoing = y <= pad ? pad - y : y - pad - 1;
        advance(y + pad, outgoing, means.get());
        for (std::size_t m = 0; m < 5; ++m) {
            const auto* row = means.get() + m * cols;
            auto* out = horizontal.get() + m * width;
            double total = 0.0;
            for (std::size_t k = pad; k > 0; --k) total += row[k - 1];
            for (std::size_t k = 0; k <= pad; ++k) total += row[k];
            for (std::size_t x = 1; x <= pad; ++x)
                total += double(row[x + pad]) - double(row[pad - x]);
            out[0] = total;
            for (std::size_t x = 1; x < width; ++x) {
                total += double(row[x + window - 1]) - double(row[x - 1]);
                out[x] = total;
            }
        }
        std::size_t x = 0;
#ifdef SKIMAGEXX_AVX2
        if (vectorized)
            x = simd::ssim_score_row(horizontal.get(), values.get(), width, divisor, norm, c1, c2);
#endif
        for (; x < width; ++x) {
            const auto mean = [&](std::size_t m) { return T(horizontal[m * width + x] / divisor); };
            values[x] = score_pixel(mean(0), mean(1), mean(2), mean(3), mean(4), norm, c1, c2);
        }
        double row_sum = 0.0;
        for (std::size_t col = 0; col < width; ++col) row_sum += values[col];
        sum += row_sum;
    }
    return sum;
}

template<class T>
bool score_for(const T* left, const T* right, std::size_t rows, std::size_t cols,
               std::size_t window, double data_range, bool sample, double& score,
               PixelType type) {
    const auto size = rows * cols;
    if (reinterpret_cast<std::uintptr_t>(left) % alignof(T) ||
        reinterpret_cast<std::uintptr_t>(right) % alignof(T))
        throw std::invalid_argument("SSIM input buffers must be aligned");

    const ScoreEnvironment environment;
    if (!environment.ready()) return false;
    const auto supported_pixel = [data_range](T value) {
        const auto magnitude = std::abs(value);
        return std::isfinite(value) && magnitude <= data_range &&
               (magnitude == 0 || magnitude >= T(1e-8));
    };
    for (std::size_t i = 0; i < size; ++i)
        if (!supported_pixel(left[i]) || !supported_pixel(right[i])) return false;

    const double pixels = static_cast<double>(window * window);
    const T norm = sample ? T(pixels / (pixels - 1.0)) : T(1.0);
    const double k1 = 0.01 * data_range, k2 = 0.03 * data_range;
    const T c1 = T(k1 * k1), c2 = T(k2 * k2);
    const auto pad = window / 2;
    double sum = 0.0;
    if (window < rows && window < cols && cols <= std::numeric_limits<std::size_t>::max() / 5 / window)
        sum = rolling_score(left, right, rows, cols, window, norm, c1, c2);
    else if (!filtered_score(left, right, rows, cols, window, norm, c1, c2, sum, type))
        return false;
    const double result = sum / static_cast<double>((rows - 2 * pad) * (cols - 2 * pad));
    if (!environment.clean()) return false;
    score = result;
    return true;
}

} // namespace

bool structural_similarity_score(const void* left, const void* right, PixelType type,
                                  std::size_t rows, std::size_t cols,
                                  std::size_t window, double data_range,
                                  bool use_sample_covariance, double& score) {
    if (!left || !right || window < 3 || window > 255 || window % 2 == 0 ||
        rows < window || cols < window)
        throw std::invalid_argument("SSIM requires non-null buffers and a valid odd window");
    if (rows > std::numeric_limits<std::size_t>::max() / cols)
        throw std::overflow_error("SSIM input shape overflows");
    if (type != PixelType::float32 && type != PixelType::float64)
        throw std::invalid_argument("SSIM requires float32 or float64 buffers");
    if (!(data_range >= 0.5 && data_range <= 4.0)) return false;
    if (type == PixelType::float32)
        return score_for(static_cast<const float*>(left), static_cast<const float*>(right),
                         rows, cols, window, data_range, use_sample_covariance, score, type);
    return score_for(static_cast<const double*>(left), static_cast<const double*>(right),
                     rows, cols, window, data_range, use_sample_covariance, score, type);
}

} // namespace skimagexx

namespace skimagexx::detail {
// The native library sees the AVX2 build definition used for dispatch.
auto metric_fast_simd() -> std::tuple<double, double, double, double, bool> (*)(
    const MetricImageView&, const MetricImageView&, std::size_t, MetricNormalization, bool, std::size_t) {
#ifdef SKIMAGEXX_AVX2
    return has_avx2() ? simd::metric_fast_reduce : nullptr;
#else
    return nullptr;
#endif
}
} // namespace skimagexx::detail
