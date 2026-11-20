#include "skimagexx/structural_similarity.hpp"
#include "skimagexx/uniform_filter.hpp"
#include "execution.hpp"
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

class ScoreEnvironment {
public:
    ScoreEnvironment() : active_(std::feholdexcept(&saved_) == 0) {}
    ~ScoreEnvironment() { if (active_) std::fesetenv(&saved_); }
    bool ready() const { return active_ && std::fegetround() == FE_TONEAREST; }
    bool clean() const {
        return std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) == 0;
    }

private:
    std::fenv_t saved_;
    bool active_;
};

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

    const double pixels = static_cast<double>(window * window);
    const T norm = sample ? T(pixels / (pixels - 1.0)) : T(1.0);
    const double k1 = 0.01 * data_range;
    const double k2 = 0.03 * data_range;
    const T c1 = T(k1 * k1);
    const T c2 = T(k2 * k2);
    const auto pad = (window - 1) / 2;
    double sum = 0.0;
    for (std::size_t y = pad; y < rows - pad; ++y) {
        double row_sum = 0.0;
        for (std::size_t x = pad; x < cols - pad; ++x) {
            const auto i = y * cols + x;
            const T a = ux[i], b = uy[i];
            const T vx = norm * (uxx[i] - a * a);
            const T vy = norm * (uyy[i] - b * b);
            const T vxy = norm * (uxy[i] - a * b);
            const T a1 = (T(2) * a) * b + c1;
            const T a2 = T(2) * vxy + c2;
            const T b1 = (a * a + b * b) + c1;
            const T b2 = (vx + vy) + c2;
            const T value = (a1 * a2) / (b1 * b2);
            row_sum += static_cast<double>(value);
        }
        sum += row_sum;
    }
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
