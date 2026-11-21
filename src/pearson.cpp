#include "skimagexx/pearson.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>

namespace skimagexx {
namespace {

constexpr std::size_t block_size = 256;
constexpr double minimum_magnitude = 0x1p-100;
constexpr double maximum_magnitude = 0x1p20;
constexpr double minimum_relative_rms = 0x1p-18;

class PearsonEnvironment {
public:
    PearsonEnvironment() : active_(std::feholdexcept(&saved_) == 0) {}
    ~PearsonEnvironment() { if (active_) std::fesetenv(&saved_); }

    bool ready() const { return active_ && std::fegetround() == FE_TONEAREST; }
    bool clean() const {
        return std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) == 0;
    }

private:
    std::fenv_t saved_;
    bool active_;
};

struct Sum {
    void add(double block) {
        const double corrected = block - error;
        const double next = value + corrected;
        error = (next - value) - corrected;
        value = next;
    }

    double value = 0;
    double error = 0;
};

bool supported(double magnitude) {
    // The upper comparison also rules out NaNs and infinities.
    return magnitude <= maximum_magnitude &&
           (magnitude >= minimum_magnitude || magnitude == 0);
}

template<bool Masked>
bool correlation(const double* left, const double* right, const std::uint8_t* mask,
                 std::size_t count, PearsonResult& result, PearsonBetaTail beta_tail,
                 const PearsonEnvironment& environment) {
    Sum left_sum, right_sum;
    double left_max = 0;
    double right_max = 0;
    std::size_t selected = 0;
    std::size_t first = 0;
    std::size_t second = 0;
    for (std::size_t start = 0; start < count; start += block_size) {
        double left_block = 0;
        double right_block = 0;
        const auto end = std::min(count, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
            if constexpr (Masked) {
                if (mask[i] == 0) continue;
            }
            const auto a = left[i], b = right[i];
            const auto ma = std::abs(a), mb = std::abs(b);
            if (!supported(ma) || !supported(mb)) return false;
            left_max = std::max(left_max, ma);
            right_max = std::max(right_max, mb);
            left_block += a;
            right_block += b;
            if (selected == 0) first = i;
            if (selected == 1) second = i;
            ++selected;
        }
        left_sum.add(left_block);
        right_sum.add(right_block);
    }
    if (selected < 2 || left_max == 0 || right_max == 0) return false;

    if (selected == 2) {
        const auto a = left[second] - left[first], b = right[second] - right[first];
        if (std::abs(a) < 2 * minimum_relative_rms * left_max ||
            std::abs(b) < 2 * minimum_relative_rms * right_max || !environment.clean())
            return false;
        result = {(a > 0) == (b > 0) ? 1.0 : -1.0, 1.0};
        return true;
    }
    if (!beta_tail.evaluate) return false;

    const auto n = static_cast<double>(selected);
    const double left_mean = left_sum.value / n;
    const double right_mean = right_sum.value / n;
    Sum left_ss, right_ss, covariance;
    // Center before squaring. The fixed local blocks limit summation error
    // and let the large positive reductions combine without losing blocks.
    for (std::size_t start = 0; start < count; start += block_size) {
        double left_square = 0;
        double right_square = 0;
        double cross = 0;
        const auto end = std::min(count, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
            if constexpr (Masked) {
                if (mask[i] == 0) continue;
            }
            const auto a = left[i] - left_mean, b = right[i] - right_mean;
            left_square += a * a;
            right_square += b * b;
            cross += a * b;
        }
        left_ss.add(left_square);
        right_ss.add(right_square);
        covariance.add(cross);
    }

    const auto left_floor = minimum_relative_rms * left_max;
    const auto right_floor = minimum_relative_rms * right_max;
    if (left_ss.value < n * left_floor * left_floor ||
        right_ss.value < n * right_floor * right_floor)
        return false;
    const double r = covariance.value / std::sqrt(left_ss.value * right_ss.value);
    const double magnitude = std::abs(r);
    // The n*r^2 bound keeps the provider away from underflow before it is
    // called. The actual returned probability still controls admission.
    if (magnitude > 0.95 || n * magnitude * magnitude > 128 || !environment.clean())
        return false;
    double upper_tail;
    if (!beta_tail.evaluate(n * 0.5 - 1, (1 + magnitude) * 0.5, upper_tail, beta_tail.context))
        return false;
    if (!(upper_tail >= 5e-13 && upper_tail <= 0.5) || !environment.clean())
        return false;
    result = {r, 2 * upper_tail};
    return true;
}

} // namespace

bool pearson_corr_coeff(const double* left, const double* right,
                        const std::uint8_t* mask, std::size_t count,
                        PearsonResult& result, PearsonBetaTail beta_tail) {
    if (!left || !right || count < 2 || count > (std::size_t{1} << 20) ||
        reinterpret_cast<std::uintptr_t>(left) % alignof(double) ||
        reinterpret_cast<std::uintptr_t>(right) % alignof(double))
        return false;
    const PearsonEnvironment environment;
    if (!environment.ready()) return false;
    if (mask)
        return correlation<true>(left, right, mask, count, result, beta_tail, environment);
    return correlation<false>(left, right, nullptr, count, result, beta_tail, environment);
}

} // namespace skimagexx
