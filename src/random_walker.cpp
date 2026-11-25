#include "skimagexx/random_walker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

// Graph weights and output classes follow scikit-image 0.26's
// random_walker_segmentation.py (third_party/scikit-image-LICENSE.txt).
namespace skimagexx {
namespace {

constexpr std::size_t max_classes = 32;

struct Markers {
    std::vector<int> classes;
    std::vector<int> unknown;
    std::size_t count;
};

template<class T>
std::optional<Markers> prepare_markers(const void* input, std::size_t size) {
    const auto* values = static_cast<const T*>(input);
    std::vector<T> seeds;
    for (std::size_t i = 0; i < size; ++i) {
        const T value = values[i];
        if constexpr (std::is_signed_v<T>) {
            if (value < 0) return std::nullopt;
        }
        if (value && std::find(seeds.begin(), seeds.end(), value) == seeds.end()) {
            if (seeds.size() == max_classes) return std::nullopt;
            seeds.push_back(value);
        }
    }
    if (seeds.empty()) return std::nullopt;
    std::sort(seeds.begin(), seeds.end());

    Markers result{{}, {}, seeds.size()};
    result.classes.resize(size);
    result.unknown.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        const T value = values[i];
        if (!value) {
            result.unknown.push_back(static_cast<int>(i));
        } else {
            const auto seed = std::lower_bound(seeds.begin(), seeds.end(), value);
            result.classes[i] = static_cast<int>(seed - seeds.begin()) + 1;
        }
    }
    if (result.unknown.empty()) return std::nullopt;
    return result;
}

std::optional<Markers> prepare_markers(const void* data, PixelType type, std::size_t count) {
    switch (type) {
        case PixelType::int8: return prepare_markers<std::int8_t>(data, count);
        case PixelType::uint8: return prepare_markers<std::uint8_t>(data, count);
        case PixelType::int16: return prepare_markers<std::int16_t>(data, count);
        case PixelType::uint16: return prepare_markers<std::uint16_t>(data, count);
        case PixelType::int32: return prepare_markers<std::int32_t>(data, count);
        case PixelType::uint32: return prepare_markers<std::uint32_t>(data, count);
        case PixelType::int64: return prepare_markers<std::int64_t>(data, count);
        case PixelType::uint64: return prepare_markers<std::uint64_t>(data, count);
        default: return std::nullopt;
    }
}

struct Laplacian {
    std::vector<int> columns;
    std::vector<double> values;
    std::vector<std::size_t> rows;
    std::vector<double> inverse_diagonal;
    std::vector<double> rhs;

    void apply(const std::vector<double>& input, std::vector<double>& output) const {
        for (std::size_t i = 0; i < output.size(); ++i) {
            double value = 0;
            for (auto at = rows[i]; at < rows[i + 1]; ++at)
                value += values[at] * input[columns[at]];
            output[i] = value;
        }
    }
};

template<class T>
double weight_scale(const T* data, std::size_t size, double beta) {
    using Float = std::conditional_t<std::is_same_v<T, float>, float, double>;
    const double mean = std::accumulate(data, data + size, 0.0) / size;
    const auto center = static_cast<Float>(mean);
    double variance = 0;
    bool varied = false;
    for (std::size_t i = 0; i < size; ++i) {
        const Float delta = data[i] - center;
        const Float square = delta * delta;
        variance += square;
        varied |= data[i] != data[0];
    }
    const auto deviation = std::sqrt(static_cast<Float>(variance / size));
    if (!varied || !(deviation > 0) || !std::isfinite(deviation))
        return std::numeric_limits<double>::quiet_NaN();
    return -static_cast<Float>(beta) / (Float{10} * deviation);
}

template<class T>
std::optional<Laplacian> build_laplacian(
    const T* data, const Markers& markers, const std::vector<std::size_t>& shape,
    const RandomWalkerOptions& options) {
    const auto pixels = markers.classes.size();
    const auto unknowns = markers.unknown.size();
    const auto scale = weight_scale(data, pixels, options.beta);
    if (!std::isfinite(scale) || (scale == 0.0 && options.beta != 0.0))
        return std::nullopt;

    const auto rank = shape.size();
    std::array<std::size_t, 3> step{};
    std::array<std::vector<double>, 3> edges;
    std::size_t stride = 1;
    for (auto axis = rank; axis-- > 0;) {
        step[axis] = stride;
        stride *= shape[axis];
        if (shape[axis] == 1) continue;
        auto& weights = edges[axis];
        weights.resize(pixels);
        for (std::size_t block = 0; block < pixels; block += stride)
            for (std::size_t j = block; j < block + stride - step[axis]; ++j) {
                const T diff = data[j + step[axis]] - data[j];
                const double gradient = diff / options.spacing[axis];
                const double weight = std::exp(scale * (gradient * gradient)) + 1e-10;
                if (!std::isfinite(weight)) return std::nullopt;
                weights[j] = weight;
            }
    }

    std::vector<int> index(pixels);
    for (std::size_t i = 0; i < unknowns; ++i)
        index[markers.unknown[i]] = static_cast<int>(i);

    Laplacian laplacian;
    laplacian.columns.reserve(unknowns * (2 * rank + 1));
    laplacian.values.reserve(unknowns * (2 * rank + 1));
    laplacian.rows.reserve(unknowns + 1);
    laplacian.inverse_diagonal.resize(unknowns);
    laplacian.rhs.resize(unknowns * markers.count);

    for (std::size_t row = 0; row < unknowns; ++row) {
        const std::size_t pixel = markers.unknown[row];
        double diagonal = 0;
        const auto add_neighbor = [&](std::size_t neighbor, double weight) {
            diagonal += weight;
            const auto label = markers.classes[neighbor];
            if (label) {
                laplacian.rhs[(label - 1) * unknowns + row] += weight;
            } else {
                laplacian.columns.push_back(index[neighbor]);
                laplacian.values.push_back(-weight);
            }
        };
        laplacian.rows.push_back(laplacian.columns.size());
        for (std::size_t axis = 0; axis < rank; ++axis)
            if ((pixel / step[axis]) % shape[axis] != 0) {
                const auto neighbor = pixel - step[axis];
                add_neighbor(neighbor, edges[axis][neighbor]);
            }
        const auto center = laplacian.values.size();
        laplacian.columns.push_back(static_cast<int>(row));
        laplacian.values.push_back(0);
        for (auto axis = rank; axis-- > 0;)
            if ((pixel / step[axis]) % shape[axis] + 1 != shape[axis])
                add_neighbor(pixel + step[axis], edges[axis][pixel]);
        laplacian.values[center] = diagonal;
        laplacian.inverse_diagonal[row] = 1.0 / diagonal;
    }
    laplacian.rows.push_back(laplacian.columns.size());
    return laplacian;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

struct ConjugateGradient {
    std::vector<double> solution;
    std::vector<double> residual;
    std::vector<double> direction;
    std::vector<double> product;

    explicit ConjugateGradient(std::size_t size)
        : solution(size), residual(size), direction(size), product(size) {}

    bool solve(const Laplacian& laplacian, const double* rhs, double tolerance) {
        const auto size = solution.size();
        std::copy_n(rhs, size, residual.begin());
        std::fill(solution.begin(), solution.end(), 0.0);
        double norm = std::sqrt(dot(residual, residual));
        if (norm == 0) return true;
        const double threshold = tolerance * norm;
        double previous = 0;
        for (std::size_t iter = 0; iter < 10 * size; ++iter) {
            if (norm < threshold || norm == 0) return true;
            double rho = 0;
            for (std::size_t i = 0; i < size; ++i) {
                product[i] = laplacian.inverse_diagonal[i] * residual[i];
                rho += residual[i] * product[i];
            }
            if (iter == 0) {
                direction = product;
            } else {
                const auto beta = rho / previous;
                for (std::size_t i = 0; i < size; ++i)
                    direction[i] = product[i] + beta * direction[i];
            }
            laplacian.apply(direction, product);
            const auto curvature = dot(direction, product);
            if (!(curvature > 0.0) || !std::isfinite(curvature)) return false;
            const auto alpha = rho / curvature;
            double norm_squared = 0;
            for (std::size_t i = 0; i < size; ++i) {
                solution[i] += alpha * direction[i];
                residual[i] -= alpha * product[i];
                norm_squared += residual[i] * residual[i];
            }
            norm = std::sqrt(norm_squared);
            previous = rho;
        }
        return false;
    }
};

std::optional<RandomWalkerResult> solve(
    const Laplacian& laplacian, const Markers& markers, const RandomWalkerOptions& options) {
    const auto size = markers.classes.size();
    const auto unknowns = markers.unknown.size();
    RandomWalkerResult result;
    result.classes = markers.count;
    if (options.return_probabilities)
        result.probabilities.resize(size * markers.count);
    else
        result.labels.assign(markers.classes.begin(), markers.classes.end());

    ConjugateGradient cg(unknowns);
    std::vector<double> best(options.return_probabilities ? 0 : unknowns);
    for (std::size_t label = 0; label < markers.count; ++label) {
        if (!cg.solve(laplacian, laplacian.rhs.data() + label * unknowns, options.tolerance))
            result.convergence_warning = true;
        for (std::size_t i = 0; i < unknowns; ++i) {
            const double probability = cg.solution[i];
            if (!std::isfinite(probability)) return std::nullopt;
            if (probability < -options.probability_tolerance ||
                probability > 1.0 + options.probability_tolerance)
                result.probability_warning = true;
            const auto pixel = markers.unknown[i];
            if (options.return_probabilities) {
                result.probabilities[label * size + pixel] = probability;
            } else if (label == 0 || probability > best[i]) {
                best[i] = probability;
                result.labels[pixel] = static_cast<std::int64_t>(label + 1);
            }
        }
    }
    if (options.return_probabilities)
        for (std::size_t pixel = 0; pixel < size; ++pixel)
            if (const auto label = markers.classes[pixel])
                result.probabilities[(label - 1) * size + pixel] = 1.0;
    return result;
}

}  // namespace

std::optional<RandomWalkerResult> random_walker(
    const void* data, PixelType data_type, const void* markers, PixelType marker_type,
    const std::vector<std::size_t>& shape, const RandomWalkerOptions& options) {
    if (!data || !markers || (shape.size() != 2 && shape.size() != 3) ||
        !std::isfinite(options.beta) || options.beta < 0 ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0 ||
        !std::isfinite(options.probability_tolerance) || options.probability_tolerance < 0)
        return std::nullopt;
    // State indices are int. The same bound leaves address space for the
    // class-major output and CSR allocation on 32-bit hosts.
    constexpr auto max_pixels = std::min(
        std::size_t(std::numeric_limits<int>::max()) / 10,
        std::numeric_limits<std::size_t>::max() / (max_classes * sizeof(double)));
    std::size_t pixels = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (!shape[axis] || shape[axis] > max_pixels / pixels ||
            !std::isfinite(options.spacing[axis]) || options.spacing[axis] <= 0)
            return std::nullopt;
        pixels *= shape[axis];
    }
    auto prepared = prepare_markers(markers, marker_type, pixels);
    if (!prepared) return std::nullopt;
    std::optional<Laplacian> laplacian;
    switch (data_type) {
        case PixelType::float64:
            laplacian = build_laplacian(static_cast<const double*>(data), *prepared, shape, options);
            break;
        case PixelType::float32:
            laplacian = build_laplacian(static_cast<const float*>(data), *prepared, shape, options);
            break;
        case PixelType::uint8: {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            std::vector<double> normalized(pixels);
            std::transform(bytes, bytes + pixels, normalized.begin(),
                           [](std::uint8_t value) { return value * (1.0 / 255.0); });
            laplacian = build_laplacian(normalized.data(), *prepared, shape, options);
            break;
        }
        default:
            return std::nullopt;
    }
    if (!laplacian) return std::nullopt;
    return solve(*laplacian, *prepared, options);
}

}  // namespace skimagexx
