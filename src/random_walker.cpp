#include "skimagexx/random_walker.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

#if defined(SKIMAGEXX_AVX2)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define WALKER_AVX2 __attribute__((target("avx2")))
#else
#define WALKER_AVX2
#endif
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

// Graph weights and output classes follow scikit-image 0.26's
// random_walker_segmentation.py (third_party/scikit-image-LICENSE.txt).
namespace skimagexx {
namespace {

constexpr std::size_t max_classes = 32;

struct Markers {
    // Positive entries are seed classes. Zeros are unknown pixels.
    std::vector<int> nodes;
    std::vector<int> unknown;
    std::size_t count;
};

template<class T>
std::optional<Markers> prepare_markers(const void* input, std::size_t size) {
    const auto* values = static_cast<const T*>(input);
    std::array<T, max_classes> seeds;
    auto seed_end = seeds.begin();
    std::size_t unknowns = 0;
    for (std::size_t i = 0; i < size; ++i) {
        const T value = values[i];
        if constexpr (std::is_signed_v<T>) {
            if (value < 0) return std::nullopt;
        }
        if (!value) {
            ++unknowns;
        } else if (std::find(seeds.begin(), seed_end, value) == seed_end) {
            if (seed_end == seeds.end()) return std::nullopt;
            *seed_end++ = value;
        }
    }
    if (seed_end == seeds.begin() || unknowns == 0) return std::nullopt;
    std::sort(seeds.begin(), seed_end);

    Markers result{{}, {}, static_cast<std::size_t>(seed_end - seeds.begin())};
    result.nodes.reserve(size);
    result.unknown.reserve(unknowns);
    for (std::size_t i = 0; i < size; ++i) {
        const T value = values[i];
        if (!value) {
            result.nodes.push_back(0);
            result.unknown.push_back(static_cast<int>(i));
        } else {
            const auto seed = std::lower_bound(seeds.begin(), seed_end, value);
            result.nodes.push_back(static_cast<int>(seed - seeds.begin()) + 1);
        }
    }
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

struct Grid {
    std::size_t depth, height, width, row, plane, storage, begin, end;

    explicit Grid(const std::vector<std::size_t>& shape)
        : depth(shape.size() == 3 ? shape[0] : 1),
          height(shape[shape.size() - 2]), width(shape.back()),
          row(width + 2), plane(row * (height + 2)),
          storage(plane * (depth > 1 ? depth + 2 : 1)),
          begin((depth > 1 ? plane : 0) + row + 1),
          end(begin + (depth - 1) * plane + (height - 1) * row + width) {}

    template<class Visit>
    bool visit_pixels(Visit visit) const {
        std::size_t i = 0;
        for (std::size_t z = 0; z < depth; ++z)
            for (std::size_t y = 0; y < height; ++y) {
                const auto start = begin + z * plane + y * row;
                for (std::size_t x = 0; x < width; ++x, ++i)
                    if (!visit(i, start + x, z, y, x)) return false;
            }
        return true;
    }
};

struct Laplacian {
    Grid grid;
    // Off-diagonal coefficients indexed by the earlier endpoint of each edge.
    std::vector<double> w_x, w_y, w_z;
    std::vector<double> diag, inv_diag;
    std::vector<std::size_t> padded_unknown;
    std::vector<double> rhs;

    Laplacian(Grid grid, std::size_t unknowns, std::size_t classes)
        : grid(grid), w_x(grid.storage), w_y(grid.storage),
          w_z(grid.depth > 1 ? grid.storage : 0), diag(grid.storage),
          inv_diag(grid.storage), padded_unknown(unknowns), rhs(unknowns * classes) {}
};

struct StencilTerm {
    const double* weight;
    std::ptrdiff_t offset;
};

template<bool Volume>
auto stencil_terms(const Laplacian& laplacian) {
    const auto begin = laplacian.grid.begin;
    const auto row = static_cast<std::ptrdiff_t>(laplacian.grid.row);
    const StencilTerm up{laplacian.w_y.data() + (begin - row), -row};
    const StencilTerm left{laplacian.w_x.data() + (begin - 1), -1};
    const StencilTerm center{laplacian.diag.data() + begin, 0};
    const StencilTerm right{laplacian.w_x.data() + begin, 1};
    const StencilTerm down{laplacian.w_y.data() + begin, row};
    // Match the original CSR column order, from the lowest to highest index.
    if constexpr (Volume) {
        const auto plane = static_cast<std::ptrdiff_t>(laplacian.grid.plane);
        return std::array<StencilTerm, 7>{{
            {laplacian.w_z.data() + (begin - plane), -plane},
            up, left, center, right, down, {laplacian.w_z.data() + begin, plane}
        }};
    } else {
        return std::array<StencilTerm, 5>{up, left, center, right, down};
    }
}

template<std::size_t Rhs, std::size_t Terms>
void apply_stencil(
    const std::array<StencilTerm, Terms>& terms, const std::array<const double*, Rhs>& p,
    const std::array<double*, Rhs>& ap, std::ptrdiff_t begin, std::ptrdiff_t end,
    std::array<double, Rhs>& curvature) {
    for (auto i = begin; i < end; ++i) {
        std::array<double, Rhs> sum{};
        for (const auto& term : terms) {
            const double weight = term.weight[i];
            for (std::size_t j = 0; j < Rhs; ++j)
                sum[j] += weight * p[j][i + term.offset];
        }
        for (std::size_t j = 0; j < Rhs; ++j) {
            ap[j][i] = sum[j];
            curvature[j] += p[j][i] * sum[j];
        }
    }
}

#if defined(SKIMAGEXX_AVX2)
WALKER_AVX2 inline double add_in_order(double sum, __m256d lanes) {
    // The original CG folds dot products in unknown-pixel order. Padding
    // and seed rows contribute zeros; the SIMD lanes must still fold in order.
    const auto low = _mm256_castpd256_pd128(lanes);
    const auto high = _mm256_extractf128_pd(lanes, 1);
    sum += _mm_cvtsd_f64(low);
    sum += _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
    sum += _mm_cvtsd_f64(high);
    return sum + _mm_cvtsd_f64(_mm_unpackhi_pd(high, high));
}

template<std::size_t Rhs, std::size_t Terms>
WALKER_AVX2 void apply_stencil_avx2(
    const std::array<StencilTerm, Terms>& terms, const std::array<const double*, Rhs>& p,
    const std::array<double*, Rhs>& ap, std::ptrdiff_t end,
    std::array<double, Rhs>& curvature) {
    std::ptrdiff_t i = 0;
    for (; i + 4 <= end; i += 4) {
        __m256d sum[Rhs];
        for (std::size_t j = 0; j < Rhs; ++j) sum[j] = _mm256_setzero_pd();
        for (const auto& term : terms) {
            const auto weight = _mm256_loadu_pd(term.weight + i);
            for (std::size_t j = 0; j < Rhs; ++j) {
                const auto neighbor = _mm256_loadu_pd(p[j] + (i + term.offset));
                sum[j] = _mm256_add_pd(sum[j], _mm256_mul_pd(weight, neighbor));
            }
        }
        for (std::size_t j = 0; j < Rhs; ++j) {
            _mm256_storeu_pd(ap[j] + i, sum[j]);
            curvature[j] = add_in_order(curvature[j], _mm256_mul_pd(_mm256_loadu_pd(p[j] + i), sum[j]));
        }
    }
    apply_stencil(terms, p, ap, i, end, curvature);
}
#endif

template<std::size_t Rhs, std::size_t Terms>
std::array<double, Rhs> apply_and_dot(
    const std::array<StencilTerm, Terms>& terms, const std::array<const double*, Rhs>& p,
    const std::array<double*, Rhs>& ap, std::ptrdiff_t size, [[maybe_unused]] bool simd) {
    std::array<double, Rhs> curvature{};
#if defined(SKIMAGEXX_AVX2)
    if (simd) {
        apply_stencil_avx2(terms, p, ap, size, curvature);
        return curvature;
    }
#endif
    apply_stencil(terms, p, ap, 0, size, curvature);
    return curvature;
}

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
    const auto pixels = markers.nodes.size();
    const auto unknowns = markers.unknown.size();
    const auto scale = weight_scale(data, pixels, options.beta);
    if (!std::isfinite(scale) || (scale == 0.0 && options.beta != 0.0))
        return std::nullopt;

    const Grid grid(shape);
    Laplacian laplacian(grid, unknowns, markers.count);
    const auto y_step = grid.width;
    const auto z_step = grid.width * grid.height;
    const auto w_x = options.spacing[shape.size() - 1];
    const auto w_y = options.spacing[shape.size() - 2];
    const auto w_z = options.spacing[0];
    if (!grid.visit_pixels([&](auto i, auto padded, auto z, auto y, auto x) {
        const auto set_edge = [&](double& coefficient, std::size_t step, double spacing) {
            const T diff = data[i + step] - data[i];
            const double gradient = diff / spacing;
            coefficient = -(std::exp(scale * (gradient * gradient)) + 1e-10);
            return std::isfinite(coefficient);
        };
        return (x + 1 == grid.width || set_edge(laplacian.w_x[padded], 1, w_x)) &&
               (y + 1 == grid.height || set_edge(laplacian.w_y[padded], y_step, w_y)) &&
               (z + 1 == grid.depth || set_edge(laplacian.w_z[padded], z_step, w_z));
    })) return std::nullopt;

    std::size_t unknown = 0;
    grid.visit_pixels([&](auto i, auto padded, auto z, auto y, auto x) {
        const bool seeded = markers.nodes[i] > 0;
        double diagonal = 0;
        const auto add = [&](int node, double coefficient) {
            diagonal -= coefficient;
            if (node > 0)
                laplacian.rhs[(node - 1) * unknowns + unknown] -= coefficient;
        };
        const auto add_back = [&](std::size_t step, double& coefficient) {
            const int node = markers.nodes[i - step];
            if (!seeded) add(node, coefficient);
            // Both endpoints have now seen this edge. Remove seed edges from
            // the stencil, retaining their diagonal and right-hand-side terms.
            if (seeded || node > 0) coefficient = 0;
        };
        if (z != 0) add_back(z_step, laplacian.w_z[padded - grid.plane]);
        if (y != 0) add_back(y_step, laplacian.w_y[padded - grid.row]);
        if (x != 0) add_back(1, laplacian.w_x[padded - 1]);
        if (!seeded) {
            if (x + 1 != grid.width) add(markers.nodes[i + 1], laplacian.w_x[padded]);
            if (y + 1 != grid.height) add(markers.nodes[i + y_step], laplacian.w_y[padded]);
            if (z + 1 != grid.depth) add(markers.nodes[i + z_step], laplacian.w_z[padded]);
            laplacian.diag[padded] = diagonal;
            laplacian.inv_diag[padded] = 1.0 / diagonal;
            laplacian.padded_unknown[unknown++] = padded;
        }
        return true;
    });
    return laplacian;
}

struct ConjugateGradient {
    std::vector<double> solution;
    std::vector<double> residual;
    std::vector<double> direction;
    std::vector<double> product;
    double rho = 0;
    double norm = 0;
    double threshold = 0;
    double previous = 0;
    bool active = false;
    bool converged = false;

    explicit ConjugateGradient(std::size_t size)
        : solution(size), residual(size), direction(size), product(size) {}

    void initialize(const Laplacian& laplacian, const double* rhs, double tolerance, bool simd);
};

struct Reductions {
    double norm_squared = 0;
    double rho = 0;
};

void initial_residual(
    const double* inverse, const double* residual, double* direction,
    std::size_t begin, std::size_t end, Reductions& sums) {
    for (auto i = begin; i < end; ++i) {
        const double r = residual[i];
        const double z = inverse[i] * r;
        direction[i] = z;
        sums.norm_squared += r * r;
        sums.rho += r * z;
    }
}

void update_residual(
    ConjugateGradient& cg, const double* inverse, double alpha,
    std::size_t begin, std::size_t end, Reductions& sums) {
    for (auto i = begin; i < end; ++i) {
        cg.solution[i] += alpha * cg.direction[i];
        const double r = cg.residual[i] - alpha * cg.product[i];
        const double z = inverse[i] * r;
        cg.residual[i] = r;
        cg.product[i] = z;  // The stencil product is consumed above.
        sums.norm_squared += r * r;
        sums.rho += r * z;
    }
}

#if defined(SKIMAGEXX_AVX2)
WALKER_AVX2 void initial_residual_avx2(
    const double* inverse, const double* residual, double* direction,
    std::size_t begin, std::size_t end, Reductions& sums) {
    auto i = begin;
    for (; i + 4 <= end; i += 4) {
        const auto r = _mm256_loadu_pd(residual + i);
        const auto z = _mm256_mul_pd(_mm256_loadu_pd(inverse + i), r);
        _mm256_storeu_pd(direction + i, z);
        sums.norm_squared = add_in_order(sums.norm_squared, _mm256_mul_pd(r, r));
        sums.rho = add_in_order(sums.rho, _mm256_mul_pd(r, z));
    }
    initial_residual(inverse, residual, direction, i, end, sums);
}

WALKER_AVX2 std::size_t update_direction_avx2(
    double* direction, const double* scaled, double beta, std::size_t begin, std::size_t end) {
    const auto b = _mm256_set1_pd(beta);
    auto i = begin;
    for (; i + 4 <= end; i += 4) {
        const auto p = _mm256_loadu_pd(direction + i);
        const auto z = _mm256_loadu_pd(scaled + i);
        _mm256_storeu_pd(direction + i, _mm256_add_pd(z, _mm256_mul_pd(b, p)));
    }
    return i;
}

WALKER_AVX2 void update_residual_avx2(
    ConjugateGradient& cg, const double* inverse, double alpha,
    std::size_t begin, std::size_t end, Reductions& sums) {
    auto* solution = cg.solution.data();
    auto* residual = cg.residual.data();
    auto* product = cg.product.data();
    const auto* direction = cg.direction.data();
    const auto a = _mm256_set1_pd(alpha);
    auto i = begin;
    for (; i + 4 <= end; i += 4) {
        const auto x = _mm256_loadu_pd(solution + i);
        const auto p = _mm256_loadu_pd(direction + i);
        const auto ap = _mm256_loadu_pd(product + i);
        const auto update = _mm256_sub_pd(_mm256_loadu_pd(residual + i), _mm256_mul_pd(a, ap));
        const auto z = _mm256_mul_pd(_mm256_loadu_pd(inverse + i), update);
        _mm256_storeu_pd(solution + i, _mm256_add_pd(x, _mm256_mul_pd(a, p)));
        _mm256_storeu_pd(residual + i, update);
        _mm256_storeu_pd(product + i, z);
        sums.norm_squared = add_in_order(sums.norm_squared, _mm256_mul_pd(update, update));
        sums.rho = add_in_order(sums.rho, _mm256_mul_pd(update, z));
    }
    update_residual(cg, inverse, alpha, i, end, sums);
}
#endif

void ConjugateGradient::initialize(
    const Laplacian& laplacian, const double* rhs, double tolerance, [[maybe_unused]] bool simd) {
    std::fill(solution.begin(), solution.end(), 0.0);
    std::fill(residual.begin(), residual.end(), 0.0);
    for (std::size_t i = 0; i < laplacian.padded_unknown.size(); ++i)
        residual[laplacian.padded_unknown[i]] = rhs[i];
    const auto* inverse = laplacian.inv_diag.data();
    const auto& grid = laplacian.grid;
    Reductions sums;
#if defined(SKIMAGEXX_AVX2)
    if (simd)
        initial_residual_avx2(inverse, residual.data(), direction.data(), grid.begin, grid.end, sums);
    else
#endif
        initial_residual(inverse, residual.data(), direction.data(), grid.begin, grid.end, sums);
    rho = sums.rho;
    norm = std::sqrt(sums.norm_squared);
    threshold = tolerance * norm;
    active = true;
    converged = false;
}

template<std::size_t Rhs, std::size_t Terms>
void advance(
    const std::array<StencilTerm, Terms>& terms, const Laplacian& laplacian,
    const std::array<ConjugateGradient*, Rhs>& states, bool simd) {
    const auto& grid = laplacian.grid;
    std::array<const double*, Rhs> direction;
    std::array<double*, Rhs> product;
    for (std::size_t j = 0; j < Rhs; ++j) {
        direction[j] = states[j]->direction.data() + grid.begin;
        product[j] = states[j]->product.data() + grid.begin;
    }
    const auto curvature = apply_and_dot(terms, direction, product, grid.end - grid.begin, simd);
    for (std::size_t j = 0; j < Rhs; ++j) {
        auto& cg = *states[j];
        if (!(curvature[j] > 0.0) || !std::isfinite(curvature[j])) {
            cg.active = false;
            continue;
        }
        const auto alpha = cg.rho / curvature[j];
        const auto* inverse = laplacian.inv_diag.data();
        Reductions sums;
#if defined(SKIMAGEXX_AVX2)
        if (simd)
            update_residual_avx2(cg, inverse, alpha, grid.begin, grid.end, sums);
        else
#endif
            update_residual(cg, inverse, alpha, grid.begin, grid.end, sums);
        cg.previous = cg.rho;
        cg.rho = sums.rho;
        cg.norm = std::sqrt(sums.norm_squared);
    }
}

template<bool Volume>
void solve_pair(const Laplacian& laplacian, std::array<ConjugateGradient, 2>& states, bool simd) {
    const auto terms = stencil_terms<Volume>(laplacian);
    const auto& grid = laplacian.grid;
    for (std::size_t iteration = 0; iteration < 10 * laplacian.padded_unknown.size(); ++iteration) {
        for (auto& cg : states) {
            if (!cg.active) continue;
            if (cg.norm < cg.threshold || cg.norm == 0) {
                cg.converged = true;
                cg.active = false;
                continue;
            }
            if (iteration != 0) {
                const auto beta = cg.rho / cg.previous;
                auto i = grid.begin;
#if defined(SKIMAGEXX_AVX2)
                if (simd)
                    i = update_direction_avx2(cg.direction.data(), cg.product.data(), beta, i, grid.end);
#endif
                for (; i < grid.end; ++i)
                    cg.direction[i] = cg.product[i] + beta * cg.direction[i];
            }
        }
        if (states[0].active && states[1].active) {
            advance(terms, laplacian, std::array{&states[0], &states[1]}, simd);
        } else if (states[0].active || states[1].active) {
            auto* remaining = states[0].active ? &states[0] : &states[1];
            advance(terms, laplacian, std::array{remaining}, simd);
        } else {
            break;
        }
    }
}

std::optional<RandomWalkerResult> solve(
    const Laplacian& laplacian, const Markers& markers, const RandomWalkerOptions& options) {
    const auto size = markers.nodes.size();
    const auto unknowns = markers.unknown.size();
    RandomWalkerResult result;
    result.classes = markers.count;
    if (options.return_probabilities)
        result.probabilities.resize(size * markers.count);
    else
        result.labels.assign(markers.nodes.begin(), markers.nodes.end());

    std::array states{ConjugateGradient(laplacian.grid.storage),
                      ConjugateGradient(markers.count > 1 ? laplacian.grid.storage : 0)};
    const bool simd = has_avx2();
    std::vector<double> best(options.return_probabilities ? 0 : unknowns);
    for (std::size_t label = 0; label < markers.count; label += 2) {
        const auto count = std::min<std::size_t>(markers.count - label, 2);
        for (std::size_t j = 0; j < count; ++j)
            states[j].initialize(laplacian, laplacian.rhs.data() + (label + j) * unknowns,
                                 options.tolerance, simd);
        if (count == 1) states[1].active = false;
        if (laplacian.grid.depth > 1)
            solve_pair<true>(laplacian, states, simd);
        else
            solve_pair<false>(laplacian, states, simd);
        for (std::size_t j = 0; j < count; ++j) {
            if (!states[j].converged) result.convergence_warning = true;
            for (std::size_t i = 0; i < unknowns; ++i) {
                const double probability = states[j].solution[laplacian.padded_unknown[i]];
                if (!std::isfinite(probability)) return std::nullopt;
                if (probability < -options.probability_tolerance ||
                    probability > 1.0 + options.probability_tolerance)
                    result.probability_warning = true;
                const auto pixel = markers.unknown[i];
                if (options.return_probabilities) {
                    result.probabilities[(label + j) * size + pixel] = probability;
                } else if (label + j == 0 || probability > best[i]) {
                    best[i] = probability;
                    result.labels[pixel] = static_cast<std::int64_t>(label + j + 1);
                }
            }
        }
    }
    if (options.return_probabilities)
        for (std::size_t pixel = 0; pixel < size; ++pixel)
            if (const auto label = markers.nodes[pixel]; label > 0)
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
    // Pixel indices are int. This also bounds byte counts for the class-major
    // output and each padded plane, which has at most 27 entries per pixel.
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

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
