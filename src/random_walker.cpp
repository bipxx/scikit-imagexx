#include "skimagexx/random_walker.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
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

#if defined(__SSE2__) || defined(_M_X64)
double add_in_order(double sum, __m128d lanes) {
    sum += _mm_cvtsd_f64(lanes);
    return sum + _mm_cvtsd_f64(_mm_unpackhi_pd(lanes, lanes));
}
#endif

template<bool Simd, std::size_t Terms, std::size_t... I>
double apply_and_dot(
    const std::array<StencilTerm, Terms>& terms, double* direction, double* product,
    std::size_t size, double beta, bool update_direction, std::index_sequence<I...>) {
    const std::array<const double*, Terms> neighbors{(direction + terms[I].offset)...};
    const auto lookahead = static_cast<std::size_t>(terms.back().offset);
    double curvature = 0;
    std::size_t ready = 0;
    constexpr std::size_t tile = 256;
    for (std::size_t i = 0; i < size;) {
        const auto end = std::min(i + tile, size);
        // The stencil needs the new direction through its farthest neighbor.
        // Update those entries before overwriting any of their previous products.
        if (update_direction) {
            const auto needed = std::min(end + lookahead, size);
#if defined(__SSE2__) || defined(_M_X64)
            if constexpr (Simd) {
                const auto b = _mm_set1_pd(beta);
                for (; ready + 2 <= needed; ready += 2) {
                    const auto p = _mm_loadu_pd(direction + ready);
                    const auto z = _mm_loadu_pd(product + ready);
                    _mm_storeu_pd(direction + ready, _mm_add_pd(z, _mm_mul_pd(b, p)));
                }
            }
#endif
            for (; ready < needed; ++ready)
                direction[ready] = product[ready] + beta * direction[ready];
        }
#if defined(__SSE2__) || defined(_M_X64)
        if constexpr (Simd) {
            for (; i + 2 <= end; i += 2) {
                auto sum = _mm_setzero_pd();
                ((sum = _mm_add_pd(sum, _mm_mul_pd(
                      _mm_loadu_pd(terms[I].weight + i), _mm_loadu_pd(neighbors[I] + i)))), ...);
                curvature = add_in_order(curvature, _mm_mul_pd(_mm_loadu_pd(direction + i), sum));
                _mm_storeu_pd(product + i, sum);
            }
        }
#endif
        for (; i < end; ++i) {
            double sum = 0;
            ((sum += terms[I].weight[i] * neighbors[I][i]), ...);
            curvature += direction[i] * sum;
            product[i] = sum;
        }
    }
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
    std::size_t unknown = 0;
    if (!grid.visit_pixels([&](auto i, auto padded, auto z, auto y, auto x) {
        const auto set_edge = [&](double& coefficient, std::size_t step, double spacing) {
            const T diff = data[i + step] - data[i];
            const double gradient = diff / spacing;
            coefficient = -(std::exp(scale * (gradient * gradient)) + 1e-10);
            return std::isfinite(coefficient);
        };
        if ((x + 1 != grid.width && !set_edge(laplacian.w_x[padded], 1, w_x)) ||
            (y + 1 != grid.height && !set_edge(laplacian.w_y[padded], y_step, w_y)) ||
            (z + 1 != grid.depth && !set_edge(laplacian.w_z[padded], z_step, w_z)))
            return false;
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
    })) return std::nullopt;
    return laplacian;
}

struct ConjugateGradient {
    std::vector<double> solution;
    std::vector<double> residual;
    std::vector<double> direction;
    std::vector<double> product;

    explicit ConjugateGradient(std::size_t size)
        : solution(size), residual(size), direction(size), product(size) {}
};

struct Reductions {
    double norm_squared = 0;
    double rho = 0;
};

template<bool Simd>
Reductions update_residual(
    double* x, double* r, const double* p, double* ap, const double* inverse,
    std::size_t size, double alpha) {
    std::size_t i = 0;
    Reductions sums;
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (Simd) {
        const auto a = _mm_set1_pd(alpha);
        auto ordered_sums = _mm_setzero_pd();
        for (; i + 2 <= size; i += 2) {
            const auto advance = _mm_mul_pd(a, _mm_loadu_pd(p + i));
            const auto update = _mm_sub_pd(_mm_loadu_pd(r + i), _mm_mul_pd(a, _mm_loadu_pd(ap + i)));
            const auto z = _mm_mul_pd(_mm_loadu_pd(inverse + i), update);
            _mm_storeu_pd(x + i, _mm_add_pd(_mm_loadu_pd(x + i), advance));
            _mm_storeu_pd(r + i, update);
            _mm_storeu_pd(ap + i, z);
            const auto squares = _mm_mul_pd(update, update);
            const auto dots = _mm_mul_pd(update, z);
            ordered_sums = _mm_add_pd(ordered_sums, _mm_unpacklo_pd(squares, dots));
            ordered_sums = _mm_add_pd(ordered_sums, _mm_unpackhi_pd(squares, dots));
        }
        sums.norm_squared = _mm_cvtsd_f64(ordered_sums);
        sums.rho = _mm_cvtsd_f64(_mm_unpackhi_pd(ordered_sums, ordered_sums));
    }
#endif
    for (; i < size; ++i) {
        x[i] += alpha * p[i];
        const double update = r[i] - alpha * ap[i];
        const double z = inverse[i] * update;
        r[i] = update;
        ap[i] = z;  // Reuse the consumed stencil product for the next direction update.
        sums.norm_squared += update * update;
        sums.rho += update * z;
    }
    return sums;
}

template<bool Simd, std::size_t Terms>
bool solve_cg(
    const std::array<StencilTerm, Terms>& terms, const Laplacian& laplacian,
    ConjugateGradient& cg, Reductions sums, double tolerance) {
    const auto& grid = laplacian.grid;
    auto* const x = cg.solution.data() + grid.begin;
    auto* const r = cg.residual.data() + grid.begin;
    auto* const p = cg.direction.data() + grid.begin;
    auto* const ap = cg.product.data() + grid.begin;
    const auto* const inverse = laplacian.inv_diag.data() + grid.begin;
    const auto span = grid.end - grid.begin;
    const auto threshold = tolerance * std::sqrt(sums.norm_squared);
    double previous = 0;
    for (std::size_t iteration = 0; iteration < 10 * laplacian.padded_unknown.size(); ++iteration) {
        const auto norm = std::sqrt(sums.norm_squared);
        if (norm < threshold || norm == 0) return true;
        const auto beta = iteration == 0 ? 0.0 : sums.rho / previous;
        const auto curvature = apply_and_dot<Simd>(
            terms, p, ap, span, beta, iteration != 0, std::make_index_sequence<Terms>{});
        if (!(curvature > 0.0) || !std::isfinite(curvature)) return false;
        previous = sums.rho;
        sums = update_residual<Simd>(x, r, p, ap, inverse, span, previous / curvature);
    }
    return false;
}

template<bool Simd, bool Volume>
bool solve_classes(
    const Laplacian& laplacian, const Markers& markers, const RandomWalkerOptions& options,
    RandomWalkerResult& result) {
    const auto size = markers.nodes.size();
    const auto unknowns = markers.unknown.size();
    const auto terms = stencil_terms<Volume>(laplacian);
    ConjugateGradient cg(laplacian.grid.storage);
    std::vector<double> best(options.return_probabilities ? 0 : unknowns);
    for (std::size_t label = 0; label < markers.count; ++label) {
        if (label != 0) {
            std::fill(cg.solution.begin(), cg.solution.end(), 0.0);
            std::fill(cg.residual.begin(), cg.residual.end(), 0.0);
            std::fill(cg.direction.begin(), cg.direction.end(), 0.0);
        }
        const auto* rhs = laplacian.rhs.data() + label * unknowns;
        Reductions sums;
        for (std::size_t i = 0; i < unknowns; ++i) {
            const auto pixel = laplacian.padded_unknown[i];
            const auto r = rhs[i];
            const auto z = laplacian.inv_diag[pixel] * r;
            cg.residual[pixel] = r;
            cg.direction[pixel] = z;
            sums.norm_squared += r * r;
            sums.rho += r * z;
        }
        if (!solve_cg<Simd>(terms, laplacian, cg, sums, options.tolerance))
            result.convergence_warning = true;
        for (std::size_t i = 0; i < unknowns; ++i) {
            const double probability = cg.solution[laplacian.padded_unknown[i]];
            if (!std::isfinite(probability)) return false;
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
    return true;
}

template<bool Simd>
bool solve_grid(
    const Laplacian& laplacian, const Markers& markers, const RandomWalkerOptions& options,
    RandomWalkerResult& result) {
    if (laplacian.grid.depth > 1)
        return solve_classes<Simd, true>(laplacian, markers, options, result);
    return solve_classes<Simd, false>(laplacian, markers, options, result);
}

std::optional<RandomWalkerResult> solve(
    const Laplacian& laplacian, const Markers& markers, const RandomWalkerOptions& options) {
    const auto size = markers.nodes.size();
    RandomWalkerResult result;
    result.classes = markers.count;
    if (options.return_probabilities)
        result.probabilities.resize(size * markers.count);
    else
        result.labels.assign(markers.nodes.begin(), markers.nodes.end());

    bool solved;
#if defined(__SSE2__) || defined(_M_X64)
    if (has_avx2())
        solved = solve_grid<true>(laplacian, markers, options, result);
    else
#endif
        solved = solve_grid<false>(laplacian, markers, options, result);
    if (!solved) return std::nullopt;

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
