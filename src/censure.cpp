#include "skimagexx/censure.hpp"
#include "execution.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace skimagexx {
namespace {

struct Shape {
    int side;
    int slope;
};

constexpr std::array<Shape, 11> outer_octagons{{
    {5, 2}, {5, 3}, {7, 3}, {9, 4}, {9, 7}, {13, 7}, {15, 10}, {15, 11}, {15, 12}, {17, 13}, {17, 14},
}};
constexpr std::array<Shape, 11> inner_octagons{{
    {3, 0}, {3, 1}, {3, 2}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {7, 5}, {7, 6}, {9, 6}, {9, 7},
}};
constexpr std::array<Shape, 12> stars{{
    {2, 1}, {4, 2}, {6, 3}, {8, 4}, {12, 6}, {16, 8}, {22, 11}, {32, 16},
    {46, 23}, {64, 32}, {90, 45}, {128, 64},
}};

struct Grid {
    int rows;
    int columns;

    std::size_t offset(int row, int column) const {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(columns)
            + static_cast<std::size_t>(column);
    }
};

int support(CensureMode mode, int scale) {
    if (mode == CensureMode::dob) {
        return 2 * scale;
    }
    const auto index = static_cast<std::size_t>(scale - 1);
    if (mode == CensureMode::octagon) {
        auto shape = outer_octagons[index];
        return (shape.side - 1) / 2 + shape.slope;
    }
    const int side = stars[index].side;
    return side + side / 2;
}

int row_extent(CensureMode mode, Shape shape, int row) {
    if (mode == CensureMode::octagon) {
        const int half_side = (shape.side - 1) / 2;
        const int radius = half_side + shape.slope;
        return row > radius ? -1 : radius - std::max(0, row - half_side);
    }
    const int radius = shape.side + shape.side / 2;
    return std::max(row <= shape.side ? shape.side : -1, radius - row);
}

int footprint_size(CensureMode mode, Shape shape, int radius) {
    int count = 0;
    for (int row = -radius; row <= radius; ++row) {
        count += 2 * row_extent(mode, shape, std::abs(row)) + 1;
    }
    return count;
}

int reflect(int coordinate, int length) {
    const int doubled = 2 * length;
    coordinate %= doubled;
    if (coordinate < 0) {
        coordinate += doubled;
    }
    return coordinate < length ? coordinate : doubled - coordinate - 1;
}

template <typename T>
std::vector<double> summed_image(const T* image, Grid grid) {
    const Grid padded{grid.rows + 1, grid.columns + 1};
    std::vector<double> integral(padded.offset(padded.rows, 0));
    for (int row = 0; row < grid.rows; ++row) {
        for (int col = 0; col < grid.columns; ++col) {
            integral[padded.offset(row + 1, col + 1)] = integral[padded.offset(row, col + 1)]
                + static_cast<double>(image[grid.offset(row, col)]);
        }
    }
    for (int row = 1; row < padded.rows; ++row) {
        for (int col = 1; col < padded.columns; ++col) {
            integral[padded.offset(row, col)] += integral[padded.offset(row, col - 1)];
        }
    }
    return integral;
}

double box_sum(const std::vector<double>& integral, Grid grid, int row, int col, int radius) {
    return integral[grid.offset(row + radius + 1, col + radius + 1)]
        + integral[grid.offset(row - radius, col - radius)]
        - integral[grid.offset(row + radius + 1, col - radius)]
        - integral[grid.offset(row - radius, col + radius + 1)];
}

void dob_response(const std::vector<double>& integral, Grid grid, int scale, std::vector<double>& plane) {
    const Grid padded{grid.rows + 1, grid.columns + 1};
    const int outer_radius = 2 * scale;
    const int inner_side = 2 * scale + 1;
    const double inner_weight = 1.0 / (inner_side * inner_side);
    const double outer_weight = 1.0 / (12 * scale * scale + 4 * scale);
    const double total_weight = inner_weight + outer_weight;
    std::fill(plane.begin(), plane.end(), 0.0);
    for (int row = outer_radius; row < grid.rows - outer_radius; ++row) {
        for (int col = outer_radius; col < grid.columns - outer_radius; ++col) {
            double inner = box_sum(integral, padded, row, col, scale);
            double outer = box_sum(integral, padded, row, col, outer_radius);
            plane[grid.offset(row, col)] = outer_weight * outer - total_weight * inner;
        }
    }
}

template <typename T>
std::vector<double> horizontal_prefix(const T* image, Grid grid, int pad) {
    const Grid padded{grid.rows, grid.columns + 2 * pad + 1};
    std::vector<double> prefix(padded.offset(padded.rows, 0));
    detail::partition(grid.rows, detail::active_workers(prefix.size()), [&](auto begin, auto end, auto) {
        for (auto row = begin; row < end; ++row) {
            const auto* source = image + grid.offset(static_cast<int>(row), 0);
            auto* sums = prefix.data() + padded.offset(static_cast<int>(row), 0);
            for (int col = 1; col < padded.columns; ++col) {
                sums[col] = sums[col - 1] + source[reflect(col - 1 - pad, grid.columns)];
            }
        }
    });
    return prefix;
}

struct Footprint {
    int radius;
    double outer_weight;
    double inner_weight;
    double error;
    std::vector<std::array<int, 2>> extents;
};

Footprint make_footprint(Grid grid, CensureMode mode, int scale, int pad, double magnitude) {
    const auto index = static_cast<std::size_t>(scale - 1);
    const int radius = support(mode, scale);
    const Shape outer = mode == CensureMode::octagon ? outer_octagons[index] : Shape{stars[index].side, 0};
    const Shape inner = mode == CensureMode::octagon ? inner_octagons[index] : Shape{stars[index].slope, 0};
    const int inner_radius = mode == CensureMode::octagon
        ? (inner.side - 1) / 2 + inner.slope : inner.side + inner.side / 2;
    const int inner_size = footprint_size(mode, inner, inner_radius);
    const int outer_size = footprint_size(mode, outer, radius);
    const double outer_weight = 1.0 / (outer_size - inner_size);
    const double inner_weight = outer_weight - (outer_weight + 1.0 / inner_size);
    const double width = grid.columns + 2 * pad;
    // A prefix accumulates at most width additions of partial sums bounded by
    // width*magnitude. Include two prefix errors per interval, weighted row
    // accumulation, and the source-ordered sum of at most outer_size terms.
    const double error = 8 * std::numeric_limits<double>::epsilon() * magnitude
        * (width * width * (2 * radius + 1) * (2 * outer_weight - inner_weight) + outer_size);
    Footprint footprint{radius, outer_weight, inner_weight, error, {}};
    footprint.extents.reserve(2 * radius + 1);
    for (int dr = -radius; dr <= radius; ++dr) {
        footprint.extents.push_back({row_extent(mode, outer, std::abs(dr)),
                                     row_extent(mode, inner, std::abs(dr))});
    }
    return footprint;
}

void footprint_response(const std::vector<double>& prefix, Grid grid, const Footprint& footprint,
                        int pad, int margin, std::vector<double>& plane) {
    const Grid padded{grid.rows, grid.columns + 2 * pad + 1};
    const int row_end = grid.rows - margin + 1;
    const int col_end = grid.columns - margin + 1;
    const auto work = static_cast<std::size_t>(row_end - margin) * (col_end - margin)
        * footprint.extents.size();
    const double correction = footprint.inner_weight - footprint.outer_weight;
    detail::partition(row_end - margin, detail::active_workers(work), [&](auto begin, auto end, auto) {
        for (int row = margin + static_cast<int>(begin); row < margin + static_cast<int>(end); ++row) {
            auto* target = plane.data() + grid.offset(row, 0);
            std::fill(target + margin, target + col_end, 0.0);
            int source_row = row - footprint.radius;
            for (const auto& [outer, inner] : footprint.extents) {
                const auto* source = prefix.data() + padded.offset(reflect(source_row++, grid.rows), pad);
                const auto* outer_start = source - outer;
                const auto* outer_end = source + outer + 1;
                if (inner < 0) {
                    for (int col = margin; col < col_end; ++col) {
                        target[col] += footprint.outer_weight * (outer_end[col] - outer_start[col]);
                    }
                } else {
                    const auto* inner_start = source - inner;
                    const auto* inner_end = source + inner + 1;
                    for (int col = margin; col < col_end; ++col) {
                        target[col] += footprint.outer_weight * (outer_end[col] - outer_start[col])
                            + correction * (inner_end[col] - inner_start[col]);
                    }
                }
            }
        }
    });
}

template <typename T>
double ordered_response(const T* image, Grid grid, const Footprint& footprint, int row, int col) {
    double value = 0;
    int source_row = row - footprint.radius;
    for (const auto& [outer, inner] : footprint.extents) {
        const auto* source = image + grid.offset(reflect(source_row++, grid.rows), 0);
        for (int dc = -outer; dc <= outer; ++dc) {
            const auto weight = std::abs(dc) <= inner ? footprint.inner_weight : footprint.outer_weight;
            value += weight * source[reflect(col + dc, grid.columns)];
        }
    }
    return static_cast<T>(value);
}

template <typename T>
bool footprint_extremum(const T* image, Grid grid, const std::array<std::vector<double>, 3>& planes,
                        const std::array<Footprint, 3>& footprints, int center,
                        int row, int col, double threshold) {
    const auto interval = [](double value, double error) {
        return std::array<double, 2>{static_cast<T>(value - error), static_cast<T>(value + error)};
    };
    const auto [low, high] = interval(planes[center][grid.offset(row, col)], footprints[center].error);
    if (high < threshold) {
        return false;
    }
    bool min_possible = true, max_possible = true;
    bool min_certain = true, max_certain = true;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            const auto index = grid.offset(row + dr, col + dc);
            for (int i = 0; i < 3; ++i) {
                if (i == center && dr == 0 && dc == 0) {
                    continue;
                }
                const auto [a, b] = interval(planes[i][index], footprints[i].error);
                min_possible = min_possible && low <= b;
                max_possible = max_possible && high >= a;
                min_certain = min_certain && high <= a;
                max_certain = max_certain && low >= b;
            }
            if (!min_possible && !max_possible) {
                return false;
            }
        }
    }
    if (low >= threshold && (min_certain || max_certain)) {
        return true;
    }
    // A close threshold or tied neighbor needs the original rounding order.
    const auto value = ordered_response(image, grid, footprints[center], row, col);
    if (value < threshold) {
        return false;
    }
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            for (int i = 0; i < 3; ++i) {
                if (i == center && dr == 0 && dc == 0) {
                    continue;
                }
                const auto index = grid.offset(row + dr, col + dc);
                const auto [a, b] = interval(planes[i][index], footprints[i].error);
                if (a == b || value < a || value > b) {
                    min_possible = min_possible && value <= b;
                    max_possible = max_possible && value >= a;
                } else {
                    const auto neighbor = ordered_response(image, grid, footprints[i], row + dr, col + dc);
                    min_possible = min_possible && value <= neighbor;
                    max_possible = max_possible && value >= neighbor;
                }
            }
            if (!min_possible && !max_possible) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
using Tensor = std::array<T, 3>;

template <typename T>
std::vector<Tensor<T>> structure_products(const T* image, Grid grid) {
    std::vector<Tensor<T>> products(grid.offset(grid.rows, 0));
    auto sample = [&](int row, int col) -> double {
        return row < 0 || col < 0 || row >= grid.rows || col >= grid.columns
            ? 0.0 : static_cast<double>(image[grid.offset(row, col)]);
    };
    auto dr = [&](int row, int col) {
        return static_cast<T>(sample(row + 1, col) - sample(row - 1, col));
    };
    auto dc = [&](int row, int col) {
        return static_cast<T>(sample(row, col + 1) - sample(row, col - 1));
    };
    for (int row = 0; row < grid.rows; ++row) {
        for (int col = 0; col < grid.columns; ++col) {
            const auto r = static_cast<T>(2.0 * dr(row, col)
                + (static_cast<double>(dr(row, col - 1)) + dr(row, col + 1)));
            const auto c = static_cast<T>(2.0 * dc(row, col)
                + (static_cast<double>(dc(row - 1, col)) + dc(row + 1, col)));
            products[grid.offset(row, col)] = {r * r, r * c, c * c};
        }
    }
    return products;
}

std::vector<double> gaussian_weights(int scale) {
    const double sigma = 1.0 + (scale - 1) / 3.0;
    const int radius = static_cast<int>(4 * sigma + 0.5);
    const double coefficient = -0.5 / (sigma * sigma);
    std::vector<double> weights(static_cast<std::size_t>(radius + 1));
    double sum = 1;
    weights[0] = 1;
    for (int delta = 1; delta <= radius; ++delta) {
        const double weight = std::exp(coefficient * (delta * delta));
        weights[static_cast<std::size_t>(delta)] = weight;
        sum += 2 * weight;
    }
    for (double& weight : weights) {
        weight /= sum;
    }
    return weights;
}

template <typename T>
bool suppress_line(const std::vector<Tensor<T>>& products, Grid grid, int row, int col,
                   const std::vector<double>& weights, double threshold) {
    const int radius = static_cast<int>(weights.size()) - 1;
    auto vertical = [&](int column) {
        std::array<double, 3> accum{};
        if (column < 0 || column >= grid.columns) {
            return Tensor<T>{};
        }
        const auto& center = products[grid.offset(row, column)];
        for (std::size_t field = 0; field < center.size(); ++field) {
            accum[field] = static_cast<double>(center[field]) * weights.front();
        }
        for (int d = radius; d > 0; --d) {
            const auto above = row >= d ? products[grid.offset(row - d, column)] : Tensor<T>{};
            const auto below = row + d < grid.rows ? products[grid.offset(row + d, column)] : Tensor<T>{};
            const double weight = weights[static_cast<std::size_t>(d)];
            for (std::size_t field = 0; field < center.size(); ++field) {
                accum[field] += (static_cast<double>(above[field]) + below[field]) * weight;
            }
        }
        return Tensor<T>{static_cast<T>(accum[0]), static_cast<T>(accum[1]), static_cast<T>(accum[2])};
    };

    const auto center = vertical(col);
    std::array<double, 3> accum{};
    for (std::size_t field = 0; field < center.size(); ++field) {
        accum[field] = static_cast<double>(center[field]) * weights.front();
    }
    for (int d = radius; d > 0; --d) {
        const auto left = vertical(col - d);
        const auto right = vertical(col + d);
        const double weight = weights[static_cast<std::size_t>(d)];
        for (std::size_t field = 0; field < center.size(); ++field) {
            accum[field] += (static_cast<double>(left[field]) + right[field]) * weight;
        }
    }
    const auto rr = static_cast<T>(accum[0]);
    const auto rc = static_cast<T>(accum[1]);
    const auto cc = static_cast<T>(accum[2]);
    const auto trace = static_cast<T>(rr + cc);
    const auto determinant = static_cast<T>(rr * cc - rc * rc);
    return trace * trace > static_cast<T>(threshold) * determinant;
}

bool extremum(const std::array<std::vector<double>, 3>& planes, Grid grid, int row, int col, double value) {
    bool minimum = true;
    bool maximum = true;
    for (int dr = -1; dr <= 1; ++dr) {
        const int neighbor_row = std::clamp(row + dr, 0, grid.rows - 1);
        for (int dc = -1; dc <= 1; ++dc) {
            const auto index = grid.offset(neighbor_row, std::clamp(col + dc, 0, grid.columns - 1));
            for (const auto& plane : planes) {
                const double neighbor = plane[index];
                minimum = minimum && value <= neighbor;
                maximum = maximum && value >= neighbor;
            }
            if (!minimum && !maximum) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
std::optional<std::vector<CensureKeypoint>> detect(const T* image, std::size_t rows, std::size_t columns,
                                                   const CensureOptions& options) {
    const auto mode = options.mode;
    int scale_limit = 64;
    switch (mode) {
        case CensureMode::dob: break;
        case CensureMode::octagon: scale_limit = static_cast<int>(outer_octagons.size()); break;
        case CensureMode::star: scale_limit = static_cast<int>(stars.size()); break;
        default: throw std::invalid_argument("invalid CENSURE mode");
    }
    if (options.min_scale < 1 || options.max_scale > scale_limit || options.min_scale > options.max_scale
        || options.max_scale - options.min_scale < 2 || !std::isfinite(options.non_max_threshold)
        || !std::isfinite(options.line_threshold) || std::abs(options.line_threshold) > 1e12) {
        throw std::invalid_argument("invalid CENSURE options");
    }
    if (image == nullptr || rows < 2 || columns < 2 || rows > 32768 || columns > 32768
        || rows * columns > 8 * 1024 * 1024) {
        throw std::invalid_argument("invalid CENSURE image dimensions");
    }
    const Grid grid{static_cast<int>(rows), static_cast<int>(columns)};
    const bool dob = mode == CensureMode::dob;
    const int pad = support(mode, options.max_scale);
    if (dob && (grid.rows <= 2 * pad || grid.columns <= 2 * pad)) {
        throw std::invalid_argument("CENSURE DoB image is too small for max_scale");
    }
    const auto size = rows * columns;
    double max_magnitude = 0;
    for (std::size_t i = 0; i < size; ++i) {
        const auto magnitude = std::abs(static_cast<double>(image[i]));
        if (!(magnitude <= 1e6) || (magnitude != 0 && magnitude < 1e-9)) {
            return std::nullopt;
        }
        max_magnitude = std::max(max_magnitude, magnitude);
    }

    const int margin = support(mode, options.min_scale + 1);
    std::vector<CensureKeypoint> result;
    if (!dob && (grid.rows < 2 * margin || grid.columns < 2 * margin)) {
        return result;
    }
    const auto prepared = dob ? summed_image(image, grid) : horizontal_prefix(image, grid, pad);
    std::array<std::vector<double>, 3> planes{
        std::vector<double>(size), std::vector<double>(size), std::vector<double>(size),
    };
    std::array<Footprint, 3> footprints{};
    std::vector<Tensor<T>> products;
    for (int scale = options.min_scale; scale <= options.max_scale; ++scale) {
        const int index = scale - options.min_scale;
        auto& plane = planes[static_cast<std::size_t>(index % 3)];
        if (dob) {
            dob_response(prepared, grid, scale, plane);
        } else {
            auto& footprint = footprints[static_cast<std::size_t>(index % 3)];
            footprint = make_footprint(grid, mode, scale, pad, max_magnitude);
            footprint_response(prepared, grid, footprint, pad, margin - 1, plane);
        }
        if (index < 2) {
            continue;
        }
        const int center_index = (index - 1) % 3;
        const auto& center = planes[static_cast<std::size_t>(center_index)];
        const int candidate_scale = scale - 1;
        // DoB's zero-initialized perimeter can be selected at nonpositive thresholds.
        const int border = dob ? 0 : support(mode, candidate_scale);
        const int row_end = dob ? grid.rows : grid.rows - border + 1;
        const int col_end = dob ? grid.columns : grid.columns - border + 1;
        const auto weights = gaussian_weights(candidate_scale);
        for (int row = border; row < row_end; ++row) {
            for (int col = border; col < col_end; ++col) {
                const double value = center[grid.offset(row, col)];
                const bool selected = dob
                    ? value >= options.non_max_threshold && extremum(planes, grid, row, col, value)
                    : footprint_extremum(image, grid, planes, footprints, center_index,
                                         row, col, options.non_max_threshold);
                if (!selected) {
                    continue;
                }
                if (products.empty()) {
                    products = structure_products(image, grid);
                }
                if (!suppress_line(products, grid, row, col, weights, options.line_threshold)) {
                    result.push_back({static_cast<std::size_t>(row), static_cast<std::size_t>(col), candidate_scale});
                }
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return std::tie(a.row, a.column, a.scale) < std::tie(b.row, b.column, b.scale);
    });
    return result;
}

}  // namespace

std::optional<std::vector<CensureKeypoint>> censure_detect(
    const float* image, std::size_t rows, std::size_t columns, const CensureOptions& options) {
    return detect(image, rows, columns, options);
}

std::optional<std::vector<CensureKeypoint>> censure_detect(
    const double* image, std::size_t rows, std::size_t columns, const CensureOptions& options) {
    return detect(image, rows, columns, options);
}

}  // namespace skimagexx
