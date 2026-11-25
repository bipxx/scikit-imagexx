#include "skimagexx/active_contour.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <type_traits>
#include <utility>

namespace skimagexx {
namespace {

// LU with a fixed half-bandwidth. Each row stores its diagonal at the center,
// with L and U sharing the slots below and above it.
template<std::size_t Width>
class BandSystem {
public:
    explicit BandSystem(std::size_t size) : rows_(size) {}

    double& at(std::size_t row, std::size_t column) {
        return rows_[row][column + Width - row];
    }

    bool factor() {
        const std::size_t size = rows_.size();
        for (std::size_t i = 0; i < size; ++i) {
            const double pivot = at(i, i);
            if (!std::isfinite(pivot) || std::abs(pivot) < 1e-8) {
                return false;
            }
            const std::size_t end = std::min(size, i + Width + 1);
            for (std::size_t j = i + 1; j < end; ++j) {
                const double multiplier = (at(j, i) /= pivot);
                for (std::size_t k = i + 1; k < end; ++k) {
                    at(j, k) -= multiplier * at(i, k);
                }
            }
        }
        return true;
    }

    template<std::size_t Count>
    void solve(std::array<double, Count>* rhs, std::size_t stride = 1) const {
        const std::size_t size = rows_.size();
        for (std::size_t i = 1; i < size; ++i) {
            auto& value = rhs[i * stride];
            for (std::size_t j = i > Width ? i - Width : 0; j < i; ++j) {
                for (std::size_t k = 0; k < Count; ++k) {
                    value[k] -= rows_[i][j + Width - i] * rhs[j * stride][k];
                }
            }
        }
        for (std::size_t i = size; i-- > 0;) {
            auto& value = rhs[i * stride];
            const std::size_t end = std::min(size, i + Width + 1);
            for (std::size_t j = i + 1; j < end; ++j) {
                for (std::size_t k = 0; k < Count; ++k) {
                    value[k] -= rows_[i][j + Width - i] * rhs[j * stride][k];
                }
            }
            for (double& v : value) {
                v /= rows_[i][Width];
            }
        }
    }

private:
    std::vector<std::array<double, 2 * Width + 1>> rows_;
};

struct Basis {
    std::size_t first;
    std::array<double, 3> weights;
    std::array<double, 3> derivatives;
};

// A quadratic interpolating spline on a uniform grid has tripled end knots
// and interior knots at 1.5, 2.5, ..., size - 2.5 (quadratic not-a-knot).
class SplineAxis {
public:
    explicit SplineAxis(std::size_t size) : size_(size), system_(size) {
        for (std::size_t i = 0; i < size; ++i) {
            const auto sample = basis(static_cast<double>(i));
            system_.at(i, i) = sample.weights[i - sample.first];
            if (i > 0 && i + 1 < size) {
                system_.at(i, i - 1) = sample.weights[0];
                system_.at(i, i + 1) = sample.weights[2];
            }
        }
        system_.factor();  // The uniform-grid collocation matrix has positive pivots.
    }

    Basis basis(double coordinate) const {
        const double x = std::clamp(coordinate, 0.0, static_cast<double>(size_ - 1));
        const std::size_t span = std::clamp(static_cast<std::size_t>(x + 0.5) + 1,
                                            std::size_t{2}, size_ - 1);
        const double a = knot(span - 1);
        const double b = knot(span);
        const double c = knot(span + 1);
        const double d = knot(span + 2);
        const double left = (c - x) / (c - b) / (c - a);
        const double right = (x - b) / (c - b) / (d - b);
        return {span - 2,
                {(c - x) * left, (x - a) * left + (d - x) * right, (x - b) * right},
                {-2.0 * left, 2.0 * (left - right), 2.0 * right}};
    }

    void interpolate(std::array<double, 1>* values, std::size_t stride) const {
        system_.solve(values, stride);
    }

private:
    double knot(std::size_t i) const {
        if (i < 3) {
            return 0.0;
        }
        return i < size_ ? static_cast<double>(i) - 1.5 : static_cast<double>(size_ - 1);
    }

    std::size_t size_;
    BandSystem<1> system_;
};

class ImageSpline {
public:
    ImageSpline(std::vector<std::array<double, 1>> values, std::size_t rows, std::size_t columns)
        : rows_(rows), columns_(columns), width_(columns), coefficients_(std::move(values)) {
        for (std::size_t row = 0; row < rows; ++row) {
            columns_.interpolate(coefficients_.data() + row * columns, 1);
        }
        for (std::size_t column = 0; column < columns; ++column) {
            rows_.interpolate(coefficients_.data() + column, columns);
        }
    }

    std::array<double, 2> gradient(double row, double column) const {
        const auto y = rows_.basis(row);
        const auto x = columns_.basis(column);
        double dy = 0.0;
        double dx = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            const auto* coefficient = coefficients_.data() + (y.first + i) * width_ + x.first;
            double horizontal = 0.0;
            double derivative = 0.0;
            for (std::size_t j = 0; j < 3; ++j) {
                horizontal += coefficient[j][0] * x.weights[j];
                derivative += coefficient[j][0] * x.derivatives[j];
            }
            dy += horizontal * y.derivatives[i];
            dx += derivative * y.weights[i];
        }
        return {dy, dx};
    }

private:
    SplineAxis rows_;
    SplineAxis columns_;
    std::size_t width_;
    std::vector<std::array<double, 1>> coefficients_;
};

template<class Pixel, class Work>
ImageSpline make_spline(const Pixel* image, std::size_t rows, std::size_t columns,
                        std::size_t channels, ActiveContourOptions options) {
    const std::size_t size = rows * columns;
    const bool use_edge = options.w_edge != 0;
    std::vector<Work> intensity(size);
    std::vector<Work> edge(use_edge ? size : 0);
    std::vector<Work> plane(edge.size()), gradient(edge.size());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t i = 0; i < size; ++i) {
            Work pixel = static_cast<Work>(image[i * channels + channel]);
            if constexpr (std::is_same_v<Pixel, std::uint8_t>) {
                pixel *= Work{1} / Work{255};
            }
            if (use_edge) {
                plane[i] = pixel;
            }
            intensity[i] += pixel;
        }
        if (use_edge) {
            sobel_into(plane.data(), gradient.data(), rows, columns);
            for (std::size_t i = 0; i < size; ++i) {
                edge[i] += gradient[i];
            }
        }
    }
    std::vector<std::array<double, 1>> values(size);
    const auto w_line = static_cast<Work>(options.w_line);
    const auto w_edge = static_cast<Work>(options.w_edge);
    for (std::size_t i = 0; i < size; ++i) {
        values[i][0] = w_line * intensity[i] + w_edge * (use_edge ? edge[i] : Work{0});
    }
    return ImageSpline(std::move(values), rows, columns);
}

bool supported_options(const ActiveContourOptions& options) {
    return options.alpha >= 0 && options.alpha <= 1
           && options.beta >= 0 && options.beta <= 1
           && std::abs(options.w_line) <= 16 && std::abs(options.w_edge) <= 16
           && options.gamma >= 0.001 && options.gamma <= 1
           && options.max_px_move >= 0 && options.max_px_move <= 16
           && options.max_num_iter > 0 && options.max_num_iter <= 10000
           && options.convergence >= 0 && options.convergence <= 16;
}

enum class Endpoint { fixed, free };

void impose_endpoint(BandSystem<2>& system, std::size_t points, bool start,
                     Endpoint endpoint, double gamma) {
    const auto index = [points, start](std::size_t offset) {
        return start ? offset : points - 1 - offset;
    };
    const std::array<double, 3> curvature{1.0, -2.0, 1.0};
    const std::array<double, 4> third_difference{-1.0, 3.0, -3.0, 1.0};
    for (std::size_t offset = 0; offset < curvature.size(); ++offset) {
        system.at(index(0), index(offset)) = endpoint == Endpoint::fixed ? 0.0 : curvature[offset];
    }
    for (std::size_t offset = 0; offset < third_difference.size(); ++offset) {
        system.at(index(1), index(offset)) = endpoint == Endpoint::fixed
                ? (offset < curvature.size() ? curvature[offset] : 0.0)
                : (start ? third_difference[offset] : -third_difference[offset]);
    }

    system.at(index(0), index(0)) += gamma;
    system.at(index(1), index(1)) += gamma;
}

template<class Work, std::size_t Width>
bool evolve(const ImageSpline& spline, const BandSystem<Width>& system,
            std::vector<Work>& snake, const std::vector<std::size_t>& order,
            bool periodic, Endpoint start, Endpoint end, ActiveContourOptions options) {
    constexpr int history_count = 10;
    const std::size_t points = order.size();
    const auto gamma = static_cast<Work>(options.gamma);
    const auto max_move = static_cast<Work>(options.max_px_move);
    const auto convergence = static_cast<Work>(options.convergence);
    const bool check_convergence = convergence > 0 && options.max_num_iter > history_count;
    const std::size_t first_moving = !periodic && start == Endpoint::fixed ? 1 : 0;
    const std::size_t end_moving = !periodic && end == Endpoint::fixed ? points - 1 : points;
    std::vector<std::array<double, 2>> rhs(points);
    std::vector<Work> history(check_convergence ? history_count * snake.size() : 0);
    const auto set_rhs = [&](std::size_t i, const std::array<double, 2>& force) {
        for (std::size_t j = 0; j < 2; ++j) {
            rhs[order[i]][j] = gamma * snake[2 * i + j] + static_cast<Work>(force[j]);
        }
    };
    const auto set_endpoint_rhs = [&](std::size_t i, Endpoint endpoint) {
        std::array<double, 2> force{};
        if (endpoint == Endpoint::free) {
            force = spline.gradient(snake[2 * i], snake[2 * i + 1]);
            for (double& component : force) {
                component = 2.0 * component;
            }
        }
        set_rhs(i, force);
    };
    for (int iteration = 0; iteration < options.max_num_iter; ++iteration) {
        if (!periodic) {
            set_endpoint_rhs(0, start);
        }
        const std::size_t first_interior = periodic ? 0 : 1;
        const std::size_t end_interior = periodic ? points : points - 1;
        for (std::size_t i = first_interior; i < end_interior; ++i) {
            set_rhs(i, spline.gradient(snake[2 * i], snake[2 * i + 1]));
        }
        if (!periodic) {
            set_endpoint_rhs(points - 1, end);
        }
        system.solve(rhs.data());
        for (std::size_t i = first_moving; i < end_moving; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                Work& coordinate = snake[2 * i + j];
                const Work fitted = static_cast<Work>(rhs[order[i]][j]);
                if (!std::isfinite(fitted)) {
                    return false;
                }
                coordinate += max_move * std::tanh(fitted - coordinate);
            }
        }

        if (!check_convergence) {
            continue;
        }
        const int slot = iteration % (history_count + 1);
        if (slot < history_count) {
            std::copy(snake.begin(), snake.end(), history.data() + std::size_t(slot) * snake.size());
        } else {
            for (int sample = 0; sample < history_count; ++sample) {
                const auto* previous = history.data() + std::size_t(sample) * snake.size();
                std::size_t i = first_moving;
                for (; i < end_moving; ++i) {
                    const Work distance = std::abs(previous[2 * i] - snake[2 * i])
                                          + std::abs(previous[2 * i + 1] - snake[2 * i + 1]);
                    if (distance >= convergence) {
                        break;
                    }
                }
                if (i == end_moving) {
                    return true;
                }
            }
        }
    }
    return true;
}

template<class Pixel, class Work>
bool run(const Pixel* image, std::size_t rows, std::size_t columns, std::size_t channels,
         const Work* initial, std::size_t points, ActiveContourOptions options,
         std::vector<Work>& output) {
    if (!image || !initial || !active_contour_shape_supported(rows, columns, channels, points)
        || !supported_options(options) || std::fegetround() != FE_TONEAREST) {
        return false;
    }
    if (!std::all_of(image, image + rows * columns * channels,
                     [](Pixel x) { return std::abs(static_cast<double>(x)) <= 65536.0; })
        || !std::all_of(initial, initial + 2 * points, [](Work x) { return std::abs(x) <= 65536.0; })) {
        return false;
    }

    const double diagonal = 2.0 * options.alpha + 6.0 * options.beta + options.gamma;
    const double adjacent = -options.alpha - 4.0 * options.beta;
    const double distant = options.beta;
    const std::array<double, 5> stencil{distant, adjacent, diagonal, adjacent, distant};
    std::vector<Work> snake(initial, initial + 2 * points);
    std::vector<std::size_t> order(points);
    if (options.boundary == ActiveContourBoundary::periodic) {
        // Fold the cycle about point zero, so even second neighbors are at
        // most four slots apart. This factors the cyclic pentadiagonal matrix
        // in linear time without a dense inverse or a boundary correction.
        for (std::size_t i = 1; i < points; ++i) {
            order[i] = i <= points / 2 ? 2 * i - 1 : 2 * (points - i);
        }
        BandSystem<4> system(points);
        for (std::size_t i = 0; i < points; ++i) {
            for (std::size_t j = 0; j < stencil.size(); ++j) {
                system.at(order[i], order[(i + points + j - 2) % points]) = stencil[j];
            }
        }
        if (!system.factor()) {
            return false;
        }
        auto spline = make_spline<Pixel, Work>(image, rows, columns, channels, options);
        if (!evolve(spline, system, snake, order, true, Endpoint::free, Endpoint::free, options)) {
            return false;
        }
    } else {
        Endpoint start;
        Endpoint end;
        switch (options.boundary) {
        case ActiveContourBoundary::fixed:
            start = end = Endpoint::fixed;
            break;
        case ActiveContourBoundary::free:
            start = end = Endpoint::free;
            break;
        case ActiveContourBoundary::fixed_free:
            start = Endpoint::fixed;
            end = Endpoint::free;
            break;
        case ActiveContourBoundary::free_fixed:
            start = Endpoint::free;
            end = Endpoint::fixed;
            break;
        default:
            return false;
        }
        BandSystem<2> system(points);
        for (std::size_t i = 0; i < points; ++i) {
            order[i] = i;
            for (std::size_t j = i > 2 ? i - 2 : 0; j < std::min(points, i + 3); ++j) {
                system.at(i, j) = stencil[j + 2 - i];
            }
        }
        impose_endpoint(system, points, true, start, options.gamma);
        impose_endpoint(system, points, false, end, options.gamma);
        if (!system.factor()) {
            return false;
        }
        auto spline = make_spline<Pixel, Work>(image, rows, columns, channels, options);
        if (!evolve(spline, system, snake, order, false, start, end, options)) {
            return false;
        }
    }
    output = std::move(snake);
    return true;
}

}  // namespace

bool active_contour_shape_supported(std::size_t rows, std::size_t columns,
                                    std::size_t channels, std::size_t points) {
    return rows >= 3 && rows <= 4096 && columns >= 3 && columns <= 4096
           && rows * columns <= 1048576 && (channels == 1 || channels == 3)
           && points >= 5 && points <= 8192;
}

bool active_contour(const double* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const double* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<double>& output) {
    return run(image, rows, columns, channels, snake, points, options, output);
}

bool active_contour(const float* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const float* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<float>& output) {
    return run(image, rows, columns, channels, snake, points, options, output);
}

bool active_contour(const std::uint8_t* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const double* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<double>& output) {
    return run(image, rows, columns, channels, snake, points, options, output);
}

}  // namespace skimagexx
