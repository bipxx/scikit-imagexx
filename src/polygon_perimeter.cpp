#include <skimagexx/polygon_perimeter.hpp>
#include <skimagexx/draw_coordinates.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <utility>

#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace skimagexx::detail {
namespace {

using Int = std::intptr_t;
using Vertices = std::vector<PerimeterVertex>;
using Member = double PerimeterVertex::*;
constexpr Int coordinate_limit = Int{1} << 26;
constexpr std::size_t point_limit = std::size_t{1} << 22;

class FloatingEnvironment {
public:
    FloatingEnvironment() : active_(std::feholdexcept(&saved_) == 0) {}
    ~FloatingEnvironment() { if (active_) std::fesetenv(&saved_); }

    bool ready() const {
#if defined(__SSE2__) || defined(_M_X64)
        if (_mm_getcsr() & 0xe040) return false;
#endif
        return active_ && std::fegetround() == FE_TONEAREST;
    }

    bool failed() const {
        return std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) != 0;
    }

private:
    std::fenv_t saved_;
    bool active_;
};

bool bounded(double value) {
    return std::isfinite(value) && std::abs(value) <= coordinate_limit;
}

// Sutherland–Hodgman traversal starts with the edge from the final vertex to
// the first. Match Path.clip_to_bbox's upper-row, lower-row, upper-column,
// lower-column pass order and retain boundary duplicates.
void clip_edge(Vertices& vertices, Vertices& scratch, Member axis, double boundary, bool upper) {
    if (vertices.empty()) return;
    const Member other = axis == &PerimeterVertex::row ? &PerimeterVertex::column : &PerimeterVertex::row;
    const auto within = [=](double v) { return upper ? v <= boundary : v >= boundary; };
    auto previous = vertices.back();
    bool previous_inside = within(previous.*axis);
    scratch.clear();
    for (const auto vertex : vertices) {
        const bool inside = within(vertex.*axis);
        if (inside != previous_inside) {
            const double fraction = (boundary - previous.*axis) / (vertex.*axis - previous.*axis);
            auto crossing = previous;
            crossing.*axis = boundary;
            crossing.*other += (vertex.*other - previous.*other) * fraction;
            scratch.push_back(crossing);
        }
        if (inside) scratch.push_back(vertex);
        previous = vertex;
        previous_inside = inside;
    }
    vertices.swap(scratch);
}

bool inside_image(Int row, Int column, const std::array<Int, 2>& shape) {
    return row >= 0 && row < shape[0] && column >= 0 && column < shape[1];
}

} // namespace

PerimeterStatus polygon_perimeter(const std::vector<PerimeterVertex>& input,
                                  PerimeterShape shape, bool clip,
                                  PerimeterCoordinates& output) {
    if (input.size() > 4096 || (clip && !shape)) return PerimeterStatus::outside_domain;
    if (shape) {
        for (auto dimension : *shape) {
            if (dimension < -coordinate_limit || dimension > coordinate_limit)
                return PerimeterStatus::outside_domain;
        }
    }

    FloatingEnvironment environment;
    if (!environment.ready()) return PerimeterStatus::outside_domain;
    if (std::any_of(input.begin(), input.end(), [](const auto& vertex) {
            return !bounded(vertex.row) || !bounded(vertex.column);
        })) return PerimeterStatus::outside_domain;

    auto vertices = input;
    if (clip) {
        Vertices scratch;
        const double last_row = static_cast<double>((*shape)[0] - 1);
        const double last_column = static_cast<double>((*shape)[1] - 1);
        clip_edge(vertices, scratch, &PerimeterVertex::row, std::max(0.0, last_row), true);
        clip_edge(vertices, scratch, &PerimeterVertex::row, std::min(0.0, last_row), false);
        clip_edge(vertices, scratch, &PerimeterVertex::column, std::max(0.0, last_column), true);
        clip_edge(vertices, scratch, &PerimeterVertex::column, std::min(0.0, last_column), false);
    }
    if (environment.failed()) return PerimeterStatus::outside_domain;
    if (vertices.size() < 3) return PerimeterStatus::no_polygon;
    const auto first = vertices.front(), last = vertices.back();
    if (first.row != last.row || first.column != last.column) vertices.push_back(first);
    // Matplotlib may simplify the rebuilt path starting at 128 vertices.
    if (vertices.size() >= 128) return PerimeterStatus::outside_domain;

    std::vector<std::array<Int, 2>> pixels;
    pixels.reserve(vertices.size());
    for (const auto& [row, column] : vertices) {
        pixels.push_back({static_cast<Int>(std::nearbyint(row)), static_cast<Int>(std::nearbyint(column))});
    }

    std::array<std::size_t, 127> lengths;
    std::size_t total = 0;
    for (std::size_t i = 1; i < pixels.size(); ++i) {
        const auto [r0, c0] = pixels[i - 1];
        const auto [r1, c1] = pixels[i];
        const auto length = static_cast<std::size_t>(std::max(std::abs(r1 - r0), std::abs(c1 - c0))) + 1;
        if (length > point_limit - total) return PerimeterStatus::outside_domain;
        lengths[i - 1] = length;
        total += length;
    }

    PerimeterCoordinates result;
    result.rows.resize(total);
    result.columns.resize(total);
    std::size_t used = 0;
    for (std::size_t i = 1; i < pixels.size(); ++i) {
        const auto [r0, c0] = pixels[i - 1];
        const auto [r1, c1] = pixels[i];
        const auto length = lengths[i - 1];
        draw_line_into(r0, c0, r1, c1, {result.rows.data() + used, result.columns.data() + used, 1, 1}, 1);
        if (!shape || (inside_image(r0, c0, *shape) && inside_image(r1, c1, *shape))) {
            used += length;
        } else {
            const auto end = used + length;
            for (auto j = used; j < end; ++j) {
                if (inside_image(result.rows[j], result.columns[j], *shape)) {
                    result.rows[used] = result.rows[j];
                    result.columns[used] = result.columns[j];
                    ++used;
                }
            }
        }
    }
    result.rows.resize(used);
    result.columns.resize(used);
    output = std::move(result);
    return PerimeterStatus::complete;
}

} // namespace skimagexx::detail
