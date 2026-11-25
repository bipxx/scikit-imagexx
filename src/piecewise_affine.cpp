#include "skimagexx/piecewise_affine.hpp"
#include "execution.hpp"

#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace skimagexx {
namespace {

using Point = std::array<double, 2>;
using Index = std::int32_t;
constexpr double geometry_tolerance = 64 * std::numeric_limits<double>::epsilon();
constexpr std::size_t max_simplices = 2 * piecewise_affine_max_controls;

Point at(const double* data, Index index) {
    return {data[2 * index], data[2 * index + 1]};
}

double orient(const Point& a, const Point& b, const Point& c) {
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
}

double in_circle(const Point& a, const Point& b, const Point& c, const Point& p) {
    const double x = a[0] - p[0], y = a[1] - p[1];
    const double u = b[0] - p[0], v = b[1] - p[1];
    const double s = c[0] - p[0], t = c[1] - p[1];
    return (x * x + y * y) * (u * t - v * s) - (u * u + v * v) * (x * t - y * s) +
           (s * s + t * t) * (x * v - y * u);
}

bool bounded(const double* values, std::size_t count, double limit) {
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i]) || std::abs(values[i]) > limit) return false;
    return true;
}

bool triangulate(const double* src, Index count, std::vector<PiecewiseSimplex>& triangles) {
    Point lo = at(src, 0), hi = lo;
    for (Index i = 1; i < count; ++i) {
        for (std::size_t k = 0; k < 2; ++k) {
            lo[k] = std::min(lo[k], src[2 * i + k]);
            hi[k] = std::max(hi[k], src[2 * i + k]);
        }
    }
    const double scale = std::max(hi[0] - lo[0], hi[1] - lo[1]);
    if (scale == 0) return false;
    std::vector<Point> points;
    points.reserve(count + 3);
    for (Index i = 0; i < count; ++i) {
        const Point point{(src[2 * i] - lo[0]) / scale, (src[2 * i + 1] - lo[1]) / scale};
        if (std::find(points.begin(), points.end(), point) != points.end()) return false;
        points.push_back(point);
    }
    // Bowyer-Watson insertion. The enclosing triangle is discarded before
    // checking the resulting planar mesh for ambiguous incircle predicates.
    points.insert(points.end(), {{-64, -32}, {65, -32}, {0.5, 64}});
    triangles.push_back({count, count + 1, count + 2});
    for (Index point = 0; point < count; ++point) {
        std::map<std::pair<Index, Index>, int> edges;
        auto kept = triangles.begin();
        for (const auto tri : triangles) {
            if (in_circle(points[tri[0]], points[tri[1]], points[tri[2]], points[point]) <= 0) {
                *kept++ = tri;
                continue;
            }
            for (std::size_t k = 0; k < 3; ++k) {
                const auto ends = std::minmax(tri[k], tri[(k + 1) % 3]);
                ++edges[{ends.first, ends.second}];
            }
        }
        triangles.erase(kept, triangles.end());
        for (const auto& [edge, number] : edges) {
            if (number != 1) continue;
            const auto [a, b] = edge;
            const double direction = orient(points[a], points[b], points[point]);
            if (direction > 0) triangles.push_back({a, b, point});
            if (direction < 0) triangles.push_back({b, a, point});
        }
    }
    triangles.erase(std::remove_if(triangles.begin(), triangles.end(), [&](const auto& tri) {
        return *std::max_element(tri.begin(), tri.end()) >= count;
    }), triangles.end());
    std::vector<bool> seen(count);
    for (const auto& tri : triangles) {
        if (orient(points[tri[0]], points[tri[1]], points[tri[2]]) <= geometry_tolerance) return false;
        for (const auto vertex : tri) seen[vertex] = true;
        for (Index i = 0; i < count; ++i) {
            if (i == tri[0] || i == tri[1] || i == tri[2]) continue;
            if (in_circle(points[tri[0]], points[tri[1]], points[tri[2]], points[i]) >= -geometry_tolerance)
                return false;
        }
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

bool fit_triangle(const double* src, const double* dst, const PiecewiseSimplex& tri,
                  PiecewiseTriangle& triangle, PiecewiseMatrix& matrix) {
    const auto a = at(src, tri[0]), b = at(src, tri[1]), c = at(src, tri[2]);
    const double ux = a[0] - c[0], uy = a[1] - c[1], vx = b[0] - c[0], vy = b[1] - c[1];
    const double det = ux * vy - uy * vx;
    const double scale = std::max({std::abs(ux), std::abs(uy), std::abs(vx), std::abs(vy)});
    if (scale == 0 || std::abs(det) <= geometry_tolerance * scale * scale || std::abs(det) < 1e-200)
        return false;
    const auto s = at(dst, tri[0]), t = at(dst, tri[1]), o = at(dst, tri[2]);
    if (s == t && s == o) return false;
    triangle = {vy / det, -vx / det, -uy / det, ux / det, c[0], c[1]};
    matrix = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    for (std::size_t k = 0; k < 2; ++k) {
        const auto row = 3 * k;
        matrix[row] = (s[k] - o[k]) * triangle[0] + (t[k] - o[k]) * triangle[2];
        matrix[row + 1] = (s[k] - o[k]) * triangle[1] + (t[k] - o[k]) * triangle[3];
        matrix[row + 2] = o[k] - (matrix[row] * c[0] + matrix[row + 1] * c[1]);
    }
    return bounded(matrix.data(), matrix.size(), 1e18);
}

} // namespace

bool piecewise_affine_fit(const double* src, const double* dst, std::size_t count,
                          PiecewiseAffineMesh& mesh) {
    if (!src || !dst || count < 3 || count > piecewise_affine_max_controls ||
        !bounded(src, 2 * count, 1e12)) return false;
    std::vector<PiecewiseSimplex> simplices;
    if (!triangulate(src, static_cast<Index>(count), simplices)) return false;
    return piecewise_affine_fit_mesh(src, dst, count, simplices.data(), simplices.size(), mesh);
}

bool piecewise_affine_fit_mesh(const double* src, const double* dst, std::size_t count,
                               const PiecewiseSimplex* simplices, std::size_t simplex_count,
                               PiecewiseAffineMesh& mesh) {
    if (!src || !dst || !simplices || count < 3 || count > piecewise_affine_max_controls ||
        !simplex_count || simplex_count > max_simplices ||
        !bounded(src, 2 * count, 1e12) || !bounded(dst, 2 * count, 1e12)) return false;
    PiecewiseAffineMesh candidate;
    candidate.triangles.resize(simplex_count);
    candidate.matrices.resize(simplex_count);
    for (std::size_t i = 0; i < simplex_count; ++i) {
        const auto& tri = simplices[i];
        for (const auto index : tri)
            if (index < 0 || static_cast<std::size_t>(index) >= count) return false;
        if (!fit_triangle(src, dst, tri, candidate.triangles[i], candidate.matrices[i])) return false;
    }
    candidate.simplices.assign(simplices, simplices + simplex_count);
    mesh = std::move(candidate);
    return true;
}

bool piecewise_affine_apply(const PiecewiseTriangle* triangles, const PiecewiseMatrix* matrices,
                            std::size_t simplex_count, const double* coords,
                            std::size_t rows, double* output) {
    if (!triangles || !matrices || !simplex_count || simplex_count > max_simplices ||
        rows > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / (2 * sizeof(double)) ||
        (rows && (!coords || !output))) return false;
    for (std::size_t i = 0; i < simplex_count; ++i)
        if (!bounded(triangles[i].data(), 6, 1e18) || !bounded(matrices[i].data(), 9, 1e18)) return false;
    std::atomic<bool> supported{true};
    const auto workload = rows * std::min(simplex_count, std::size_t(16));
    detail::partition(rows, detail::active_workers(workload), [&](auto begin, auto end, auto) {
        constexpr double tolerance = 100 * std::numeric_limits<double>::epsilon();
        for (auto row = begin; row < end; ++row) {
            const double x = coords[2 * row], y = coords[2 * row + 1];
            if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 1e12 || std::abs(y) > 1e12) {
                supported.store(false, std::memory_order_relaxed);
                return;
            }
            double* out = output + row * 2;
            out[0] = out[1] = -1;
            for (std::size_t index = 0; index < simplex_count; ++index) {
                const auto& t = triangles[index];
                const double dx = x - t[4], dy = y - t[5];
                const double u = t[0] * dx + t[1] * dy, v = t[2] * dx + t[3] * dy;
                if (u < -tolerance || v < -tolerance || 1 - u - v < -tolerance) continue;
                const auto& m = matrices[index];
                double w = (m[6] * x + m[7] * y) + m[8];
                if (w == 0) w = std::numeric_limits<double>::epsilon();
                out[0] = ((m[0] * x + m[1] * y) + m[2]) / w;
                out[1] = ((m[3] * x + m[4] * y) + m[5]) / w;
                break;
            }
        }
    });
    return supported.load(std::memory_order_relaxed);
}

} // namespace skimagexx
