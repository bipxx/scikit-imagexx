#include "polygon.hpp"
#include "execution.hpp"
#include <atomic>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace skimagexx::detail {
// Literal dual-ray classifier from scikit-image 0.26 geometry.pyx.
// See third_party/scikit-image-LICENSE.txt. Keep the two divisions ordered;
// a cross-product predicate or a canonical boolean changes source behavior.
std::uint8_t polygon_label(const PolygonView& polygon, double x, double y) noexcept {
    constexpr double eps = 1e-12;
    unsigned int left = 0, right = 0;
    double x1 = polygon.x[polygon.vertices-1]-x;
    double y1 = polygon.y[polygon.vertices-1]-y;
    for (std::size_t i=0; i<polygon.vertices; ++i) {
        const double x0 = polygon.x[i]-x;
        const double y0 = polygon.y[i]-y;
        if ((-eps < x0 && x0 < eps) && (-eps < y0 && y0 < eps)) return 2;
        if ((y0 > 0) != (y1 > 0))
            if ((x0*y1-x1*y0)/(y1-y0) > 0) ++right;
        if ((y0 < 0) != (y1 < 0))
            if ((x0*y1-x1*y0)/(y1-y0) < 0) ++left;
        x1 = x0;
        y1 = y0;
    }
    if ((right&1) != (left&1)) return 3;
    return (right&1) ? 1 : 0;
}

namespace {
class Capture {
    std::fenv_t saved_;
    bool valid_;
public:
    explicit Capture(int rounding):valid_(std::feholdexcept(&saved_)==0) {
        if (valid_) std::fesetround(rounding);
    }
    ~Capture() { if (valid_) std::fesetenv(&saved_); }
    bool valid() const { return valid_; }
};
constexpr auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
void validate(const PolygonView& polygon, std::size_t count, std::uint8_t* output, int workers) {
    if (workers<1 || workers>1024) throw std::invalid_argument("polygon worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if (workers!=1) throw std::invalid_argument("this build has no OpenMP support");
#endif
    if (polygon.vertices>limit/sizeof(double) || count>limit)
        throw std::invalid_argument("polygon buffer extent overflow");
    if (count && (!polygon.vertices || !polygon.x || !polygon.y || !output))
        throw std::invalid_argument("polygon requires vertices and nonnull buffers for nonempty work");
}
template<class F> void run(std::size_t count, int workers, F body) {
    if (!count) return;
    const int rounding = std::fegetround();
    std::atomic<bool> capture_failed{false};
    partition(count,count>=parallel_min?workers:1,[&](std::size_t begin,std::size_t end,std::size_t) {
        Capture capture(rounding);
        if (!capture.valid()) { capture_failed.store(true,std::memory_order_relaxed); return; }
        body(begin,end);
    });
    if (capture_failed.load(std::memory_order_relaxed))
        throw std::runtime_error("cannot capture polygon floating environment");
}
}

void points_in_poly(const PolygonView& polygon, const double* x, const double* y,
                    std::size_t count, std::uint8_t* output, int workers) {
    validate(polygon,count,output,workers);
    if (count>limit/sizeof(double)) throw std::invalid_argument("polygon query extent overflow");
    if (count && (!x || !y)) throw std::invalid_argument("polygon requires nonnull query columns");
    const bool vectorized = has_avx2();
    run(count,workers,[&](std::size_t begin,std::size_t end) {
#ifdef SKIMAGEXX_AVX2
        if (vectorized) { simd::polygon_points(polygon,x,y,output,begin,end); return; }
#else
        (void)vectorized;
#endif
        for (auto i=begin; i<end; ++i) output[i]=polygon_label(polygon,x[i],y[i]);
    });
}

void grid_points_in_poly(const PolygonView& polygon, std::size_t rows, std::size_t columns,
                         std::uint8_t* output, int workers) {
    if (rows>limit || columns>limit || (columns && rows>limit/columns))
        throw std::invalid_argument("polygon grid extent overflow");
    const auto count=rows*columns;
    validate(polygon,count,output,workers);
    const bool vectorized = has_avx2();
    run(count,workers,[&](std::size_t begin,std::size_t end) {
#ifdef SKIMAGEXX_AVX2
        if (vectorized) { simd::polygon_grid(polygon,columns,output,begin,end); return; }
#else
        (void)vectorized;
#endif
        for (auto i=begin; i<end; ++i)
            output[i]=polygon_label(polygon,static_cast<double>(i/columns),static_cast<double>(i%columns));
    });
}
}
