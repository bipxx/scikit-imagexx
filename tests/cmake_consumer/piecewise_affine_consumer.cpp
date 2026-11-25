#include <skimagexx/piecewise_affine.hpp>

#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("piecewise affine SDK assertion failed");
}

} // namespace

int main() {
    const double src[]{0, 0, 4, 0, 0, 3, 4, 4, 1, 1};
    const double dst[]{1, 2, 5, 2, 1, 5, 5, 6, 1.5, 3.5};
    skimagexx::PiecewiseAffineMesh mesh;
    require(skimagexx::piecewise_affine_fit(src, dst, 5, mesh));
    require(mesh.simplices.size() == 4);
    double result[10]{};
    require(skimagexx::piecewise_affine_apply(mesh.triangles.data(), mesh.matrices.data(),
             mesh.simplices.size(), src, 5, result));
    for (std::size_t i = 0; i < 10; ++i) require(std::abs(result[i] - dst[i]) < 1e-12);
    const double outside[]{-3, -10};
    require(skimagexx::piecewise_affine_apply(mesh.triangles.data(), mesh.matrices.data(),
             mesh.simplices.size(), outside, 1, result));
    require(result[0] == -1 && result[1] == -1);

    skimagexx::PiecewiseAffineMesh ordered;
    const skimagexx::PiecewiseSimplex triangle{1, 0, 4};
    require(skimagexx::piecewise_affine_fit_mesh(src, dst, 5, &triangle, 1, ordered));
    require(ordered.simplices.front() == triangle);
    const skimagexx::PiecewiseSimplex invalid{0, -1, 4};
    require(!skimagexx::piecewise_affine_fit_mesh(src, dst, 5, &invalid, 1, ordered));
    require(ordered.simplices.front() == triangle);
    const double cocircular[]{0, 0, 1, 0, 1, 1, 0, 1};
    require(!skimagexx::piecewise_affine_fit(cocircular, cocircular, 4, mesh));
    require(mesh.simplices.size() == 4);
}
