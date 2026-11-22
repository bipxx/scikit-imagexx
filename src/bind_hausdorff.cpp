#include "skimagexx/hausdorff.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

bool mask_dtype(const py::array& array) {
    return py::array_t<bool, py::array::c_style>::check_(array) ||
           py::array_t<std::uint8_t, py::array::c_style>::check_(array);
}

skimagexx::HausdorffMask mask(const py::array& array) {
    skimagexx::bindings::check_array_storage(array);
    skimagexx::HausdorffMask input{static_cast<const std::uint8_t*>(array.data()), {}};
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis)
        input.shape[axis] = static_cast<std::size_t>(array.shape(axis));
    return input;
}

py::array pair_point(const skimagexx::HausdorffResult& result,
                     std::size_t side, py::ssize_t rank) {
    const auto count = static_cast<py::ssize_t>(result.count[side]);
    py::array_t<py::ssize_t> coordinates(py::array::ShapeContainer{rank, count});
    auto* data = coordinates.mutable_data();
    for (py::ssize_t axis = 0; axis < rank; ++axis)
        for (py::ssize_t i = 0; i < count; ++i)
            data[axis * count + i] = result.points[side][i][axis];
    return py::array(coordinates.dtype(), py::array::ShapeContainer{rank},
                     py::array::StridesContainer{count * py::ssize_t(sizeof(py::ssize_t))},
                     data + result.attaining[side], coordinates);
}

} // namespace

void bind_hausdorff(py::module_& module) {
    module.def("hausdorff_mask", [](const py::array& a, const py::array& b, int mode) -> py::object {
        if (a.ndim() < 1 || a.ndim() > 4 || a.ndim() != b.ndim() ||
            !mask_dtype(a) || !mask_dtype(b) || mode < 0 || mode > 2)
            throw py::value_error("Hausdorff requires matching mask ranks and a supported mode");
        const auto operation = static_cast<skimagexx::HausdorffOperation>(mode);
        skimagexx::HausdorffResult result;
        if (!skimagexx::hausdorff(mask(a), mask(b), static_cast<std::size_t>(a.ndim()), operation, result))
            return py::none();
        const bool both_nonempty = result.count[0] && result.count[1];
        if (operation == skimagexx::HausdorffOperation::pair) {
            if (!both_nonempty) return py::tuple();
            return py::make_tuple(pair_point(result, 0, a.ndim()), pair_point(result, 1, a.ndim()));
        }
        if (!result.count[0] && !result.count[1]) return py::int_(0);
        if (!both_nonempty) return py::float_(result.distance);
        return py::module_::import("numpy").attr("float64")(result.distance);
    }, py::arg("image0").noconvert(), py::arg("image1").noconvert(), py::arg("mode"));
}
