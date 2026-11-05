#include <skimagexx/line_nd.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace {
using skimagexx::detail::LineNDType;
bool vector(const py::object& object, skimagexx::detail::LineNDVector& output) {
    if (!py::isinstance<py::array>(object)) return false;
    const auto array = py::reinterpret_borrow<py::array>(object);
    // A separate direct builtin entry is a boundary too. Public conversion
    // supplies owned C arrays; snapshots are read before releasing the GIL.
    if (array.ndim() > 1 || !array.owndata() || !(array.flags() & py::array::c_style)
            || !array.dtype().attr("isnative").cast<bool>()) return false;
    const char kind = array.dtype().kind();
    const auto bytes = array.itemsize();
    LineNDType type;
    if (kind == 'f' && bytes == 2) type = LineNDType::float16;
    else if (kind == 'f' && bytes == 4) type = LineNDType::float32;
    else if (kind == 'f' && bytes == 8) type = LineNDType::float64;
    else if (kind == 'i' && bytes == 1) type = LineNDType::int8;
    else if (kind == 'u' && bytes == 1) type = LineNDType::uint8;
    else if (kind == 'i' && bytes == 8) type = LineNDType::int64;
    else if (kind == 'u' && bytes == 8) type = LineNDType::uint64;
    else return false;
    output = {array.data(), std::size_t(array.size()),
              array.ndim() ? array.strides(0) : bytes, type};
    return true;
}
}

void bind_line_nd(py::module_& module) {
    module.def("draw_line_nd", [](const py::object& start, const py::object& stop,
                                bool endpoint, bool integer, int workers, bool scalar_modulo_native) -> py::object {
        if (workers < 1 || workers > 1024) throw py::value_error("line ND worker snapshot outside [1,1024]");
        skimagexx::detail::LineNDVector first{}, last{};
        if (!vector(start, first) || !vector(stop, last) || first.type != last.type) return py::none();
        const bool first_scalar = py::reinterpret_borrow<py::array>(start).ndim() == 0;
        const bool last_scalar = py::reinterpret_borrow<py::array>(stop).ndim() == 0;
        if (first_scalar != last_scalar || (first_scalar && first.type != LineNDType::float64)) return py::none();
        skimagexx::detail::LineNDPlan plan;
        if (!skimagexx::detail::line_nd_prepare(first, last, endpoint, integer, plan,
                scalar_modulo_native ? skimagexx::detail::LineNDScalarPromotion::weak_input
                                     : skimagexx::detail::LineNDScalarPromotion::legacy_float64)) return py::none();
        const bool scalar = first_scalar;
        const auto dtype = integer ? py::dtype("intp")
            : plan.precision == LineNDType::float16 ? py::dtype("float16")
            : plan.precision == LineNDType::float32 ? py::dtype("float32") : py::dtype("float64");
        const auto bytes = dtype.itemsize();
        if (plan.points > std::size_t(std::numeric_limits<py::ssize_t>::max())
                          / std::size_t(bytes) / plan.dimensions) return py::none();
        const auto points = py::ssize_t(plan.points), dimensions = py::ssize_t(plan.dimensions);
        std::vector<py::ssize_t> shape, strides;
        py::ssize_t point_step, dimension_step;
        if (scalar) {
            shape = {points}; strides = {points ? bytes : 0};
            point_step = bytes; dimension_step = bytes;
        } else if (integer) {
            shape = {dimensions, points};
            if (!points) strides = {0, 0};
            else if (dimensions > 1 && points > 1) strides = {bytes, dimensions * bytes};
            else strides = {points * bytes, bytes};
            dimension_step = strides[0]; point_step = strides[1];
        } else {
            shape = {points, dimensions};
            strides = points ? std::vector<py::ssize_t>{dimensions * bytes, bytes}
                             : std::vector<py::ssize_t>{0, 0};
            point_step = strides[0]; dimension_step = strides[1];
        }
        py::array base(dtype, shape, strides);
        auto* output = base.mutable_data();
        skimagexx::detail::LineNDFPEvents events;
        {
            py::gil_scoped_release release;
            events = skimagexx::detail::line_nd_into(plan, output, point_step,
                                                   dimension_step, workers);
        }
        return py::make_tuple(base, events.divide_underflow, events.multiply_underflow);
    }, py::arg("start"), py::arg("stop"), py::arg("endpoint"), py::arg("integer"), py::arg("workers"), py::arg("scalar_modulo_native"));
}
