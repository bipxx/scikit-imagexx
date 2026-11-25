#include "skimagexx/canny.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

bool exact_array(py::handle value) {
    return Py_TYPE(value.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

bool native_dtype(const py::dtype& dtype) {
    const std::uint16_t one = 1;
    const auto host_order = *reinterpret_cast<const std::uint8_t*>(&one) ? '<' : '>';
    const auto order = dtype.byteorder();
    return (order == '=' || order == '|' || order == host_order) && dtype.attr("metadata").is_none();
}

bool bounded_array(const py::array& value, bool writeable) {
    if (!(value.flags() & py::array::c_style) || (writeable && !value.writeable()) ||
        reinterpret_cast<std::uintptr_t>(value.data()) % static_cast<std::size_t>(value.itemsize()) != 0)
        return false;
    try {
        skimagexx::bindings::check_array_storage(value);
    } catch (const py::value_error&) {
        return false;
    }
    return true;
}

// The dtype character distinguishes MSVC's eight-byte longdouble from float64.
bool image_type(const py::dtype& dtype, skimagexx::PixelType& type) {
    using skimagexx::PixelType;
    if (!native_dtype(dtype)) return false;
    const auto bytes = dtype.itemsize();
    const auto kind = dtype.kind();
    if (kind == 'b' && bytes == 1) type = PixelType::boolean;
    else if (kind == 'i' && bytes == 1) type = PixelType::int8;
    else if (kind == 'u' && bytes == 1) type = PixelType::uint8;
    else if (kind == 'i' && bytes == 2) type = PixelType::int16;
    else if (kind == 'u' && bytes == 2) type = PixelType::uint16;
    else if (kind == 'i' && bytes == 4) type = PixelType::int32;
    else if (kind == 'u' && bytes == 4) type = PixelType::uint32;
    else if (dtype.char_() == 'f' && bytes == 4) type = PixelType::float32;
    else if (dtype.char_() == 'd' && bytes == 8) type = PixelType::float64;
    else return false;
    return true;
}

}  // namespace

void bind_canny(py::module_& module) {
    module.def("canny_native", [](py::handle input, py::handle mask_value, py::handle weights_value,
                                int border, double cval, double low, double high, bool quantiles,
                                bool wide_quantile, bool constant_bleed) -> py::object {
        if (!exact_array(input) || (!mask_value.is_none() && !exact_array(mask_value)) ||
            !exact_array(weights_value)) return py::none();
        const auto image = py::reinterpret_borrow<py::array>(input);
        const auto weights = py::reinterpret_borrow<py::array>(weights_value);
        skimagexx::PixelType type{};
        if (image.ndim() != 2 || image.shape(0) <= 0 || image.shape(1) <= 0 ||
            !image_type(image.dtype(), type) || !bounded_array(image, true) ||
            weights.ndim() != 1 || weights.shape(0) < 0 ||
            weights.shape(0) > static_cast<py::ssize_t>(2 * skimagexx::canny_max_radius + 1) ||
            weights.dtype().char_() != 'd' || !native_dtype(weights.dtype()) || !bounded_array(weights, false))
            return py::none();

        const auto rows = image.shape(0), cols = image.shape(1);
        if (static_cast<std::size_t>(rows) > skimagexx::canny_max_pixels / static_cast<std::size_t>(cols))
            return py::none();
        const auto* kernel = static_cast<const double*>(weights.data());
        const skimagexx::CannyOptions options{
            weights.size() ? std::vector<double>(kernel, kernel + weights.size()) : std::vector<double>{},
            static_cast<skimagexx::GaussianBorder>(border), cval, low, high, quantiles, wide_quantile, constant_bleed,
        };

        py::array_t<bool> output(py::array::ShapeContainer{rows, cols});
        // Allocation can run Python callbacks. Capture the live views after it.
        skimagexx::PixelType current_type{};
        if (image.ndim() != 2 || image.shape(0) != rows || image.shape(1) != cols ||
            !image_type(image.dtype(), current_type) || type != current_type || !bounded_array(image, true))
            return py::none();
        const std::uint8_t* mask = nullptr;
        if (!mask_value.is_none()) {
            const auto array = py::reinterpret_borrow<py::array>(mask_value);
            const auto dtype = array.dtype();
            if (array.ndim() != 2 || array.shape(0) != rows || array.shape(1) != cols ||
                dtype.char_() != '?' || !native_dtype(dtype) || !bounded_array(array, true)) return py::none();
            mask = static_cast<const std::uint8_t*>(array.data());
        }
        const auto* values = image.data();
        auto* result = reinterpret_cast<std::uint8_t*>(output.mutable_data());
        bool accepted;
        {
            py::gil_scoped_release release;
            accepted = skimagexx::canny_into(values, type, mask, static_cast<std::size_t>(rows),
                                             static_cast<std::size_t>(cols), options, result);
        }
        if (!accepted) return py::none();
        return output;
    });
}
