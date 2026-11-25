#include "skimagexx/chan_vese.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;
namespace {

template<class T>
void check_array(const py::array& value) {
    const auto dtype = value.dtype();
    const auto expected = py::dtype::of<T>();
    if (value.ndim() != 2 || !dtype.attr("isnative").cast<bool>() ||
        dtype.num() != expected.num() || !dtype.equal(expected) || !dtype.attr("metadata").is_none() ||
        !(value.flags() & py::array::c_style) || !value.writeable() ||
        reinterpret_cast<std::uintptr_t>(value.data()) % alignof(T))
        throw py::value_error("chan_vese requires exact writable aligned native C-order arrays");
    skimagexx::bindings::check_array_storage(value);
}

template<class T>
py::object evolve(const py::array& image, const py::array& level_set,
                  const skimagexx::ChanVeseOptions& options) {
    check_array<T>(image);
    check_array<T>(level_set);
    const auto height = image.shape(0);
    const auto width = image.shape(1);
    if (height != level_set.shape(0) || width != level_set.shape(1))
        throw py::value_error("chan_vese image and level set shapes must match");

    skimagexx::ChanVeseResult<T> result;
    try {
        py::gil_scoped_release release;
        result = skimagexx::chan_vese(static_cast<const T*>(image.data()),
                                     static_cast<const T*>(level_set.data()),
                                     static_cast<std::size_t>(image.size()),
                                     static_cast<std::size_t>(height), static_cast<std::size_t>(width), options);
    } catch (const std::domain_error&) {
        return py::none();
    }

    py::array_t<bool> segmentation({height, width});
    std::copy(result.segmentation.begin(), result.segmentation.end(), segmentation.mutable_data());
    if (!options.extended_output)
        return segmentation;

    py::array phi = level_set;
    if (!result.energies.empty()) {
        py::array_t<T> updated({height, width});
        std::copy(result.level_set.begin(), result.level_set.end(), updated.mutable_data());
        phi = std::move(updated);
    }
    const auto scalar = py::dtype::of<T>().attr("type");
    py::list energies;
    for (const T value : result.energies)
        energies.append(scalar(value));
    return py::make_tuple(segmentation, phi, energies);
}

}  // namespace

void bind_chan_vese(py::module_& module) {
    module.def("chan_vese_evolve", [](const py::array& image, const py::array& level_set,
        double mu, double lambda1, double lambda2, double tol, std::size_t max_num_iter,
        double dt, bool extended_output, std::size_t reduction_buffer) -> py::object {
        const auto* ndarray = py::detail::npy_api::get().PyArray_Type_;
        if (Py_TYPE(image.ptr()) != ndarray || Py_TYPE(level_set.ptr()) != ndarray)
            throw py::value_error("chan_vese requires exact NumPy arrays");
        const skimagexx::ChanVeseOptions options{
            mu, lambda1, lambda2, tol, max_num_iter, dt, extended_output, reduction_buffer};
        if (image.dtype().num() == py::dtype::of<float>().num())
            return evolve<float>(image, level_set, options);
        return evolve<double>(image, level_set, options);
    }, py::arg("image").noconvert(), py::arg("level_set").noconvert(),
       py::arg("mu"), py::arg("lambda1"), py::arg("lambda2"), py::arg("tol"), py::arg("max_num_iter"),
       py::arg("dt"), py::arg("extended_output"), py::arg("reduction_buffer"));
}
