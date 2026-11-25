#include "skimagexx/slic.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cfenv>
#include <cstdint>
#include <type_traits>

namespace py = pybind11;

namespace {

enum class Input { unsupported, u8, f32, f64 };

Input admissible(const py::array& image, const py::type& ndarray, unsigned spatial_rank) {
    if (!py::type::of(image).is(ndarray) || !(image.flags() & py::array::c_style) ||
        !image.writeable() || image.size() <= 0 || image.size() > (1 << 26) ||
        (spatial_rank != 2 && spatial_rank != 3) ||
        (image.ndim() != spatial_rank && image.ndim() != spatial_rank + 1) ||
        std::fegetround() != FE_TONEAREST)
        return Input::unsupported;
    for (py::ssize_t axis = 0; axis < image.ndim(); ++axis)
        if (image.shape(axis) <= 0 || image.shape(axis) > (1 << 20))
            return Input::unsupported;
    const auto dtype = image.dtype();
    if (dtype.char_() == 'g' || !dtype.attr("metadata").is_none() ||
        reinterpret_cast<std::uintptr_t>(image.data()) % std::size_t(dtype.alignment()))
        return Input::unsupported;
    Input kind = Input::unsupported;
    if (dtype.equal(py::dtype::of<std::uint8_t>()))
        kind = Input::u8;
    else if (dtype.equal(py::dtype::of<float>()))
        kind = Input::f32;
    else if (dtype.equal(py::dtype::of<double>()))
        kind = Input::f64;
    if (kind == Input::unsupported)
        return kind;

    const auto base = image.base();
    if (base && !base.is_none() &&
        (!py::type::of(base).is(ndarray) || !py::reinterpret_borrow<py::array>(base).owndata()))
        return Input::unsupported;
    skimagexx::bindings::check_array_storage(image);
    return kind;
}

template<class T>
bool call(const py::array& image, const std::array<std::size_t, 4>& shape,
          const skimagexx::SlicOptions& options, py::array_t<py::ssize_t>& labels,
          bool prepared) {
    const auto* input = static_cast<const T*>(image.data());
    const auto image_size = static_cast<std::size_t>(image.size());
    auto* output = labels.mutable_data();
    const auto label_size = static_cast<std::size_t>(labels.size());
    py::gil_scoped_release release;
    if constexpr (!std::is_same_v<T, std::uint8_t>)
        if (prepared)
            return skimagexx::slic_prepared_into(input, image_size, shape, options, output, label_size);
    return skimagexx::slic_into(input, image_size, shape, options, output, label_size);
}

}

void bind_slic(py::module_& module) {
    const auto np = py::module_::import("numpy");
    const auto ndarray = np.attr("ndarray").cast<py::type>();
    const auto asarray = np.attr("asarray").cast<py::object>();
    module.def("slic_accept_image", [ndarray](const py::array& image, unsigned rank) {
        return admissible(image, ndarray, rank) != Input::unsupported;
    }, py::arg("image").noconvert(), py::arg("spatial_rank"));
    module.def("slic_grid", [ndarray, asarray](
        const py::array& image, unsigned rank, std::size_t count, double compactness,
        std::size_t iterations, const std::array<double, 3>& spacing, bool connectivity,
        double min_size_factor, double max_size_factor, bool zero, py::ssize_t start_label,
        bool prepared) -> py::object {
        const auto kind = admissible(image, ndarray, rank);
        if (kind == Input::unsupported || (prepared && kind == Input::u8))
            return py::none();
        const skimagexx::SlicOptions options{count, compactness, iterations, spacing, connectivity,
                                            min_size_factor, max_size_factor, zero, start_label};
        std::array<std::size_t, 4> shape{1, 1, 1, 1};
        for (unsigned axis = 0; axis < rank; ++axis)
            shape[3 - rank + axis] = static_cast<std::size_t>(image.shape(axis));
        if (image.ndim() != rank)
            shape[3] = static_cast<std::size_t>(image.shape(rank));
        py::array_t<py::ssize_t> labels({py::ssize_t(shape[0]), py::ssize_t(shape[1]), py::ssize_t(shape[2])});
        bool success = false;
        switch (kind) {
        case Input::u8:
            success = call<std::uint8_t>(image, shape, options, labels, prepared);
            break;
        case Input::f32:
            success = call<float>(image, shape, options, labels, prepared);
            break;
        case Input::f64:
            success = call<double>(image, shape, options, labels, prepared);
            break;
        case Input::unsupported:
            break;
        }
        if (!success)
            return py::none();
        auto memory = py::reinterpret_steal<py::object>(PyMemoryView_FromObject(labels.ptr()));
        if (!memory)
            throw py::error_already_set();
        auto result = asarray(memory);
        if (rank == 2)
            return result.attr("__getitem__")(0);
        return result;
    }, py::arg("image").noconvert(), py::arg("spatial_rank"), py::arg("n_segments"),
       py::arg("compactness"), py::arg("max_num_iter"), py::arg("spacing"),
       py::arg("enforce_connectivity"), py::arg("min_size_factor"), py::arg("max_size_factor"),
       py::arg("slic_zero"), py::arg("start_label"), py::arg("prepared") = false);
}
