#include "skimagexx/brief.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void check_buffer(const py::array& value) {
    if (value.ndim() != 2 || !(value.flags() & py::array::c_style) ||
        !value.attr("flags").attr("aligned").cast<bool>())
        throw py::type_error("BRIEF requires aligned contiguous 2D buffers");
    if (!value.owndata()) skimagexx::bindings::check_array_storage(value);
}

template <typename Pixel>
void sample(const py::array& image, const py::array& keypoints, const py::array& first,
            const py::array& second, py::array& descriptors) {
    skimagexx::brief_sample(
        static_cast<const Pixel*>(image.data()),
        static_cast<std::size_t>(image.shape(0)), static_cast<std::size_t>(image.shape(1)),
        static_cast<const std::int64_t*>(keypoints.data()),
        static_cast<std::size_t>(keypoints.shape(0)),
        static_cast<const std::int32_t*>(first.data()),
        static_cast<const std::int32_t*>(second.data()),
        static_cast<std::size_t>(first.shape(0)),
        static_cast<std::uint8_t*>(descriptors.mutable_data()));
}

} // namespace

void bind_brief(py::module_& module) {
    module.def("brief_sample", [](const py::array& image, const py::array& keypoints,
                                 const py::array& first, const py::array& second,
                                 py::array& descriptors) {
        check_buffer(image);
        check_buffer(keypoints);
        check_buffer(first);
        check_buffer(second);
        check_buffer(descriptors);
        if (keypoints.shape(1) != 2 || first.shape(1) != 2 || second.shape(1) != 2 ||
            first.shape(0) != second.shape(0) || descriptors.shape(0) != keypoints.shape(0) ||
            descriptors.shape(1) != first.shape(0) ||
            !keypoints.dtype().equal(py::dtype::of<std::int64_t>()) ||
            !first.dtype().equal(py::dtype::of<std::int32_t>()) ||
            !second.dtype().equal(py::dtype::of<std::int32_t>()) ||
            !descriptors.dtype().equal(py::dtype::of<bool>()))
            throw py::type_error("invalid BRIEF keypoint, offset or descriptor buffers");
        if (image.dtype().equal(py::dtype::of<float>()))
            sample<float>(image, keypoints, first, second, descriptors);
        else if (image.dtype().equal(py::dtype::of<double>()))
            sample<double>(image, keypoints, first, second, descriptors);
        else
            throw py::type_error("BRIEF requires a float32 or float64 image");
    }, py::arg("image").noconvert(), py::arg("keypoints").noconvert(),
       py::arg("first").noconvert(), py::arg("second").noconvert(),
       py::arg("descriptors").noconvert());
}

#ifdef SKIMAGEXX_PRIVATE_BRIEF_MODULE
PYBIND11_MODULE(_wave18_brief, module) {
    bind_brief(module);
}
#endif
