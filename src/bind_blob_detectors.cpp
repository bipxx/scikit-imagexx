#include "skimagexx/blob_detectors.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_blob_detectors(py::module_& module) {
    module.def("blob_detect", [](const py::array& image, int method,
               const std::vector<std::array<double, 2>>& scales, bool scalar_sigma,
               double normalization, std::optional<double> threshold,
               std::optional<double> threshold_relative, double overlap,
               const std::array<std::size_t, 2>& exclude_border) -> py::object {
        const auto dtype = image.dtype();
        const auto kind = py::cast<std::string>(dtype.attr("char"));
        if (Py_TYPE(image.ptr()) != py::detail::npy_api::get().PyArray_Type_ || image.ndim() != 2 ||
            !dtype.attr("metadata").is_none() || !py::cast<bool>(dtype.attr("isnative")) ||
            !(image.flags() & py::array::c_style) || !image.writeable() ||
            !(image.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_) ||
            (kind != "B" && kind != "f" && kind != "d") || method < 0 || method > 2 ||
            (method == 2 && !scalar_sigma))
            throw py::value_error("blob_detect requires an ordinary aligned native 2D image and method");
        skimagexx::bindings::check_array_storage(image);
        const auto type = kind == "B" ? skimagexx::PixelType::uint8 :
                          kind == "f" ? skimagexx::PixelType::float32 : skimagexx::PixelType::float64;
        if (std::size_t(dtype.itemsize()) != skimagexx::pixel_size(type))
            throw py::value_error("blob_detect requires the expected native dtype itemsize");
        const skimagexx::BlobDetectionOptions options{static_cast<skimagexx::BlobMethod>(method),
            scales, normalization, threshold, threshold_relative, overlap, exclude_border};
        const auto* input = image.data();
        const auto count = std::size_t(image.size()), height = std::size_t(image.shape(0)),
                   width = std::size_t(image.shape(1));
        std::optional<std::vector<skimagexx::BlobDetection>> blobs;
        {
            py::gil_scoped_release release;
            blobs = skimagexx::blob_detect(input, count, type, height, width, options);
        }
        if (!blobs)
            return py::none();

        const auto rows = py::ssize_t(blobs->size());
        const py::ssize_t columns = scalar_sigma ? 3 : 4;
        const auto copy_output = [&](auto example) -> py::object {
            using T = decltype(example);
            py::array_t<T> result({rows, columns});
            auto* output = result.mutable_data();
            for (const auto& blob : *blobs) {
                *output++ = T(blob.row);
                *output++ = T(blob.column);
                *output++ = T(blob.sigma[0]);
                if (!scalar_sigma)
                    *output++ = T(blob.sigma[1]);
            }
            return result;
        };
        if (rows && method != 2 && type == skimagexx::PixelType::float32)
            return copy_output(float{});
        return copy_output(double{});
    }, py::arg("image").noconvert(), py::arg("method"), py::arg("scales"), py::arg("scalar_sigma"),
       py::arg("normalization"), py::arg("threshold"), py::arg("threshold_relative"),
       py::arg("overlap"), py::arg("exclude_border"));
}
