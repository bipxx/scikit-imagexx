#include "skimagexx/structural_similarity.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_structural_similarity(py::module_& module) {
    module.def("structural_similarity_score", [](const py::array& left, const py::array& right,
               std::size_t window, double data_range, bool use_sample_covariance) -> py::object {
        if (left.ndim() != 2 || right.ndim() != 2 ||
            left.shape(0) != right.shape(0) || left.shape(1) != right.shape(1))
            throw py::value_error("SSIM inputs must have matching 2D shapes");
        using Float = py::array_t<float, py::array::c_style>;
        using Double = py::array_t<double, py::array::c_style>;
        skimagexx::PixelType type;
        if (Float::check_(left) && Float::check_(right)) type = skimagexx::PixelType::float32;
        else if (Double::check_(left) && Double::check_(right)) type = skimagexx::PixelType::float64;
        else throw py::type_error("SSIM expects matching native C-order float32 or float64 inputs");

        skimagexx::bindings::check_array_storage(left);
        skimagexx::bindings::check_array_storage(right);
        const auto rows = static_cast<std::size_t>(left.shape(0));
        const auto cols = static_cast<std::size_t>(left.shape(1));
        const auto* left_data = left.data();
        const auto* right_data = right.data();
        double score;
        bool accepted;
        {
            py::gil_scoped_release release;
            accepted = skimagexx::structural_similarity_score(left_data, right_data, type, rows, cols,
                                                             window, data_range, use_sample_covariance, score);
        }
        if (!accepted) return py::none();
        return py::float_(score);
    }, py::arg("left").noconvert(), py::arg("right").noconvert(), py::arg("window"),
       py::arg("data_range"), py::arg("use_sample_covariance"));
}
