#include "skimagexx/random_noise.hpp"
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"

#include <cfenv>

namespace py = pybind11;

namespace {

void check_buffer(const py::array& array) {
    if (!array.dtype().equal(py::dtype::of<double>()) || !(array.flags() & py::array::c_style) ||
        !array.attr("flags").attr("aligned").cast<bool>() || !array.dtype().attr("metadata").is_none() ||
        array.ndim() < 1 || array.ndim() > 3 || array.size() < 1 || array.size() > 524288)
        throw py::value_error("random_noise requires bounded C-order float64 storage");
    skimagexx::bindings::check_array_storage(array);
}

py::array_t<double> make_output(const py::array& image) {
    return skimagexx::bindings::empty_c_array<double>(image.shape(), static_cast<int>(image.ndim()));
}

skimagexx::RandomNoiseDraw prepare_draw(const py::object& input, double probability, const py::array& image) {
    if (input.is_none())
        return {nullptr, 0, probability};
    if (!py::isinstance<py::array>(input))
        throw py::value_error("random_noise requires an ndarray provider draw");
    const auto array = py::reinterpret_borrow<py::array>(input);
    check_buffer(array);
    if (array.size() != image.size())
        throw py::value_error("random_noise draw has the wrong number of elements");
    return {static_cast<const double*>(array.data()), std::size_t(array.size()), probability};
}

}  // namespace

void bind_random_noise(py::module_& module) {
    module.def("random_noise_round_to_nearest", [] { return std::fegetround() == FE_TONEAREST; });
    module.def("random_noise_apply_normal", [](const py::array& image, const py::array& noise, bool speckle) {
        check_buffer(image);
        check_buffer(noise);
        if (image.size() != noise.size())
            throw py::value_error("random_noise provider draw has the wrong number of elements");
        auto output = make_output(image);
        const auto* values = static_cast<const double*>(image.data());
        const auto* samples = static_cast<const double*>(noise.data());
        const auto count = std::size_t(image.size());
        auto* result = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::random_noise_normal_into(values, samples, result, count, speckle);
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("noise").noconvert(), py::arg("speckle"));
    module.def("random_noise_apply_impulse", [](const py::array& image, const py::object& flipped,
               const py::object& salted, double p, double q, double low) {
        check_buffer(image);
        const auto flip = prepare_draw(flipped, p, image);
        const auto salt = prepare_draw(salted, q, image);
        auto output = make_output(image);
        const auto* values = static_cast<const double*>(image.data());
        const auto count = std::size_t(image.size());
        auto* result = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::random_noise_impulse_into(values, result, count, flip, salt, low);
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("flipped"), py::arg("salted"),
       py::arg("amount"), py::arg("salt_ratio"), py::arg("low"));
}
