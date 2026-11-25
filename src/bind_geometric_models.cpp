#include "skimagexx/geometric_models.hpp"
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace bindings = skimagexx::bindings;
namespace {

bool real_buffer(const py::array& input) {
    const auto dtype = input.dtype();
    if (Py_TYPE(input.ptr()) != py::detail::npy_api::get().PyArray_Type_
        || !(input.flags() & py::array::c_style)
        || reinterpret_cast<std::uintptr_t>(input.data()) % alignof(double)
        || (!dtype.is(py::dtype::of<double>())
            && (dtype.kind() != 'f' || dtype.itemsize() != sizeof(double)
                || !bindings::is_native_byteorder(dtype) || !dtype.attr("metadata").is_none()))) {
        return false;
    }
    if (!input.owndata()) skimagexx::bindings::check_array_storage(input);
    return true;
}

bool point_buffer(const py::array& input) {
    return input.ndim() == 2 && input.shape(1) == 2 && real_buffer(input);
}

std::array<double, 2> pair_from_array(const py::array& array) {
    if (array.ndim() != 1 || array.shape(0) != 2 || !real_buffer(array)) {
        throw py::value_error("expected a contiguous float64 coordinate pair");
    }
    const auto* data = static_cast<const double*>(array.data());
    return {data[0], data[1]};
}

py::object fit(const py::array& data, bool ellipse) {
    if (!point_buffer(data)) return py::none();
    const auto* points = static_cast<const double*>(data.data());
    const auto count = static_cast<std::size_t>(data.shape(0));
    if (ellipse) {
        skimagexx::EllipseParameters model;
        if (!skimagexx::fit_ellipse(points, count, model)) return py::none();
        return py::make_tuple(model.center, model.axis_lengths, model.theta);
    }
    skimagexx::CircleParameters model;
    if (!skimagexx::fit_circle(points, count, model)) return py::none();
    return py::make_tuple(model.center, model.radius);
}

py::object residuals(const py::array& data, const py::array& center, double radius,
                     const py::object& axis_lengths, double theta) {
    if (!point_buffer(data)) return py::none();
    const auto origin = pair_from_array(center);
    const auto axes = axis_lengths.is_none() ? std::array<double, 2>{}
                                            : pair_from_array(axis_lengths.cast<py::array>());
    const py::ssize_t shape[]{data.shape(0)};
    auto result = bindings::empty_c_array<double>(shape);
    // NumPy allocation can call GC and Python finalizers. Check again before
    // reading the buffer and keep the GIL over the native computation.
    if (data.shape(0) != result.shape(0) || !point_buffer(data)) return py::none();
    const auto* points = static_cast<const double*>(data.data());
    const auto count = static_cast<std::size_t>(data.shape(0));
    if (axis_lengths.is_none()) {
        skimagexx::circle_residuals(points, count, {origin, radius}, result.mutable_data());
    } else if (!skimagexx::ellipse_residuals(points, count, {origin, axes, theta}, result.mutable_data())) {
        return py::none();
    }
    return result;
}

py::object predict(const py::array& cosine, const py::array& sine, const py::array& center,
                   double radius, const py::object& axis_lengths, double theta) {
    if (!real_buffer(cosine) || !real_buffer(sine) ||
        cosine.ndim() != sine.ndim() ||
        !std::equal(cosine.shape(), cosine.shape() + cosine.ndim(), sine.shape())) return py::none();
    const auto origin = pair_from_array(center);
    const auto axes = axis_lengths.is_none() ? std::array<double, 2>{}
                                            : pair_from_array(axis_lengths.cast<py::array>());
    // NumPy 2 supports at most 64 input axes; the trailing output axis is extra.
    py::ssize_t shape[bindings::numpy_max_rank + 1];
    const auto dimensions = cosine.ndim();
    std::copy_n(cosine.shape(), dimensions, shape);
    shape[dimensions] = 2;
    auto result = bindings::empty_c_array<double>(shape, static_cast<int>(dimensions + 1));
    if (!real_buffer(cosine) || !real_buffer(sine) ||
        cosine.ndim() != dimensions || sine.ndim() != dimensions ||
        !std::equal(shape, shape + dimensions, cosine.shape()) ||
        !std::equal(shape, shape + dimensions, sine.shape())) return py::none();
    const auto* c = static_cast<const double*>(cosine.data());
    const auto* s = static_cast<const double*>(sine.data());
    const auto count = static_cast<std::size_t>(cosine.size());
    if (axis_lengths.is_none()) {
        skimagexx::predict_circle(c, s, count, {origin, radius}, result.mutable_data());
    } else {
        skimagexx::predict_ellipse(c, s, count, {origin, axes, theta}, result.mutable_data());
    }
    return result;
}

bool truth(const py::handle& value) {
    const int result = PyObject_IsTrue(value.ptr());
    if (result < 0) throw py::error_already_set();
    return result != 0;
}

py::tuple select(const py::object& data, const py::object& indices) {
    // Keep the application's list live across callbacks, just as the public
    // reference loop does. Only a trial's selected arrays are snapshotted.
    py::list arguments;
    for (const auto& array : data) arguments.append(array[indices]);
    return py::tuple(arguments);
}

py::tuple prepend(const py::object& value, const py::tuple& rest) {
    py::tuple arguments(rest.size() + 1);
    arguments[0] = value;
    for (std::size_t i = 0; i < rest.size(); ++i) arguments[i + 1] = rest[i];
    return arguments;
}

// Keep NumPy as the RNG and residual reduction provider. Python callbacks
// run in the reference order, including the draw before each validation.
py::tuple ransac(const py::object& data, const py::object& model_class,
                 const py::object& rng, py::object indices, std::size_t samples,
                 std::size_t minimum, const py::object& threshold, double trials,
                 double stop_samples, const py::object& stop_residuals, double probability,
                 const py::object& validate_data, const py::object& validate_model,
                 const py::object& warn, const py::module_& numpy, const py::module_& builtins) {
    const auto absolute = numpy.attr("abs");
    const auto count_nonzero = numpy.attr("count_nonzero");
    const auto choice = rng.attr("choice");
    py::object best = py::none();
    auto best_count = std::size_t{0};
    auto best_error = py::cast(std::numeric_limits<double>::infinity());
    for (std::size_t trial = 0; static_cast<double>(trial) < trials; ++trial) {
        const auto subset = select(data, indices);
        indices = choice(samples, minimum, py::arg("replace") = false);
        if (!validate_data.is_none() && !truth(validate_data(*subset))) continue;
        const auto model = model_class.attr("from_estimate")(*subset);
        if (!truth(model)) continue;
        if (!validate_model.is_none() && !truth(validate_model(*prepend(model, subset)))) continue;

        const auto residual = absolute(model.attr("residuals")(*data));
        const auto inliers = py::reinterpret_steal<py::object>(
            PyObject_RichCompare(residual.ptr(), threshold.ptr(), Py_LT));
        if (!inliers) throw py::error_already_set();
        const auto error = residual.attr("dot")(residual);
        const auto count = count_nonzero(inliers).cast<std::size_t>();
        if (count > best_count || (count == best_count && error < best_error)) {
            best = inliers;
            best_count = count;
            best_error = error;
            trials = std::min(trials, skimagexx::ransac_trial_limit(count, samples, minimum, probability));
            if (static_cast<double>(count) >= stop_samples || error <= stop_residuals) break;
        }
    }
    py::object model = py::none();
    if (!best.is_none() && truth(builtins.attr("any")(best))) {
        const auto subset = select(data, best);
        model = model_class.attr("from_estimate")(*subset);
        if (!validate_model.is_none() && !truth(validate_model(*prepend(model, subset)))) {
            warn("Estimated model is not valid. Try increasing max_trials.");
        }
    } else {
        warn("No inliers found. Model not fitted");
        best = py::none();
    }
    return py::make_tuple(py::getattr(model, "model", model), best);
}

}  // namespace

void bind_geometric_models(py::module_& module) {
    module.def("measure_geometric_fit", &fit, py::arg("data").noconvert(), py::arg("ellipse"));
    module.def("measure_geometric_residuals", &residuals);
    module.def("measure_geometric_predict", &predict);
    const auto numpy = py::module_::import("numpy");
    const auto builtins = py::module_::import("builtins");
    module.def("measure_ransac_loop", [numpy, builtins](
        const py::object& data, const py::object& model_class, const py::object& rng,
        py::object indices, std::size_t samples, std::size_t minimum, const py::object& threshold,
        double trials, double stop_samples, const py::object& stop_residuals, double probability,
        const py::object& validate_data, const py::object& validate_model, const py::object& warn) {
        return ransac(data, model_class, rng, std::move(indices), samples, minimum, threshold,
                     trials, stop_samples, stop_residuals, probability, validate_data, validate_model,
                     warn, numpy, builtins);
    });
}
