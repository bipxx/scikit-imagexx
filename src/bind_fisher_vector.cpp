#include "skimagexx/fisher_vector.hpp"
#include "bind_array_storage.hpp"

#include <pybind11/numpy.h>

namespace py = pybind11;
namespace {

using Array = py::array_t<double, py::array::c_style>;

bool bounded(const py::array& array, py::ssize_t rank) {
    if (!Array::check_(array) || array.ndim() != rank ||
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
        return false;
    if (array.owndata()) return true;
    try {
        skimagexx::bindings::check_array_storage(array);
        return true;
    } catch (const py::value_error&) {
        return false;
    }
}

bool descriptors_supported(const py::array& descriptors) {
    return bounded(descriptors, 2) && descriptors.shape(0) > 0 &&
           descriptors.shape(1) > 0 && descriptors.shape(1) <= 2048;
}

bool matrix(const py::array& array, py::ssize_t rows, py::ssize_t cols) {
    return bounded(array, 2) && array.shape(0) == rows && array.shape(1) == cols;
}

bool mixture_supported(const py::array& weights, const py::array& means,
                       const py::array& precision, py::ssize_t dimensions) {
    if (!bounded(weights, 1)) return false;
    const auto count = weights.shape(0);
    return count > 0 && count <= 256 && matrix(means, count, dimensions) &&
           matrix(precision, count, dimensions);
}

const double* values(const py::array& array) {
    return static_cast<const double*>(array.data());
}

skimagexx::DiagonalMixtureView mixture(const py::array& weights, const py::array& means,
                                      const py::array& precision) {
    return {values(weights), values(means), values(precision),
            static_cast<std::size_t>(means.shape(0)), static_cast<std::size_t>(means.shape(1))};
}

py::object expectation(const py::array& data, const py::array& weights,
                       const py::array& means, const py::array& precision) {
    if (!descriptors_supported(data) ||
        !mixture_supported(weights, means, precision, data.shape(1))) return py::none();
    const auto rows = data.shape(0), components = weights.shape(0);
    py::array_t<double> density(rows);
    py::array_t<double> log_resp({rows, components});
    if (!skimagexx::diagonal_gmm_expectation(values(data), static_cast<std::size_t>(rows),
            mixture(weights, means, precision), density.mutable_data(), log_resp.mutable_data()))
        return py::none();
    return py::make_tuple(std::move(density), std::move(log_resp));
}

py::object maximization(const py::array& data, const py::array& log_resp, double reg) {
    if (!descriptors_supported(data) || !bounded(log_resp, 2)) return py::none();
    const auto rows = data.shape(0), dimensions = data.shape(1), components = log_resp.shape(1);
    if (log_resp.shape(0) != rows || components <= 0 || components > 256) return py::none();
    py::array_t<double> weights(components);
    py::array_t<double> means({components, dimensions}), covars({components, dimensions}),
                        precision({components, dimensions});
    if (!skimagexx::diagonal_gmm_maximization(values(data), static_cast<std::size_t>(rows),
            static_cast<std::size_t>(dimensions), static_cast<std::size_t>(components),
            values(log_resp), reg, weights.mutable_data(), means.mutable_data(),
            covars.mutable_data(), precision.mutable_data())) return py::none();
    return py::make_tuple(std::move(weights), std::move(means), std::move(covars),
                          std::move(precision));
}

py::object fisher(const py::array& data, const py::array& weights, const py::array& means,
                  const py::array& precision, const py::array& covars, bool improved,
                  double alpha) {
    if (!descriptors_supported(data) ||
        !mixture_supported(weights, means, precision, data.shape(1)) ||
        !matrix(covars, weights.shape(0), data.shape(1))) return py::none();
    py::array_t<double> result(weights.shape(0) * (1 + 2 * data.shape(1)));
    if (!skimagexx::fisher_vector(values(data), static_cast<std::size_t>(data.shape(0)),
            mixture(weights, means, precision), values(covars), improved, alpha,
            result.mutable_data())) return py::none();
    return result;
}

} // namespace

void bind_fisher_vector(py::module_& module) {
    module.def("_diagonal_gmm_expectation", &expectation);
    module.def("_diagonal_gmm_maximization", &maximization);
    module.def("_diagonal_fisher_vector", &fisher);
}

#ifdef SKIMAGEXX_PRIVATE_FISHER_MODULE
PYBIND11_MODULE(_wave18_fisher, module) {
    bind_fisher_vector(module);
}
#endif
