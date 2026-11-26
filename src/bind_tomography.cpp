#include <skimagexx/tomography.hpp>

#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace py = pybind11;
namespace {
using namespace skimagexx;
constexpr int max_extent = 16384;

bool plain_array(const py::handle& input) {
    return Py_TYPE(input.ptr()) == py::detail::npy_api::get().PyArray_Type_;
}

template<class T> bool descriptor(const py::array& a) {
    const auto dtype = a.dtype();
    constexpr char code = std::is_same_v<T, double> ? 'd' : 'f';
    return dtype.char_() == code && dtype.equal(py::dtype::of<T>())
        && dtype.attr("metadata").is_none()
        && reinterpret_cast<std::uintptr_t>(a.data()) % alignof(T) == 0;
}

void storage(const py::array& a) {
    if (!a.owndata()) bindings::check_array_storage(a);
}

bool layout(const py::array& array, int ndim) {
    if (array.ndim() != ndim || !array.writeable() || !(array.flags() & py::array::c_style)) return false;
    for (int axis = 0; axis < ndim; ++axis) {
        if (array.shape(axis) <= 0 || array.shape(axis) > max_extent) return false;
    }
    return true;
}

template<class T> bool valid(const py::array& array, int ndim) {
    return layout(array, ndim) && descriptor<T>(array);
}

template<class T> const T* data(const py::array& a) {
    return static_cast<const T*>(a.data());
}

template<class T> bool finite(const T* values, py::ssize_t size) {
    // Keep the arithmetic away from overflow in the native domain. Exceptional
    // inputs retain upstream handling of nonfinite values and NumPy FP policy.
    constexpr auto bound = std::is_same_v<T, float> ? T(1e18) : T(1e140);
    return std::all_of(values, values + size, [bound](T value) { return std::abs(value) <= bound; });
}

template<class T>
py::object forward(const py::array& a, const py::array& coordinates, bool circle) {
    if (!valid<T>(a, 2)) return py::none();
    storage(a);
    const int rows = int(a.shape(0)), columns = int(a.shape(1));
    const int angles = int(coordinates.shape(1));
    const int size = radon_detector_size(rows, columns, circle);
    auto output = bindings::empty_c_array<T>({size, angles});
    auto* out = output.mutable_data();
    const auto* input = data<T>(a);
    const auto* values = data<double>(coordinates);
    const auto input_count = a.size(), coordinate_count = coordinates.size();
    bool warn = false;
    const bool admissible = [&] {
        py::gil_scoped_release release;
        if (!finite(input, input_count) || !finite(values, coordinate_count)) return false;
        std::vector<RadonRotation> rotations;
        rotations.reserve(angles);
        for (int k = 0; k < angles; ++k) {
            if (std::abs(values[k]) > 1 || std::abs(values[angles + k]) > 1
                || std::abs(values[2 * angles + k]) > 2 * size
                || std::abs(values[3 * angles + k]) > 2 * size) {
                return false;
            }
            rotations.push_back({values[k], values[angles + k], values[2 * angles + k], values[3 * angles + k]});
        }
        warn = circle && radon_outside_circle(input, rows, columns);
        radon_into(input, rows, columns, rotations.data(), angles, circle, out);
        return true;
    }();
    if (!admissible) return py::none();
    return py::make_tuple(std::move(output), warn);
}

template<class T>
py::object inverse(const py::array& a, const py::array& coordinates, int size, TomographyFilter filter,
                   TomographyInterpolation interpolation, bool circle) {
    if (!valid<T>(a, 2)) return py::none();
    storage(a);
    const int detectors = int(a.shape(0)), angles = int(a.shape(1));
    if (coordinates.shape(1) != angles || size < 1 || size > max_extent
        || (interpolation == TomographyInterpolation::cubic
            && radon_detector_size(detectors, detectors, !circle) < 4)) return py::none();
    auto output = bindings::empty_c_array<T>({size, size});
    auto* out = output.mutable_data();
    const auto* input = data<T>(a);
    const auto* values = data<double>(coordinates);
    const auto input_count = a.size(), coordinate_count = coordinates.size();
    bool admissible;
    {
        py::gil_scoped_release release;
        admissible = finite(input, input_count) && finite(values, coordinate_count);
        if (admissible) {
            iradon_into(input, detectors, angles, values, values + angles, size, filter, interpolation, circle, out);
        }
    }
    if (!admissible) return py::none();
    return output;
}

template<class T>
py::object sart(const py::array& a, const py::array& theta, const py::array& shifts,
                const py::object& initial, const py::object& clipping, double relaxation, bool weak_scalars) {
    if (!valid<T>(a, 2) || !valid<T>(theta, 1) || !valid<T>(shifts, 1)
        || a.shape(1) != theta.shape(0) || a.shape(1) != shifts.shape(0)
        || a.shape(0) < 4 || !(std::abs(relaxation) <= 1)) return py::none();
    storage(a);
    storage(theta);
    storage(shifts);
    const int size = int(a.shape(0)), angles = int(a.shape(1));
    const bool has_clip = !clipping.is_none();
    const bool has_initial = !initial.is_none();
    if ((has_clip && !plain_array(clipping)) || (has_initial && !plain_array(initial))) return py::none();
    py::array clip, image;
    const T* bounds = nullptr;
    if (has_clip) {
        clip = py::reinterpret_borrow<py::array>(clipping);
        if (!valid<T>(clip, 1) || clip.shape(0) != 2) return py::none();
        storage(clip);
        bounds = data<T>(clip);
    }
    if (has_initial) {
        image = py::reinterpret_borrow<py::array>(initial);
        if (!valid<T>(image, 2) || image.shape(0) != size || image.shape(1) != size) return py::none();
        storage(image);
    } else {
        image = bindings::empty_c_array<T>({size, size});
    }
    py::array_t<T> clipped;
    T* final_image = nullptr;
    if (has_clip) {
        clipped = bindings::empty_c_array<T>({size, size});
        final_image = clipped.mutable_data();
    }
    auto* estimate = static_cast<T*>(image.mutable_data());
    const auto* input = data<T>(a);
    const auto* angle_values = data<T>(theta);
    const auto* shift_values = data<T>(shifts);
    const auto input_count = a.size(), image_count = image.size();
    bool admissible;
    {
        py::gil_scoped_release release;
        admissible = finite(input, input_count) && (!has_initial || finite(estimate, image_count))
            && (!has_clip || finite(bounds, 2))
            && std::all_of(angle_values, angle_values + angles, [](T x) { return std::abs(x) <= T(360); })
            && std::all_of(shift_values, shift_values + angles, [=](T x) { return std::abs(x) <= T(size); });
        if (admissible) {
            std::vector<T> sorted(angle_values, angle_values + angles);
            std::sort(sorted.begin(), sorted.end());
            admissible = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        }
        if (admissible) {
            if (!has_initial) std::fill_n(estimate, std::size_t(size) * size, T(0));
            iradon_sart_into(input, size, angles, angle_values, shift_values, T(relaxation), weak_scalars,
                             estimate, bounds, final_image);
        }
    }
    if (!admissible) return py::none();
    if (has_clip) return clipped;
    return image;
}
}

void bind_tomography(py::module_& module) {
    module.def("_tomography_storage", [](const py::handle& input) {
        if (!plain_array(input)) throw py::type_error("tomography requires an ordinary ndarray");
        storage(py::reinterpret_borrow<py::array>(input));
    });
    module.def("_tomography_radon", [](const py::handle& input, const py::handle& transform, bool circle) -> py::object {
        if (!plain_array(input) || !plain_array(transform)) return py::none();
        const auto a = py::reinterpret_borrow<py::array>(input);
        const auto coordinates = py::reinterpret_borrow<py::array>(transform);
        if (!valid<double>(coordinates, 2) || coordinates.shape(0) != 4) return py::none();
        storage(coordinates);
        if (a.dtype().char_() == 'f') return forward<float>(a, coordinates, circle);
        return forward<double>(a, coordinates, circle);
    });
    module.def("_tomography_iradon", [](const py::handle& input, const py::handle& transform, int size,
                                        int filter, int interpolation, bool circle) -> py::object {
        if (!plain_array(input) || !plain_array(transform)) return py::none();
        const auto a = py::reinterpret_borrow<py::array>(input);
        const auto coordinates = py::reinterpret_borrow<py::array>(transform);
        if (!valid<double>(coordinates, 2) || coordinates.shape(0) != 2
            || filter < 0 || filter > int(TomographyFilter::none)
            || interpolation < 0 || interpolation > int(TomographyInterpolation::cubic)) return py::none();
        storage(coordinates);
        const auto f = static_cast<TomographyFilter>(filter);
        const auto i = static_cast<TomographyInterpolation>(interpolation);
        if (a.dtype().char_() == 'f') return inverse<float>(a, coordinates, size, f, i, circle);
        return inverse<double>(a, coordinates, size, f, i, circle);
    });
    module.def("_tomography_sart", [](const py::handle& input, const py::handle& angles, const py::handle& offset,
                                      const py::object& initial, const py::object& clipping, double relaxation,
                                      bool weak_scalars) -> py::object {
        if (!plain_array(input) || !plain_array(angles) || !plain_array(offset)) return py::none();
        const auto a = py::reinterpret_borrow<py::array>(input);
        const auto theta = py::reinterpret_borrow<py::array>(angles);
        const auto shifts = py::reinterpret_borrow<py::array>(offset);
        if (a.dtype().char_() == 'f') return sart<float>(a, theta, shifts, initial, clipping, relaxation, weak_scalars);
        return sart<double>(a, theta, shifts, initial, clipping, relaxation, weak_scalars);
    });
}
