#include <skimagexx/order_angles_golden_ratio.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <limits>
#include <optional>

#include "bind_array_storage.hpp"

namespace py = pybind11;
namespace {
using skimagexx::ComponentsView;
using skimagexx::PixelType;

std::optional<ComponentsView> view(const py::array& array) {
    if (array.ndim() != 1 || array.shape(0) > 16) return std::nullopt;
    const auto kind = array.dtype().kind();
    const auto item = array.itemsize();
    ComponentsView result;
    if (kind == 'b' && item == 1) result.type = PixelType::boolean;
    else if (kind == 'u' && item == 1) result.type = PixelType::uint8;
    else if (kind == 'i' && item == 1) result.type = PixelType::int8;
    else if (kind == 'u' && item == 2) result.type = PixelType::uint16;
    else if (kind == 'i' && item == 2) result.type = PixelType::int16;
    else if (kind == 'u' && item == 4) result.type = PixelType::uint32;
    else if (kind == 'i' && item == 4) result.type = PixelType::int32;
    else if (kind == 'u' && item == 8) result.type = PixelType::uint64;
    else if (kind == 'i' && item == 8) result.type = PixelType::int64;
    else if (kind == 'f' && item == 2) {
        result.type = PixelType::float32;
        result.half = true;
    } else if (kind == 'f' && item == 4) result.type = PixelType::float32;
    else if (kind == 'f' && item == 8) result.type = PixelType::float64;
    else return std::nullopt;

    const auto base = array.base();
    if (base && (PyMemoryView_Check(base.ptr()) || PyByteArray_CheckExact(base.ptr())))
        return std::nullopt;
    skimagexx::bindings::check_array_storage(array);
    result.data = array.data();
    result.shape = {static_cast<std::size_t>(array.shape(0))};
    result.strides = {array.strides(0)};
    result.swap = !py::cast<bool>(array.dtype().attr("isnative"));
    return result;
}

template<class T> T load(const ComponentsView& array, std::size_t i) {
    T result;
    const auto* source = static_cast<const char*>(array.data) + std::ptrdiff_t(i) * array.strides[0];
    std::memcpy(&result, source, sizeof(result));
    if (array.swap) {
        auto* bytes = reinterpret_cast<unsigned char*>(&result);
        std::reverse(bytes, bytes + sizeof(result));
    }
    return result;
}

template<class Word>
bool normal_or_zero(Word word, Word smallest_normal, Word limit) {
    // Classify the raw IEEE magnitude. Even a signaling NaN must not touch FP
    // state before the selected upstream generator begins.
    const auto magnitude = word & (std::numeric_limits<Word>::max() >> 1);
    return magnitude == 0 || (magnitude >= smallest_normal && magnitude <= limit);
}

template<class T> bool signed_angle(T angle) {
    return angle >= -360 && angle <= 360;
}

bool supported_value(const ComponentsView& array, std::size_t i) {
    if (array.half)
        return normal_or_zero<std::uint16_t>(load<std::uint16_t>(array, i), 0x0400, 0x5da0);
    switch (array.type) {
    case PixelType::boolean: return load<std::uint8_t>(array, i) <= 1;
    case PixelType::int8:
    case PixelType::uint8: return true;
    case PixelType::int16: return signed_angle(load<std::int16_t>(array, i));
    case PixelType::int32: return signed_angle(load<std::int32_t>(array, i));
    case PixelType::int64: return signed_angle(load<std::int64_t>(array, i));
    case PixelType::uint16: return load<std::uint16_t>(array, i) <= 360;
    case PixelType::uint32: return load<std::uint32_t>(array, i) <= 360;
    case PixelType::uint64: return load<std::uint64_t>(array, i) <= 360;
    case PixelType::float32:
        return normal_or_zero<std::uint32_t>(load<std::uint32_t>(array, i), 0x00800000, 0x43b40000);
    case PixelType::float64:
        return normal_or_zero<std::uint64_t>(load<std::uint64_t>(array, i),
                                             UINT64_C(0x0010000000000000), UINT64_C(0x4076800000000000));
    }
    return false;
}

struct State {
    py::array owner;
    skimagexx::GoldenAngleOrder order;
    std::int64_t first;

    State(const py::array& array, const ComponentsView& input, bool weak)
        : owner(array),
          order(input, weak ? skimagexx::GoldenScalarPolicy::weak_input
                            : skimagexx::GoldenScalarPolicy::wide),
          first(order.begin()) {}
};
}

void bind_order_angles_golden_ratio(py::module_& module) {
    py::class_<State>(module, "GoldenAngleState")
        .def_property_readonly("first", [](const State& state) { return state.first; })
        .def_property_readonly("worker_budget", [](const State& state) { return state.order.worker_budget(); })
        .def_property_readonly("remaining_indices", [](const State& state) { return state.order.remaining(); });
    module.def("_order_angles_begin", [](const py::array& array, bool weak) -> py::object {
        if (std::fegetround() != FE_TONEAREST) return py::none();
        const auto input = view(array);
        if (!input) return py::none();
        for (std::size_t i = 0; i < input->shape[0]; ++i)
            if (!supported_value(*input, i)) return py::none();
        return py::cast(State(array, *input, weak));
    }, py::arg("input").noconvert(), py::arg("weak"));
    module.def("_order_angles_resume", [](State& state) { return state.order.resume(); });
    module.def("_order_angles_next", [](State& state, const py::array& array) {
        const auto input = view(array);
        if (!input || std::fegetround() != FE_TONEAREST)
            throw py::value_error("golden-angle live input left the native domain");
        for (auto i : state.order.remaining()) {
            if (i >= input->shape[0] || !supported_value(*input, i))
                throw py::value_error("golden-angle live input left the native domain");
        }
        return state.order.next(*input);
    }, py::arg("state"), py::arg("live").noconvert());
}