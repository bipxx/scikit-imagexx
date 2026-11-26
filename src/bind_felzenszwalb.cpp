#include "bind_array_storage.hpp"
#include "skimagexx/felzenszwalb.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

namespace py = pybind11;

namespace {

using Index = std::ptrdiff_t;
static_assert(sizeof(double) == sizeof(std::uint64_t));

struct RadixEdge {
    std::uint64_t key;
    std::uint32_t index;
};

std::unique_ptr<Index[]> unique_cost_order(const double* costs, Index count) {
    if (static_cast<std::uint64_t>(count) > std::numeric_limits<std::uint32_t>::max())
        return {};

    constexpr unsigned bits = 11;
    constexpr unsigned buckets = 1u << bits;
    constexpr unsigned passes = (64u + bits - 1) / bits;
    constexpr auto mask = buckets - 1;
    std::array<std::array<std::uint32_t, buckets>, passes> histograms{};
    std::unique_ptr<RadixEdge[]> edges(new RadixEdge[count]);
    std::unique_ptr<RadixEdge[]> scratch(new RadixEdge[count]);
    for (Index i = 0; i < count; ++i) {
        // Costs are nonnegative finite square roots. Unsigned IEEE-754 bits
        // order them numerically, and the zero check coalesces signed zeros.
        std::uint64_t key = 0;
        if (costs[i] != 0)
            std::memcpy(&key, costs + i, sizeof(key));
        edges[i] = {key, static_cast<std::uint32_t>(i)};
        for (unsigned pass = 0; pass < passes; ++pass)
            ++histograms[pass][(key >> (pass * bits)) & mask];
    }

    for (unsigned pass = 0; pass < passes; ++pass) {
        auto& positions = histograms[pass];
        if (std::find(positions.begin(), positions.end(), count) != positions.end())
            continue;
        std::uint32_t offset = 0;
        for (auto& position : positions) {
            const auto frequency = position;
            position = offset;
            offset += frequency;
        }
        const auto shift = pass * bits;
        for (Index i = 0; i < count; ++i) {
            const auto edge = edges[i];
            scratch[positions[(edge.key >> shift) & mask]++] = edge;
        }
        edges.swap(scratch);
    }

    for (Index i = 1; i < count; ++i)
        if (edges[i - 1].key == edges[i].key)
            return {};
    std::unique_ptr<Index[]> order(new Index[count]);
    for (Index i = 0; i < count; ++i)
        order[i] = edges[i].index;
    return order;
}

} // namespace

void bind_felzenszwalb(py::module_& module) {
    module.def("felzenszwalb_prepared", [](const py::array& image, double scale,
                                         std::ptrdiff_t min_size) -> py::object {
        const auto dtype = image.dtype();
        if (Py_TYPE(image.ptr()) != py::detail::npy_api::get().PyArray_Type_ ||
            dtype.char_() != 'd' || !dtype.equal(py::dtype::of<double>()) ||
            !dtype.attr("metadata").is_none() || !(image.flags() & py::array::c_style) ||
            !image.writeable() || reinterpret_cast<std::uintptr_t>(image.data()) % alignof(double)) {
            throw py::type_error("felzenszwalb requires writable contiguous float64 storage");
        }
        if (image.ndim() != 3 || image.size() == 0 || image.shape(2) > 3 ||
            image.size() > std::numeric_limits<std::ptrdiff_t>::max() / 4 || !std::isfinite(scale)) {
            throw py::value_error("invalid felzenszwalb image or scale");
        }
        skimagexx::bindings::check_array_storage(image);
        const auto height = image.shape(0);
        const auto width = image.shape(1);
        const auto channels = image.shape(2);
        const auto count = skimagexx::felzenszwalb_edge_count(height, width);
        py::array_t<double> costs(count);
        const auto* values = static_cast<const double*>(image.data());
        auto* weights = costs.mutable_data();
        bool finite;
        std::unique_ptr<Index[]> native_order;
        {
            py::gil_scoped_release release;
            skimagexx::felzenszwalb_costs_into(values, weights, height, width, channels);
            finite = std::all_of(weights, weights + count, [](double v) { return std::isfinite(v); });
            if (finite)
                native_order = unique_cost_order(weights, count);
        }
        if (!finite) return py::none();

        // Equal-cost edges retain the runtime's NumPy quicksort tie ordering.
        py::array numpy_order;
        const auto* indices = native_order.get();
        if (!indices) {
            numpy_order = costs.attr("argsort")().cast<py::array>();
            indices = static_cast<const Index*>(numpy_order.data());
        }
        py::array_t<std::ptrdiff_t> output({height, width});
        auto* labels = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::felzenszwalb_labels_into(weights, indices, labels, height, width, scale, min_size);
        }
        return output;
    }, py::arg("image").noconvert(), py::arg("scale"), py::arg("min_size"));
}
