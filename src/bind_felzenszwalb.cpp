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
    std::uint32_t low;
    std::uint32_t high;
    std::uint32_t index;
};

std::unique_ptr<Index[]> unique_cost_order(const double* costs, Index count, bool& finite) {
    if (!count || static_cast<std::uint64_t>(count) > std::numeric_limits<std::uint32_t>::max()) {
        finite = std::all_of(costs, costs + count, [](double v) { return std::isfinite(v); });
        return {};
    }

    constexpr unsigned bits = 11;
    constexpr unsigned buckets = 1u << bits;
    constexpr unsigned mask = buckets - 1;
    std::array<std::array<std::uint32_t, buckets>, 6> histograms{};
    std::unique_ptr<RadixEdge[]> edges(new RadixEdge[count]);
    for (Index i = 0; i < count; ++i) {
        // Square roots of sums of squares are nonnegative, including +0.
        // Their unsigned IEEE-754 bits have the same order as their values.
        std::uint64_t key;
        std::memcpy(&key, costs + i, sizeof(key));
        const auto low = static_cast<std::uint32_t>(key);
        const auto high = static_cast<std::uint32_t>(key >> 32);
        if (high >= UINT32_C(0x7ff00000)) {
            finite = false;
            return {};
        }
        edges[i] = {low, high, static_cast<std::uint32_t>(i)};
        ++histograms[0][low & mask];
        ++histograms[1][(low >> bits) & mask];
        ++histograms[2][low >> (2 * bits)];
        ++histograms[3][high & mask];
        ++histograms[4][(high >> bits) & mask];
        ++histograms[5][high >> (2 * bits)];
    }

    std::unique_ptr<RadixEdge[]> scratch;
    const auto scatter = [&](unsigned pass, unsigned shift, unsigned size, auto word) {
        auto& positions = histograms[pass];
        std::uint32_t offset = 0;
        for (unsigned i = 0; i < size; ++i) {
            const auto frequency = positions[i];
            if (frequency == count)
                return;
            positions[i] = offset;
            offset += frequency;
        }
        if (!scratch)
            scratch.reset(new RadixEdge[count]);
        for (Index i = 0; i < count; ++i) {
            const auto edge = edges[i];
            scratch[positions[((edge.*word) >> shift) & mask]++] = edge;
        }
        edges.swap(scratch);
    };
    constexpr auto top_buckets = 1u << (32 - 2 * bits);
    scatter(0, 0, buckets, &RadixEdge::low);
    scatter(1, bits, buckets, &RadixEdge::low);
    scatter(2, 2 * bits, top_buckets, &RadixEdge::low);
    scatter(3, 0, buckets, &RadixEdge::high);
    scatter(4, bits, buckets, &RadixEdge::high);
    scatter(5, 2 * bits, top_buckets, &RadixEdge::high);

    for (Index i = 1; i < count; ++i)
        if (edges[i - 1].high == edges[i].high && edges[i - 1].low == edges[i].low)
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
        bool finite = true;
        std::unique_ptr<Index[]> native_order;
        {
            py::gil_scoped_release release;
            skimagexx::felzenszwalb_costs_into(values, weights, height, width, channels);
            native_order = unique_cost_order(weights, count, finite);
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
