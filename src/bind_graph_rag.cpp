#include "bind_array_storage.hpp"
#include "skimagexx/graph_rag.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace py = pybind11;
namespace rag = skimagexx::graph_rag;

namespace skimagexx::graph_rag {
std::vector<std::size_t> component_order(const std::vector<Label>& components);
}

namespace {

template<class T>
using Array = py::array_t<T, py::array::c_style>;

template<class T>
void check(const Array<T>& array) {
    skimagexx::bindings::check_array_storage(array);
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T)) {
        throw py::value_error("RAG requires aligned arrays");
    }
}

std::vector<std::size_t> shape(const Array<rag::Label>& labels) {
    return {labels.shape(), labels.shape() + labels.ndim()};
}

template<class T>
py::array_t<T> to_array(const std::vector<T>& values) {
    py::array_t<T> output(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return output;
}

void check_csr(const Array<std::int32_t>& offsets, const Array<std::int32_t>& indices,
               const Array<double>& weights) {
    check(offsets);
    check(indices);
    check(weights);
    if (offsets.ndim() != 1 || offsets.size() < 1 || indices.ndim() != 1 ||
        weights.ndim() != 1 || indices.size() != weights.size()) {
        throw py::value_error("invalid RAG sparse buffers");
    }
}

PyObject* dict_item(PyObject* dict, PyObject* key) {
    auto* result = PyDict_GetItemWithError(dict, key);
    if (!result && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return result;
}

bool exact_type(PyObject* object, py::handle type) {
    return reinterpret_cast<PyObject*>(Py_TYPE(object)) == type.ptr();
}

bool integer_node(PyObject* node, const py::tuple& types) {
    return PyLong_CheckExact(node) ||
        std::any_of(types.begin(), types.end(), [&](auto type) { return exact_type(node, type); });
}

std::size_t bounded_index(PyObject* value, std::size_t slots, const py::tuple& types) {
    py::object index;
    if (!PyLong_CheckExact(value)) {
        if (!integer_node(value, types)) {
            return slots;
        }
        index = py::reinterpret_steal<py::object>(PyNumber_Index(value));
        if (!index) {
            throw py::error_already_set();
        }
        value = index.ptr();
    }
    int overflow = 0;
    const auto number = PyLong_AsLongLongAndOverflow(value, &overflow);
    return !overflow && number >= 0 && static_cast<unsigned long long>(number) < slots
        ? static_cast<std::size_t>(number) : slots;
}

void set_dict_item(PyObject* dict, PyObject* key, PyObject* value) {
    if (PyDict_SetItem(dict, key, value) < 0) {
        throw py::error_already_set();
    }
}

py::dict copy_dict(PyObject* source) {
    auto copy = py::reinterpret_steal<py::dict>(PyDict_Copy(source));
    if (!copy) {
        throw py::error_already_set();
    }
    return copy;
}

py::object copy_graph_data(const py::dict& nodes, const py::dict& adjacency,
                           const py::tuple& integer_types) {
    py::dict node_copy, adjacency_copy;
    Py_ssize_t cursor = 0;
    PyObject* node = nullptr;
    PyObject* attrs = nullptr;
    while (PyDict_Next(nodes.ptr(), &cursor, &node, &attrs)) {
        if (!integer_node(node, integer_types) || !PyDict_CheckExact(attrs)) {
            return py::none();
        }
        set_dict_item(node_copy.ptr(), node, copy_dict(attrs).ptr());
        set_dict_item(adjacency_copy.ptr(), node, py::dict().ptr());
    }
    cursor = 0;
    PyObject* neighbors = nullptr;
    while (PyDict_Next(adjacency.ptr(), &cursor, &node, &neighbors)) {
        if (!integer_node(node, integer_types) || !PyDict_CheckExact(neighbors)) {
            return py::none();
        }
        auto* target = dict_item(adjacency_copy.ptr(), node);
        if (!target) {
            return py::none();
        }
        Py_ssize_t edge = 0;
        PyObject* neighbor = nullptr;
        while (PyDict_Next(neighbors, &edge, &neighbor, &attrs)) {
            if (!integer_node(neighbor, integer_types) || !PyDict_CheckExact(attrs)) {
                return py::none();
            }
            if (dict_item(target, neighbor)) {
                continue;
            }
            auto* reverse = dict_item(adjacency_copy.ptr(), neighbor);
            if (!reverse) {
                return py::none();
            }
            const auto data = copy_dict(attrs);
            set_dict_item(target, neighbor, data.ptr());
            set_dict_item(reverse, node, data.ptr());
        }
    }
    return py::make_tuple(node_copy, adjacency_copy);
}

template<class T>
std::size_t label_slots(const Array<T>& labels) {
    check(labels);
    const auto size = static_cast<std::size_t>(labels.size());
    const auto* data = labels.data();
    if (!size || size > (std::size_t{1} << 30)) {
        return 0;
    }
    const auto scan = [&]() -> std::size_t {
        const auto limit = 4 * std::clamp(size, std::size_t{16384}, std::size_t{1} << 18);
        T low = data[0];
        T high = low;
        for (std::size_t i = 1; i < size; ++i) {
            low = std::min(low, data[i]);
            high = std::max(high, data[i]);
        }
        return low < 0 || static_cast<std::uint64_t>(high) >= limit
            ? 0 : static_cast<std::size_t>(high) + 1;
    };
    if (size < 65536) {
        return scan();
    }
    py::gil_scoped_release release;
    return scan();
}

py::object merge_mapping(const py::dict& nodes, std::size_t slots,
                         const py::tuple& integer_types) {
    py::array_t<py::ssize_t> mapping(static_cast<py::ssize_t>(slots));
    auto* output = mapping.mutable_data();
    std::iota(output, output + slots, py::ssize_t{0});
    const py::str labels("labels");
    Py_ssize_t cursor = 0;
    PyObject* attrs = nullptr;
    py::ssize_t region = 0;
    while (PyDict_Next(nodes.ptr(), &cursor, nullptr, &attrs)) {
        auto* values = PyDict_CheckExact(attrs) ? dict_item(attrs, labels.ptr()) : nullptr;
        if (!values || !PyList_CheckExact(values)) {
            return py::none();
        }
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(values); ++i) {
            const auto id = bounded_index(PyList_GET_ITEM(values, i), slots, integer_types);
            if (id == slots) {
                return py::none();
            }
            output[id] = region;
        }
        ++region;
    }
    return mapping;
}

py::list heap_entry(PyObject* weight, PyObject* first, PyObject* second) {
    py::list entry(4);
    PyList_SET_ITEM(entry.ptr(), 0, Py_NewRef(weight));
    PyList_SET_ITEM(entry.ptr(), 1, Py_NewRef(first));
    PyList_SET_ITEM(entry.ptr(), 2, Py_NewRef(second));
    PyList_SET_ITEM(entry.ptr(), 3, Py_NewRef(Py_True));
    return entry;
}

py::object normalized_buffers(const py::dict& adjacency, double diagonal,
                              const py::tuple& integer_types, py::handle float32,
                              py::handle float64) {
    constexpr std::size_t limit = std::size_t{1} << 30;
    const auto size = adjacency.size();
    if (size >= limit) {
        return py::none();
    }
    py::dict positions;
    PyObject* node = nullptr;
    PyObject* neighbors = nullptr;
    Py_ssize_t cursor = 0;
    std::int32_t index = 0;
    while (PyDict_Next(adjacency.ptr(), &cursor, &node, &neighbors)) {
        if (!integer_node(node, integer_types) || !PyDict_CheckExact(neighbors)) {
            return py::none();
        }
        positions[py::handle(node)] = py::int_(index++);
    }
    std::vector<std::int32_t> offsets{0};
    std::vector<std::int32_t> indices;
    std::vector<double> weights;
    std::vector<std::pair<std::int32_t, double>> column;
    bool has_off_diagonal = false;
    offsets.reserve(size + 1);
    const py::str weight("weight");
    cursor = 0;
    index = 0;
    while (PyDict_Next(adjacency.ptr(), &cursor, &node, &neighbors)) {
        column.clear();
        column.emplace_back(index, diagonal);
        PyObject* neighbor = nullptr;
        PyObject* attrs = nullptr;
        Py_ssize_t edge = 0;
        while (PyDict_Next(neighbors, &edge, &neighbor, &attrs)) {
            if (!PyDict_CheckExact(attrs) || !integer_node(neighbor, integer_types)) {
                return py::none();
            }
            auto* position = dict_item(positions.ptr(), neighbor);
            if (!position) {
                return py::none();
            }
            const auto row = static_cast<std::int32_t>(PyLong_AsLong(position));
            auto* value = dict_item(attrs, weight.ptr());
            double number = 1.0;
            if (value) {
                if (PyFloat_CheckExact(value) || (exact_type(value, float64) && PyFloat_Check(value))) {
                    number = PyFloat_AS_DOUBLE(value);
                } else if (exact_type(value, float32)) {
                    number = PyFloat_AsDouble(value);
                    if (PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                } else {
                    return py::none();
                }
                if (!(0.0 <= number && number <= 1e100)) {
                    return py::none();
                }
            }
            if (row != index) {
                column.emplace_back(row, number);
                has_off_diagonal |= number != 0.0;
            }
        }
        if (column.size() >= limit - weights.size()) {
            return py::none();
        }
        // NetworkX's COO conversion emits sorted row indices within each column.
        std::sort(column.begin(), column.end());
        for (const auto& [row, value] : column) {
            indices.push_back(row);
            weights.push_back(value);
        }
        offsets.push_back(static_cast<std::int32_t>(weights.size()));
        ++index;
    }
    return py::make_tuple(py::make_tuple(to_array(weights), to_array(indices), to_array(offsets)),
                          has_off_diagonal);
}

py::object initial_heap(const py::dict& adjacency, const py::tuple& integer_types,
                        py::handle float64, const py::function& heapify) {
    py::set seen;
    py::list heap;
    std::vector<py::handle> attributes;
    const py::str weight("weight");
    const py::str heap_item("heap item");
    PyObject* node = nullptr;
    PyObject* neighbors = nullptr;
    Py_ssize_t cursor = 0;
    while (PyDict_Next(adjacency.ptr(), &cursor, &node, &neighbors)) {
        if (!integer_node(node, integer_types) || !PyDict_CheckExact(neighbors)) {
            return py::none();
        }
        PyObject* neighbor = nullptr;
        PyObject* attrs = nullptr;
        Py_ssize_t edge = 0;
        while (PyDict_Next(neighbors, &edge, &neighbor, &attrs)) {
            if (!PyDict_CheckExact(attrs) || !integer_node(neighbor, integer_types)) {
                return py::none();
            }
            const auto previous = PySet_Contains(seen.ptr(), neighbor);
            if (previous < 0) {
                throw py::error_already_set();
            }
            if (previous) {
                continue;
            }
            auto* value = dict_item(attrs, weight.ptr());
            // Keep sequential insertion for unusual comparison semantics.
            if (!value || (!PyFloat_CheckExact(value) &&
                !(exact_type(value, float64) && PyFloat_Check(value))) ||
                std::isnan(PyFloat_AS_DOUBLE(value))) {
                return py::none();
            }
            heap.append(heap_entry(value, node, neighbor));
            attributes.emplace_back(attrs);
        }
        if (PySet_Add(seen.ptr(), node) < 0) {
            throw py::error_already_set();
        }
    }
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        if (PyDict_SetItem(attributes[i].ptr(), heap_item.ptr(), PyList_GET_ITEM(heap.ptr(), i)) < 0) {
            throw py::error_already_set();
        }
    }
    heapify(heap);
    return heap;
}

void refresh_heap(const py::dict& neighbors, py::handle node, const py::list& heap,
                  py::handle push) {
    const py::str heap_item("heap item");
    const py::str weight("weight");
    const py::int_ valid(3);
    // Use a Python items iterator here: a custom weight can mutate the graph
    // during a heap comparison, and the iterator must detect size changes.
    for (auto pair : neighbors.attr("items")()) {
        const auto entries = py::reinterpret_borrow<py::tuple>(pair);
        const auto attrs = entries[1];
        if (attrs.contains(heap_item)) {
            attrs[heap_item][valid] = py::bool_(false);
        }
        const py::object value = attrs[weight];
        auto item = heap_entry(value.ptr(), node.ptr(), entries[0].ptr());
        attrs[heap_item] = item;
        PyObject* args[] = {heap.ptr(), item.ptr()};
        auto* pushed = PyObject_Vectorcall(push.ptr(), args, 2, nullptr);
        if (!pushed) {
            throw py::error_already_set();
        }
        Py_DECREF(pushed);
    }
}

void invalidate_heap(py::handle graph, py::handle src, py::handle dst, py::handle atlas_type) {
    const py::str adjacency("adj");
    const py::str atlas("_atlas");
    const py::str values("values");
    const py::str heap_item("heap item");
    const py::int_ valid(3);
    for (auto node : {src, dst}) {
        py::object neighbors = graph.attr(adjacency)[node];
        if (exact_type(neighbors.ptr(), atlas_type)) {
            const py::object backing = neighbors.attr(atlas);
            if (PyDict_CheckExact(backing.ptr())) {
                neighbors = backing;
            }
        }
        // Use the dict's iterator so custom attribute objects can still run
        // Python and have size-changing mutations detected.
        for (auto attrs : neighbors.attr(values)()) {
            auto* item = PyDict_CheckExact(attrs.ptr()) ? dict_item(attrs.ptr(), heap_item.ptr()) : nullptr;
            if (item && PyList_CheckExact(item) && PyList_GET_SIZE(item) > 3) {
                if (PyList_SetItem(item, 3, Py_NewRef(Py_False)) < 0) {
                    throw py::error_already_set();
                }
            } else {
                attrs[heap_item][valid] = py::bool_(false);
            }
        }
    }
}

} // namespace

void bind_graph_rag(py::module_& module) {
    module.def("graph_rag_copy_data", &copy_graph_data);
    module.def("graph_rag_normalized_buffers", &normalized_buffers);
    module.def("graph_rag_initial_heap", &initial_heap);
    module.def("graph_rag_refresh_heap", &refresh_heap);
    module.def("graph_rag_invalidate_heap", &invalidate_heap);
    module.def("graph_rag_merge_mapping", &merge_mapping);
    module.def("graph_rag_label_slots", &label_slots<std::int32_t>, py::arg("labels").noconvert());
    module.def("graph_rag_label_slots", &label_slots<std::int64_t>, py::arg("labels").noconvert());
    module.def("graph_rag_check", &skimagexx::bindings::check_array_storage,
               py::arg("array").noconvert());
    module.def("graph_rag_adjacency", [](const Array<rag::Label>& labels, int connectivity) {
        check(labels);
        const auto extents = shape(labels);
        const auto* data = labels.data();
        std::vector<rag::Edge> edges;
        {
            py::gil_scoped_release release;
            edges = rag::adjacency(data, extents, connectivity);
        }
        py::list result;
        for (const auto& edge : edges) {
            result.append(py::make_tuple(edge.first, edge.second));
        }
        return result;
    }, py::arg("labels").noconvert(), py::arg("connectivity"));
    module.def("graph_rag_colors", [](const Array<rag::Label>& labels, const Array<double>& rgb,
                                      std::size_t label_count) {
        check(labels);
        check(rgb);
        if (rgb.size() % 3 || rgb.size() / 3 != labels.size()) {
            throw py::value_error("RAG RGB image and labels differ in size");
        }
        const auto* label_data = labels.data();
        const auto* pixels = rgb.data();
        const auto size = static_cast<std::size_t>(labels.size());
        std::vector<rag::RegionColor> regions;
        {
            py::gil_scoped_release release;
            regions = rag::region_colors(label_data, pixels, size, label_count);
        }
        py::dict result;
        for (const auto& region : regions) {
            if (region.count) {
                py::array_t<double> total(3);
                std::copy(region.total.begin(), region.total.end(), total.mutable_data());
                result[py::int_(region.label)] = py::make_tuple(region.count, total);
            }
        }
        return result;
    }, py::arg("labels").noconvert(), py::arg("rgb").noconvert(), py::arg("label_count"));
    module.def("graph_rag_boundary", [](const Array<rag::Label>& labels, const Array<double>& values,
                                         int connectivity, bool single_precision) -> py::object {
        check(labels);
        check(values);
        if (labels.size() != values.size()) {
            throw py::value_error("RAG edge map and labels differ in size");
        }
        const auto extents = shape(labels);
        const auto* label_data = labels.data();
        const auto* data = values.data();
        const auto size = static_cast<std::size_t>(values.size());
        std::vector<rag::BoundaryEdge> edges;
        bool exact = true;
        {
            py::gil_scoped_release release;
            // All partial sums must be independent of the sparse duplicate
            // order. First bound the lattice globally, then for float32 by edge.
            const double limit = 0x1p53 / (512.0 * static_cast<double>(size));
            double maximum = 0.0;
            for (std::size_t i = 0; i < size; ++i) {
                const double magnitude = std::abs(data[i]);
                if (!(magnitude <= limit)) {
                    exact = false;
                    break;
                }
                maximum = std::max(maximum, magnitude);
                const auto grid = 256.0 * data[i];
                if (grid != static_cast<double>(static_cast<std::int64_t>(grid))) {
                    exact = false;
                    break;
                }
            }
            if (exact) {
                edges = rag::boundary(label_data, data, extents, connectivity, false);
                exact = !single_precision || std::none_of(edges.begin(), edges.end(), [&](const auto& edge) {
                    return static_cast<double>(edge.count) * maximum > 0x1p16;
                });
            }
        }
        if (!exact) {
            return py::none();
        }
        py::list result;
        for (const auto& item : edges) {
            result.append(py::make_tuple(item.edge.first, item.edge.second, item.count, item.total));
        }
        return result;
    }, py::arg("labels").noconvert(), py::arg("values").noconvert(),
       py::arg("connectivity"), py::arg("single_precision"));
    module.def("graph_rag_boundary_samples", [](const Array<rag::Label>& labels, int connectivity) {
        check(labels);
        const auto extents = shape(labels);
        const auto* data = labels.data();
        std::vector<rag::BoundarySample> samples;
        {
            py::gil_scoped_release release;
            samples = rag::boundary_samples(data, extents, connectivity);
        }
        const auto size = static_cast<py::ssize_t>(samples.size());
        py::array_t<rag::Label> first(size), second(size);
        py::array_t<py::ssize_t> pixels(size);
        auto* lo = first.mutable_data();
        auto* hi = second.mutable_data();
        auto* positions = pixels.mutable_data();
        {
            py::gil_scoped_release release;
            for (const auto& sample : samples) {
                *lo++ = sample.edge.first;
                *hi++ = sample.edge.second;
                *positions++ = static_cast<py::ssize_t>(sample.pixel);
            }
        }
        return py::make_tuple(first, second, pixels);
    }, py::arg("labels").noconvert(), py::arg("connectivity"));
    module.def("graph_rag_centroids", [](const Array<rag::Label>& labels, std::size_t label_count) {
        check(labels);
        if (labels.ndim() != 2 || labels.size() > std::numeric_limits<std::int32_t>::max()) {
            throw py::value_error("RAG centroids require a bounded two-dimensional array");
        }
        const auto* data = labels.data();
        const auto height = static_cast<std::size_t>(labels.shape(0));
        const auto width = static_cast<std::size_t>(labels.shape(1));
        std::vector<std::array<rag::Label, 2>> centers;
        {
            py::gil_scoped_release release;
            centers = rag::centroids(data, height, width, label_count);
        }
        py::list result;
        for (const auto& center : centers) {
            result.append(py::make_tuple(center[0], center[1]));
        }
        return result;
    }, py::arg("labels").noconvert(), py::arg("label_count"));
    module.def("graph_rag_threshold", [](const py::dict& adjacency, const py::dict& node_data,
                                          const py::object& threshold, const py::tuple& integer_types,
                                          const py::tuple& scalar_types, std::size_t slots) -> py::object {
        const auto count = adjacency.size();
        std::vector<py::handle> nodes;
        nodes.reserve(count);
        auto scalar = [&](PyObject* value) {
            if (PyLong_CheckExact(value) || PyFloat_CheckExact(value)) {
                return true;
            }
            for (auto type : scalar_types) {
                if (reinterpret_cast<PyObject*>(Py_TYPE(value)) == type.ptr()) {
                    return true;
                }
            }
            return false;
        };
        PyObject* node = nullptr;
        PyObject* neighbors = nullptr;
        Py_ssize_t iteration = 0;
        while (PyDict_Next(adjacency.ptr(), &iteration, &node, &neighbors)) {
            if (!scalar(node) || !PyDict_CheckExact(neighbors)) {
                return py::none();
            }
            nodes.emplace_back(node);
        }
        std::vector<std::size_t> ids;
        if (slots && count <= slots && slots <= std::max<std::size_t>(64, 4 * count)) {
            ids.reserve(count);
            for (auto key : nodes) {
                const auto id = bounded_index(key.ptr(), slots, integer_types);
                if (id == slots) {
                    ids.clear();
                    break;
                }
                ids.push_back(id);
            }
        }
        const bool dense = !ids.empty();
        std::vector<std::size_t> dense_positions;
        py::object positions;
        if (dense) {
            dense_positions.assign(slots, count);
            for (std::size_t i = 0; i < count; ++i) {
                dense_positions[ids[i]] = i;
            }
        }
        const auto position_of = [&](PyObject* key) {
            if (dense) {
                const auto id = bounded_index(key, slots, integer_types);
                if (id != slots && dense_positions[id] != count) {
                    return dense_positions[id];
                }
            }
            if (!scalar(key)) {
                return count;
            }
            // NetworkX may retain a neighbor key such as 1.0 for an
            // integer node 1. Resolve these aliases with Python equality.
            if (!positions) {
                positions = py::dict();
                for (std::size_t i = 0; i < count; ++i) {
                    positions[nodes[i]] = py::int_(i);
                }
            }
            auto* position = dict_item(positions.ptr(), key);
            return position ? static_cast<std::size_t>(PyLong_AsSsize_t(position)) : count;
        };
        const py::str weight("weight");
        py::list removed;
        std::vector<rag::Edge> kept;
        kept.reserve(count);
        iteration = 0;
        for (std::size_t i = 0; PyDict_Next(adjacency.ptr(), &iteration, &node, &neighbors); ++i) {
            PyObject* other = nullptr;
            PyObject* data = nullptr;
            Py_ssize_t neighbor_index = 0;
            while (PyDict_Next(neighbors, &neighbor_index, &other, &data)) {
                if (!PyDict_CheckExact(data)) {
                    return py::none();
                }
                const auto j = position_of(other);
                if (j == count) {
                    return py::none();
                }
                if (j < i) {
                    continue;
                }
                auto* value = PyDict_GetItemWithError(data, weight.ptr());
                if (!value) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetObject(PyExc_KeyError, weight.ptr());
                    }
                    throw py::error_already_set();
                }
                if (!scalar(value)) {
                    return py::none();
                }
                const bool both_float = PyFloat_CheckExact(value) && PyFloat_CheckExact(threshold.ptr());
                const int compare = both_float ? PyFloat_AS_DOUBLE(value) >= PyFloat_AS_DOUBLE(threshold.ptr())
                    : PyObject_RichCompareBool(value, threshold.ptr(), Py_GE);
                if (compare < 0) {
                    throw py::error_already_set();
                }
                if (compare) {
                    removed.append(py::make_tuple(py::handle(node), py::handle(other)));
                } else {
                    kept.push_back({static_cast<rag::Label>(i), static_cast<rag::Label>(j)});
                }
            }
        }
        const auto components = rag::components(count, kept);
        if (dense) {
            const py::str labels("labels");
            std::size_t singletons = 0;
            for (; singletons < count; ++singletons) {
                auto key = nodes[singletons];
                auto* attrs = dict_item(node_data.ptr(), key.ptr());
                auto* values = attrs && PyDict_CheckExact(attrs) ? dict_item(attrs, labels.ptr()) : nullptr;
                if (!values || !PyList_CheckExact(values) || PyList_GET_SIZE(values) != 1) {
                    break;
                }
                auto* value = PyList_GET_ITEM(values, 0);
                if (value != key.ptr() && bounded_index(value, slots, integer_types) != ids[singletons]) {
                    break;
                }
            }
            if (singletons == count) {
                py::array_t<rag::Label> mapping(static_cast<py::ssize_t>(slots));
                auto* destination = mapping.mutable_data();
                std::iota(destination, destination + slots, rag::Label{0});
                for (std::size_t i = 0; i < count; ++i) {
                    destination[ids[i]] = components[i];
                }
                return py::make_tuple(removed, mapping);
            }
        }
        py::list assignments(count);
        std::size_t next = 0;
        for (auto index : rag::component_order(components)) {
            assignments[next++] = py::make_tuple(nodes[index], components[index]);
        }
        return py::make_tuple(removed, assignments);
    });
    module.def("graph_rag_central", [](const Array<std::int32_t>& offsets,
                                       const Array<std::int32_t>& indices, const Array<double>& weights) {
        check_csr(offsets, indices, weights);
        const auto size = static_cast<std::size_t>(offsets.size() - 1);
        const auto entries = static_cast<std::size_t>(weights.size());
        const auto* rows = offsets.data();
        const auto* columns = indices.data();
        const auto* data = weights.data();
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = rag::central_distances(size, rows, columns, data, entries);
        }
        return to_array(result);
    }, py::arg("offsets").noconvert(), py::arg("indices").noconvert(), py::arg("weights").noconvert());
    module.def("graph_rag_spectral", [](const Array<double>& vector, const Array<double>& degrees,
                                        const Array<std::int32_t>& offsets, const Array<std::int32_t>& indices,
                                        const Array<double>& weights, std::size_t cuts) {
        check(vector);
        check(degrees);
        check_csr(offsets, indices, weights);
        if (vector.ndim() != 1 || degrees.ndim() != 1 || vector.size() != degrees.size() ||
            vector.size() != offsets.size() - 1) {
            throw py::value_error("RAG spectrum and graph differ in size");
        }
        const auto size = vector.size();
        const auto entries = weights.size();
        const auto* rows = offsets.data();
        const auto* columns = indices.data();
        const auto* data = weights.data();
        const auto* eigenvector = vector.data();
        const auto* diagonal = degrees.data();
        rag::SpectralCut result;
        {
            py::gil_scoped_release release;
            if (rows[0] != 0 || rows[size] != entries ||
                !std::is_sorted(rows, rows + size + 1) ||
                std::any_of(columns, columns + entries, [&](auto index) {
                    return index < 0 || index >= size;
                })) {
                throw std::invalid_argument("invalid RAG spectral indices");
            }
            result = rag::spectral_cut(eigenvector, diagonal, static_cast<std::size_t>(size),
                                       rows, columns, data, cuts);
        }
        return py::make_tuple(to_array(result.mask), result.cost);
    }, py::arg("vector").noconvert(), py::arg("degrees").noconvert(),
       py::arg("offsets").noconvert(), py::arg("indices").noconvert(),
       py::arg("weights").noconvert(), py::arg("cuts"));
}

#ifdef SKIMAGEXX_GRAPH_RAG_STANDALONE
PYBIND11_MODULE(_graph_rag_native, module) {
    bind_graph_rag(module);
}
#endif
