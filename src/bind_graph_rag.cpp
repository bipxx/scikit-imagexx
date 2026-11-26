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
    return py::make_tuple(to_array(weights), to_array(indices), to_array(offsets));
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

} // namespace

void bind_graph_rag(py::module_& module) {
    module.def("graph_rag_normalized_buffers", &normalized_buffers);
    module.def("graph_rag_initial_heap", &initial_heap);
    module.def("graph_rag_refresh_heap", &refresh_heap);
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
                                          const py::object& threshold, const py::tuple& scalar_types,
                                          std::size_t slots) -> py::object {
        py::dict positions;
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
            positions[py::handle(node)] = py::int_(nodes.size());
            nodes.emplace_back(node);
        }
        const py::str weight("weight");
        py::list removed;
        std::vector<rag::Edge> kept;
        iteration = 0;
        for (std::size_t i = 0; PyDict_Next(adjacency.ptr(), &iteration, &node, &neighbors); ++i) {
            PyObject* other = nullptr;
            PyObject* data = nullptr;
            Py_ssize_t neighbor_index = 0;
            while (PyDict_Next(neighbors, &neighbor_index, &other, &data)) {
                if (!scalar(other) || !PyDict_CheckExact(data)) {
                    return py::none();
                }
                auto* position = PyDict_GetItemWithError(positions.ptr(), other);
                if (!position) {
                    if (PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                    return py::none();
                }
                const auto j = static_cast<std::size_t>(PyLong_AsSsize_t(position));
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
        if (slots && count <= slots) {
            const py::str labels("labels");
            std::vector<std::size_t> ids;
            ids.reserve(count);
            for (auto key : nodes) {
                auto* attrs = dict_item(node_data.ptr(), key.ptr());
                auto* values = attrs && PyDict_CheckExact(attrs) ? dict_item(attrs, labels.ptr()) : nullptr;
                if (!values || !PyList_CheckExact(values) || PyList_GET_SIZE(values) != 1 ||
                    !PyLong_CheckExact(key.ptr())) {
                    break;
                }
                auto* value = PyList_GET_ITEM(values, 0);
                if (value != key.ptr() &&
                    (!PyLong_CheckExact(value) || PyObject_RichCompareBool(value, key.ptr(), Py_EQ) != 1)) {
                    break;
                }
                int overflow = 0;
                const auto id = PyLong_AsLongLongAndOverflow(key.ptr(), &overflow);
                if (overflow || id < 0 || static_cast<std::size_t>(id) >= slots) {
                    break;
                }
                ids.push_back(static_cast<std::size_t>(id));
            }
            if (ids.size() == count) {
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
