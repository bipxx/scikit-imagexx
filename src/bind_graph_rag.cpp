#include "bind_array_storage.hpp"
#include "skimagexx/graph_rag.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace rag = skimagexx::graph_rag;

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

} // namespace

void bind_graph_rag(py::module_& module) {
    module.def("graph_rag_check", &skimagexx::bindings::check_array_storage,
               py::arg("array").noconvert());
    module.def("graph_rag_adjacency", [](const Array<rag::Label>& labels, int connectivity) {
        check(labels);
        const auto extents = shape(labels);
        std::vector<rag::Edge> edges;
        {
            py::gil_scoped_release release;
            edges = rag::adjacency(labels.data(), extents, connectivity);
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
        std::vector<rag::RegionColor> regions;
        {
            py::gil_scoped_release release;
            regions = rag::region_colors(labels.data(), rgb.data(),
                                         static_cast<std::size_t>(labels.size()), label_count);
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
        std::vector<rag::BoundaryEdge> edges;
        bool exact = true;
        {
            py::gil_scoped_release release;
            // All partial sums must be independent of the sparse duplicate
            // order. First bound the lattice globally, then for float32 by edge.
            const auto* data = values.data();
            const auto size = static_cast<std::size_t>(values.size());
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
                edges = rag::boundary(labels.data(), data, extents, connectivity, false);
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
        std::vector<rag::BoundarySample> samples;
        {
            py::gil_scoped_release release;
            samples = rag::boundary_samples(labels.data(), extents, connectivity);
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
        std::vector<std::array<rag::Label, 2>> centers;
        {
            py::gil_scoped_release release;
            centers = rag::centroids(labels.data(), static_cast<std::size_t>(labels.shape(0)),
                                     static_cast<std::size_t>(labels.shape(1)), label_count);
        }
        py::list result;
        for (const auto& center : centers) {
            result.append(py::make_tuple(center[0], center[1]));
        }
        return result;
    }, py::arg("labels").noconvert(), py::arg("label_count"));
    module.def("graph_rag_threshold", [](const py::dict& adjacency, const py::object& threshold,
                                          const py::tuple& scalar_types) -> py::object {
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
        std::vector<std::vector<std::size_t>> groups(count);
        for (std::size_t i = 0; i < count; ++i) {
            groups[static_cast<std::size_t>(components[i])].push_back(i);
        }
        py::list assignments(count);
        std::size_t next = 0;
        for (std::size_t component = 0; component < count && !groups[component].empty(); ++component) {
            const py::int_ label(component);
            for (auto index : groups[component]) {
                assignments[next++] = py::make_tuple(nodes[index], label);
            }
        }
        return py::make_tuple(removed, assignments);
    });
    module.def("graph_rag_central", [](const Array<std::int32_t>& offsets,
                                       const Array<std::int32_t>& indices, const Array<double>& weights) {
        check_csr(offsets, indices, weights);
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = rag::central_distances(static_cast<std::size_t>(offsets.size() - 1),
                offsets.data(), indices.data(), weights.data(), static_cast<std::size_t>(weights.size()));
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
        rag::SpectralCut result;
        {
            py::gil_scoped_release release;
            if (offsets.data()[0] != 0 || offsets.data()[vector.size()] != weights.size() ||
                !std::is_sorted(offsets.data(), offsets.data() + offsets.size()) ||
                std::any_of(indices.data(), indices.data() + indices.size(), [&](auto index) {
                    return index < 0 || index >= vector.size();
                })) {
                throw std::invalid_argument("invalid RAG spectral indices");
            }
            result = rag::spectral_cut(vector.data(), degrees.data(), static_cast<std::size_t>(vector.size()),
                                       offsets.data(), indices.data(), weights.data(), cuts);
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
