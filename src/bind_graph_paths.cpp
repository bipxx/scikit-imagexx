#include "skimagexx/graph_paths.hpp"

#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace py = pybind11;
namespace graph = skimagexx::graph_paths;

namespace {

template<class T>
using Flat = py::array_t<T, py::array::c_style>;

template<class T>
void check_flat(const Flat<T>& array, graph::Index size) {
    skimagexx::bindings::check_array_storage(array);
    if (array.ndim() != 1 || array.size() != size
        || reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T)) {
        throw py::value_error("graph paths requires aligned flat buffers of the grid size");
    }
}

bool supported_costs(const double* values, graph::Index size) {
    return std::none_of(values, values + size, [](double value) { return std::isnan(value); });
}

py::tuple point(const graph::Grid& grid, graph::Index node) {
    auto coordinates = grid.coordinates(node);
    py::tuple result(coordinates.size());
    for (std::size_t i = 0; i < coordinates.size(); ++i) result[i] = py::int_(coordinates[i]);
    return result;
}

py::list points(const graph::Grid& grid, const graph::Shape& path) {
    py::list result(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) result[i] = point(grid, path[i]);
    return result;
}

void check_indices(const graph::Shape& indices, graph::Index size) {
    for (const auto index : indices) {
        if (index < 0 || index >= size) throw py::value_error("graph index out of bounds");
    }
}

py::object solve(const graph::Grid& grid, const Flat<double>& costs, const graph::Shape& starts,
           const graph::Shape& ends, int mode, bool all_ends, bool use_ends, std::ptrdiff_t max_visits,
           double max_step, Flat<double>& distance, Flat<graph::Offset>& predecessor,
           const py::tuple& callbacks, graph::Shape pending, const py::module_& numpy) {
    check_flat(costs, grid.size());
    check_flat(distance, grid.size());
    check_flat(predecessor, grid.size());
    check_indices(starts, grid.size());
    check_indices(ends, grid.size());
    check_indices(pending, grid.size());
    if (!supported_costs(costs.data(), grid.size())) return py::none();
    const graph::Search options{static_cast<graph::Cost>(mode), all_ends, use_ends, max_visits, max_step};
    if (callbacks.empty()) {
        graph::find_costs(grid, costs.data(), starts, ends, options,
                          distance.mutable_data(), predecessor.mutable_data(), {}, &pending);
        return py::cast(std::move(pending));
    }
    if (callbacks.size() != 5) throw py::value_error("expected five graph callback slots");
    graph::Hooks hooks;
    const auto goal = callbacks[0];
    const auto travel = callbacks[1];
    const auto examine = callbacks[2];
    const auto update = callbacks[3];
    const auto connect = callbacks[4];
    const auto* cost_data = costs.data();
    auto* distance_data = distance.mutable_data();
    auto* predecessor_data = predecessor.mutable_data();
    // Hooks observe previous result views and may edit the original costs.
    // Validate the spans again when Python returns before using them in C++.
    const auto check_views = [&] {
        check_flat(costs, grid.size());
        check_flat(distance, grid.size());
        check_flat(predecessor, grid.size());
        if (costs.data() != cost_data || distance.mutable_data() != distance_data
            || predecessor.mutable_data() != predecessor_data) {
            throw py::value_error("graph buffers changed during a callback");
        }
    };
    if (!goal.is_none()) {
        hooks.goal = [&](graph::Index node, double cost) {
            auto result = goal(node, cost);
            if (!PyNumber_Check(result.ptr())) throw py::type_error("an integer is required");
            auto integer = py::reinterpret_steal<py::object>(PyNumber_Long(result.ptr()));
            if (!integer) throw py::error_already_set();
            const auto flag = PyLong_AsLongLong(integer.ptr());
            if (flag == -1 && PyErr_Occurred()) throw py::error_already_set();
            if (flag < std::numeric_limits<int>::min() || flag > std::numeric_limits<int>::max()) {
                PyErr_SetString(PyExc_OverflowError, "value too large to convert to int");
                throw py::error_already_set();
            }
            result = py::object();
            check_views();
            return static_cast<int>(flag);
        };
    }
    if (!travel.is_none()) {
        hooks.travel = [&](double old_cost, double new_cost, double length) {
            auto result = travel(old_cost, new_cost, length);
            const auto step = PyFloat_AsDouble(result.ptr());
            if (step == -1.0 && PyErr_Occurred()) throw py::error_already_set();
            result = py::object();
            check_views();
            return step;
        };
    }
    if (!examine.is_none()) {
        hooks.examine = [&](graph::Index node, graph::Index next, double length) {
            try { examine(node, next, length); }
            catch (py::error_already_set& error) { error.discard_as_unraisable(examine); }
            check_views();
        };
    }
    if (!update.is_none()) {
        hooks.update = [&](graph::Index node, graph::Index next, double length) {
            try { update(node, next, length); }
            catch (py::error_already_set& error) { error.discard_as_unraisable(update); }
            check_views();
        };
    }
    py::object numpy_index;
    if (!connect.is_none()) {
        numpy_index = numpy.attr("intp");
        hooks.connect = [&](graph::Index first, graph::Index second, graph::Index node,
                            graph::Index next, double cost, double next_cost) {
            auto connection_point = [&](graph::Index position) {
                const auto coordinates = grid.coordinates(position);
                py::tuple result(coordinates.size());
                for (std::size_t i = 0; i < coordinates.size(); ++i) result[i] = numpy_index(coordinates[i]);
                return result;
            };
            try { connect(first, second, connection_point(node), connection_point(next), cost, next_cost); }
            catch (py::error_already_set& error) { error.discard_as_unraisable(connect); }
            check_views();
        };
    }
    check_views();
    graph::find_costs(grid, cost_data, starts, ends, options,
                      distance_data, predecessor_data, hooks, &pending);
    return py::cast(std::move(pending));
}

py::list trace(const graph::Grid& grid, const Flat<double>& distance,
               const Flat<graph::Offset>& predecessor, graph::Index end) {
    check_flat(distance, grid.size());
    check_flat(predecessor, grid.size());
    if (end < 0 || end >= grid.size()) throw py::value_error("graph index out of bounds");
    return points(grid, graph::traceback(grid, distance.data(), predecessor.data(), end));
}

py::object shortest(const Flat<double>& costs, const graph::Shape& shape, int reach,
                    int axis, bool output_indexlist) {
    if (shape.size() < 2 || axis < 0 || static_cast<std::size_t>(axis) >= shape.size()
        || reach < 0 || reach > 127) return py::none();
    graph::Shape move(shape.size(), -reach);
    move[axis] = 1;
    graph::Offsets offsets;
    for (;;) {
        if (offsets.size() == 32767) return py::none();
        offsets.push_back(move);
        std::size_t i = 0;
        for (; i < shape.size(); ++i) {
            if (i == static_cast<std::size_t>(axis)) continue;
            if (move[i] < reach) {
                ++move[i];
                break;
            }
            move[i] = -reach;
        }
        if (i == shape.size()) break;
    }
    const auto count = offsets.size();
    const graph::Grid grid(shape, std::move(offsets), std::vector<double>(count, 1));
    check_flat(costs, grid.size());
    if (!supported_costs(costs.data(), grid.size())) return py::none();
    std::pair<graph::Shape, double> result;
    try {
        result = graph::shortest_path(grid, costs.data(), axis);
    } catch (const std::invalid_argument&) {
        return py::none();
    }
    if (output_indexlist) return py::make_tuple(points(grid, result.first), result.second);
    const auto dimensions = static_cast<py::ssize_t>(shape.size()) - 1;
    const py::ssize_t output_shape[]{static_cast<py::ssize_t>(result.first.size()), dimensions};
    auto path = skimagexx::bindings::empty_c_array<py::ssize_t>(output_shape);
    auto* out = path.mutable_data();
    for (const auto node : result.first) {
        auto coordinates = grid.coordinates(node);
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i != static_cast<std::size_t>(axis)) *out++ = coordinates[i];
        }
    }
    return py::make_tuple(path.attr("squeeze")(), result.second);
}

template<class T>
py::array_t<T> take_vector(std::vector<T>&& vector) {
    auto owner = std::make_unique<std::vector<T>>(std::move(vector));
    auto* values = owner.get();
    py::capsule capsule(values, [](void* ptr) { delete static_cast<std::vector<T>*>(ptr); });
    owner.release();
    return py::array_t<T>(values->size(), values->data(), capsule);
}

py::tuple pixel(const graph::Shape& shape, const py::object& mask, graph::Offsets offsets,
                std::vector<double> lengths) {
    const graph::Grid grid(shape, std::move(offsets), std::move(lengths), true);
    Flat<bool> mask_array;
    if (!mask.is_none()) {
        if (!Flat<bool>::check_(mask)) throw py::value_error("expected a contiguous boolean mask");
        mask_array = py::reinterpret_borrow<Flat<bool>>(mask);
        check_flat(mask_array, grid.size());
    }
    const auto* bytes = mask.is_none() ? nullptr : reinterpret_cast<const std::uint8_t*>(mask_array.data());
    auto edges = graph::pixel_edges(grid, bytes);
    return py::make_tuple(take_vector(std::move(edges.nodes)), take_vector(std::move(edges.indptr)),
                          take_vector(std::move(edges.indices)), take_vector(std::move(edges.distances)));
}

} // namespace

void bind_graph_paths(py::module_& module) {
    const auto numpy = py::module_::import("numpy");
    py::class_<graph::Grid>(module, "GraphPathsGrid")
        .def(py::init<graph::Shape, graph::Offsets, std::vector<double>, bool>())
        .def("solve", [numpy](const graph::Grid& grid, const Flat<double>& costs,
             const graph::Shape& starts, const graph::Shape& ends, int mode, bool all_ends,
             bool use_ends, std::ptrdiff_t max_visits, double max_step, Flat<double>& distance,
             Flat<graph::Offset>& predecessor, const py::tuple& callbacks, graph::Shape pending) {
                return solve(grid, costs, starts, ends, mode, all_ends, use_ends, max_visits,
                             max_step, distance, predecessor, callbacks, std::move(pending), numpy);
             }, py::arg("costs").noconvert(), py::arg("starts"), py::arg("ends"),
             py::arg("mode"), py::arg("all_ends"), py::arg("use_ends"), py::arg("max_visits"),
             py::arg("max_step"), py::arg("distance").noconvert(), py::arg("predecessor").noconvert(),
             py::arg("callbacks"), py::arg("pending"))
        .def("trace", &trace, py::arg("distance").noconvert(), py::arg("predecessor").noconvert(),
             py::arg("end"));
    module.def("graph_paths_offsets", &graph::lattice_offsets);
    module.def("graph_paths_check_storage", &skimagexx::bindings::check_array_storage,
               py::arg("array").noconvert());
    module.def("graph_paths_shortest", &shortest, py::arg("costs").noconvert(), py::arg("shape"),
               py::arg("reach"), py::arg("axis"), py::arg("output_indexlist"));
    module.def("graph_paths_pixel", &pixel);
}

#ifdef SKIMAGEXX_GRAPH_PATHS_PRIVATE_MODULE
PYBIND11_MODULE(_graph_paths_probe, module) {
    bind_graph_paths(module);
}
#endif
