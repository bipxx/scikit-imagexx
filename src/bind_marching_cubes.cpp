#include "skimagexx/marching_cubes.hpp"
#include "bind_array_storage.hpp"

#include <cstring>
#include <limits>
#include <numeric>

namespace py = pybind11;
namespace {

using Point = std::array<float, 3>;
using Face = std::array<std::int32_t, 3>;

template<class T, std::size_t N>
py::array_t<T> to_array(const std::vector<std::array<T, N>>& data) {
    static_assert(sizeof(std::array<T, N>) == sizeof(T) * N);
    py::array_t<T> output({static_cast<py::ssize_t>(data.size()), static_cast<py::ssize_t>(N)});
    if (!data.empty()) {
        std::memcpy(output.mutable_data(), data.data(), data.size() * sizeof(data[0]));
    }
    return output;
}

py::array_t<float> to_array(const std::vector<float>& data) {
    py::array_t<float> output(static_cast<py::ssize_t>(data.size()));
    if (!data.empty()) {
        std::memcpy(output.mutable_data(), data.data(), data.size() * sizeof(float));
    }
    return output;
}

py::tuple extract(const py::array& volume, double level, int step, bool classic, const py::object& mask) {
    if (volume.ndim() != 3 || !(volume.flags() & py::array::c_style)
        || !volume.dtype().is(py::dtype::of<float>())
        || reinterpret_cast<std::uintptr_t>(volume.data()) % alignof(float) || step < 1) {
        throw py::value_error("invalid native marching cubes volume or step");
    }
    skimagexx::bindings::check_array_storage(volume);
    const std::uint8_t* active = nullptr;
    if (!mask.is_none()) {
        const auto array = py::cast<py::array>(mask);
        if (array.ndim() != 3 || !(array.flags() & py::array::c_style) ||
            (!array.dtype().is(py::dtype::of<bool>()) && !array.dtype().is(py::dtype::of<std::uint8_t>()))) {
            throw py::value_error("invalid native marching cubes mask");
        }
        for (int i = 0; i < 3; ++i) {
            if (volume.shape(i) != array.shape(i)) throw py::value_error("invalid marching cubes mask shape");
        }
        skimagexx::bindings::check_array_storage(array);
        active = static_cast<const std::uint8_t*>(array.data());
    }
    const auto mesh = skimagexx::marching_cubes(static_cast<const float*>(volume.data()),
        {static_cast<std::size_t>(volume.shape(0)), static_cast<std::size_t>(volume.shape(1)),
         static_cast<std::size_t>(volume.shape(2))},
        level, static_cast<std::size_t>(step), classic, active);
    return py::make_tuple(to_array(mesh.vertices), to_array(mesh.faces), to_array(mesh.normals),
                          to_array(mesh.values));
}

// Replicate the representative selection and face order of the public API.
py::tuple remove_degenerate(const py::array_t<float, 0>& vertices,
                            const py::array_t<std::int32_t, 0>& faces,
                            const py::array_t<float, 0>& normals,
                            const py::array_t<float, 0>& values) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3 ||
        faces.ndim() != 2 || faces.shape(1) != 3 ||
        normals.ndim() != 2 || normals.shape(1) != 3 || normals.shape(0) != vertices.shape(0) ||
        values.ndim() != 1 || values.shape(0) != vertices.shape(0)) {
        throw py::value_error("invalid native marching cubes mesh");
    }
    for (const auto* array : {static_cast<const py::array*>(&vertices),
                              static_cast<const py::array*>(&faces),
                              static_cast<const py::array*>(&normals),
                              static_cast<const py::array*>(&values)}) {
        skimagexx::bindings::check_array_storage(*array);
        if (reinterpret_cast<std::uintptr_t>(array->data()) % alignof(float)) {
            throw py::value_error("unaligned marching cubes mesh");
        }
        for (py::ssize_t d = 0; d < array->ndim(); ++d) {
            if (array->strides(d) % static_cast<py::ssize_t>(alignof(float))) {
                throw py::value_error("unaligned marching cubes mesh stride");
            }
        }
    }
    const auto points = vertices.unchecked<2>();
    const auto triangles = faces.unchecked<2>();
    const auto gradients = normals.unchecked<2>();
    const auto strengths = values.unchecked<1>();
    const auto count = points.shape(0);
    std::vector<py::ssize_t> representative(static_cast<std::size_t>(count));
    std::iota(representative.begin(), representative.end(), 0);
    std::vector<Face> kept;
    kept.reserve(static_cast<std::size_t>(triangles.shape(0)));
    for (py::ssize_t i = 0; i < triangles.shape(0); ++i) {
        Face face{triangles(i, 0), triangles(i, 1), triangles(i, 2)};
        for (auto vertex : face) {
            if (vertex < 0 || vertex >= count) throw py::value_error("invalid marching cubes vertex index");
        }
        bool valid = true;
        for (int first = 0; first < 2; ++first) {
            const auto a = face[first];
            for (int second = first + 1; second < 3; ++second) {
                const auto b = face[second];
                if (points(a, 0) == points(b, 0) && points(a, 1) == points(b, 1) &&
                    points(a, 2) == points(b, 2)) {
                    representative[a] = representative[b] = std::min(representative[a], representative[b]);
                    valid = false;
                }
            }
        }
        if (valid) kept.push_back(face);
    }
    std::vector<py::ssize_t> compact(static_cast<std::size_t>(count));
    std::vector<Point> out_points, out_normals;
    std::vector<float> out_values;
    for (py::ssize_t i = 0; i < count; ++i) {
        if (representative[i] == i) {
            out_points.push_back({points(i, 0), points(i, 1), points(i, 2)});
            out_normals.push_back({gradients(i, 0), gradients(i, 1), gradients(i, 2)});
            out_values.push_back(strengths(i));
        }
        compact[i] = static_cast<py::ssize_t>(out_points.size()) - 1;
    }
    py::array_t<py::ssize_t> out_faces({static_cast<py::ssize_t>(kept.size()), py::ssize_t{3}});
    auto* destination = out_faces.mutable_data();
    for (const auto& face : kept) {
        for (auto a : face) *destination++ = compact[representative[a]];
    }
    return py::make_tuple(to_array(out_points), out_faces, to_array(out_normals), to_array(out_values));
}

}  // namespace

void bind_marching_cubes(py::module_& module) {
    module.def("measure_marching_cubes", &extract, py::arg("volume").noconvert(),
               py::arg("level"), py::arg("step"), py::arg("classic"), py::arg("mask"));
    module.def("measure_marching_cubes_remove_degenerate", &remove_degenerate,
               py::arg("vertices").noconvert(), py::arg("faces").noconvert(),
               py::arg("normals").noconvert(), py::arg("values").noconvert());
    module.def("measure_marching_cubes_check_storage", &skimagexx::bindings::check_array_storage,
               py::arg("array").noconvert());
}
