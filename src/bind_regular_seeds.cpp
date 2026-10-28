#include "regular_seeds.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <limits>

namespace py = pybind11;

void bind_regular_seeds(py::module_& module) {
    module.def("regular_seeds_apply", [](py::array output,
               const py::tuple& grid, int workers) {
        if (workers < 1 || workers > 1024)
            throw py::value_error("regular-seeds worker snapshot outside [1,1024]");
        if (output.ndim() > 32
                || grid.size() != std::size_t(output.ndim())
                || !(output.flags() & py::array::c_style) || !output.writeable()) return false;
        const auto dtype = output.dtype();
        const auto kind = dtype.kind();
        const auto bytes = std::size_t(output.itemsize());
        using skimagexx::detail::RegularSeedsType;
        RegularSeedsType type;
        if (kind == 'b' && bytes == 1) type = RegularSeedsType::boolean;
        else if (kind == 'i' || kind == 'u') {
            switch (bytes) {
                case 1: type = kind == 'i' ? RegularSeedsType::i8 : RegularSeedsType::u8; break;
                case 2: type = kind == 'i' ? RegularSeedsType::i16 : RegularSeedsType::u16; break;
                case 4: type = kind == 'i' ? RegularSeedsType::i32 : RegularSeedsType::u32; break;
                case 8: type = kind == 'i' ? RegularSeedsType::i64 : RegularSeedsType::u64; break;
                default: return false;
            }
        } else if (kind == 'f' && bytes == 4) type = RegularSeedsType::f32;
        else if (kind == 'f' && bytes == 8) type = RegularSeedsType::f64;
        else return false;
        if (!dtype.attr("isnative").cast<bool>()
                || reinterpret_cast<std::uintptr_t>(output.data()) % bytes) return false;
        skimagexx::bindings::check_array_storage(output);
        std::array<skimagexx::detail::RegularSeedsAxis, 32> axes;
        bool empty = false;
        for (py::ssize_t axis = 0; axis < output.ndim(); ++axis) {
            if (!py::isinstance<py::slice>(grid[axis]) || output.strides(axis) < 0
                    || std::size_t(output.strides(axis)) % bytes) return false;
            py::ssize_t start, stop, step, count;
            if (PySlice_GetIndicesEx(grid[axis].ptr(), output.shape(axis),
                    &start, &stop, &step, &count) < 0) throw py::error_already_set();
            if (step <= 0) return false;
            axes[axis] = {std::size_t(start), std::size_t(step), std::size_t(count),
                          std::size_t(output.strides(axis)) / bytes};
            empty = empty || count == 0;
        }
        const auto output_count = std::size_t(output.size());
        std::size_t selected = empty ? 0 : 1;
        if (!empty) for (py::ssize_t axis = 0; axis < output.ndim(); ++axis) {
            if (selected > output_count / axes[axis].count) return false;
            selected *= axes[axis].count;
        }
        if (selected > std::size_t(std::numeric_limits<std::int64_t>::max())
                || (type == RegularSeedsType::f32 && selected > (std::uint64_t(1) << 24))
                || (type == RegularSeedsType::f64 && selected > (std::uint64_t(1) << 53))) return false;
        const skimagexx::detail::RegularSeedsGrid descriptor{
            axes.data(), std::size_t(output.ndim()), selected, output_count};
        auto* destination = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::detail::regular_seeds_assign(descriptor, destination, type, workers);
        }
        return true;
    }, py::arg("output").noconvert(), py::arg("grid").noconvert(), py::arg("workers"));
}
