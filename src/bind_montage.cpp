#include <skimagexx/montage.hpp>
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstdint>
#include <limits>

namespace py = pybind11;

void bind_montage(py::module_& module) {
    module.def("montage_apply", [](const py::array& input, const py::array& fill,
               py::array output, const std::array<std::size_t, 3>& grid, int workers) {
        if (workers < 1 || workers > 1024)
            throw py::value_error("montage worker snapshot outside [1,1024]");
        if (input.ndim() != 4 || fill.ndim() != 1 || output.ndim() != 3
                || !output.owndata() || !output.writeable()
                || !(output.flags() & py::array::c_style)
                || !(fill.flags() & py::array::c_style)
                || !grid[0] || !grid[1]) return false;
        const auto dtype = input.dtype();
        const auto kind = dtype.kind();
        const auto bytes = std::size_t(input.itemsize());
        const bool pod = (kind == 'b' && bytes == 1)
            || ((kind == 'i' || kind == 'u') && (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8))
            || (kind == 'f' && (bytes == 2 || bytes == 4 || bytes == 8))
            || (kind == 'c' && (bytes == 8 || bytes == 16));
        if (!pod || !dtype.equal(fill.dtype()) || !dtype.equal(output.dtype())) return false;
        const auto images = std::size_t(input.shape(0));
        const auto rows = std::size_t(input.shape(1));
        const auto columns = std::size_t(input.shape(2));
        const auto channels = std::size_t(input.shape(3));
        const auto height = std::size_t(output.shape(0));
        const auto width = std::size_t(output.shape(1));
        if (std::size_t(fill.size()) < channels || std::size_t(output.shape(2)) != channels
                || grid[2] > height || grid[2] > width
                || (height-grid[2]) % grid[0] || (width-grid[2]) % grid[1]
                || (height-grid[2]) / grid[0] < grid[2]
                || (width-grid[2]) / grid[1] < grid[2]
                || (height-grid[2]) / grid[0] - grid[2] != rows
                || (width-grid[2]) / grid[1] - grid[2] != columns
                || (images && (images-1) / grid[1] >= grid[0])) return false;
        // All providers/callbacks precede this boundary. Validate the actual
        // three buffers once, then pass fixed byte descriptors without Python.
        skimagexx::bindings::check_array_storage(input);
        skimagexx::bindings::check_array_storage(fill);
        skimagexx::bindings::check_array_storage(output);
        skimagexx::detail::MontageInput descriptor{input.data(), {}, {}, bytes};
        std::size_t low = 0, high = 0;
        const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        for (int axis = 0; axis < 4; ++axis) {
            descriptor.shape[axis] = std::size_t(input.shape(axis));
            descriptor.strides[axis] = input.strides(axis);
            if (!input.size()) continue;
            const auto stride = input.strides(axis);
            const auto magnitude = stride < 0 ? std::size_t(-(stride+1))+1 : std::size_t(stride);
            auto& span = stride < 0 ? low : high;
            const auto count = descriptor.shape[axis]-1;
            if (count && magnitude > (limit-span) / count) return false;
            span += count*magnitude;
        }
        if (std::size_t(output.nbytes()) > limit || std::size_t(fill.nbytes()) > limit) return false;
        const auto source = reinterpret_cast<std::uintptr_t>(input.data());
        const auto target = reinterpret_cast<std::uintptr_t>(output.data());
        const auto pattern = reinterpret_cast<std::uintptr_t>(fill.data());
        const auto end = target + std::size_t(output.nbytes());
        if (output.size() && ((input.size() && target < source+high+bytes && source-low < end)
                || (fill.size() && target < pattern+fill.nbytes() && pattern < end))) return false;
        auto* destination = output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::detail::montage_into(descriptor, {grid[0],grid[1],grid[2]}, fill.data(),
                                          {destination,height,width}, workers);
        }
        return true;
    }, py::arg("input").noconvert(), py::arg("fill").noconvert(),
       py::arg("output").noconvert(), py::arg("grid"), py::arg("workers"));
}
