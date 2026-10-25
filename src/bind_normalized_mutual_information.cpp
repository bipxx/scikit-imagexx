#include "skimagexx/normalized_mutual_information.hpp"
#include "bind_numeric.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace py = pybind11;
namespace {
// Use the ndarray C base link, never an overridable Python .base property.
// Keep owners/export leases alive across the released-GIL numerical region.
struct Intake {
    struct Lease {
        Py_buffer value{};
        ~Lease() { if (value.obj) PyBuffer_Release(&value); }
    } lease;
    struct Stamp {
        py::array array;
        void* data;
        PyObject* base;
        PyObject* dtype;
        int flags;
        std::vector<py::ssize_t> shape, strides;
    };
    std::vector<Stamp> stamps;
    std::vector<py::object> owners;
    std::uintptr_t allocation = 0;
    std::size_t allocation_bytes = 0;

    Intake(const py::array& image, const py::object& mmap_type) {
        py::object current = image;
        for (unsigned depth = 0; depth < 64; ++depth) {
            if (std::any_of(owners.begin(), owners.end(), [&](const auto& owner) { return owner.ptr() == current.ptr(); }))
                throw py::value_error("NMI array owner cycle");
            owners.push_back(current);
            if (!py::isinstance<py::array>(current)) {
                const bool trusted = PyBytes_CheckExact(current.ptr()) || PyByteArray_CheckExact(current.ptr()) ||
                    PyMemoryView_Check(current.ptr()) || Py_TYPE(current.ptr()) == reinterpret_cast<PyTypeObject*>(mmap_type.ptr());
                if (!trusted) throw py::value_error("NMI requires bounded ndarray or builtin-buffer storage");
                if (PyObject_GetBuffer(current.ptr(), &lease.value, PyBUF_SIMPLE) < 0) throw py::error_already_set();
                if (lease.value.len < 0) throw py::value_error("NMI invalid buffer extent");
                allocation = reinterpret_cast<std::uintptr_t>(lease.value.buf);
                allocation_bytes = std::size_t(lease.value.len);
                validate(); return;
            }
            auto array = py::reinterpret_borrow<py::array>(current);
            const auto* raw = py::detail::array_proxy(array.ptr());
            Stamp stamp{array, raw->data, raw->base, reinterpret_cast<PyObject*>(raw->descr), raw->flags, {}, {}};
            for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
                stamp.shape.push_back(array.shape(axis)); stamp.strides.push_back(array.strides(axis));
            }
            stamps.push_back(std::move(stamp));
            if (raw->base) { current = py::reinterpret_borrow<py::object>(raw->base); continue; }
            if (!(array.flags() & py::detail::npy_api::NPY_ARRAY_OWNDATA_) ||
                    !(array.flags() & (py::array::c_style | py::array::f_style)))
                throw py::value_error("NMI requires an owned contiguous root allocation");
            if (array.nbytes() < 0) throw py::value_error("NMI invalid root extent");
            allocation = reinterpret_cast<std::uintptr_t>(raw->data);
            allocation_bytes = std::size_t(array.nbytes());
            validate(); return;
        }
        throw py::value_error("NMI array owner chain exceeds bound");
    }

    void validate() const {
        for (const auto& stamp : stamps) {
            const auto* raw = py::detail::array_proxy(stamp.array.ptr());
            if (raw->data != stamp.data || raw->base != stamp.base || reinterpret_cast<PyObject*>(raw->descr) != stamp.dtype ||
                    raw->flags != stamp.flags || stamp.array.ndim() != py::ssize_t(stamp.shape.size()))
                throw py::value_error("NMI array metadata changed during admission");
            for (py::ssize_t axis = 0; axis < stamp.array.ndim(); ++axis)
                if (stamp.array.shape(axis) != stamp.shape[std::size_t(axis)] || stamp.array.strides(axis) != stamp.strides[std::size_t(axis)])
                    throw py::value_error("NMI array metadata changed during admission");
            if (stamp.array.size() == 0) continue;
            std::size_t negative = 0, positive = 0;
            const auto limit = std::numeric_limits<std::uintptr_t>::max();
            for (std::size_t axis = 0; axis < stamp.shape.size(); ++axis) {
                if (stamp.shape[axis] < 0) throw py::value_error("NMI negative array extent");
                const auto stride = stamp.strides[axis];
                const auto step = stride < 0 ? std::size_t(-(stride + 1)) + 1 : std::size_t(stride);
                const auto count = std::size_t(stamp.shape[axis] - 1);
                auto& extent = stride < 0 ? negative : positive;
                if (count && step > (limit - extent) / count) throw py::value_error("NMI byte extent overflows");
                extent += count * step;
            }
            const auto pointer = reinterpret_cast<std::uintptr_t>(stamp.data);
            const auto itemsize = std::size_t(stamp.array.itemsize());
            if (negative > pointer || positive > limit - pointer || itemsize > limit - pointer - positive ||
                    allocation_bytes > limit - allocation || pointer - negative < allocation ||
                    pointer + positive + itemsize > allocation + allocation_bytes)
                throw py::value_error("NMI array byte extent escapes its current owner allocation");
        }
    }
};

skimagexx::MetricImageView view(const py::array& image) {
    skimagexx::MetricImageView result; result.data = image.data();
    result.half = image.dtype().kind() == 'f' && image.itemsize() == 2;
    result.type = result.half ? skimagexx::PixelType::uint16 : skimagexx::bindings::real_pixel_type(image.dtype());
    result.swap = !image.dtype().attr("isnative").cast<bool>() && image.itemsize() > 1;
    for (py::ssize_t axis = 0; axis < image.ndim(); ++axis) {
        result.shape.push_back(std::size_t(image.shape(axis)));
        result.strides.push_back(std::ptrdiff_t(image.strides(axis)));
    }
    return result;
}
}
void bind_normalized_mutual_information(py::module_& module) {
    const py::object mmap_type = py::module_::import("mmap").attr("mmap");
    module.def("nmi_validate_inputs", [mmap_type](py::array a, py::array b, const std::vector<double>&) {
        Intake first(a, mmap_type), second(b, mmap_type);
        first.validate(); second.validate(); return true;
    }, py::arg("image0").noconvert(), py::arg("image1").noconvert(), py::arg("metadata") = std::vector<double>{});
    module.def("nmi_prepare", [mmap_type](py::array a, py::array b) {
        Intake first_owner(a, mmap_type), second_owner(b, mmap_type);
        const auto first = view(a), second = view(b);
        skimagexx::NmiPreparation result;
        first_owner.validate(); second_owner.validate();
        { py::gil_scoped_release release; result = skimagexx::nmi_prepare(first, second); }
        return py::make_tuple(result.admitted, result.samples, result.bounds);
    }, py::arg("image0").noconvert(), py::arg("image1").noconvert());
    module.def("nmi_statistics", [mmap_type](py::array a, py::array b,
            const std::vector<double>& edges0, const std::vector<double>& edges1,
            const std::vector<double>& widths0, const std::vector<double>& widths1, bool diagnostics) -> py::object {
        Intake first_owner(a, mmap_type), second_owner(b, mmap_type);
        const auto first = view(a), second = view(b);
        skimagexx::NmiStatistics result;
        first_owner.validate(); second_owner.validate();
        { py::gil_scoped_release release;
          result = skimagexx::nmi_statistics(first, second, edges0, edges1, widths0, widths1); }
        if (!diagnostics) return result.admitted ? py::cast(result.entropy) : py::none();
        py::dict output;
        output["admitted"] = result.admitted; output["entropy"] = result.entropy;
        py::array_t<std::uint64_t> counts({py::ssize_t(result.bins[0]), py::ssize_t(result.bins[1])});
        if (!result.counts.empty()) std::memcpy(counts.mutable_data(), result.counts.data(), result.counts.size() * sizeof(std::uint64_t));
        else std::memset(counts.mutable_data(), 0, std::size_t(counts.nbytes()));
        output["counts"] = counts;
        return output;
    }, py::arg("image0").noconvert(), py::arg("image1").noconvert(), py::arg("edges0"), py::arg("edges1"),
       py::arg("widths0"), py::arg("widths1"), py::arg("diagnostics") = false);
}
