#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx::bindings {
namespace py = pybind11;

// Metadata-only construction and validation. Unknown allocation owners remain
// an explicit provider boundary; a known owner with escaping bytes is rejected.
// Strong references and direct ndarray C metadata never call Python .base.
class LiU16Intake {
    struct Stamp {
        py::array array;
        void* data;
        PyObject* base;
        PyObject* dtype;
        int flags;
        py::ssize_t itemsize;
        std::vector<py::ssize_t> shape, strides;
    };
    std::vector<py::object> owners_;
    std::vector<Stamp> stamps_;
    std::uintptr_t allocation_ = 0;
    std::size_t allocation_bytes_ = 0, count_ = 0;
    bool supported_ = false, swap_ = false;

    static std::size_t count(const Stamp& stamp) {
        for (auto length : stamp.shape)
            if (length < 0) throw py::value_error("Li uint16 negative array dimension");
        if (std::find(stamp.shape.begin(), stamp.shape.end(), 0) != stamp.shape.end()) return 0;
        std::size_t total = 1;
        for (auto length : stamp.shape) {
            if (std::size_t(length) > std::numeric_limits<std::size_t>::max() / total)
                throw py::value_error("Li uint16 array dimensions overflow");
            total *= std::size_t(length);
        }
        return total;
    }

public:
    static constexpr std::size_t maximum_samples = 2147483647;

    explicit LiU16Intake(const py::array& image) {
        if (image.dtype().kind() != 'u' || image.itemsize() != 2)
            throw py::type_error("Li uint16 intake requires uint16 storage");
        const char order = image.dtype().byteorder();
        const std::uint16_t native = 1;
        const bool little = *reinterpret_cast<const unsigned char*>(&native) != 0;
        swap_ = order == '>' ? little : order == '<' ? !little : false;
        py::object current = image;
        for (unsigned depth = 0; depth < 64; ++depth) {
            if (std::any_of(owners_.begin(), owners_.end(), [&](const auto& owner) { return owner.ptr() == current.ptr(); }))
                throw py::value_error("Li uint16 array owner cycle");
            owners_.push_back(current);
            if (!py::isinstance<py::array>(current)) return;
            auto array = py::reinterpret_borrow<py::array>(current);
            const auto* raw = py::detail::array_proxy(array.ptr());
            Stamp stamp{array, raw->data, raw->base, reinterpret_cast<PyObject*>(raw->descr), raw->flags,
                        array.itemsize(), {}, {}};
            for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
                stamp.shape.push_back(array.shape(axis));
                stamp.strides.push_back(array.strides(axis));
            }
            stamps_.push_back(std::move(stamp));
            if (raw->base) { current = py::reinterpret_borrow<py::object>(raw->base); continue; }
            if (!(raw->flags & py::detail::npy_api::NPY_ARRAY_OWNDATA_) ||
                !(raw->flags & (py::array::c_style | py::array::f_style))) return;
            const auto root_count = count(stamps_.back());
            if (array.itemsize() < 0 || (root_count && std::size_t(array.itemsize()) > std::numeric_limits<std::size_t>::max() / root_count))
                throw py::value_error("Li uint16 root allocation size overflows");
            allocation_ = reinterpret_cast<std::uintptr_t>(raw->data);
            allocation_bytes_ = root_count * std::size_t(array.itemsize());
            count_ = count(stamps_.front());
            supported_ = true;
            validate();
            if (count_ > maximum_samples) supported_ = false;
            return;
        }
        throw py::value_error("Li uint16 array owner chain exceeds bound");
    }

    bool supported() const noexcept { return supported_; }
    std::size_t size() const noexcept { return count_; }

    void validate() const {
        if (!supported_) return;
        const auto limit = std::numeric_limits<std::uintptr_t>::max();
        if (allocation_bytes_ && !allocation_)
            throw py::value_error("Li uint16 nonempty root has null storage");
        if (allocation_bytes_ > limit - allocation_)
            throw py::value_error("Li uint16 root allocation address overflows");
        for (const auto& stamp : stamps_) {
            const auto* raw = py::detail::array_proxy(stamp.array.ptr());
            if (raw->data != stamp.data || raw->base != stamp.base || reinterpret_cast<PyObject*>(raw->descr) != stamp.dtype ||
                raw->flags != stamp.flags || stamp.array.itemsize() != stamp.itemsize || stamp.array.ndim() != py::ssize_t(stamp.shape.size()))
                throw py::value_error("Li uint16 owner metadata changed during admission");
            for (py::ssize_t axis = 0; axis < stamp.array.ndim(); ++axis)
                if (stamp.array.shape(axis) != stamp.shape[std::size_t(axis)] || stamp.array.strides(axis) != stamp.strides[std::size_t(axis)])
                    throw py::value_error("Li uint16 owner metadata changed during admission");
            if (!count(stamp)) continue;
            std::uintptr_t negative = 0, positive = 0;
            for (std::size_t axis = 0; axis < stamp.shape.size(); ++axis) {
                const auto stride = stamp.strides[axis];
                const auto magnitude = stride < 0 ? std::uintptr_t(-(stride + 1)) + 1 : std::uintptr_t(stride);
                const auto length = std::uintptr_t(stamp.shape[axis] - 1);
                auto& extent = stride < 0 ? negative : positive;
                if (length && magnitude > (limit - extent) / length)
                    throw py::value_error("Li uint16 byte extent overflows");
                extent += length * magnitude;
            }
            if (stamp.itemsize < 0) throw py::value_error("Li uint16 negative element size");
            const auto pointer = reinterpret_cast<std::uintptr_t>(stamp.data);
            const auto itemsize = std::uintptr_t(stamp.itemsize);
            if (negative > pointer || positive > limit - pointer || itemsize > limit - pointer - positive ||
                pointer - negative < allocation_ || pointer + positive + itemsize > allocation_ + allocation_bytes_)
                throw py::value_error("Li uint16 view escapes its current owner allocation");
        }
    }

    // Call only after validate(), retaining this object across released-GIL work.
    // All indexing uses stored metadata; no Python API or conversion provider.
    std::vector<std::uint16_t> copy_logical() const {
        if (!supported_) throw std::logic_error("unsupported Li uint16 owner");
        std::vector<std::uint16_t> output(count_);
        const auto& view = stamps_.front();
        const auto origin = reinterpret_cast<std::uintptr_t>(view.data);
        for (std::size_t index = 0; index < count_; ++index) {
            auto remaining = index;
            std::uintptr_t positive = 0, negative = 0;
            for (std::size_t axis = view.shape.size(); axis-- > 0;) {
                const auto coordinate = remaining % std::size_t(view.shape[axis]);
                remaining /= std::size_t(view.shape[axis]);
                const auto stride = view.strides[axis];
                if (stride < 0) negative += coordinate * (std::uintptr_t(-(stride + 1)) + 1);
                else positive += coordinate * std::uintptr_t(stride);
            }
            std::uint16_t value;
            std::memcpy(&value, reinterpret_cast<const void*>(origin + positive - negative), sizeof(value));
            output[index] = swap_ ? std::uint16_t((value >> 8) | (value << 8)) : value;
        }
        return output;
    }
};
} // namespace skimagexx::bindings
