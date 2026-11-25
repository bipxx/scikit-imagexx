#include "skimagexx/io_suite.hpp"

#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx::io {
namespace {

void copy_rows(const char* source, char*& output, const std::ptrdiff_t* shape,
               const std::ptrdiff_t* strides, std::size_t axes, std::size_t block) {
    if (axes == 0) {
        std::memcpy(output, source, block);
        output += block;
        return;
    }
    for (std::ptrdiff_t i = 0; i < *shape; ++i) {
        copy_rows(source + i * *strides, output, shape + 1, strides + 1, axes - 1, block);
    }
}

constexpr bool whitespace(char c) {
    return c == ' ' || (c >= '\t' && c <= '\r');
}

void skip_space(std::string_view& text) {
    while (!text.empty() && whitespace(text.front())) text.remove_prefix(1);
}

bool take_line(std::string_view& text, std::string_view& line) {
    const auto end = text.find('\n');
    if (end == text.npos) return false;
    line = text.substr(0, end);
    text.remove_prefix(end + 1);
    return true;
}

bool take_size(std::string_view& text, std::size_t& value, std::ptrdiff_t max_digits = 20) {
    skip_space(text);
    if (text.empty()) return false;
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr - text.data() > max_digits
        || (parsed.ptr != end && !whitespace(*parsed.ptr))) {
        return false;
    }
    text.remove_prefix(static_cast<std::size_t>(parsed.ptr - text.data()));
    return true;
}

}  // namespace

void copy_image(const void* source, void* output, const std::ptrdiff_t* shape,
                const std::ptrdiff_t* strides, std::size_t rank, std::size_t item_size) {
    if (item_size == 0) return;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] == 0) return;
    }
    auto block = item_size;
    while (rank && (shape[rank - 1] == 1 ||
                    strides[rank - 1] == static_cast<std::ptrdiff_t>(block))) {
        block *= static_cast<std::size_t>(shape[--rank]);
    }
    auto target = static_cast<char*>(output);
    copy_rows(static_cast<const char*>(source), target, shape, strides, rank, block);
}

std::optional<NetpbmImage> netpbm_view(std::string_view bytes) {
    if (bytes.size() < 3 || bytes[0] != 'P' || !whitespace(bytes[2])) return {};
    const std::size_t channels = bytes[1] == '6' ? 3 : bytes[1] == '5' ? 1 : 0;
    if (channels == 0) return {};
    bytes.remove_prefix(3);
    std::size_t width, height, maximum;
    if (!take_size(bytes, width, 10) || !take_size(bytes, height, 10) || !take_size(bytes, maximum, 10)
        || maximum != 255 || width == 0 || height == 0 || bytes.empty()) {
        return {};
    }
    // Pillow stops at the first separator after maxval. Raster bytes can be whitespace.
    bytes.remove_prefix(1);
    if (height > bytes.size() / channels / width || width * height * channels != bytes.size()) {
        return {};
    }
    return NetpbmImage{width, height, channels, bytes};
}

std::string netpbm_header(std::size_t width, std::size_t height, std::size_t channels) {
    if (!width || !height || (channels != 1 && channels != 3)) {
        throw std::invalid_argument("Expected a nonempty grayscale or RGB image");
    }
    return (channels == 3 ? "P6\n" : "P5\n") + std::to_string(width)
           + ' ' + std::to_string(height) + "\n255\n";
}

std::optional<FeatureLayout> feature_layout(std::string_view text, FeatureFormat format) {
    std::string_view line;
    std::size_t count, width;
    if (!take_line(text, line)) return {};
    if (format == FeatureFormat::sift) {
        if (!take_size(line, count) || !take_size(line, width)) return {};
    } else {
        if (!take_size(line, width) || width == 0) return {};
        --width;
        skip_space(line);
        if (!line.empty() || !take_line(text, line) || !take_size(line, count)) return {};
    }
    skip_space(line);
    const std::size_t fields = format == FeatureFormat::sift ? 4 : 6;
    if (!line.empty() || width > std::numeric_limits<std::size_t>::max() - fields) return {};
    const auto stride = width + fields;
    // Even single-digit values need at least one byte each, before separators.
    if (count > text.size() / stride) return {};
    return FeatureLayout{count, width, stride, text};
}

bool read_feature_values(std::string_view text, double* output, std::size_t size) {
    const auto* next = text.data();
    const auto* end = text.empty() ? next : next + text.size();
    for (std::size_t i = 0; i < size; ++i) {
        while (next != end && whitespace(*next)) ++next;
        if (next == end) return false;
        if (*next == '+') {
            ++next;
            if (next == end || *next == '-') return false;
        }
        // Descriptor bins are commonly short integers. Keep the decimal parser for real values.
        const auto* start = next;
        unsigned integer = 0;
        while (next != end && *next >= '0' && *next <= '9' && next - start < 9) {
            integer = integer * 10u + static_cast<unsigned>(*next++ - '0');
        }
        if (next != start && (next == end || whitespace(*next))) {
            output[i] = integer;
            continue;
        }
        const auto first_digit = *start == '-' ? start + 1 : start;
        if (first_digit == end || (*first_digit != '.' && (*first_digit < '0' || *first_digit > '9'))) {
            return false;
        }
        const auto parsed = std::from_chars(start, end, output[i]);
        if (parsed.ec != std::errc{} || (parsed.ptr != end && !whitespace(*parsed.ptr))) {
            return false;
        }
        next = parsed.ptr;
    }
    while (next != end && whitespace(*next)) ++next;
    return next == end;
}

}  // namespace skimagexx::io
