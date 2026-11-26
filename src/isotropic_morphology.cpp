#include "skimagexx/isotropic_morphology.hpp"
#include "skimagexx/expand_labels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

void sentinel_mask(std::size_t count, std::size_t rank, const std::size_t* shape,
                   const double* spacing, double radius, bool erosion, std::uint8_t* output) {
    // An EDT with no zero-valued site uses the external feature (-1, 0, ...).
    for (std::size_t index = 0; index < count; ++index) {
        std::size_t coordinate[3]{};
        auto offset = index;
        for (std::size_t axis = rank; axis-- > 0;) {
            coordinate[axis] = offset % shape[axis];
            offset /= shape[axis];
        }
        ++coordinate[0];
        double squared = 0;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const auto scaled = static_cast<double>(coordinate[axis]) * spacing[axis];
            squared += scaled * scaled;
        }
        output[index] = static_cast<std::uint8_t>((std::sqrt(squared) <= radius) != erosion);
    }
}

void threshold(const std::uint8_t* sites, std::size_t count, std::size_t rank,
               const std::size_t* shape, const double* spacing, double radius,
               bool erosion, std::uint8_t* output) {
    if (std::none_of(sites, sites + count, [](auto value) { return value != 0; })) {
        sentinel_mask(count, rank, shape, spacing, radius, erosion, output);
        return;
    }
    expand_labels_into(sites, count, rank, shape, spacing, radius, output, count);
    std::transform(output, output + count, output, [erosion](auto value) {
        return static_cast<std::uint8_t>((value != 0) != erosion);
    });
}

void apply(const std::uint8_t* image, std::size_t count, std::size_t rank,
           const std::size_t* shape, const double* spacing, double radius,
           bool erosion, std::uint8_t* output) {
    if (erosion) {
        std::vector<std::uint8_t> sites(count);
        std::transform(image, image + count, sites.begin(), [](auto value) { return value == 0; });
        threshold(sites.data(), count, rank, shape, spacing, radius, true, output);
    } else {
        threshold(image, count, rank, shape, spacing, radius, false, output);
    }
}

}  // namespace

void isotropic_morphology_into(const std::uint8_t* image, std::size_t image_count,
                               std::size_t rank, const std::size_t* shape,
                               const double* spacing, double radius, IsotropicOperation operation,
                               std::uint8_t* output, std::size_t output_count) {
    using Op = IsotropicOperation;
    if (!image || !output || !shape || !spacing || rank < 1 || rank > 3 ||
        (operation != Op::erosion && operation != Op::dilation &&
         operation != Op::opening && operation != Op::closing))
        throw std::invalid_argument("isotropic morphology requires valid buffers, rank and operation");
    constexpr std::size_t max_count = 64 * 1024 * 1024;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto length = shape[axis];
        if (length == 0 || length > 65536 || count > max_count / length ||
            !(spacing[axis] >= 1.0 / 65536.0 && spacing[axis] <= 65536.0))
            throw std::invalid_argument("isotropic morphology shape or spacing is outside the native domain");
        count *= length;
    }
    if (image_count < count || output_count < count)
        throw std::invalid_argument("isotropic morphology buffers have insufficient extent");

    const bool final_erosion = operation == Op::erosion || operation == Op::closing;
    if (std::isnan(radius)) {
        std::fill_n(output, count, std::uint8_t{0});
    } else if (radius < 0 || std::isinf(radius)) {
        std::fill_n(output, count, static_cast<std::uint8_t>((radius < 0) == final_erosion));
    } else if (radius == 0) {
        std::transform(image, image + count, output, [](auto value) { return value != 0; });
    } else if (operation == Op::opening || operation == Op::closing) {
        std::vector<std::uint8_t> intermediate(count);
        apply(image, count, rank, shape, spacing, radius, !final_erosion, intermediate.data());
        apply(intermediate.data(), count, rank, shape, spacing, radius, final_erosion, output);
    } else {
        apply(image, count, rank, shape, spacing, radius, final_erosion, output);
    }
}

}  // namespace skimagexx
