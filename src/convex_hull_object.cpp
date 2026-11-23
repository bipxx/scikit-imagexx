#include "skimagexx/convex_hull_object.hpp"

#include "skimagexx/components.hpp"
#include "skimagexx/convex_hull_image.hpp"

#include <algorithm>
#include <vector>

namespace skimagexx {
namespace {

struct Bounds {
    std::size_t top;
    std::size_t left;
    std::size_t bottom = 0;
    std::size_t right = 0;

    std::size_t height() const { return bottom - top + 1; }
    std::size_t width() const { return right - left + 1; }
};

} // namespace

bool convex_hull_object_into(const std::uint8_t* image, std::size_t height,
                             std::size_t width, int connectivity, std::uint8_t* output) {
    if (!image || !output || height == 0 || width == 0 ||
        height > convex_hull_image_max_extent || width > convex_hull_image_max_extent ||
        (connectivity != 1 && connectivity != 2))
        return false;

    const auto area = height * width;
    if (std::any_of(image, image + area, [](std::uint8_t pixel) { return pixel > 1; }))
        return false;

    std::vector<std::int32_t> labels(area);
    const auto count = components_label_bool_into(
        image, labels.data(), false, {height, width}, 0, connectivity);
    std::vector<Bounds> bounds(count, {height, width});
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            const auto id = labels[row * width + col];
            if (id == 0) continue;
            auto& box = bounds[static_cast<std::size_t>(id - 1)];
            box.top = std::min(box.top, row);
            box.left = std::min(box.left, col);
            box.bottom = row;
            box.right = std::max(box.right, col);
        }
    }

    std::size_t cropped_area = 0;
    for (const auto& box : bounds) {
        cropped_area += box.height() * box.width();
        if (cropped_area > 8 * area) return false;
    }

    // Every foreground pixel belongs to its hull, including isolated pixels.
    std::copy_n(image, area, output);
    std::vector<std::uint8_t> component, hull;
    for (std::size_t index = 0; index < bounds.size(); ++index) {
        const auto& box = bounds[index];
        const auto h = box.height(), w = box.width();
        if (h * w == 1) continue;
        component.resize(h * w);
        hull.resize(h * w);
        const auto id = static_cast<std::int32_t>(index + 1);
        for (std::size_t row = 0; row < h; ++row) {
            const auto origin = (box.top + row) * width + box.left;
            for (std::size_t col = 0; col < w; ++col)
                component[row * w + col] = labels[origin + col] == id;
        }
        // The half-pixel hull cannot reach an integer pixel outside these
        // bounds. The admitted image kernel already uses the same translation.
        if (!convex_hull_image_into(component.data(), h, w, true, true, hull.data()))
            return false;
        for (std::size_t row = 0; row < h; ++row) {
            const auto origin = (box.top + row) * width + box.left;
            for (std::size_t col = 0; col < w; ++col)
                output[origin + col] |= hull[row * w + col];
        }
    }
    return true;
}

} // namespace skimagexx
