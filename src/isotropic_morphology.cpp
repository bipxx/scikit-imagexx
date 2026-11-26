#include "skimagexx/isotropic_morphology.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

constexpr double absent = std::numeric_limits<double>::infinity();

double squared_cutoff(double radius) {
    if (radius < 1.0 / 65536.0)
        return 0.0;  // Every positive distance on the native grid is larger.
    if (radius >= 0x1p33)
        return 0x1p66;  // Larger than any squared distance on the native grid.

    // A rounded sqrt can equal radius on either side of radius * radius.
    // Correct the cutoff once so exact boundary hits keep the EDT semantics.
    auto cutoff = radius * radius;
    while (std::sqrt(cutoff) > radius)
        cutoff = std::nextafter(cutoff, 0.0);
    auto next = std::nextafter(cutoff, absent);
    while (std::sqrt(next) <= radius) {
        cutoff = next;
        next = std::nextafter(cutoff, absent);
    }
    return cutoff;
}

double squared_offset(int delta, double spacing) {
    const double scaled = double(delta) * spacing;
    return scaled * scaled;
}

void sentinel_mask(std::size_t count, std::size_t rank, const std::size_t* shape,
                   const double* spacing, double cutoff, bool erosion, std::uint8_t* output) {
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
        output[index] = static_cast<std::uint8_t>((squared <= cutoff) != erosion);
    }
}

template<bool Erosion>
bool first_axis(const std::uint8_t* image, std::size_t count, int length, double spacing,
                double* distances) {
    const auto stride = count / length;
    std::vector<int> nearest(stride, -1);
    for (int pos = 0; pos < length; ++pos) {
        const auto start = pos * stride;
        for (std::size_t j = 0; j < stride; ++j) {
            if ((image[start + j] != 0) != Erosion)
                nearest[j] = pos;
            distances[start + j] = nearest[j] < 0 ? absent : double(pos - nearest[j]);
        }
    }
    if (std::none_of(nearest.begin(), nearest.end(), [](int site) { return site >= 0; }))
        return false;

    // The last forward site is also the first site of the backward sweep.
    for (int pos = length - 1; pos >= 0; --pos) {
        const auto start = pos * stride;
        for (std::size_t j = 0; j < stride; ++j) {
            if ((image[start + j] != 0) != Erosion)
                nearest[j] = pos;
            auto& distance = distances[start + j];
            const int upper = nearest[j] - pos;
            if (upper >= 0 && upper < distance)
                distance = squared_offset(upper, spacing);
            else if (distance != absent)
                distance = squared_offset(-static_cast<int>(distance), spacing);
        }
    }
    return true;
}

struct Site {
    int position;
    double orthogonal;
};

// Use the same axis order and parabola comparisons as the feature transform
// in expand_labels. The mask needs only the distance, so discard coordinates
// from earlier axes and write the threshold during the last sweep.
template<bool Final>
void other_axis(double* distances, std::size_t count, int length, std::size_t stride,
                double spacing, double cutoff, bool erosion, std::uint8_t* output) {
    std::vector<Site> envelope;
    envelope.reserve(length);
    const auto span = length * stride;
    for (std::size_t block = 0; block < count; block += span) {
        for (std::size_t offset = 0; offset < stride; ++offset) {
            auto* line = distances + block + offset;
            envelope.clear();
            for (int pos = 0; pos < length; ++pos) {
                const auto orthogonal = line[pos * stride];
                if (orthogonal == absent)
                    continue;
                while (envelope.size() >= 2) {
                    const auto& a = envelope[envelope.size() - 2];
                    const auto& b = envelope.back();
                    const double left = double(b.position - a.position) * spacing;
                    const double right = double(pos - b.position) * spacing;
                    const double width = left + right;
                    const double gap = width * b.orthogonal - right * a.orthogonal - left * orthogonal;
                    if (gap - left * right * width <= 0.0)
                        break;
                    envelope.pop_back();
                }
                envelope.push_back({pos, orthogonal});
            }
            if (envelope.empty()) {
                if constexpr (Final)
                    std::fill_n(output + block + offset, length, static_cast<std::uint8_t>(erosion));
                continue;
            }

            std::size_t nearest = 0;
            for (int pos = 0; pos < length; ++pos) {
                const auto squared_distance = [&](const Site& site) {
                    return site.orthogonal + squared_offset(site.position - pos, spacing);
                };
                auto best = squared_distance(envelope[nearest]);
                while (nearest + 1 < envelope.size()) {
                    const auto next = squared_distance(envelope[nearest + 1]);
                    if (best <= next)
                        break;
                    best = next;
                    ++nearest;
                }
                const auto index = pos * stride;
                if constexpr (Final)
                    output[block + offset + index] = static_cast<std::uint8_t>((best <= cutoff) != erosion);
                else
                    line[index] = best;
            }
        }
    }
}

void apply(const std::uint8_t* image, std::size_t count, std::size_t rank,
           const std::size_t* shape, const double* spacing, double cutoff,
           bool erosion, double* distances, std::uint8_t* output) {
    const auto length = static_cast<int>(shape[0]);
    const bool any_site = erosion ? first_axis<true>(image, count, length, spacing[0], distances)
                                 : first_axis<false>(image, count, length, spacing[0], distances);
    if (!any_site) {
        sentinel_mask(count, rank, shape, spacing, cutoff, erosion, output);
    } else if (rank == 1) {
        std::transform(distances, distances + count, output, [=](double value) {
            return static_cast<std::uint8_t>((value <= cutoff) != erosion);
        });
    } else {
        if (rank == 3)
            other_axis<false>(distances, count, static_cast<int>(shape[1]), shape[2],
                              spacing[1], cutoff, erosion, output);
        other_axis<true>(distances, count, static_cast<int>(shape[rank - 1]), 1,
                         spacing[rank - 1], cutoff, erosion, output);
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
    } else {
        const auto cutoff = squared_cutoff(radius);
        std::vector<double> distances(count);
        std::vector<std::uint8_t> intermediate;
        if (operation == Op::opening || operation == Op::closing) {
            intermediate.resize(count);
            apply(image, count, rank, shape, spacing, cutoff, !final_erosion, distances.data(), intermediate.data());
            image = intermediate.data();
        }
        apply(image, count, rank, shape, spacing, cutoff, final_erosion, distances.data(), output);
    }
}

}  // namespace skimagexx
