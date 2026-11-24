#include <skimagexx/corner_moravec.hpp>

#include <algorithm>
#include <cfenv>
#include <limits>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace skimagexx {

bool corner_moravec_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, int window, double* response) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    if (!image || !response || !corner_moravec_shape_supported(rows, columns, window)
        || std::fegetround() != FE_TONEAREST) {
        return false;
    }

    constexpr double reciprocal = 1.0 / 255.0;
    std::vector<double> pixels(rows * columns);
    std::transform(image, image + pixels.size(), pixels.begin(),
        [reciprocal](std::uint8_t pixel) { return double(pixel) * reciprocal; });
    std::fill_n(response, pixels.size(), 0.0);

    const int height = static_cast<int>(rows);
    const int width = static_cast<int>(columns);
    const int border = 2 * window;
    const int diameter = 2 * window + 1;
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            const double* patch = pixels.data() + (y - window) * width + x - window;
            double best = std::numeric_limits<double>::max();
            for (int dy = -window; dy <= window; ++dy) {
                if (dy == 0) {
                    continue;
                }
                for (int dx = -window; dx <= window; ++dx) {
                    if (dx == 0) {
                        continue;
                    }
                    const double* shifted = patch + dy * width + dx;
                    double score = 0.0;
                    // Round subtraction, multiplication and addition separately.
                    // The source adds every patch word in logical row-major order.
                    for (int row = 0; row < diameter; ++row) {
                        const int start = row * width;
                        for (int col = 0; col < diameter; ++col) {
                            const int pixel = start + col;
                            const double difference = patch[pixel] - shifted[pixel];
                            const double square = difference * difference;
                            score += square;
                        }
                    }
                    best = std::min(best, score);
                }
            }
            response[y * width + x] = best;
        }
    }
    return true;
}

}  // namespace skimagexx

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
