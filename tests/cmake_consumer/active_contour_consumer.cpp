#include <skimagexx/active_contour.hpp>

#include <array>
#include <cmath>
#include <vector>

int main() {
    std::array<double, 12> image{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            const auto y = static_cast<double>(row);
            const auto x = static_cast<double>(column);
            image[4 * row + column] = 0.02 * y * y + 0.01 * x * x;
        }
    }
    const std::array<double, 10> snake{0.2, 0.9, 0.8, 1.7, 1.1, 2.3, 1.5, 2.7, 1.9, 2.9};
    skimagexx::ActiveContourOptions options;
    options.alpha = 0;
    options.beta = 0;
    options.gamma = 0.5;
    options.w_line = 1;
    options.w_edge = 0;
    options.max_num_iter = 1;
    std::vector<double> output;
    if (!skimagexx::active_contour(image.data(), 3, 4, 1, snake.data(), 5, options, output)
        || output.size() != snake.size()) {
        return 1;
    }
    for (std::size_t i = 0; i < snake.size(); ++i) {
        const double derivative = (i % 2 == 0 ? 0.04 : 0.02) * snake[i];
        if (std::abs(output[i] - (snake[i] + std::tanh(derivative / options.gamma))) > 1e-12) {
            return 2;
        }
    }
    options.boundary = skimagexx::ActiveContourBoundary::fixed;
    if (!skimagexx::active_contour(image.data(), 3, 4, 1, snake.data(), 5, options, output)
        || output[0] != snake[0] || output[1] != snake[1]
        || output[8] != snake[8] || output[9] != snake[9]) {
        return 3;
    }
    const auto previous = output;
    if (skimagexx::active_contour(image.data(), 2, 4, 1, snake.data(), 5, options, output)
        || output != previous) {
        return 4;
    }
}
