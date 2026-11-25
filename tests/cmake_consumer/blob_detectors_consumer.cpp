#include <skimagexx/blob_detectors.hpp>
#include <skimagexx/image.hpp>

#include <array>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("blob detector SDK result or validation differs");
}

}  // namespace

int main() {
    constexpr std::size_t height = 13, width = 19;
    std::array<float, height * width> image{};
    image[6 * width + 9] = 1;
    const auto before = image;
    using skimagexx::BlobDetectionOptions;
    using skimagexx::BlobMethod;
    using skimagexx::PixelType;

    auto detect = [&](const BlobDetectionOptions& options) {
        return skimagexx::blob_detect(image.data(), image.size(), PixelType::float32, height, width, options);
    };
    auto check = [&](const BlobDetectionOptions& options, std::size_t column, double sigma) {
        const auto result = detect(options);
        require(result && result->size() == 1);
        const auto& blob = result->front();
        require(blob.row == 6 && blob.column == column && blob.sigma == std::array<double, 2>{sigma, sigma});
    };

    require(skimagexx::get_num_threads() == 1);
    const BlobDetectionOptions dog{BlobMethod::dog, {{1, 1}, {1.6, 1.6}}, 1 / (1.6 - 1), .001, {}, .5, {}};
    const BlobDetectionOptions log{BlobMethod::log, {{1, 1}}, 1, .001, {}, .5, {}};
    const BlobDetectionOptions doh{BlobMethod::doh, {{2.2, 2.2}}, 1, .001, {}, .5, {}};
    check(dog, 9, 1);
    check(log, 9, 1);
    check(doh, 8, 2.2);
    require(image == before);

    auto excluded = log;
    excluded.exclude_border = {7, 0};
    const auto empty = detect(excluded);
    require(empty && empty->empty());

    bool rejected = false;
    try {
        skimagexx::blob_detect(image.data(), image.size() - 1, PixelType::float32, height, width, log);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);
    image[0] = std::numeric_limits<float>::quiet_NaN();
    require(!detect(log));
}
