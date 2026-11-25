#include <skimagexx/image.hpp>
#include <skimagexx/sift.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("SIFT SDK assertion failed");
}

} // namespace

int main() {
    using namespace skimagexx;
    constexpr int h = 192, w = 208;
    std::vector<float> image(h * w);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const float y = float(r % 17 - 8), x = float(c % 21 - 10);
            const int pattern = ((r / 17) * 11 + (c / 21) * 7) % 13;
            image[r * w + c] = float(0.25 + 0.05 * pattern) * std::exp(-(y * y + x * x) / (4 + pattern));
        }
    }
    require(get_num_threads() == 1);
    SiftOptions p;
    p.n_hist = 3;
    p.n_ori = 6;
    const auto combined = sift_detect(image.data(), h, w, PixelType::float32, p, true);
    require(combined.keypoints.size() > 32);
    require(combined.descriptors.size() == combined.keypoints.size() * 54);
    const auto detected = sift_detect(image.data(), h, w, PixelType::float32, p);
    require(detected.descriptors.empty() && detected.keypoints.size() == combined.keypoints.size());
    const auto extracted = sift_extract(image.data(), h, w, PixelType::float32, p, detected.keypoints);
    require(extracted == combined.descriptors);
#ifdef _OPENMP
    set_num_threads(2);
    const auto parallel = sift_detect(image.data(), h, w, PixelType::float32, p, true);
    set_num_threads(1);
    require(parallel.descriptors == combined.descriptors);
    require(parallel.keypoints.size() == combined.keypoints.size());
    for (std::size_t i = 0; i < parallel.keypoints.size(); ++i) {
        const auto& a = parallel.keypoints[i];
        const auto& b = combined.keypoints[i];
        require(a.row == b.row && a.col == b.col && a.sigma == b.sigma &&
                a.orientation == b.orientation && a.scale == b.scale && a.octave == b.octave);
    }
#endif
    std::vector<std::uint8_t> flat(h * w, 100);
    require(sift_detect(flat.data(), h, w, PixelType::uint8).keypoints.empty());
    bool zero_norm = false;
    const auto empty_descriptors = sift_extract(flat.data(), h, w, PixelType::uint8, p,
                                                detected.keypoints, &zero_norm);
    require(zero_norm && empty_descriptors == std::vector<std::uint8_t>(empty_descriptors.size()));
    try {
        p.upsampling = 3;
        (void)sift_detect(flat.data(), h, w, PixelType::uint8, p);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    throw std::runtime_error("SIFT SDK accepted unsupported upsampling");
}
