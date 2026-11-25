#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace skimagexx {

struct CascadeDetection {
    int r, c, width, height;
};

// An XML syntax error. Unsupported or invalid model data raises invalid_argument.
class CascadeXmlError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// OpenCV BOOST cascade with MB-LBP or axis-aligned Haar stump features.
class Cascade {
public:
    explicit Cascade(std::string_view xml, float eps = 1e-5f);
    static Cascade from_file(const std::filesystem::path& path, float eps = 1e-5f);

    float eps;
    std::ptrdiff_t window_height, window_width, stages_number, stumps_number, features_number;

    // Input is a C-order, unpadded, inclusive float32 integral image. Scale
    // factors are processed in supplied order. Clustering is serial.
    std::vector<CascadeDetection> detect(const float* integral, int rows, int columns,
                                         const std::vector<float>& scales, float step_ratio,
                                         int min_neighbor_number = 4,
                                         float intersection_score_threshold = 0.5f) const;

private:
    struct Model;
    std::shared_ptr<const Model> model_;
};

}
