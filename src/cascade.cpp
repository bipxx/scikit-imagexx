#include "skimagexx/cascade.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <utility>

namespace skimagexx {
namespace {

struct XmlNode {
    std::string_view name, value;
    std::vector<XmlNode> children;

    const XmlNode* find(std::string_view key) const {
        const auto item = std::find_if(children.begin(), children.end(),
                                       [key](const auto& child) { return child.name == key; });
        return item == children.end() ? nullptr : &*item;
    }

    const XmlNode& get(std::string_view key) const {
        if (const auto* node = find(key)) return *node;
        throw std::invalid_argument("Missing cascade element: " + std::string(key));
    }
};

std::string_view trim(std::string_view input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    return input.substr(first, input.find_last_not_of(" \t\r\n") - first + 1);
}

class XmlReader {
public:
    explicit XmlReader(std::string_view xml) : xml_(xml) {}

    XmlNode read() {
        take("\xef\xbb\xbf");
        skip();
        auto root = element(0);
        skip();
        if (!xml_.empty()) fail("Trailing XML content");
        return root;
    }

private:
    std::string_view xml_;
    std::size_t offset_ = 0;

    [[noreturn]] void fail(const char* message) const {
        throw CascadeXmlError(std::string(message) + " at byte " + std::to_string(offset_));
    }

    void advance(std::size_t length) {
        xml_.remove_prefix(length);
        offset_ += length;
    }

    bool take(std::string_view token) {
        if (xml_.substr(0, token.size()) != token) return false;
        advance(token.size());
        return true;
    }

    void spaces() {
        const auto first = xml_.find_first_not_of(" \t\r\n");
        advance(first == std::string_view::npos ? xml_.size() : first);
    }

    void through(std::string_view end) {
        const auto pos = xml_.find(end);
        if (pos == std::string_view::npos) fail("Unterminated XML construct");
        advance(pos + end.size());
    }

    void skip() {
        for (;;) {
            spaces();
            if (take("<!--")) through("-->");
            else if (take("<?")) through("?>");
            else return;
        }
    }

    std::string_view name() {
        std::size_t len = 0;
        for (const char value : xml_) {
            const auto ch = static_cast<unsigned char>(value);
            if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != ':' && ch != '.') break;
            ++len;
        }
        if (!len) fail("Expected XML name");
        const auto result = xml_.substr(0, len);
        advance(len);
        return result;
    }

    XmlNode element(int depth) {
        if (depth >= 64 || !take("<")) fail("Expected cascade XML element");
        XmlNode node{name(), {}, {}};
        spaces();
        while (!take(">")) {
            if (take("/>")) return node;
            name();
            spaces();
            if (!take("=")) fail("Expected XML attribute value");
            spaces();
            if (take("\"")) through("\"");
            else if (take("'")) through("'");
            else fail("Expected quoted XML attribute");
            spaces();
        }
        skip();
        while (!take("</")) {
            if (xml_.empty()) fail("Unclosed XML element");
            if (xml_.front() == '<') {
                node.children.push_back(element(depth + 1));
            } else {
                const auto end = xml_.find('<');
                if (end == std::string_view::npos) fail("Unclosed XML element");
                if (!node.value.empty()) fail("Mixed cascade XML text");
                node.value = trim(xml_.substr(0, end));
                advance(end);
            }
            skip();
        }
        if (name() != node.name) fail("Mismatched XML element");
        spaces();
        if (!take(">")) fail("Expected end of XML element");
        return node;
    }
};

std::istringstream words(std::string_view text) {
    std::istringstream input{std::string(text)};
    input.imbue(std::locale::classic());
    return input;
}

template<class T>
T number(std::istream& input) {
    T value{};
    if (!(input >> value)) throw std::invalid_argument("Invalid numeric value in cascade XML");
    return value;
}

void end_of_numbers(std::istream& input) {
    if ((input >> std::ws).peek() != std::char_traits<char>::eof()) {
        throw std::invalid_argument("Unexpected numeric values in cascade XML");
    }
}

template<class T>
T scalar(const XmlNode& node) {
    auto input = words(node.value);
    const auto value = number<T>(input);
    end_of_numbers(input);
    return value;
}

struct Rectangle {
    int x, y, width, height;
    float weight;
};

struct Stump {
    std::size_t feature;
    float left, right, threshold;
    std::array<std::uint32_t, 8> lut{};
};

struct Stage {
    std::size_t begin, end;
    float threshold;
};

struct Cluster {
    std::int64_t r, c, width, height, count;
    CascadeDetection mean;

    void add(const CascadeDetection& box) {
        r += box.r;
        c += box.c;
        width += box.width;
        height += box.height;
        ++count;
        mean = {static_cast<int>(r / count), static_cast<int>(c / count),
                static_cast<int>(width / count), static_cast<int>(height / count)};
    }
};

float intersection(const CascadeDetection& a, const CascadeDetection& b) {
    const int width = std::min(a.c + a.width, b.c + b.width) - std::max(a.c, b.c);
    const int height = std::min(a.r + a.height, b.r + b.height) - std::max(a.r, b.r);
    if (width <= 0 || height <= 0) return 0;
    const auto smaller_area = std::min(static_cast<float>(a.width) * static_cast<float>(a.height),
                                       static_cast<float>(b.width) * static_cast<float>(b.height));
    return static_cast<float>(width) * static_cast<float>(height) / smaller_area;
}

void add_detection(std::vector<Cluster>& clusters, const CascadeDetection& box, float threshold) {
    Cluster* best = nullptr;
    for (auto& cluster : clusters) {
        const auto score = intersection(box, cluster.mean);
        if (score > threshold) {
            threshold = score;
            best = &cluster;
        }
    }
    if (best) best->add(box);
    else clusters.push_back({box.r, box.c, box.width, box.height, 1, box});
}

using Offset = std::ptrdiff_t;

struct Corners {
    Offset bottom_right, top_left, top_right, bottom_left;
};

Corners corners(int x, int y, int width, int height, Offset stride) {
    const auto top = stride * y;
    const auto bottom = stride * (y + height);
    return {bottom + x + width, top + x, top + x + width, bottom + x};
}

template<class T>
T area_sum(const T* origin, const Corners& a) {
    const T sum = origin[a.bottom_right] + origin[a.top_left];
    return (sum - origin[a.top_right]) - origin[a.bottom_left];
}

struct LbpGrid {
    std::array<Offset, 4> x, y;
};

unsigned lbp_code(const float* img, const LbpGrid& grid) {
    const auto& x = grid.x;
    const auto& y = grid.y;
    const auto* p0 = img + y[0];
    const auto* p1 = img + y[1];
    const auto* p2 = img + y[2];
    const auto* p3 = img + y[3];
    const auto block = [&x](const float* top, const float* bottom, std::size_t col) {
        return ((bottom[x[col + 1]] + top[x[col]]) - top[x[col + 1]]) - bottom[x[col]];
    };
    const float center = block(p1, p2, 1);
    return (static_cast<unsigned>(block(p0, p1, 0) >= center) << 7)
         | (static_cast<unsigned>(block(p0, p1, 1) >= center) << 6)
         | (static_cast<unsigned>(block(p0, p1, 2) >= center) << 5)
         | (static_cast<unsigned>(block(p1, p2, 2) >= center) << 4)
         | (static_cast<unsigned>(block(p2, p3, 2) >= center) << 3)
         | (static_cast<unsigned>(block(p2, p3, 1) >= center) << 2)
         | (static_cast<unsigned>(block(p2, p3, 0) >= center) << 1)
         | static_cast<unsigned>(block(p1, p2, 0) >= center);
}

int scaled(int length, float scale) {
    return static_cast<int>(static_cast<float>(length) * scale);
}

struct HaarRectangle {
    Corners area;
    float weight;
};

using HaarFeature = std::vector<HaarRectangle>;

} // namespace

struct Cascade::Model {
    int height = 0, width = 0;
    bool haar = false;
    std::vector<Rectangle> lbp_rectangles;
    std::vector<std::vector<Rectangle>> haar_rectangles;
    std::vector<Stage> stages;
    std::vector<Stump> stumps;
    std::size_t total_features = 0;

    explicit Model(std::string_view xml) {
        const auto root = XmlReader(xml).read();
        const auto* child = root.find("cascade");
        const auto& cascade = child ? *child : root;
        if (const auto* kind = cascade.find("featureType")) {
            if (kind->value != "LBP" && kind->value != "HAAR") {
                throw std::invalid_argument("Unsupported cascade feature type");
            }
            haar = kind->value == "HAAR";
        }
        height = scalar<int>(cascade.get("height"));
        width = scalar<int>(cascade.get("width"));
        if (height <= (haar ? 2 : 0) || width <= (haar ? 2 : 0)) {
            throw std::invalid_argument("Invalid cascade window");
        }
        const auto& features = cascade.get("features").children;
        total_features = features.size();
        for (const auto& feature : features) {
            if (!haar) {
                lbp_rectangles.push_back(rectangle(feature.get("rect"), false));
                continue;
            }
            if (const auto* tilted = feature.find("tilted"); tilted && scalar<int>(*tilted) != 0) {
                throw std::invalid_argument("Tilted Haar cascade features are unsupported");
            }
            auto& rectangles = haar_rectangles.emplace_back();
            for (const auto& rect : feature.get("rects").children) {
                rectangles.push_back(rectangle(rect, true));
            }
            if (rectangles.empty()) throw std::invalid_argument("Empty Haar feature");
        }
        const auto& source_stages = cascade.get("stages").children;
        const auto count = scalar<std::ptrdiff_t>(cascade.get("stageNum"));
        if (count < 0 || static_cast<std::size_t>(count) > source_stages.size()) {
            throw std::invalid_argument("Invalid cascade stage count");
        }
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            const auto& stage = source_stages[static_cast<std::size_t>(i)];
            const auto& weak = stage.get("weakClassifiers").children;
            const auto weak_count = scalar<std::ptrdiff_t>(stage.get("maxWeakCount"));
            if (weak_count < 0 || static_cast<std::size_t>(weak_count) > weak.size()) {
                throw std::invalid_argument("Invalid cascade stump count");
            }
            const auto begin = stumps.size();
            for (std::ptrdiff_t j = 0; j < weak_count; ++j) {
                stumps.push_back(stump(weak[static_cast<std::size_t>(j)]));
            }
            stages.push_back({begin, stumps.size(), scalar<float>(stage.get("stageThreshold"))});
        }
    }

private:
    Rectangle rectangle(const XmlNode& node, bool weighted) const {
        auto input = words(node.value);
        Rectangle result{number<int>(input), number<int>(input), number<int>(input), number<int>(input),
                         weighted ? number<float>(input) : 0.f};
        end_of_numbers(input);
        const int cells = weighted ? 1 : 3;
        if (result.x < 0 || result.x >= width || result.y < 0 || result.y >= height ||
            result.width <= 0 || result.width > (width - result.x) / cells ||
            result.height <= 0 || result.height > (height - result.y) / cells) {
            throw std::invalid_argument("Cascade feature lies outside its window");
        }
        return result;
    }

    Stump stump(const XmlNode& node) const {
        auto input = words(node.get("internalNodes").value);
        const auto left_child = number<int>(input);
        const auto right_child = number<int>(input);
        const auto feature = number<std::size_t>(input);
        if (feature >= total_features) throw std::invalid_argument("Invalid cascade feature index");
        auto leaves = words(node.get("leafValues").value);
        Stump result{feature, number<float>(leaves), number<float>(leaves), 0.f, {}};
        if (haar) {
            if (left_child != 0 || right_child != -1) {
                throw std::invalid_argument("Haar cascades require decision stumps");
            }
            result.threshold = number<float>(input);
        } else {
            for (auto& bits : result.lut) bits = static_cast<std::uint32_t>(number<std::int64_t>(input));
        }
        end_of_numbers(input);
        end_of_numbers(leaves);
        return result;
    }
};

Cascade::Cascade(std::string_view xml, float tolerance)
    : eps(tolerance), model_(std::make_shared<Model>(xml)) {
    window_height = model_->height;
    window_width = model_->width;
    stages_number = static_cast<std::ptrdiff_t>(model_->stages.size());
    stumps_number = static_cast<std::ptrdiff_t>(model_->stumps.size());
    features_number = static_cast<std::ptrdiff_t>(model_->total_features);
}

Cascade Cascade::from_file(const std::filesystem::path& path, float eps) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::ios_base::failure("Cannot open cascade XML");
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return Cascade(contents, eps);
}

std::vector<CascadeDetection> Cascade::detect(const float* integral, int rows, int columns,
                                              const std::vector<float>& scales, float step_ratio,
                                              int min_neighbor_number, float threshold) const {
    if (rows < 0 || columns < 0 || (rows && columns && !integral)) {
        throw std::invalid_argument("Invalid cascade integral image");
    }
    if (!model_ || window_height < model_->height || window_width < model_->width ||
        window_height > std::numeric_limits<int>::max() || window_width > std::numeric_limits<int>::max() ||
        stages_number > static_cast<std::ptrdiff_t>(model_->stages.size())) {
        throw std::invalid_argument("Invalid cascade detection configuration");
    }
    if (scales.empty() || !rows || !columns) return {};

    const auto stride = static_cast<Offset>(columns) + 1;
    const auto padded_rows = static_cast<std::size_t>(rows) + 1;
    if (static_cast<std::size_t>(stride) > static_cast<std::size_t>(std::numeric_limits<Offset>::max()) / padded_rows) {
        throw std::length_error("Cascade integral image exceeds addressable storage");
    }
    const auto pixel_count = padded_rows * static_cast<std::size_t>(stride);
    std::vector<float> padded(pixel_count);
    for (int r = 0; r < rows; ++r) {
        std::copy_n(integral + static_cast<Offset>(r) * columns, columns,
                    padded.data() + (static_cast<Offset>(r) + 1) * stride + 1);
    }

    std::vector<double> squared;
    if (model_->haar) {
        // Recover source pixels from the supplied integral image to construct
        // the squared integral required for Haar variance normalization.
        squared.resize(pixel_count);
        for (int r = 0; r < rows; ++r) {
            double accumulated = 0;
            for (int c = 0; c < columns; ++c) {
                const auto offset = static_cast<Offset>(r) * stride + c;
                const auto pixel = area_sum(padded.data() + offset, corners(0, 0, 1, 1, stride));
                accumulated += static_cast<double>(pixel) * pixel;
                squared[static_cast<std::size_t>(offset + stride + 1)] =
                    accumulated + squared[static_cast<std::size_t>(offset + 1)];
            }
        }
    }

    std::vector<Cluster> clusters;
    std::vector<LbpGrid> grids(model_->lbp_rectangles.size());
    std::vector<HaarFeature> haar_features(model_->haar_rectangles.size());
    for (const float scale : scales) {
        if (!std::isfinite(scale) || scale < 1) throw std::invalid_argument("Invalid cascade scale");
        const auto height = static_cast<float>(window_height) * scale;
        const auto width = static_cast<float>(window_width) * scale;
        if (static_cast<double>(height) >= rows || static_cast<double>(width) >= columns) continue;
        const int current_height = static_cast<int>(height);
        const int current_width = static_cast<int>(width);
        const int max_r = rows - current_height;
        const int max_c = columns - current_width;
        const float requested_step = std::round(scale * step_ratio);
        if (!std::isfinite(requested_step) || requested_step < 1) {
            throw std::invalid_argument("Cascade search step must be positive");
        }
        const int step = static_cast<double>(requested_step) >= std::max(max_r, max_c)
            ? std::max(max_r, max_c) : static_cast<int>(requested_step);
        for (std::size_t i = 0; i < grids.size(); ++i) {
            const auto& rect = model_->lbp_rectangles[i];
            const int x = scaled(rect.x, scale), y = scaled(rect.y, scale);
            const int w = scaled(rect.width, scale), h = scaled(rect.height, scale);
            for (int j = 0; j != 4; ++j) {
                grids[i].x[static_cast<std::size_t>(j)] = x + static_cast<Offset>(j) * w;
                grids[i].y[static_cast<std::size_t>(j)] = (y + static_cast<Offset>(j) * h) * stride;
            }
        }
        Corners norm_area{};
        double norm_pixels = 0;
        if (model_->haar) {
            const int border = scaled(1, scale);
            const int norm_width = scaled(model_->width - 2, scale);
            const int norm_height = scaled(model_->height - 2, scale);
            norm_area = corners(border, border, norm_width, norm_height, stride);
            norm_pixels = static_cast<double>(norm_width) * norm_height;
            for (std::size_t i = 0; i < haar_features.size(); ++i) {
                auto& feature = haar_features[i];
                feature.clear();
                for (const auto& rect : model_->haar_rectangles[i]) {
                    feature.push_back({corners(scaled(rect.x, scale), scaled(rect.y, scale),
                                               scaled(rect.width, scale), scaled(rect.height, scale), stride),
                                       rect.weight});
                }
            }
        }
        for (Offset r = 0; r < max_r; r += step) {
            for (Offset c = 0; c < max_c; c += step) {
                const auto offset = r * stride + c;
                const auto* origin = padded.data() + offset;
                float variance_norm = 1.f;
                if (model_->haar) {
                    const double sum = area_sum(origin, norm_area);
                    const double sum_squares = area_sum(squared.data() + offset, norm_area);
                    const double variance = norm_pixels * sum_squares - sum * sum;
                    if (variance > 0) variance_norm = static_cast<float>(std::sqrt(variance));
                }
                bool passed = true;
                for (std::ptrdiff_t i = 0; i < stages_number; ++i) {
                    const auto& stage = model_->stages[static_cast<std::size_t>(i)];
                    float score = 0;
                    for (auto j = stage.begin; j < stage.end; ++j) {
                        const auto& stump = model_->stumps[j];
                        bool left;
                        if (model_->haar) {
                            float value = 0;
                            for (const auto& rect : haar_features[stump.feature]) {
                                value += area_sum(origin, rect.area) * rect.weight;
                            }
                            left = value < stump.threshold * variance_norm;
                        } else {
                            const unsigned code = lbp_code(origin, grids[stump.feature]);
                            left = ((stump.lut[code >> 5] >> (code & 31)) & 1U) != 0;
                        }
                        score += left ? stump.left : stump.right;
                    }
                    if (score < stage.threshold - eps) {
                        passed = false;
                        break;
                    }
                }
                if (passed) add_detection(clusters, {static_cast<int>(r), static_cast<int>(c),
                                                     current_width, current_height}, threshold);
            }
        }
    }
    std::vector<CascadeDetection> result;
    for (const auto& cluster : clusters) {
        if (cluster.count >= min_neighbor_number) result.push_back(cluster.mean);
    }
    return result;
}

}
