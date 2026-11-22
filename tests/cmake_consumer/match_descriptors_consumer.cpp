#include <skimagexx/match_descriptors.hpp>

#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("match_descriptors SDK assertion failed");
}

} // namespace

int main() {
    using skimagexx::DescriptorMatch;
    using skimagexx::DescriptorMatchOptions;
    using skimagexx::match_descriptors_bool;
    const std::array<std::uint8_t, 8> left{};
    const std::array<std::uint8_t, 12> right{1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1};
    std::vector<DescriptorMatch> matches;
    require(match_descriptors_bool(left.data(), 2, right.data(), 3, 4, {}, matches));
    require(matches == std::vector<DescriptorMatch>{{0, 0}});
    DescriptorMatchOptions options;
    options.cross_check = false;
    require(match_descriptors_bool(left.data(), 2, right.data(), 3, 4, options, matches));
    require(matches == std::vector<DescriptorMatch>{{0, 0}, {1, 0}});
    options.max_distance = 0.25;
    require(match_descriptors_bool(left.data(), 2, right.data(), 3, 4, options, matches));
    require(matches.empty());
    options.max_distance = std::nextafter(0.25, 1.0);
    options.max_ratio = 0.5;
    require(match_descriptors_bool(left.data(), 2, right.data(), 3, 4, options, matches));
    require(matches.empty());
    options.max_ratio = std::nextafter(0.5, 1.0);
    require(match_descriptors_bool(left.data(), 2, right.data(), 3, 4, options, matches));
    require(matches == std::vector<DescriptorMatch>{{0, 0}, {1, 0}});
    const auto previous = matches;
    const std::array<std::uint8_t, 4> noncanonical{2, 0, 0, 0};
    require(!match_descriptors_bool(noncanonical.data(), 1, right.data(), 3, 4, options, matches));
    require(!match_descriptors_bool(left.data(), 1, right.data(), 1, 7, options, matches));
    require(matches == previous);

    // Multiple packed words, including the highest bit in the final word.
    std::array<std::uint8_t, 256> wide_left{};
    std::array<std::uint8_t, 512> wide_right{};
    wide_left.back() = 1;
    wide_right.back() = 1;
    require(match_descriptors_bool(wide_left.data(), 1, wide_right.data(), 2, 256, {}, matches));
    require(matches == std::vector<DescriptorMatch>{{0, 1}});
}
