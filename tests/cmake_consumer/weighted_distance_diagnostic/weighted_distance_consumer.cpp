#include <skimagexx/distance.hpp>
#include <skimagexx/image.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace skimagexx;
    if (get_num_threads() != 1) throw std::runtime_error("installed SDK startup workers must be 1");
    // Exact Lab row retained by root-admitted witness run 44174979377.
    const std::array<std::uint64_t, 3> words{{
        0xc042127fc6ef280bULL, 0xc0425175bbcde9a5ULL, 0x404c7feda6e2bf20ULL}};
    static_assert(sizeof(double) == sizeof(std::uint64_t), "binary64 SDK witness required");
    std::array<double, 3> first{}, second{};
    std::memcpy(first.data(), words.data(), sizeof(words));
    std::memcpy(second.data(), words.data(), sizeof(words));
    const bool available = has_avx2();
    try {
        for (int workers : {1, 4}) {
            set_num_threads(workers);
            for (auto metric : {DistanceMetric::cie94, DistanceMetric::cmc}) {
                double output = -1.;
                delta_e_into(first.data(), second.data(), &output, 1,
                    PixelType::float64, PixelType::float64, metric);
                std::uint64_t output_word = 0;
                std::memcpy(&output_word, &output, sizeof(output_word));
                if (output_word != 0 || get_num_threads() != workers ||
                    std::memcmp(first.data(), words.data(), sizeof(words)) != 0 ||
                    std::memcmp(second.data(), words.data(), sizeof(words)) != 0)
                    throw std::runtime_error("weighted installed SDK self-pair/worker/input mismatch");
                std::cout << "metric=" << (metric == DistanceMetric::cie94 ? "CIE94" : "CMC")
                          << " workers=" << workers << " output_raw64=" << output_word << '\n';
            }
        }
    } catch (...) {
        set_num_threads(1);
        throw;
    }
    set_num_threads(1);
    if (get_num_threads() != 1) throw std::runtime_error("installed SDK worker restoration failed");
    std::cout << "installed weighted SDK calls=4; default/restored=1; AVX2=" << available << '\n';
}
