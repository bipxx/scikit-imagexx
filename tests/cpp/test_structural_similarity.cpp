#include "skimagexx/structural_similarity.hpp"
#include "skimagexx/image.hpp"
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void score_matches(skimagexx::PixelType dtype, double expected, double tolerance) {
    constexpr std::size_t rows = 3, cols = 3;
    std::vector<T> left(rows * cols), right(rows * cols);
    for (int i = 0; i < 9; ++i) {
        left[i] = T((i * 19 % 127 - 63) / 128.0);
        right[i] = T((i * 29 % 151 - 75) / 152.0);
    }
    double score = 42;
    require(std::feclearexcept(FE_ALL_EXCEPT) == 0 && std::feraiseexcept(FE_INVALID) == 0,
            "cannot prime floating status");
    const int initial_flags = std::fetestexcept(FE_ALL_EXCEPT);
    const auto valid = skimagexx::structural_similarity_score(left.data(), right.data(), dtype,
                                                              rows, cols, 3, 0.5, true, score);
    const int final_flags = std::fetestexcept(FE_ALL_EXCEPT);
    require(std::feclearexcept(FE_ALL_EXCEPT) == 0, "cannot clear floating status");
    require(valid, "declined valid score");
    require(initial_flags == final_flags, "valid score changed caller floating status");
    require(std::abs(score - expected) <= tolerance, "score differs from original reference");
    require(skimagexx::structural_similarity_score(left.data(), left.data(), dtype,
                                                    rows, cols, 3, 0.5, false, score)
            && score == 1, "equal aliased inputs must score one");
    score = 42;
    require(std::fesetround(FE_DOWNWARD) == 0, "cannot set downward rounding");
    const auto accepted = skimagexx::structural_similarity_score(left.data(), right.data(), dtype,
                                                                 rows, cols, 3, 0.5, true, score);
    const bool preserved_rounding = std::fegetround() == FE_DOWNWARD;
    require(std::fesetround(FE_TONEAREST) == 0, "cannot restore nearest rounding");
    require(!accepted && score == 42 && preserved_rounding, "unsupported rounding modified outputs");
    right[0] = T(1e-12);
    require(!skimagexx::structural_similarity_score(left.data(), right.data(), dtype,
                                                     rows, cols, 3, 0.5, true, score)
            && score == 42, "out-of-domain result must remain unchanged");
}

} // namespace

int main(int argc, char** argv) try {
    require(argc == 3, "provide original float32 and float64 scores as hexadecimal arguments");
    require(skimagexx::get_num_threads() == 1, "expected default worker count to be one");
    score_matches<float>(skimagexx::PixelType::float32, std::stod(argv[1]), 2e-7);
    score_matches<double>(skimagexx::PixelType::float64, std::stod(argv[2]), 1e-13);
    std::cout << "SSIM SDK oracle checks passed\n";
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}
