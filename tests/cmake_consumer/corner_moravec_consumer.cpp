#include <skimagexx/corner_moravec.hpp>

#include <array>
#include <cfenv>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("corner_moravec SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::corner_moravec_u8_into;

    std::array<std::uint8_t, 49> impulse{};
    impulse[3 * 7 + 3] = 255;
    std::array<double, 49> response{};
    const auto input = impulse;
    require(corner_moravec_u8_into(impulse.data(), 7, 7, 1, response.data()));
    require(impulse == input);
    for (std::size_t r = 0; r < 7; ++r) {
        for (std::size_t c = 0; c < 7; ++c) {
            double value = (r >= 2 && r <= 4 && c >= 2 && c <= 4) ? 1.0 : 0.0;
            if (r == 3 && c == 3) value = 2.0;
            require(response[r * 7 + c] == value);
        }
    }

    std::array<std::uint8_t, 25> column_step{};
    for (std::size_t r = 0; r < 5; ++r) {
        for (std::size_t c = 2; c < 5; ++c) column_step[r * 5 + c] = 255;
    }
    std::array<double, 25> smallest{};
    require(corner_moravec_u8_into(column_step.data(), 5, 5, 1, smallest.data()));
    require(smallest[2 * 5 + 2] == 3.0);  // A vertical-only shift would give zero.

    const auto prior = response;
    require(!corner_moravec_u8_into(impulse.data(), 7, 7, 2, response.data()));
    require(!corner_moravec_u8_into(impulse.data(), 7, 7, -1, response.data()));
    require(!corner_moravec_u8_into(impulse.data(), 7, 321, 1, response.data()));
    require(!corner_moravec_u8_into(nullptr, 7, 7, 1, response.data()));
    require(response == prior);

    const int nearest = std::fegetround();
    require(nearest == FE_TONEAREST && std::fesetround(FE_DOWNWARD) == 0);
    const bool refused = !corner_moravec_u8_into(impulse.data(), 7, 7, 1, response.data());
    require(std::fesetround(nearest) == 0 && refused && response == prior);
}
