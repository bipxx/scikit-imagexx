#include <skimagexx/line_aa.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks = 0, records = 0;
std::uint64_t digest = UINT64_C(1469598103934665603);
std::string context;
void require(bool value) {
    ++checks;
    if (!value) throw std::runtime_error(context + " check=" + std::to_string(checks));
}
std::uint64_t bits(double value) {
    std::uint64_t word;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}
void hash(std::uint64_t word) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        digest ^= (word >> shift) & 255;
        digest *= UINT64_C(1099511628211);
    }
}
template<class T> T read(std::ifstream& input) {
    T value;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(input));
    return value;
}
template<class T> std::vector<T> read_values(std::ifstream& input, std::size_t count) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(T)));
    require(bool(input));
    return values;
}

template<class T> struct Buffer {
    std::vector<T> storage;
    std::ptrdiff_t step;
    std::size_t start;
    T sentinel;
    Buffer(std::size_t count, std::ptrdiff_t stride, T guard)
        : storage((count - 1) * std::size_t(std::abs(stride)) + 3, guard),
          step(stride), start(stride > 0 ? 1 : 1 + (count - 1) * std::size_t(-stride)), sentinel(guard) {}
    T* data() { return storage.data() + start; }
    T value(std::size_t index) const { return storage[std::ptrdiff_t(start) + std::ptrdiff_t(index) * step]; }
    void guards(std::size_t count) const {
        for (std::size_t i = 0; i < storage.size(); ++i) {
            const auto delta = std::ptrdiff_t(i) - std::ptrdiff_t(start);
            const bool written = delta % step == 0 && delta / step >= 0 && std::size_t(delta / step) < count;
            if (!written) require(storage[i] == sentinel);
        }
    }
};

void saved(const char* filename, int workers) {
    std::ifstream input(filename, std::ios::binary);
    require(bool(input));
    std::array<char, 8> magic;
    input.read(magic.data(), magic.size());
    require(std::string(magic.data(), magic.size()) == std::string("LAA1RAW\0", 8));
    require(read<std::uint64_t>(input) == 36);
    for (unsigned record = 0; record < 36; ++record) {
        const auto label_size = read<std::uint64_t>(input);
        context.assign(label_size, ' ');
        input.read(context.data(), std::streamsize(label_size));
        require(bool(input));
        const auto r0 = read<std::int64_t>(input), c0 = read<std::int64_t>(input);
        const auto r1 = read<std::int64_t>(input), c1 = read<std::int64_t>(input);
        const auto count = read<std::uint64_t>(input);
        const auto expected_rows = read_values<std::intptr_t>(input, count);
        const auto expected_columns = read_values<std::intptr_t>(input, count);
        const auto expected_weights = read_values<std::uint64_t>(input, count);
        DrawLineAAGeometry geometry;
        require(draw_line_aa_prepare(r0, c0, r1, c1, geometry));
        require(draw_line_aa_count(geometry, workers) == count);
        for (const auto strides : {std::array<std::ptrdiff_t, 3>{1, 1, 1},
                                   std::array<std::ptrdiff_t, 3>{2, 3, 2},
                                   std::array<std::ptrdiff_t, 3>{-2, -3, -2}}) {
            Buffer<std::intptr_t> rows(count, strides[0], 123456789);
            Buffer<std::intptr_t> columns(count, strides[1], 123456789);
            Buffer<double> weights(count, strides[2], -123456789.25);
            draw_line_aa_into(geometry, {rows.data(), columns.data(), strides[0], strides[1]},
                              weights.data(), strides[2], workers);
            for (std::size_t i = 0; i < count; ++i) {
                require(rows.value(i) == expected_rows[i]);
                require(columns.value(i) == expected_columns[i]);
                require(bits(weights.value(i)) == expected_weights[i]);
                hash(std::uint64_t(rows.value(i)));
                hash(std::uint64_t(columns.value(i)));
                hash(bits(weights.value(i)));
            }
            rows.guards(count); columns.guards(count); weights.guards(count);
        }
        ++records;
    }
    require(input.peek() == std::ifstream::traits_type::eof());
}

void spatial(std::intptr_t r0, std::intptr_t c0, std::intptr_t r1, std::intptr_t c1, int workers) {
    context = "spatial " + std::to_string(r0) + "," + std::to_string(c0) + ":" + std::to_string(r1) + "," + std::to_string(c1);
    DrawLineAAGeometry geometry;
    require(draw_line_aa_prepare(r0, c0, r1, c1, geometry));
    const auto count = draw_line_aa_count(geometry, workers);
    const std::int64_t dr = std::abs(std::int64_t(r1) - r0), dc = std::abs(std::int64_t(c1) - c0);
    require(count >= 1 && count <= std::size_t(3 * (dr + dc + 1)));
    std::vector<std::intptr_t> rows(count), columns(count);
    std::vector<double> weights(count);
    draw_line_aa_into(geometry, {rows.data(), columns.data(), 1, 1}, weights.data(), 1, workers);
    require(rows.front() == r0 && columns.front() == c0 && weights.front() == 1.0);
    require(rows.back() == r1 && columns.back() == c1 && weights.back() == 1.0);
    const auto scale = dr + dc == 0 ? 1.0 : std::sqrt(double(dr * dr + dc * dc));
    const auto sign_r = r1 > r0 ? 1 : -1, sign_c = c1 > c0 ? 1 : -1;
    std::map<std::pair<std::intptr_t, std::intptr_t>, std::uint64_t> actual, expected;
    for (std::size_t i = 0; i < count; ++i) {
        require(std::isfinite(weights[i]) && weights[i] >= 0 && weights[i] <= 1);
        require(actual.emplace(std::make_pair(rows[i], columns[i]), bits(weights[i])).second);
    }
    // Independent geometric strip oracle; it does not repeat the recurrence.
    for (auto r = std::min(r0, r1); r <= std::max(r0, r1); ++r)
        for (auto c = std::min(c0, c1); c <= std::max(c0, c1); ++c) {
            const auto distance = std::abs(dc * sign_r * (r - r0) - dr * sign_c * (c - c0));
            if (distance < scale) expected.emplace(std::make_pair(r, c), bits(1.0 - distance / scale));
        }
    require(actual == expected);
    require(get_num_threads() == workers);
}

void admission() {
    context = "admission";
    DrawLineAAGeometry initial;
    require(draw_line_aa_prepare(1, 2, 3, 4, initial));
    const auto fields = [](const DrawLineAAGeometry& g) {
        return std::array<std::int32_t, 8>{g.r0, g.c0, g.r1, g.c1, g.dr, g.dc, g.sign_r, g.sign_c};
    };
    constexpr auto low = std::numeric_limits<std::int32_t>::min();
    constexpr auto high = std::numeric_limits<std::int32_t>::max();
    for (const auto values : {std::array<std::intptr_t, 4>{low, 0, 0, 0},
                              {high, 0, 0, 0}, {0, low, 0, 0}, {0, 0, high, 0}, {0, 0, 0, low},
                              {0, 0, 46341, 0}, {0, 0, 32768, 32768}, {low + 1, 0, high - 1, 0},
                              {std::numeric_limits<std::intptr_t>::max(), 0, 0, 0},
                              {std::numeric_limits<std::intptr_t>::min(), 0, 0, 0}}) {
        auto output = initial;
        require(!draw_line_aa_prepare(values[0], values[1], values[2], values[3], output));
        require(fields(output) == fields(initial) && bits(output.distance) == bits(initial.distance));
    }
    require(draw_line_aa_prepare(0, 0, 46340, 0, initial));
    require(draw_line_aa_count(initial, 1) == 46341);
    require(draw_line_aa_prepare(0, 0, 32767, 32767, initial));
    require(draw_line_aa_count(initial, 1) <= std::size_t(3 * (65534 + 1)));
}
}

int main(int argc, char** argv) {
    static_assert(sizeof(std::intptr_t) == 8, "frozen reference uses64-bit intp");
    require((argc == 1 || argc == 3) && get_num_threads() == 1);
    admission();
    for (int workers : {1, 4}) {
#ifdef SKIMAGEXX_LINE_AA_PORTABLE
        if (workers == 4) {
            bool rejected = false;
            try { set_num_threads(4); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && get_num_threads() == 1);
            continue;
        }
#endif
        set_num_threads(workers);
        if (argc == 3) { saved(argv[1], workers); saved(argv[2], workers); }
        for (std::intptr_t r0 = -4; r0 <= 4; ++r0)
            for (std::intptr_t c0 = -4; c0 <= 4; ++c0)
                for (std::intptr_t r1 = -4; r1 <= 4; ++r1)
                    for (std::intptr_t c1 = -4; c1 <= 4; ++c1)
                        spatial(r0, c0, r1, c1, workers);
        spatial(std::numeric_limits<std::int32_t>::min() + 1,
                std::numeric_limits<std::int32_t>::max() - 1,
                std::numeric_limits<std::int32_t>::min() + 2,
                std::numeric_limits<std::int32_t>::max() - 3, workers);
    }
    set_num_threads(1);
    std::cout << "line_aa frozen records=" << records << "; word hash=" << digest << '\n';
    std::cout << "line_aa native controls=" << checks << "; default worker=1; AVX2=" << has_avx2() << '\n';
}
