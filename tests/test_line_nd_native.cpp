#include <skimagexx/line_nd.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx::detail;
namespace {
std::size_t checks = 0, records = 0;
std::string context;
void require(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error(context + " check=" + std::to_string(checks));
}
std::uint64_t read(std::ifstream& stream) {
    std::uint64_t word; stream.read(reinterpret_cast<char*>(&word), 8);
    require(bool(stream)); return word;
}
std::vector<unsigned char> blob(std::ifstream& stream) {
    std::vector<unsigned char> result(read(stream));
    stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(result.size()));
    require(bool(stream)); return result;
}
std::size_t itemsize(LineNDType type) {
    return type == LineNDType::float16 ? 2 : type == LineNDType::float32 ? 4 :
        type == LineNDType::int8 || type == LineNDType::uint8 ? 1 : 8;
}
struct Input {
    std::vector<unsigned char> backing;
    LineNDVector view;
    Input(const std::vector<unsigned char>& raw, LineNDType type, int layout) {
        const auto item = itemsize(type), count = raw.size() / item;
        backing.assign(count * item * 2 + 17, 0xa7);
        const std::ptrdiff_t step = layout == 0 ? std::ptrdiff_t(item) :
            layout == 1 ? std::ptrdiff_t(item * 2) : -std::ptrdiff_t(item * 2);
        const auto offset = std::size_t(layout == 2 ? 1 + (count - 1) * item * 2 : 1);
        for (std::size_t i = 0; i < count; ++i)
            std::memcpy(backing.data() + std::ptrdiff_t(offset) + std::ptrdiff_t(i) * step,
                        raw.data() + i * item, item);
        view = {backing.data() + offset, count, step, type};
    }
};
void saved(const char* path, int workers) {
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream));
    const auto count = read(stream); require(count == 92);
    for (std::size_t row = 0; row < count; ++row) {
        const auto id = blob(stream); context.assign(id.begin(), id.end());
        const auto type = LineNDType(read(stream));
        const auto a = blob(stream), b = blob(stream);
        const auto endpoint = bool(read(stream)), integer = bool(read(stream));
        const auto dimensions = read(stream), points = read(stream), item = read(stream);
        const auto scalar = bool(read(stream)), underflow = bool(read(stream));
        (void)scalar;
        std::vector<std::vector<unsigned char>> expected;
        for (std::size_t d = 0; d < dimensions; ++d) expected.push_back(blob(stream));
        for (int layout = 0; layout < 3; ++layout) {
            Input start(a, type, layout), stop(b, type, layout);
            const auto a_before = start.backing, b_before = stop.backing;
            LineNDPlan plan;
            std::feraiseexcept(FE_INVALID);
            const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
            const auto promotion = context.rfind("2.5.4:",0) == 0
                ? LineNDScalarPromotion::weak_input : LineNDScalarPromotion::legacy_float64;
            require(line_nd_prepare(start.view, stop.view, endpoint, integer, plan, promotion));
            require(std::fetestexcept(FE_ALL_EXCEPT) == flags);
            require(plan.dimensions == dimensions && plan.points == points);
            require(plan.events.divide_underflow == underflow);
            require(start.backing == a_before && stop.backing == b_before);
            // Mutating inputs after prepare cannot alter its fixed snapshots.
            std::fill(start.backing.begin(), start.backing.end(), 0x57);
            std::fill(stop.backing.begin(), stop.backing.end(), 0x62);
            std::ptrdiff_t point_step = std::ptrdiff_t(dimensions * item);
            std::ptrdiff_t dimension_step = std::ptrdiff_t(item);
            if (layout == 1) { point_step = std::ptrdiff_t(item); dimension_step = std::ptrdiff_t((points + 1) * item); }
            if (layout == 2) point_step = -point_step;
            const auto span = dimensions * (points + 2) * item;
            std::vector<unsigned char> output(span + 33, 0xc5), written(output.size(), 0);
            auto origin = std::size_t(1 + (layout == 2 && points ? (points - 1) * dimensions * item : 0));
            const auto events = line_nd_into(plan, output.data() + origin, point_step, dimension_step, workers);
            require(events.divide_underflow == underflow && events.multiply_underflow == underflow);
            require(std::fetestexcept(FE_ALL_EXCEPT) == flags);
            for (std::size_t d = 0; d < dimensions; ++d) {
                require(expected[d].size() == points * item);
                for (std::size_t i = 0; i < points; ++i) {
                    const auto offset = std::ptrdiff_t(origin) + std::ptrdiff_t(i) * point_step + std::ptrdiff_t(d) * dimension_step;
                    require(std::memcmp(output.data() + offset, expected[d].data() + i * item, item) == 0);
                    for (std::size_t byte = 0; byte < item; ++byte) written[std::size_t(offset) + byte] = 1;
                }
            }
            for (std::size_t i = 0; i < output.size(); ++i) if (!written[i]) require(output[i] == 0xc5);
            std::feclearexcept(FE_ALL_EXCEPT);
        }
        ++records;
    }
}
void holdouts(int workers) {
    context = "independent-large-and-refusal";
    double start[2] = {0, 0}, stop[2] = {131073, -131073};
    LineNDPlan plan;
    require(line_nd_prepare({start,2,8,LineNDType::float64}, {stop,2,8,LineNDType::float64}, true, false, plan));
    std::vector<double> out(plan.points * 2);
    const auto flags = line_nd_into(plan, out.data(), 16, 8, workers);
    require(!flags.divide_underflow && !flags.multiply_underflow);
    for (std::size_t i = 0; i < plan.points; ++i) {
        require(out[2*i] == double(i)); require(out[2*i+1] == -double(i));
    }
    const auto before = plan;
    require(!line_nd_prepare({start,0,8,LineNDType::float64}, {stop,2,8,LineNDType::float64}, true, false, plan));
    require(plan.points == before.points && plan.start == before.start);
    require(!line_nd_prepare({start,2,8,LineNDType::float64}, {stop,2,8,LineNDType::float32}, true, false, plan));
    // Half software-rounding underflow: known IEEE words, no new originals.
    const std::uint16_t half_start[2] = {0,0}, half_stop[2] = {0x4400,1};
    require(line_nd_prepare({half_start,2,2,LineNDType::float16},
                           {half_stop,2,2,LineNDType::float16}, false, false, plan));
    require(plan.points == 4 && plan.events.divide_underflow && plan.zero_step);
    std::uint16_t half_output[8]{};
    const auto half_events = line_nd_into(plan, half_output, 4, 2, workers);
    const std::uint16_t half_expected[8] = {0,0,0x3c00,0,0x4000,0,0x4200,1};
    require(half_events.divide_underflow && half_events.multiply_underflow);
    require(std::memcmp(half_output,half_expected,sizeof(half_output)) == 0);
    // Same NumPy scalar expression has two promotion policies. These IEEE
    // half neighbors are independent arithmetic controls, not new originals.
    const std::uint16_t near_half = 0xb7ff, half_end = 0x4480;
    const std::intptr_t legacy[6] = {0,0,2,2,4,4}, weak[6] = {-1,0,1,2,3,4};
    for (auto promotion : {LineNDScalarPromotion::legacy_float64,LineNDScalarPromotion::weak_input}) {
        require(line_nd_prepare({&near_half,1,2,LineNDType::float16},
            {&half_end,1,2,LineNDType::float16},true,true,plan,promotion));
        std::intptr_t result[6]{};
        line_nd_into(plan,result,8,8,workers);
        require(std::memcmp(result,promotion==LineNDScalarPromotion::weak_input?weak:legacy,sizeof(result))==0);
        const double near_double = -0.4999999701976776, double_end = 4.5;
        require(line_nd_prepare({&near_double,1,8,LineNDType::float64},
            {&double_end,1,8,LineNDType::float64},true,true,plan,promotion));
        line_nd_into(plan,result,8,8,workers);
        const std::intptr_t double_expected[6]={0,1,2,3,4,4};
        require(std::memcmp(result,double_expected,sizeof(result))==0);
        const float descending_start=4.5f, descending_stop=-0.4999999701976776f;
        require(line_nd_prepare({&descending_start,1,4,LineNDType::float32},
            {&descending_stop,1,4,LineNDType::float32},true,true,plan,promotion));
        line_nd_into(plan,result,8,8,workers);
        const std::intptr_t descending[6]={4,4,2,2,0,0};
        require(std::memcmp(result,descending,sizeof(result))==0);
    }
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2);
        require(skimagexx::get_num_threads() == 1);
        for (int workers : {1,4}) {
            skimagexx::set_num_threads(workers);
            const auto snapshot = skimagexx::get_num_threads();
            saved(argv[1], snapshot); holdouts(snapshot);
        }
        skimagexx::set_num_threads(1); require(skimagexx::get_num_threads() == 1);
        std::cout << "PASS records=" << records << " checks=" << checks
                  << " saved_payloads=240arrays+6scalars layouts=3 workers=1,4 default_restored=1 promotion=legacy,weak\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
