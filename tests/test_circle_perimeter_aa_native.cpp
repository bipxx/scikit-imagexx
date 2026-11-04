#include <skimagexx/circle_perimeter_aa.hpp>
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
        : storage((count ? count - 1 : 0) * std::size_t(std::abs(stride)) + 3, guard),
          step(stride), start(stride > 0 || !count ? 1 : 1 + (count - 1) * std::size_t(-stride)), sentinel(guard) {}
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
    std::array<char,8> magic;
    input.read(magic.data(),magic.size());
    require(std::string(magic.data(),magic.size())=="CAARAW01");
    require(read<std::uint64_t>(input)==40);
    for (unsigned record=0;record<40;++record) {
        const auto label_size=read<std::uint64_t>(input);
        context.assign(label_size,' ');
        input.read(context.data(),std::streamsize(label_size));
        require(bool(input));
        CirclePerimeterAARequest q;
        q.row=read<std::int64_t>(input); q.column=read<std::int64_t>(input); q.radius=read<std::int64_t>(input);
        q.clipped=bool(read<std::uint64_t>(input));
        q.height=read<std::int64_t>(input); q.width=read<std::int64_t>(input);
        const auto n=read<std::uint64_t>(input);
        const auto expected_rows=read_values<std::intptr_t>(input,n);
        const auto expected_columns=read_values<std::intptr_t>(input,n);
        const auto expected_weights=read_values<std::uint64_t>(input,n);
        CirclePerimeterAA prepared;
        require(prepare_circle_perimeter_aa(q,prepared));
        require(circle_perimeter_aa_count(prepared,workers)==n && n<=prepared.max_points);
        for (const auto strides : {std::array<std::ptrdiff_t,3>{1,1,1}, {2,3,2}, {-2,-3,-2}}) {
            Buffer<std::intptr_t> rows(n,strides[0],123456789), columns(n,strides[1],123456789);
            Buffer<double> weights(n,strides[2],-123456789.25);
            circle_perimeter_aa_fill(prepared,{rows.data(),columns.data(),strides[0],strides[1]},weights.data(),strides[2],workers);
            for (std::size_t i=0;i<n;++i) {
                require(rows.value(i)==expected_rows[i]); require(columns.value(i)==expected_columns[i]);
                require(bits(weights.value(i))==expected_weights[i]);
                hash(std::uint64_t(rows.value(i))); hash(std::uint64_t(columns.value(i))); hash(bits(weights.value(i)));
            }
            rows.guards(n); columns.guards(n); weights.guards(n);
        }
        ++records;
    }
    require(input.peek()==std::ifstream::traits_type::eof());
}

struct Points {
    std::vector<std::intptr_t> rows,columns;
    std::vector<double> weights;
};
Points output(const CirclePerimeterAA& prepared,int workers) {
    const auto n=circle_perimeter_aa_count(prepared,workers);
    require(n<=prepared.max_points);
    Points result{std::vector<std::intptr_t>(n),std::vector<std::intptr_t>(n),std::vector<double>(n)};
    circle_perimeter_aa_fill(prepared,{result.rows.data(),result.columns.data(),1,1},result.weights.data(),1,workers);
    for (auto weight : result.weights) require(std::isfinite(weight) && weight>=0 && weight<=1);
    return result;
}

void controls(int workers) {
    context="initial8, stable clipping, independent output invariants";
    for (std::intptr_t radius=-3;radius<=64;++radius) {
        CirclePerimeterAARequest q{7,-3,radius,false,0,0};
        CirclePerimeterAA prepared;
        require(prepare_circle_perimeter_aa(q,prepared));
        const auto whole=output(prepared,workers);
        require(whole.rows.size()>=8 && (whole.rows.size()-8)%16==0);
        const std::intptr_t rr[]{radius,0,radius,0,-radius,0,-radius,0};
        const std::intptr_t cc[]{0,radius,0,-radius,0,radius,0,-radius};
        for (unsigned i=0;i<8;++i) {
            require(whole.rows[i]==rr[i]+q.row && whole.columns[i]==cc[i]+q.column);
            require(bits(whole.weights[i])==bits(1.0));
        }
        if (radius<=1) require(whole.rows.size()==8);
        for (auto bounds : {std::pair<std::intptr_t,std::intptr_t>{0,9},{9,0},{-1,9},{9,-1},{8,9},{50,60}}) {
            auto clipped=q;clipped.clipped=true;clipped.height=bounds.first;clipped.width=bounds.second;
            CirclePerimeterAA cut;
            require(prepare_circle_perimeter_aa(clipped,cut));
            const auto actual=output(cut,workers);
            std::size_t j=0;
            for (std::size_t i=0;i<whole.rows.size();++i) {
                if (whole.rows[i]>=0 && whole.rows[i]<bounds.first && whole.columns[i]>=0 && whole.columns[i]<bounds.second) {
                    require(j<actual.rows.size());
                    require(actual.rows[j]==whole.rows[i] && actual.columns[j]==whole.columns[i]);
                    require(bits(actual.weights[j])==bits(whole.weights[i])); ++j;
                }
            }
            require(j==actual.rows.size());
        }
    }
    context="prospective long radius admission, not timing";
    CirclePerimeterAA prepared;
    require(prepare_circle_perimeter_aa({0,0,4096,false,0,0},prepared));
    const auto long_result=output(prepared,workers);
    require(long_result.rows.size()>40000);
    require(get_num_threads()==workers);
}

void admission() {
    context="single boundary, unchanged refusal, empty spans";
    constexpr auto high=std::numeric_limits<std::intptr_t>::max(),low=std::numeric_limits<std::intptr_t>::min();
    const CirclePerimeterAA sentinel{{1,2,3,false,4,5},19,9};
    for (auto q : {CirclePerimeterAARequest{high,0,1}, {low,0,1}, {0,high,1}, {0,low,1},
                   {0,0,low}, {0,0,high}, {0,0,3037000500LL}}) {
        auto prepared=sentinel;
        require(!prepare_circle_perimeter_aa(q,prepared));
        require(prepared.request.row==1 && prepared.request.column==2 && prepared.request.radius==3);
        require(prepared.request.height==4 && prepared.request.width==5 && !prepared.request.clipped);
        require(prepared.max_points==19 && prepared.squared_radius==9);
    }
    CirclePerimeterAA prepared;
    require(prepare_circle_perimeter_aa({high,low,0},prepared));
    const auto extreme=output(prepared,1);
    require(extreme.rows==std::vector<std::intptr_t>(8,high) && extreme.columns==std::vector<std::intptr_t>(8,low));
    require(prepare_circle_perimeter_aa({0,0,-high},prepared));
    require(circle_perimeter_aa_count(prepared,1)==8);
    const auto negative=output(prepared,1);require(negative.rows[0]==-high && negative.rows[4]==high);
    require(prepare_circle_perimeter_aa({0,0,3037000499LL},prepared)); // no huge traversal/allocation
    require(prepared.squared_radius==std::intptr_t(3037000499LL)*3037000499LL);
    require(prepare_circle_perimeter_aa({0,0,0,true,0,9},prepared));
    require(circle_perimeter_aa_count(prepared,1)==0);
    circle_perimeter_aa_fill(prepared,{nullptr,nullptr,1,1},nullptr,1,1);
}
}

int main(int argc,char** argv) {
    static_assert(sizeof(std::intptr_t)==8,"frozen reference uses64-bit intp");
    require((argc==1 || argc==3) && get_num_threads()==1);
    admission();
    for (int workers : {1,4}) {
#ifdef SKIMAGEXX_CIRCLE_AA_PORTABLE
        if (workers==4) {
            bool rejected=false;
            try { set_num_threads(4); } catch (const std::invalid_argument&) { rejected=true; }
            require(rejected && get_num_threads()==1);
            continue;
        }
#endif
        set_num_threads(workers);
        if (argc==3) { saved(argv[1],workers);saved(argv[2],workers); }
        controls(workers);
    }
    set_num_threads(1);
    require(get_num_threads()==1);
    std::cout<<"circle_perimeter_aa frozen records="<<records<<"; word hash="<<digest<<'\n';
    std::cout<<"circle_perimeter_aa native controls="<<checks<<"; default worker=1; AVX2="<<has_avx2()<<'\n';
}
