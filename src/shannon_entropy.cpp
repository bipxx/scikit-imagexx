#include "shannon_entropy.hpp"
#include "execution.hpp"
#include "skimagexx/histogram_matching.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace skimagexx::detail {
namespace {
template<class UInt>
UInt float_key(UInt bits) {
    constexpr auto sign = UInt{1} << (std::numeric_limits<UInt>::digits - 1);
    constexpr UInt infinity = sizeof(UInt) == 4 ? UInt{0x7f800000} :
        static_cast<UInt>(0x7ff0000000000000ULL);
    const auto magnitude = bits & (sign - 1);
    // Group NaNs after positive infinity and canonicalize both signed zeros.
    // Inspect bits only, so even a signaling NaN never enters FP arithmetic.
    if (magnitude > infinity) return std::numeric_limits<UInt>::max();
    if (magnitude == 0) return sign;
    return bits ^ ((bits & sign) ? std::numeric_limits<UInt>::max() : sign);
}

template<class UInt>
std::vector<std::ptrdiff_t> radix_counts(const void* input, std::size_t size) {
    constexpr std::size_t bins = 256;
    std::array<std::array<std::size_t, bins>, sizeof(UInt)> offsets{};
    std::unique_ptr<UInt[]> keys{new UInt[size]};
    const auto* bytes = static_cast<const unsigned char*>(input);
    for (std::size_t i = 0; i < size; ++i) {
        UInt bits;
        std::memcpy(&bits, bytes + i * sizeof(UInt), sizeof(UInt));
        const auto key = float_key(bits);
        keys[i] = key;
        for (std::size_t digit = 0; digit < sizeof(UInt); ++digit)
            ++offsets[digit][(key >> (digit * 8)) & 255];
    }

    std::unique_ptr<UInt[]> scratch;
    for (std::size_t digit = 0; digit < sizeof(UInt); ++digit) {
        // Byte frequencies are independent of all preceding scatter passes.
        auto& next = offsets[digit];
        if (next[(keys[0] >> (digit * 8)) & 255] == size) continue;
        std::size_t start = 0;
        for (auto& position : next) {
            const auto count = position;
            position = start;
            start += count;
        }
        if (!scratch) scratch.reset(new UInt[size]);
        const auto shift = digit * 8;
        for (std::size_t i = 0; i < size; ++i) {
            const auto key = keys[i];
            scratch[next[(key >> shift) & 255]++] = key;
        }
        keys.swap(scratch);
    }
    if (!scratch) return {static_cast<std::ptrdiff_t>(size)};

    std::vector<std::ptrdiff_t> counts;
    counts.reserve(size);
    std::size_t begin = 0;
    for (std::size_t i = 1; i < size; ++i) {
        if (keys[i] != keys[begin]) {
            counts.push_back(static_cast<std::ptrdiff_t>(i - begin));
            begin = i;
        }
    }
    counts.push_back(static_cast<std::ptrdiff_t>(size - begin));
    return counts;
}

template<class T>
std::vector<std::ptrdiff_t> small_integer_counts(const T* input, std::size_t size) {
    const auto populated_counts = [size](const auto& bins) {
        std::vector<std::ptrdiff_t> counts;
        counts.reserve(std::min(size, bins.size()));
        for (auto count : bins)
            if (count) counts.push_back(count);
        return counts;
    };
    if constexpr (sizeof(T) == 1) {
        std::array<std::ptrdiff_t, 256> bins{};
        for (std::size_t i = 0; i < size; ++i) ++bins[input[i]];
        return populated_counts(bins);
    } else {
        const auto [min_value, max_value] = std::minmax_element(input, input + size);
        const int low = *min_value, high = *max_value;
        if (low <= high) {
            std::vector<std::ptrdiff_t> bins(std::size_t(high - low) + 1);
            std::size_t i = 0;
            for (; i < size; ++i) {
                const auto index = static_cast<std::size_t>(int(input[i]) - low);
                if (index >= bins.size()) break;
                ++bins[index];
            }
            if (i == size) return populated_counts(bins);
        }
        // The input may be writable through another view. A changed range
        // requires bins that can accommodate every value of the dtype.
        constexpr int lowest = std::numeric_limits<T>::min();
        std::vector<std::ptrdiff_t> bins(65536);
        for (std::size_t i = 0; i < size; ++i) ++bins[std::size_t(int(input[i]) - lowest)];
        return populated_counts(bins);
    }
}

void validate_workers(int workers) {
    if (workers < 1 || workers > 1024)
        throw std::invalid_argument("entropy worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if (workers != 1) throw std::invalid_argument("this build has no OpenMP support");
#endif
}

// NumPy DOUBLE_pairwise_sum (128 block), same association as colocalization.
// Primary BSD notice: third_party/numpy-LICENSE.txt.
template<class Get>double pairwise(std::size_t start,std::size_t n,const Get& get) {
    if(n<8){double sum=-0.;for(std::size_t i=0;i<n;++i)sum+=get(start+i);return sum;}
    if(n<=128){
        double v[8];for(std::size_t i=0;i<8;++i)v[i]=get(start+i);
        std::size_t i=8;for(;i<n-n%8;i+=8)for(std::size_t j=0;j<8;++j)v[j]+=get(start+i+j);
        double sum=((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7]));
        for(;i<n;++i)sum+=get(start+i);return sum;
    }
    auto left=n/2;left-=left%8;
    const double a=pairwise(start,left,get),b=pairwise(start+left,n-left,get);return a+b;
}
class Capture {
    std::fenv_t saved_;
public:
    Capture(){if(std::feholdexcept(&saved_))throw std::runtime_error("cannot capture entropy floating environment");}
    ~Capture(){std::fesetenv(&saved_);}
};
}
std::vector<std::ptrdiff_t> shannon_sorted_counts(
    const void* data, std::size_t size, PixelType type, int workers) {
    validate_workers(workers);
    if (size > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / pixel_size(type)
        || (size && !data))
        throw std::invalid_argument("entropy input extent overflow or null storage");
    if (size == 0) return {};
    switch (type) {
        case PixelType::float64: return radix_counts<std::uint64_t>(data, size);
        case PixelType::float32: return radix_counts<std::uint32_t>(data, size);
        case PixelType::uint8: return small_integer_counts(static_cast<const std::uint8_t*>(data), size);
        case PixelType::int16: return small_integer_counts(static_cast<const std::int16_t*>(data), size);
        case PixelType::uint16: return small_integer_counts(static_cast<const std::uint16_t*>(data), size);
        default: {
            const auto counts = histogram_sorted_counts(data, size, type, workers);
            return {counts.begin(), counts.end()};
        }
    }
}

double shannon_entropy_counts(const std::ptrdiff_t* counts,std::size_t size,int workers){
    validate_workers(workers);
    if(size>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)||(size&&!counts))
        throw std::invalid_argument("entropy count extent overflow or null storage");
    std::size_t integer_total=0;
    auto min_c=std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t max_c=0;
    for(std::size_t i=0;i<size;++i){
        const auto count=counts[i];
        if(count<=0||std::size_t(count)>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-integer_total)
            throw std::invalid_argument("entropy requires positive addressable counts");
        integer_total+=std::size_t(count);
        min_c=std::min(min_c,count);
        max_c=std::max(max_c,count);
    }
    Capture capture;
    if(!size)return 0.;
    // Every partial sum is exact when the integer total fits in binary64.
    const double total=integer_total<=9007199254740992ULL ? double(integer_total) :
        0.+pairwise(0,size,[&](auto i){return double(counts[i]);});
    const auto term=[&](std::ptrdiff_t count) {
        const double p=double(count)/total;
        return -p*std::log(p);
    };
    if(min_c==max_c) {
        const double uniform_term=term(min_c);
        return 0.+pairwise(0,size,[&](auto){return uniform_term;});
    }
    constexpr std::ptrdiff_t max_cached_count=256;
    using TermCache=std::array<double,max_cached_count+1>;
    const auto cached_term=[&](TermCache& term_cache,std::ptrdiff_t count) {
        if(count>max_cached_count)return term(count);
        auto& cached=term_cache[std::size_t(count)];
        // A nonuniform distribution has at least two symbols, so a cached
        // count has 0 < p < 1 and contributes a strictly positive term.
        if(cached==0.)cached=term(count);
        return cached;
    };
    if(workers==1 || size<parallel_min || max_c<=max_cached_count) {
        if(min_c>max_cached_count)return 0.+pairwise(0,size,[&](auto i){return term(counts[i]);});
        TermCache term_cache{};
        return 0.+pairwise(0,size,[&](auto i){return cached_term(term_cache,counts[i]);});
    }

    std::vector<double> terms(size);
    partition(size,workers,[&](auto begin,auto end,auto){
        Capture worker_capture;
        if(min_c>max_cached_count) {
            for(auto i=begin;i<end;++i)terms[i]=term(counts[i]);
        } else {
            TermCache term_cache{};
            for(auto i=begin;i<end;++i)terms[i]=cached_term(term_cache,counts[i]);
        }
    });
    return 0.+pairwise(0,size,[&](auto i){return terms[i];});
}
}
