#include "skimagexx/hysteresis.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::uint64_t hysteresis_pack64(const std::uint8_t* input);
}
#endif

namespace skimagexx { namespace {
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
constexpr bool little_endian = true;
#else
constexpr bool little_endian = false;
#endif

std::size_t extent(const std::vector<std::size_t>& shape) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t n=1;for(auto d:shape){if(d>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(std::size_t)/n)throw std::overflow_error("hysteresis dimensions overflow");n*=d;}return n;
}
std::vector<std::size_t> pitches(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> out(shape.size(),1);for(auto d=shape.size();d-->1;)out[d-1]=out[d]*shape[d];return out;
}
std::ptrdiff_t linear(const HysteresisView& v,const std::vector<std::size_t>& shape) {
    if(std::all_of(v.strides.begin(),v.strides.end(),[](auto s){return s==0;}))return 0;
    std::ptrdiff_t expected=1;for(auto d=shape.size();d-->0;){if(shape[d]>1 && v.strides[d]!=expected)return -1;expected*=std::ptrdiff_t(shape[d]);}return 1;
}
std::ptrdiff_t offset(std::size_t i,const HysteresisView& v,const std::vector<std::size_t>& shape,std::ptrdiff_t step) {
    if(step>=0)return std::ptrdiff_t(i)*step;
    std::ptrdiff_t out=0;for(auto d=shape.size();d-->0;){out+=std::ptrdiff_t(i%shape[d])*v.strides[d];i/=shape[d];}return out;
}
template<class T> T value(const HysteresisView& view,std::ptrdiff_t i) {
    const auto x=static_cast<const T*>(view.data)[i];
    if constexpr(std::is_same_v<T,std::uint8_t>)return view.type==PixelType::boolean ? T(x!=0) : x;
    else return x;
}
template<class T> T minimum_value(T x, T y) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(x)) return x;
        if (std::isnan(y)) return y;
    }
    return x < y ? x : y;
}
template<class T> void map(const HysteresisView& a,const HysteresisView& b,void* output,const std::vector<std::size_t>& shape,bool minimum) {
    const auto n=extent(shape);if(!n)return;
    const auto ap=linear(a,shape),bp=linear(b,shape);
    const bool flat=ap>=0 && bp>=0;
    std::size_t inner=shape.size();while(inner && shape[inner-1]==1)--inner;
    const auto width=inner ? shape[inner-1] : n;
    const auto as=flat ? ap : a.strides[inner-1],bs=flat ? bp : b.strides[inner-1];
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        for(auto position=begin;position<end;) {
            const auto length=flat ? end-position : std::min(end-position,width-position%width);
            const auto ai=offset(position,a,shape,ap),bi=offset(position,b,shape,bp);std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(as>=0 && as<=1 && bs>=0 && bs<=1 && has_avx2())done=simd::hysteresis_map(static_cast<const T*>(a.data)+ai,
                static_cast<const T*>(b.data)+bi,static_cast<unsigned char*>(output)+position*(minimum ? sizeof(T) : 1),length,a.type,as,bs,minimum);
#endif
        for(auto k=done;k<length;++k) {
            const auto i=position+k;
            const auto x=value<T>(a,ai+std::ptrdiff_t(k)*as),y=value<T>(b,bi+std::ptrdiff_t(k)*bs);
            if(minimum) static_cast<T*>(output)[i]=minimum_value(x,y);
            else static_cast<std::uint8_t*>(output)[i]=x>y;
        }
        position+=length;
        }
    });
}
void dispatch(const HysteresisView& a,const HysteresisView& b,void* out,const std::vector<std::size_t>& shape,bool minimum) {
    if(a.strides.size()!=shape.size() || b.strides.size()!=shape.size())throw std::invalid_argument("hysteresis view rank mismatch");
    if(extent(shape) && (!a.data || !b.data || !out))throw std::invalid_argument("nonnull hysteresis buffers required");
    if(a.type!=b.type) {
        if(minimum || !((a.type==PixelType::int64 && b.type==PixelType::uint64) || (a.type==PixelType::uint64 && b.type==PixelType::int64)))throw std::invalid_argument("unsupported mixed hysteresis comparison");
        const auto n=extent(shape);const auto ap=linear(a,shape),bp=linear(b,shape);
        detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){
            const auto ai=offset(i,a,shape,ap),bi=offset(i,b,shape,bp);bool result;
            if(a.type==PixelType::int64){const auto x=value<std::int64_t>(a,ai);const auto y=value<std::uint64_t>(b,bi);result=x>=0 && std::uint64_t(x)>y;}
            else {const auto x=value<std::uint64_t>(a,ai);const auto y=value<std::int64_t>(b,bi);result=y<0 || x>std::uint64_t(y);}
            static_cast<std::uint8_t*>(out)[i]=result;
        }});return;
    }
    switch(a.type) {
#define MAP(tag,T) case PixelType::tag:map<T>(a,b,out,shape,minimum);return;
    MAP(boolean,std::uint8_t) MAP(uint8,std::uint8_t) MAP(int8,std::int8_t) MAP(uint16,std::uint16_t) MAP(int16,std::int16_t)
    MAP(uint32,std::uint32_t) MAP(int32,std::int32_t) MAP(uint64,std::uint64_t) MAP(int64,std::int64_t) MAP(float32,float) MAP(float64,double)
#undef MAP
    }
    throw std::invalid_argument("unsupported hysteresis dtype");
}
std::size_t root(std::vector<std::size_t>& parents,std::size_t i) {
    while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;
}
void unite(std::vector<std::size_t>& parents,std::size_t a,std::size_t b) {
    a=root(parents,a);b=root(parents,b);if(a>b)std::swap(a,b);parents[b]=a;
}
std::uint64_t pack_mask(const std::uint8_t* input, std::size_t length) {
    std::uint64_t bits = 0;
    std::size_t i = 0;
    if constexpr (little_endian) {
        constexpr std::uint64_t high = 0x8080808080808080;
        for (; i + 8 <= length; i += 8) {
            std::uint64_t bytes;
            std::memcpy(&bytes, input + i, 8);
            // Set the high bit for nonzero bytes without a carry between lanes.
            const auto nonzero = (((bytes & ~high) + ~high) | bytes) & high;
            bits |= (((nonzero >> 7) * 0x0102040810204080) >> 56) << i;
        }
    }
    for (; i < length; ++i) bits |= std::uint64_t(input[i] != 0) << i;
    return bits;
}

void unpack_mask(std::uint8_t* output, std::size_t length, std::uint64_t bits) {
    std::size_t i = 0;
    if constexpr (little_endian) {
        for (; i + 8 <= length; i += 8) {
            auto bytes = bits & 0xff;
            bytes = (bytes | (bytes << 28)) & 0x0000000f0000000f;
            bytes = (bytes | (bytes << 14)) & 0x0003000300030003;
            bytes = (bytes | (bytes << 7)) & 0x0101010101010101;
            std::memcpy(output + i, &bytes, 8);
            bits >>= 8;
        }
    }
    for (; i < length; ++i, bits >>= 1) output[i] = std::uint8_t(bits & 1);
}

std::uint64_t flood_word(std::uint64_t seeds, std::uint64_t weak) {
    auto reached = (seeds | (seeds << 1) | (seeds >> 1)) & weak;
    if (reached == seeds || reached == weak) return reached;
    // Double the propagation distance while keeping every intermediate bit weak.
    auto to_high = weak & (weak << 1);
    auto to_low = weak & (weak >> 1);
    for (unsigned shift = 2; shift < 64; shift *= 2) {
        reached |= ((reached << shift) & to_high) | ((reached >> shift) & to_low);
        to_high &= to_high << shift;
        to_low &= to_low >> shift;
    }
    return reached;
}

unsigned popcount(std::uint64_t bits) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bits);
#else
    bits -= (bits >> 1) & 0x5555555555555555;
    bits = (bits & 0x3333333333333333) + ((bits >> 2) & 0x3333333333333333);
    bits = (bits + (bits >> 4)) & 0x0f0f0f0f0f0f0f0f;
    return unsigned((bits * 0x0101010101010101) >> 56);
#endif
}

void flood_2d(const std::uint8_t* weak, const std::uint8_t* strong, std::uint8_t* output,
              std::size_t rows, std::size_t width) {
    const auto words = (width + 63) / 64;
    std::vector<std::uint64_t> weak_bits(rows * words), strong_bits(rows * words), out_bits(rows * words);
    std::size_t weak_count = 0, seed_count = 0, background_count = 0;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized = has_avx2();
#endif
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t word = 0; word < words; ++word) {
            const auto i = row * width + word * 64;
            const auto k = row * words + word;
            const auto length = std::min<std::size_t>(64, width - word * 64);
            std::uint64_t w, s;
#ifdef SKIMAGEXX_AVX2
            if (vectorized && length == 64) {
                w = simd::hysteresis_pack64(weak + i);
                s = simd::hysteresis_pack64(strong + i);
            } else
#endif
            {
                w = pack_mask(weak + i, length);
                s = pack_mask(strong + i, length);
            }
            weak_bits[k] = w;
            out_bits[k] = w & s;
            // Reuse the strong plane for new bits awaiting vertical propagation.
            strong_bits[k] = out_bits[k];
            weak_count += popcount(w);
            seed_count += popcount(w & s);
            background_count += popcount(s & ~w);
        }
    }

    const bool all_weak_reached = seed_count && (seed_count == weak_count || weak_count == rows * width);
    if (seed_count && !all_weak_reached) {
        // This ring holds at most one entry per row, including revisited rows.
        std::vector<std::size_t> queue;
        queue.reserve(rows);
        queue.resize(rows);
        std::vector<std::uint8_t> queued(rows);
        std::size_t head = 0, tail = 0, pending = 0;
        auto enqueue = [&](std::size_t row) {
            if (queued[row]) return;
            queued[row] = 1;
            queue[tail] = row;
            if (++tail == rows) tail = 0;
            ++pending;
        };
        for (std::size_t row = 0; row < rows; ++row) {
            const auto begin = strong_bits.begin() + row * words;
            if (std::any_of(begin, begin + words, [](auto bits) { return bits != 0; })) enqueue(row);
        }
        while (pending) {
            const auto row = queue[head];
            if (++head == rows) head = 0;
            --pending;
            queued[row] = 0;
            const auto begin = row * words;
            const auto end = begin + words;

            auto expand = [&](std::size_t k, std::uint64_t carry, bool dirty) {
                const auto old = out_bits[k];
                const auto seed = old | (carry & weak_bits[k]);
                if (dirty || seed != old) {
                    out_bits[k] = flood_word(seed, weak_bits[k]);
                    strong_bits[k] |= out_bits[k] ^ old;
                }
                return out_bits[k];
            };
            std::uint64_t carry = 0;
            for (auto k = begin; k < end; ++k)
                carry = expand(k, carry, strong_bits[k] != 0) >> 63;
            carry = 0;
            // After the forward pass, only new carries can open paths to the left.
            for (auto k = end; k-- > begin;)
                carry = expand(k, carry, false) << 63;

            auto propagate = [&](std::size_t neighbor) {
                bool changed = false;
                auto adjacent = neighbor * words;
                for (auto k = begin; k < end; ++k, ++adjacent) {
                    const auto added = strong_bits[k] & weak_bits[adjacent] & ~out_bits[adjacent];
                    out_bits[adjacent] |= added;
                    strong_bits[adjacent] |= added;
                    changed |= added != 0;
                }
                if (changed) enqueue(neighbor);
            };
            if (row) propagate(row - 1);
            if (row + 1 < rows) propagate(row + 1);
            std::fill_n(strong_bits.data() + begin, words, std::uint64_t(0));
        }
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t word = 0; word < words; ++word) {
            const auto k = row * words + word;
            auto bits = all_weak_reached ? weak_bits[k] : out_bits[k];
            if (background_count) bits |= ~weak_bits[k];
            unpack_mask(output + row * width + word * 64,
                        std::min<std::size_t>(64, width - word * 64), bits);
        }
    }
}

struct Run { std::uint32_t row_begin, left, right; };
struct WideRun { std::size_t row_begin, left, right; };

template<class Span>
void flood_runs(const std::uint8_t* weak,const std::uint8_t* strong,std::uint8_t* out,
    std::size_t n,const std::vector<std::size_t>& shape,const std::vector<std::size_t>& pitch,bool background) {
    const auto width = shape.back();
    std::vector<Span> queue;
    if (background) {
        for (std::size_t i = 0; i < n; ++i) out[i] = !weak[i];
    } else {
        std::fill_n(out, n, std::uint8_t(0));
    }
    auto discover=[&](std::size_t seed,std::size_t row_begin) {
        auto left=seed,right=seed+1;const auto row_end=row_begin+width;
        while(left>row_begin && weak[left-1] && !out[left-1])--left;
        while(right<row_end && weak[right] && !out[right])++right;
        std::fill(out+left,out+right,std::uint8_t(1));
        using Index = decltype(Span::row_begin);
        queue.push_back({Index(row_begin), Index(left - row_begin), Index(right - row_begin)});
        return right;
    };
    auto flood_seed = [&](std::size_t seed, std::size_t seed_row) {
        if (!strong[seed] || out[seed] || !weak[seed]) return;
        queue.clear();
        discover(seed, seed_row);
        for(std::size_t head=0;head<queue.size();++head) {
            // discover may reallocate the queue; keep this run by value.
            const auto run = queue[head];
            auto scan = [&](std::size_t adjacent_row) {
                const auto end = adjacent_row + run.right;
                for (auto i = adjacent_row + run.left; i < end;) {
                    if(weak[i] && !out[i])i=discover(i,adjacent_row);
                    else ++i;
                }
            };
            for (std::size_t d = 0; d + 1 < shape.size(); ++d) {
                const auto coordinate = run.row_begin / pitch[d] % shape[d];
                if (coordinate) scan(run.row_begin - pitch[d]);
                if (coordinate + 1 < shape[d]) scan(run.row_begin + pitch[d]);
            }
        }
    };
    for (std::size_t row = 0; row < n; row += width) {
        const auto end = row + width;
        auto seed = row;
        // A word load skips sparse seed blocks and supports noncanonical mask bytes.
        for (; end - seed >= sizeof(std::uint64_t); seed += sizeof(std::uint64_t)) {
            std::uint64_t block;
            std::memcpy(&block, strong + seed, sizeof(block));
            if (block) {
                for (std::size_t lane = 0; lane < sizeof(block); ++lane) flood_seed(seed + lane, row);
            }
        }
        for (; seed < end; ++seed) flood_seed(seed, row);
    }
}
}
void hysteresis_minimum_into(const HysteresisView& a,const HysteresisView& b,void* out,const std::vector<std::size_t>& shape){dispatch(a,b,out,shape,true);}
void hysteresis_greater_into(const HysteresisView& a,const HysteresisView& b,std::uint8_t* out,const std::vector<std::size_t>& shape){dispatch(a,b,out,shape,false);}
void hysteresis_connect_into(const std::uint8_t* weak,const std::uint8_t* strong,std::uint8_t* out,const std::vector<std::size_t>& shape) {
    const auto n=extent(shape);if(!n)return;
    if(!weak || !strong || !out)throw std::invalid_argument("nonnull hysteresis masks required");
    const auto workers=detail::active_workers(n);
    if (workers == 1 && shape.size() == 2) {
        flood_2d(weak, strong, out, shape[0], shape[1]);
        return;
    }
    struct Counts{std::size_t weak=0,seeds=0;bool background=false;};
    std::vector<Counts> totals(std::size_t(workers),Counts{});
    detail::partition(n,workers,[&](auto begin,auto end,auto worker){Counts c;for(auto i=begin;i<end;++i){c.weak+=weak[i]!=0;c.seeds+=weak[i] && strong[i];c.background|=!weak[i] && strong[i];}totals[worker]=c;});
    Counts counts;for(auto c:totals){counts.weak+=c.weak;counts.seeds+=c.seeds;counts.background|=c.background;}
    if(!counts.seeds || counts.seeds==counts.weak || counts.weak==n) {
        detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)out[i]=weak[i] ? counts.seeds && (counts.seeds==counts.weak || counts.weak==n) : counts.background;});return;
    }
    const auto pitch=pitches(shape);
    if(workers==1) {
        if (n <= std::numeric_limits<std::uint32_t>::max())
            flood_runs<Run>(weak, strong, out, n, shape, pitch, counts.background);
        else
            flood_runs<WideRun>(weak, strong, out, n, shape, pitch, counts.background);
        return;
    }
    std::vector<std::size_t> parents(n);std::vector<std::pair<std::size_t,std::size_t>> segments(std::size_t(workers),{0,0});
    detail::partition(n,workers,[&](auto begin,auto end,auto worker){segments[worker]={begin,end};
        for(auto i=begin;i<end;++i)parents[i]=i;
        if(shape.size()==2){const auto width=shape[1];for(auto i=begin;i<end;++i)if(weak[i]){
            if(i>=width && i-width>=begin && weak[i-width])unite(parents,i,i-width);
            if(i%width && i>begin && weak[i-1])unite(parents,i,i-1);
        }} else if(shape.size()==1){for(auto i=begin;i<end;++i)if(weak[i] && i>begin && weak[i-1])unite(parents,i,i-1);}
        else for(auto i=begin;i<end;++i)if(weak[i])for(std::size_t d=0;d<shape.size();++d)if((i/pitch[d])%shape[d] && i-pitch[d]>=begin && weak[i-pitch[d]])unite(parents,i,i-pitch[d]);
    });
    for(auto segment:segments)for(std::size_t d=0;d<shape.size();++d)if(shape[d]>1){const auto end=segment.first+std::min(pitch[d],segment.second-segment.first);
        for(auto i=segment.first;i<end;++i)if((i/pitch[d])%shape[d] && weak[i] && weak[i-pitch[d]])unite(parents,i,i-pitch[d]);
    }
    std::vector<std::uint8_t> selected(n,0);
    for(std::size_t i=0;i<n;++i)if(weak[i]){parents[i]=root(parents,i);if(strong[i])selected[parents[i]]=1;}
    detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)out[i]=weak[i] ? selected[parents[i]] : counts.background;});
}
void hysteresis_threshold_into(const void* image,const void* low,const void* high,std::uint8_t* out,PixelType type,
    const std::vector<std::size_t>& shape,bool low_scalar,bool high_scalar) {
    const auto n=extent(shape);if(!n)return;
    if (!image || !low || !high || !out) throw std::invalid_argument("nonnull hysteresis buffers required");
    const HysteresisView a{image, type, {}}, l{low, type, {}}, h{high, type, {}};
    auto run = [&](auto sample) {
        using T = decltype(sample);
        std::vector<std::uint8_t> weak(n), strong(n);
#ifdef SKIMAGEXX_AVX2
        const bool vectorized = has_avx2();
#endif
        detail::partition(n, detail::active_workers(n), [&](auto begin, auto end, auto) {
#ifdef SKIMAGEXX_AVX2
            if (vectorized) {
                // Keep the existing SIMD comparison semantics while sharing a
                // cache-sized scratch buffer between the two mask calculations.
                T clipped[256];
                constexpr std::size_t lanes = 32 / sizeof(T);
                while (end - begin >= lanes) {
                    const auto* x = static_cast<const T*>(image) + begin;
                    const auto* lo = static_cast<const T*>(low) + (low_scalar ? 0 : begin);
                    const auto* hi = static_cast<const T*>(high) + (high_scalar ? 0 : begin);
                    const auto length = std::min<std::size_t>(end - begin, 256);
                    const auto done = simd::hysteresis_map(lo, hi, clipped, length, type,
                                                          !low_scalar, !high_scalar, true);
                    simd::hysteresis_map(x, clipped, weak.data() + begin, done, type, 1, 1, false);
                    simd::hysteresis_map(x, hi, strong.data() + begin, done, type, 1, !high_scalar, false);
                    begin += done;
                }
            }
#endif
            for (auto i = begin; i < end; ++i) {
                const T x = value<T>(a, i);
                const T lo = value<T>(l, low_scalar ? 0 : i);
                const T hi = value<T>(h, high_scalar ? 0 : i);
                weak[i] = x > minimum_value(lo, hi);
                strong[i] = x > hi;
            }
        });
        hysteresis_connect_into(weak.data(), strong.data(), out, shape);
    };
    switch(type) {
#define RUN(tag,T) case PixelType::tag:run(T{});return;
    RUN(boolean,std::uint8_t) RUN(uint8,std::uint8_t) RUN(int8,std::int8_t) RUN(uint16,std::uint16_t) RUN(int16,std::int16_t)
    RUN(uint32,std::uint32_t) RUN(int32,std::int32_t) RUN(uint64,std::uint64_t) RUN(int64,std::int64_t) RUN(float32,float) RUN(float64,double)
#undef RUN
    }
    throw std::invalid_argument("unsupported hysteresis dtype");
}
}
