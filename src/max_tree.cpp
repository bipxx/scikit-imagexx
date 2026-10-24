#include "skimagexx/max_tree.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
void max_tree_copy(const void*,void*,std::size_t);
void max_tree_invert(const void*,void*,std::size_t,MaxTreeType);
}
#endif
namespace {
struct Span { std::uintptr_t begin,end; };
struct GuardEnvironment {
    std::fenv_t saved{},held{}; bool valid=false,stored=false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    unsigned csr=_mm_getcsr();
#endif
    GuardEnvironment() noexcept { stored=std::fegetenv(&saved)==0; valid=stored && std::feholdexcept(&held)==0; }
    ~GuardEnvironment() noexcept {
        if(stored)std::fesetenv(&saved);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        _mm_setcsr(csr);
#endif
    }
};
bool fp_profile() {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr() & ((1u<<15)|(1u<<6)|(3u<<13)))return false;
#endif
    return true;
}
bool multiply(std::size_t a,std::size_t b,std::size_t& result) {
    if(a && b>std::numeric_limits<std::size_t>::max()/a)return false;
    result=a*b;return true;
}
bool add_span(const void* pointer,std::size_t count,std::size_t expected,
              std::size_t width,std::array<Span,14>& spans,std::size_t& used) {
    if(!pointer || count!=expected || count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/width)return false;
    auto begin=reinterpret_cast<std::uintptr_t>(pointer);auto bytes=count*width;
    if(begin%width || bytes>std::numeric_limits<std::uintptr_t>::max()-begin)return false;
    Span next{begin,begin+bytes};
    for(std::size_t k=0;k<used;++k)if(next.begin<spans[k].end && spans[k].begin<next.end)return false;
    if(used==spans.size())return false;
    spans[used++]=next;return true;
}
bool closing(MaxTreeOperation op) { return op==MaxTreeOperation::area_closing || op==MaxTreeOperation::diameter_closing; }
bool diameter(MaxTreeOperation op) { return op==MaxTreeOperation::diameter_opening || op==MaxTreeOperation::diameter_closing; }
bool filtered(MaxTreeOperation op) { return op!=MaxTreeOperation::tree && op!=MaxTreeOperation::local_maxima; }
bool provided_tree(MaxTreeProvided p) { return p.parent.data || p.traverser.data || p.parent.elements || p.traverser.elements; }

template<class T> T inverse(T value) {
    if constexpr(std::is_floating_point_v<T>)return T(T(1)-value);
    else if constexpr(std::is_unsigned_v<T>)return T(std::numeric_limits<T>::max()-value);
    else return T(T(-1)-value);
}
template<class T> const char* typed_guard(const MaxTreeImage& image,const MaxTreeParameters& parameters,MaxTreeProvided p) {
    const auto* values=static_cast<const T*>(image.data);
    if constexpr(std::is_floating_point_v<T>)
        for(std::size_t i=0;i<image.elements;++i)if(!std::isfinite(values[i]))return "max-tree input must be finite";
    if(!provided_tree(p))return nullptr;
    const bool inverted=closing(parameters.operation);
    auto value=[&](std::int64_t pixel) {return inverted?inverse(values[pixel]):values[pixel];};
    auto less=[&](std::int64_t left,std::int64_t right) {
        T a=value(left),b=value(right);return a<b || (!(b<a) && left<right);
    };
    for(std::size_t i=0;i<image.elements;++i) {
        const auto pixel=p.traverser.data[i];
        if(pixel<0 || std::uint64_t(pixel)>=image.elements)return "provided traverser index is out of bounds";
        if(i && !less(p.traverser.data[i-1],pixel))return "provided traverser must be the stable intensity permutation";
    }
    const auto root=p.traverser.data[0];
    if(p.parent.data[root]!=root)return "provided tree must have a self root";
    for(std::size_t i=0;i<image.elements;++i) {
        const auto q=p.parent.data[i];
        if(q<0 || std::uint64_t(q)>=image.elements)return "provided parent index is out of bounds";
        if(std::int64_t(i)!=root && !less(q,std::int64_t(i)))return "provided parent must precede its child";
        if(value(std::int64_t(i))<value(q))return "provided parent intensity exceeds child intensity";
    }
    return nullptr;
}

template<class F> int independent(std::size_t n,int requested,F body) {
    const int workers=n>=detail::parallel_min?requested:1;std::atomic<int> team{1};
    detail::partition(n,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
#ifdef _OPENMP
        if(id==0)team.store(omp_get_num_threads());
#else
        (void)id;
#endif
        body(begin,end);
    });
    return team.load();
}
void copy_words(const std::int64_t* source,std::int64_t* output,std::size_t n,bool vectorized) {
#ifdef SKIMAGEXX_AVX2
    if(vectorized && n>=4){simd::max_tree_copy(source,output,n*sizeof(std::int64_t));return;}
#else
    (void)vectorized;
#endif
    if(n)std::memcpy(output,source,n*sizeof(std::int64_t));
}
template<class T> void invert_range(const T* source,T* output,std::size_t n,MaxTreeType type,bool vectorized) {
#ifdef SKIMAGEXX_AVX2
    if(vectorized){simd::max_tree_invert(source,output,n,type);return;}
#else
    (void)vectorized;(void)type;
#endif
    for(std::size_t i=0;i<n;++i)output[i]=inverse(source[i]);
}
std::array<std::int64_t,3> coordinates(std::size_t pixel,const MaxTreeImage& image) {
    std::array<std::int64_t,3> result{};
    for(std::size_t k=image.rank;k-->0;){result[k]=std::int64_t(pixel%image.shape[k]);pixel/=image.shape[k];}
    return result;
}
struct Neighbor {std::array<int,3> delta;std::int64_t flat;int distance;};
struct Neighborhood {std::array<Neighbor,26> neighbors{};std::size_t size=0;};
Neighborhood neighborhood(const MaxTreeImage& image,int connectivity) {
    Neighborhood result;
    std::array<std::int64_t,3> strides{{0,0,1}};
    for(std::size_t k=image.rank;k-->0;)strides[k]=k+1==image.rank?1:strides[k+1]*std::int64_t(image.shape[k+1]);
    for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)
        for(int c=image.rank==3?-1:0;c<=(image.rank==3?1:0);++c) {
            const int distance=a*a+b*b+c*c;
            if(!distance || distance>connectivity)continue;
            std::array<int,3> delta{{a,b,c}};std::int64_t offset=0;
            for(std::size_t k=0;k<image.rank;++k)offset+=delta[k]*strides[k];
            Neighbor next{delta,offset,distance};auto position=result.size++;
            while(position && result.neighbors[position-1].distance>distance){result.neighbors[position]=result.neighbors[position-1];--position;}
            result.neighbors[position]=next;
        }
    return result;
}
template<class T> void build_tree(const T* values,const MaxTreeImage& image,int connectivity,
                                 const MaxTreeWorkspace& w,bool vectorized) {
    const auto n=image.elements;auto* parent=w.parent.data;auto* roots=w.roots.data;
    auto* order=w.traverser.data;auto* scratch=w.sort_scratch.data;
    if constexpr(std::is_same_v<T,std::uint8_t>) {
        // Ascending source indices make equal-value placement exactly stable.
        std::array<std::size_t,256> next{};
        for(std::size_t i=0;i<n;++i)++next[values[i]];
        std::size_t prefix=0;
        for(auto& count:next){const auto frequency=count;count=prefix;prefix+=frequency;}
        for(std::size_t i=0;i<n;++i)order[next[values[i]]++]=std::int64_t(i);
    }else if constexpr(std::is_same_v<T,std::uint16_t>) {
        // Bounded radix storage; both passes preserve the preceding tie order.
        std::array<std::size_t,256> low{},high{};
        for(std::size_t i=0;i<n;++i){++low[values[i]&255u];++high[values[i]>>8];}
        std::size_t low_prefix=0,high_prefix=0;
        for(std::size_t bin=0;bin<256;++bin) {
            const auto low_count=low[bin],high_count=high[bin];
            low[bin]=low_prefix;low_prefix+=low_count;
            high[bin]=high_prefix;high_prefix+=high_count;
        }
        for(std::size_t i=0;i<n;++i)scratch[low[values[i]&255u]++]=std::int64_t(i);
        for(std::size_t i=0;i<n;++i){const auto pixel=scratch[i];order[high[values[pixel]>>8]++]=pixel;}
    }else {
        auto less=[&](std::int64_t a,std::int64_t b) {return values[a]<values[b] || (!(values[b]<values[a]) && a<b);};
        auto* source=order;auto* target=scratch;
        for(std::size_t width=1;width<n;width*=2) {
            for(std::size_t begin=0;begin<n;begin+=2*width) {
                auto middle=std::min(begin+width,n),end=std::min(begin+2*width,n);
                auto left=begin,right=middle,out=begin;
                while(left<middle && right<end)target[out++]=less(source[right],source[left])?source[right++]:source[left++];
                copy_words(source+left,target+out,middle-left,vectorized);out+=middle-left;
                copy_words(source+right,target+out,end-right,vectorized);
            }
            std::swap(source,target);
        }
        if(source!=order)copy_words(source,order,n,vectorized);
    }
    const auto neighbors=neighborhood(image,connectivity);
    for(std::size_t index=n;index-->0;) {
        const auto p=order[index];parent[p]=roots[p]=p;
        const auto position=coordinates(std::size_t(p),image);
        bool interior=true;
        for(std::size_t k=0;k<image.rank;++k)interior=interior && position[k]>0 && std::size_t(position[k]+1)<image.shape[k];
        for(std::size_t k=0;k<neighbors.size;++k) {
            const auto& neighbor=neighbors.neighbors[k];bool valid=true;
            if(!interior)for(std::size_t d=0;d<image.rank;++d) {
                auto coordinate=position[d]+neighbor.delta[d];
                if(coordinate<0 || std::uint64_t(coordinate)>=image.shape[d]){valid=false;break;}
            }
            if(!valid)continue;
            const auto q=p+neighbor.flat;
            if(parent[q]<0)continue;
            // Advance as well as shorten the path. Only the internal scratch
            // forest changes; its representative and parent writes stay fixed.
            auto cursor=q;
            while(roots[cursor]!=roots[roots[cursor]]) {
                roots[cursor]=roots[roots[cursor]];
                cursor=roots[cursor];
            }
            const auto root=roots[cursor];
            if(root!=p){roots[root]=p;parent[root]=p;}
        }
    }
    for(std::size_t i=0;i<n;++i) {
        const auto p=order[i],q=parent[p];
        if(values[q]==values[parent[q]])parent[p]=parent[q];
    }
}

template<class T> MaxTreeResult process(const MaxTreeImage& image,const MaxTreeParameters& parameters,
    const MaxTreeWorkspace& w,MaxTreeBuffer output,MaxTreeProvided supplied,int workers) {
    const auto n=image.elements,rank=image.rank;const auto op=parameters.operation;
    const bool reuse=provided_tree(supplied),inverted=closing(op),extent=diameter(op),filter=filtered(op);
    const bool vectorized=has_avx2();const auto* input=static_cast<const T*>(image.data);
    auto* inverted_image=static_cast<T*>(w.inverted.data);
    auto* maxima=static_cast<std::uint64_t*>(output.data);
    const int team=independent(n,workers,[&](std::size_t begin,std::size_t end) {
        if(inverted)invert_range(input+begin,inverted_image+begin,end-begin,image.type,vectorized);
        for(auto i=begin;i<end;++i) {
            if(!reuse){w.parent.data[i]=-1;w.roots.data[i]=-1;w.traverser.data[i]=std::int64_t(i);}
            if(op==MaxTreeOperation::local_maxima)maxima[i]=1;
            if(filter)w.attribute.data[i]=1.0;
            if(extent) {
                const auto position=coordinates(i,image);
                for(std::size_t d=0;d<rank;++d)w.minimum_coordinates.data[i*rank+d]=w.maximum_coordinates.data[i*rank+d]=position[d];
            }
        }
    });
    const auto* values=inverted?inverted_image:input;
    if(!reuse)build_tree(values,image,parameters.connectivity,w,vectorized);
    const auto* parent=reuse?supplied.parent.data:w.parent.data;
    const auto* order=reuse?supplied.traverser.data:w.traverser.data;
    if(op==MaxTreeOperation::tree)return {false,team};
    const auto root=order[0];
    if(op==MaxTreeOperation::local_maxima) {
        std::uint64_t label=1;
        for(std::size_t i=n;i-->0;) {
            auto p=order[i];if(p==root)continue;auto q=parent[p];
            if(values[p]!=values[q]){maxima[q]=0;if(maxima[p]>0)maxima[p]=label++;}
        }
        for(std::size_t i=n;i-->0;) {
            auto p=order[i];if(p!=root && values[p]==values[parent[p]])maxima[p]=maxima[parent[p]];
        }
        return {reuse,team};
    }
    auto* attribute=w.attribute.data;
    for(std::size_t i=n;i-->0;) {
        auto p=order[i];if(p==root)continue;auto q=parent[p];
        if(!extent)attribute[q]=attribute[q]+attribute[p];
        else {
            std::int64_t longest=0;
            for(std::size_t d=0;d<rank;++d) {
                auto qi=std::size_t(q)*rank+d,pi=std::size_t(p)*rank+d;
                w.minimum_coordinates.data[qi]=std::min(w.minimum_coordinates.data[qi],w.minimum_coordinates.data[pi]);
                w.maximum_coordinates.data[qi]=std::max(w.maximum_coordinates.data[qi],w.maximum_coordinates.data[pi]);
                longest=std::max(longest,w.maximum_coordinates.data[qi]-w.minimum_coordinates.data[qi]);
            }
            attribute[q]=double(longest)+1.0;
        }
    }
    auto* result=static_cast<T*>(output.data);
    result[root]=attribute[root]<parameters.threshold?T(0):values[root];
    for(std::size_t i=0;i<n;++i) {
        auto p=order[i];if(p==root)continue;auto q=parent[p];
        result[p]=(values[p]==values[q] || attribute[p]<parameters.threshold)?result[q]:values[p];
    }
    if(inverted)independent(n,workers,[&](std::size_t begin,std::size_t end) {
        invert_range(result+begin,result+begin,end-begin,image.type,vectorized);
    });
    return {reuse,team};
}

#define MAX_TREE_TYPES(F) \
    F(u8,std::uint8_t) F(u16,std::uint16_t) F(u32,std::uint32_t) F(u64,std::uint64_t) \
    F(i8,std::int8_t) F(i16,std::int16_t) F(i32,std::int32_t) F(i64,std::int64_t) F(f32,float) F(f64,double)
}

const char* max_tree_ineligibility(const MaxTreeImage& image,const MaxTreeParameters& p,
    const MaxTreeWorkspace& w,MaxTreeBuffer output,MaxTreeProvided provided) noexcept {
    GuardEnvironment environment;
    if(!environment.valid || !fp_profile())return "max-tree requires nearest gradual floating point";
    if(image.rank!=2 && image.rank!=3)return "max-tree prepared rank must be 2 or 3";
    std::size_t n=1,coordinates_count=0;
    for(std::size_t d=0;d<image.rank;++d) {
        if(image.shape[d]<3 || image.shape[d]>std::size_t(std::numeric_limits<std::int32_t>::max()))return "max-tree shape must be int32 dimensions >=3";
        if(!multiply(n,image.shape[d],n))return "max-tree shape product overflows";
    }
    if(n!=image.elements || n>std::size_t(std::numeric_limits<std::int64_t>::max()) || !multiply(n,image.rank,coordinates_count))return "max-tree input element count is invalid";
    const bool reuse=provided_tree(provided);
    if(!reuse) {
        if(p.connectivity<1 || std::size_t(p.connectivity)>image.rank)return "max-tree connectivity must be in 1..rank";
        std::array<std::size_t,3> strides{{0,0,1}};
        for(std::size_t k=image.rank;k-->0;)strides[k]=k+1==image.rank?1:strides[k+1]*image.shape[k+1];
        std::size_t largest_offset=0;
        for(int d=0;d<p.connectivity;++d) {
            const auto stride=strides[std::size_t(d)];
            if(stride>std::size_t(std::numeric_limits<std::int32_t>::max())-largest_offset)return "max-tree raveled neighborhood overflows source int32";
            largest_offset+=stride;
        }
    }
    if(int(p.operation)<0 || int(p.operation)>int(MaxTreeOperation::diameter_closing))return "max-tree operation is invalid";
    std::size_t width=0;
    switch(image.type) {
#define WIDTH(name,type) case MaxTreeType::name:width=sizeof(type);break;
        MAX_TREE_TYPES(WIDTH)
#undef WIDTH
        default:return "max-tree dtype is unsupported";
    }
    std::array<Span,14> spans{};std::size_t used=0;
    if(!add_span(image.data,image.elements,n,width,spans,used))return "max-tree input span is invalid";
    if(reuse) {
        if(p.operation==MaxTreeOperation::tree)return "max-tree builder does not accept a provided tree";
        if(!add_span(provided.parent.data,provided.parent.elements,n,8,spans,used) ||
           !add_span(provided.traverser.data,provided.traverser.elements,n,8,spans,used))return "provided max-tree spans are invalid or overlap";
    } else {
        for(auto buffer:{w.parent,w.traverser,w.roots,w.sort_scratch})
            if(!add_span(buffer.data,buffer.elements,n,8,spans,used))return "max-tree index workspace spans are invalid or overlap";
    }
    if(p.operation==MaxTreeOperation::tree) {
        if(output.data || output.elements)return "max-tree builder output must be empty";
    } else if(!add_span(output.data,output.elements,n,p.operation==MaxTreeOperation::local_maxima?8:width,spans,used))return "max-tree output span is invalid or overlaps";
    if(closing(p.operation) && !add_span(w.inverted.data,w.inverted.elements,n,width,spans,used))return "max-tree inverted workspace is invalid or overlaps";
    if(filtered(p.operation) && !add_span(w.attribute.data,w.attribute.elements,n,8,spans,used))return "max-tree attribute workspace is invalid or overlaps";
    if(diameter(p.operation))
        for(auto buffer:{w.minimum_coordinates,w.maximum_coordinates})
            if(!add_span(buffer.data,buffer.elements,coordinates_count,8,spans,used))return "max-tree coordinate workspace is invalid or overlaps";
    switch(image.type) {
#define GUARD(name,type) case MaxTreeType::name:return typed_guard<type>(image,p,provided);
        MAX_TREE_TYPES(GUARD)
#undef GUARD
        default:return "max-tree dtype is unsupported";
    }
}
MaxTreeResult max_tree_into(const MaxTreeImage& image,const MaxTreeParameters& p,
    const MaxTreeWorkspace& w,MaxTreeBuffer output,MaxTreeProvided provided) {
    if(const auto* reason=max_tree_ineligibility(image,p,w,output,provided))throw std::invalid_argument(reason);
    const int workers=get_num_threads();
    switch(image.type) {
#define RUN(name,type) case MaxTreeType::name:return process<type>(image,p,w,output,provided,workers);
        MAX_TREE_TYPES(RUN)
#undef RUN
        default:throw std::invalid_argument("max-tree dtype is unsupported");
    }
}
#undef MAX_TREE_TYPES
} // namespace skimagexx
