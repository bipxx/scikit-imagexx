#include "max_tree_stages.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx::max_tree_stages {
namespace {
struct Environment {
    std::fenv_t saved{},held{};bool valid=false,stored=false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    unsigned csr=_mm_getcsr();
#endif
    Environment() {stored=std::fegetenv(&saved)==0;valid=stored && std::feholdexcept(&held)==0;}
    ~Environment() {
        if(stored)std::fesetenv(&saved);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        _mm_setcsr(csr);
#endif
    }
};
void check(bool valid,const char* reason) {if(!valid)throw std::invalid_argument(reason);}
struct Checked {
    struct Span {std::uintptr_t begin,end;};
    std::array<Span,10> spans{};std::size_t used=0;
    void add(const void* pointer,std::size_t count,std::size_t width,std::size_t expected) {
        check(pointer && count==expected && count && count<=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/width,"invalid max-tree stage span/count");
        auto begin=reinterpret_cast<std::uintptr_t>(pointer),bytes=count*width;
        check(begin%width==0 && bytes<=std::numeric_limits<std::uintptr_t>::max()-begin,"max-tree stage alignment/extent overflow");
        for(std::size_t k=0;k<used;++k)check(!(begin<spans[k].end && spans[k].begin<begin+bytes),"max-tree stage arrays must be disjoint");
        check(used<spans.size(),"too many max-tree stage spans");spans[used++]={begin,begin+bytes};
    }
};
#define TYPES(F) \
    F(u8,std::uint8_t) F(u16,std::uint16_t) F(u32,std::uint32_t) F(u64,std::uint64_t) \
    F(i8,std::int8_t) F(i16,std::int16_t) F(i32,std::int32_t) F(i64,std::int64_t) F(f32,float) F(f64,double)
std::size_t width(MaxTreeType type) {
    switch(type) {
#define W(name,T) case MaxTreeType::name:return sizeof(T);
        TYPES(W)
#undef W
        default:throw std::invalid_argument("unsupported max-tree stage dtype");
    }
}
template<class T> void finite(Image image) {
    if constexpr(std::is_floating_point_v<T>)
        for(std::size_t i=0;i<image.elements;++i)check(std::isfinite(static_cast<const T*>(image.data)[i]),"max-tree stage image must be finite");
}
void input(Checked& checked,Image image) {
    check(image.elements && image.elements<=std::size_t(std::numeric_limits<std::int64_t>::max()),"max-tree stage requires a nonempty signed64 input extent");
    checked.add(image.data,image.elements,width(image.type),image.elements);
    switch(image.type) {
#define FINITE(name,T) case MaxTreeType::name:finite<T>(image);break;
        TYPES(FINITE)
#undef FINITE
        default:throw std::invalid_argument("unsupported max-tree stage dtype");
    }
}
void tree(Checked& checked,Image image,Tree supplied) {
    checked.add(supplied.parent.data,supplied.parent.elements,8,image.elements);
    checked.add(supplied.traverser.data,supplied.traverser.elements,8,image.elements);
    for(std::size_t i=0;i<image.elements;++i) {
        const auto p=supplied.traverser.data[i],q=supplied.parent.data[i];
        check(p>=0 && std::uint64_t(p)<image.elements && q>=0 && std::uint64_t(q)<image.elements,"max-tree stage index out of bounds");
    }
}
std::size_t geometry(Image image,MaxTreeSpan<const std::int32_t> shape) {
    check(shape.elements==2 || shape.elements==3,"max-tree stage spatial rank must be2or3");
    std::size_t product=1;
    for(std::size_t d=0;d<shape.elements;++d) {
        check(shape.data[d]>0 && product<=std::numeric_limits<std::size_t>::max()/std::size_t(shape.data[d]),"max-tree stage shape overflow");
        product*=std::size_t(shape.data[d]);
    }
    check(product==image.elements,"max-tree stage shape product mismatch");return shape.elements;
}
void environment(Environment& value) {
    check(value.valid && std::fegetround()==FE_TONEAREST,"max-tree stages require nearest floating point");
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    check((_mm_getcsr() & ((1u<<15)|(1u<<6)|(3u<<13)))==0,"max-tree stages require gradual floating point");
#endif
}
void coordinate_span(Checked& checked,Image image,Coordinates value,std::size_t rank) {
    check(rank==value.rank && image.elements<=std::numeric_limits<std::size_t>::max()/rank,"max-tree coordinate count overflow");
    checked.add(value.data,value.elements,8,image.elements*rank);
    check((value.row_stride==rank && value.column_stride==1) ||
          (value.row_stride==1 && value.column_stride==image.elements),"max-tree coordinates require denseC/F(N,rank)");
}
std::array<std::int64_t,3> position(std::size_t pixel,MaxTreeSpan<const std::int32_t> shape) {
    std::array<std::int64_t,3> result{};
    for(std::size_t d=shape.elements;d-->0;){result[d]=std::int64_t(pixel%std::size_t(shape.data[d]));pixel/=std::size_t(shape.data[d]);}
    return result;
}
template<class T> void build_body(Image image,MaxTreeSpan<const std::uint8_t> mask,
    MaxTreeSpan<const std::int32_t> structure,MaxTreeSpan<const std::int32_t> shape,
    MaxTreeSpan<const std::int32_t> points,MaxTreeSpan<std::int64_t> parent,
    MaxTreeSpan<const std::int64_t> order,MaxTreeSpan<std::int64_t> roots) {
    const auto* values=static_cast<const T*>(image.data);
    std::fill_n(parent.data,image.elements,-1);std::fill_n(roots.data,image.elements,-1);
    for(std::size_t i=image.elements;i-->0;) {
        const auto p=order.data[i];parent.data[p]=roots.data[p]=p;
        const auto coords=position(std::size_t(p),shape);
        for(std::size_t k=0;k<structure.elements;++k) {
            if(!mask.data[p]) {
                bool valid=true;
                for(std::size_t d=0;d<shape.elements;++d) {
                    auto c=coords[d]+points.data[k*shape.elements+d];
                    if(c<0 || c>=shape.data[d]){valid=false;break;}
                }
                if(!valid)continue;
            }
            const auto q=p+structure.data[k];if(parent.data[q]<0)continue;
            while(roots.data[q]!=roots.data[roots.data[q]])roots.data[q]=roots.data[roots.data[q]];
            const auto root=roots.data[q];if(root!=p){roots.data[root]=p;parent.data[root]=p;}
        }
    }
    for(std::size_t i=0;i<image.elements;++i) {
        const auto p=order.data[i],q=parent.data[p];
        if(values[q]==values[parent.data[q]])parent.data[p]=parent.data[q];
    }
}
template<class T> void filter_body(Image image,MaxTreeBuffer output,Tree supplied,
    MaxTreeSpan<const double> attribute,double threshold) {
    const auto* values=static_cast<const T*>(image.data);auto* result=static_cast<T*>(output.data);
    const auto root=supplied.traverser.data[0];result[root]=attribute.data[root]<threshold?T(0):values[root];
    for(std::size_t i=0;i<image.elements;++i) {
        const auto p=supplied.traverser.data[i];if(p==root)continue;const auto q=supplied.parent.data[p];
        result[p]=(values[p]==values[q] || attribute.data[p]<threshold)?result[q]:values[p];
    }
}
template<class T> void maxima_body(Image image,MaxTreeSpan<std::uint64_t> output,Tree supplied) {
    const auto* values=static_cast<const T*>(image.data);const auto root=supplied.traverser.data[0];std::uint64_t label=1;
    for(std::size_t i=0;i<image.elements;++i) {
        const auto p=supplied.traverser.data[image.elements-1-i];if(p==root)continue;const auto q=supplied.parent.data[p];
        if(values[p]!=values[q]){output.data[q]=0;if(output.data[p]>0)output.data[p]=label++;}
    }
    for(std::size_t i=image.elements;i-->0;) {
        const auto p=supplied.traverser.data[i];if(p!=root && values[p]==values[supplied.parent.data[p]])output.data[p]=output.data[supplied.parent.data[p]];
    }
}
}

void build(Image image,MaxTreeSpan<const std::uint8_t> mask,MaxTreeSpan<const std::int32_t> structure,
    MaxTreeSpan<const std::int32_t> shape,MaxTreeSpan<const std::int32_t> points,
    MaxTreeSpan<std::int64_t> parent,MaxTreeSpan<const std::int64_t> order,MaxTreeSpan<std::int64_t> roots) {
    {
        Environment held;environment(held);Checked checked;input(checked,image);
        checked.add(shape.data,shape.elements,4,shape.elements);const auto rank=geometry(image,shape);
        for(std::size_t d=0;d<rank;++d)check(shape.data[d]>=3,"max-tree builder shape must be>=3");
        check(structure.elements && structure.elements<=26,"max-tree stage radius-one neighbor count invalid");
        checked.add(mask.data,mask.elements,1,image.elements);checked.add(structure.data,structure.elements,4,structure.elements);
        checked.add(points.data,points.elements,4,structure.elements*rank);
        checked.add(parent.data,parent.elements,8,image.elements);checked.add(order.data,order.elements,8,image.elements);checked.add(roots.data,roots.elements,8,image.elements);
        std::array<std::int64_t,3> strides{{0,0,1}};
        for(std::size_t d=rank;d-->0;)strides[d]=d+1==rank?1:strides[d+1]*shape.data[d+1];
        for(std::size_t k=0;k<structure.elements;++k) {
            std::int64_t flat=0;
            for(std::size_t d=0;d<rank;++d){auto delta=points.data[k*rank+d];check(delta>=-1 && delta<=1,"max-tree stage supports radius-one points only");flat+=delta*strides[d];}
            check(flat==structure.data[k],"max-tree stage offset/point mismatch");
        }
        std::vector<unsigned char> seen(image.elements,0);
        for(std::size_t i=0;i<image.elements;++i) {
            const auto p=order.data[i];check(p>=0 && std::uint64_t(p)<image.elements && !seen[std::size_t(p)],"max-tree builder requires a bounded permutation");seen[std::size_t(p)]=1;
            if(mask.data[i]) {
                const auto coords=position(i,shape);
                for(std::size_t k=0;k<structure.elements;++k)for(std::size_t d=0;d<rank;++d) {
                    const auto c=coords[d]+points.data[k*rank+d];check(c>=0 && c<shape.data[d],"max-tree mask does not protect its boundary");
                }
            }
        }
    }
    switch(image.type) {
#define BUILD(name,T) case MaxTreeType::name:build_body<T>(image,mask,structure,shape,points,parent,order,roots);return;
        TYPES(BUILD)
#undef BUILD
        default:throw std::invalid_argument("unsupported max-tree stage dtype");
    }
}
void area(Image image,Tree supplied,MaxTreeSpan<double> attribute) {
    {Environment held;environment(held);Checked c;input(c,image);tree(c,image,supplied);c.add(attribute.data,attribute.elements,8,image.elements);}
    const auto root=supplied.traverser.data[0];
    for(std::size_t i=image.elements;i-->0;) {
        const auto p=supplied.traverser.data[i];if(p==root)continue;const auto q=supplied.parent.data[p];attribute.data[q]=attribute.data[q]+attribute.data[p];
    }
}
void extension(Image image,MaxTreeSpan<const std::int32_t> shape,Tree supplied,MaxTreeSpan<double> attribute,Coordinates low,Coordinates high) {
    {Environment held;environment(held);Checked c;input(c,image);tree(c,image,supplied);c.add(shape.data,shape.elements,4,shape.elements);auto rank=geometry(image,shape);
     c.add(attribute.data,attribute.elements,8,image.elements);coordinate_span(c,image,low,rank);coordinate_span(c,image,high,rank);}
    const auto root=supplied.traverser.data[0];
    for(std::size_t i=image.elements;i-->0;) {
        const auto p=supplied.traverser.data[i];if(p==root)continue;const auto q=supplied.parent.data[p];double longest=0;
        for(std::size_t d=0;d<shape.elements;++d) {
            auto pi=std::size_t(p)*low.row_stride+d*low.column_stride,qi=std::size_t(q)*low.row_stride+d*low.column_stride;
            auto ph=std::size_t(p)*high.row_stride+d*high.column_stride,qh=std::size_t(q)*high.row_stride+d*high.column_stride;
            // numpy maximum/minimum propagate NaN and select the second zero
            // on ties; default coordinate providers are finite exact integers.
            high.data[qh]=(high.data[ph]>=high.data[qh] || std::isnan(high.data[ph]))?high.data[ph]:high.data[qh];
            low.data[qi]=(low.data[pi]<=low.data[qi] || std::isnan(low.data[pi]))?low.data[pi]:low.data[qi];
            const double delta=high.data[qh]-low.data[qi];
            longest=d==0?delta:(delta>=longest || std::isnan(delta))?delta:longest;
        }
        attribute.data[q]=longest+1;
    }
}
void direct_filter(Image image,MaxTreeBuffer output,Tree supplied,MaxTreeSpan<const double> attribute,double threshold) {
    {Environment held;environment(held);Checked c;input(c,image);tree(c,image,supplied);c.add(output.data,output.elements,width(image.type),image.elements);c.add(attribute.data,attribute.elements,8,image.elements);}
    switch(image.type) {
#define FILTER(name,T) case MaxTreeType::name:filter_body<T>(image,output,supplied,attribute,threshold);return;
        TYPES(FILTER)
#undef FILTER
        default:throw std::invalid_argument("unsupported max-tree stage dtype");
    }
}
void local_maxima(Image image,MaxTreeSpan<std::uint64_t> output,Tree supplied) {
    {Environment held;environment(held);Checked c;input(c,image);tree(c,image,supplied);c.add(output.data,output.elements,8,image.elements);}
    switch(image.type) {
#define MAXIMA(name,T) case MaxTreeType::name:maxima_body<T>(image,output,supplied);return;
        TYPES(MAXIMA)
#undef MAXIMA
        default:throw std::invalid_argument("unsupported max-tree stage dtype");
    }
}
#undef TYPES
} // namespace skimagexx::max_tree_stages
