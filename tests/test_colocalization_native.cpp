#include "skimagexx/colocalization.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace skimagexx;
std::uint64_t checks=0;
void require(bool condition){++checks;if(!condition)throw std::runtime_error("colocalization native control failed at check "+std::to_string(checks));}
template<class Call>void reject(const Call& call){bool caught=false;try{call();}catch(const std::invalid_argument&){caught=true;}require(caught);}
template<class T>std::uint64_t raw(T value){std::uint64_t out=0;std::memcpy(&out,&value,sizeof(value));return out;}
struct One {
    std::size_t length;
    std::ptrdiff_t stride;
    ColocView value;
    One(const void* data,std::size_t n,ColocType type,std::size_t item,bool swap=false):length(n),stride(std::ptrdiff_t(item)),
        value{data,data,n*item,{&length,1},{&stride,1},type,swap}{}
};
void bool_counts() {
    const std::array<std::size_t,10> lengths{0,1,7,31,32,33,63,64,65,262151};
    for(auto n:lengths) {
        std::vector<unsigned char> a(n+2,0),b(n+2,0);
        std::uint64_t expected=0,joint=0;
        for(std::size_t i=0;i<n;++i) {
            a[i+1]=i%3==0 ? 0:i%3==1 ? 2:255;
            b[i+1]=i%5==0 ? 0:128;
            expected+=i%3!=0;joint+=(i%3!=0 && i%5!=0);
        }
        One first(a.data()+1,n,ColocType::boolean,1),second(b.data()+1,n,ColocType::boolean,1);
        first.value.allocation=a.data();first.value.allocation_bytes=a.size();
        second.value.allocation=b.data();second.value.allocation_bytes=b.size();
        require(coloc_count_nonzero(first.value)==expected);
        require(coloc_joint_count(first.value,second.value)==joint);
        // Negative stride visits the exact same bounded bytes; truth remains
        // nonzero even when the two raw bool bit patterns have no shared bit.
        if(n){first.value.data=a.data()+n;first.stride=-1;require(coloc_count_nonzero(first.value)==expected);}
    }
}
void byte_binary_conversion() {
    for(auto type:{ColocType::uint8,ColocType::int8})for(unsigned value=0;value<256;++value) {
        const auto byte=static_cast<unsigned char>(value);One view(&byte,1,type,1);
        require(coloc_binary_valid(view.value).value==(value<=1));
    }
    One empty(nullptr,0,ColocType::uint8,1);
    require(coloc_binary_valid(empty.value).value);
    require(coloc_binary_into(empty.value,coloc_keep_order(empty.value),{nullptr,0}).worker_limit==1);
    const std::array<unsigned char,6> logical{0,1,1,0,1,0};
    const std::array<unsigned char,6> fortran_order{0,0,1,1,1,0};
    for(auto type:{ColocType::uint8,ColocType::int8})for(int layout=0;layout<4;++layout) {
        std::array<std::size_t,2> shape{2,3};
        const std::array<std::ptrdiff_t,2> strides=layout==1 ? std::array<std::ptrdiff_t,2>{1,2}:
            layout==2 ? std::array<std::ptrdiff_t,2>{3,-1}:layout==3 ? std::array<std::ptrdiff_t,2>{6,2}:std::array<std::ptrdiff_t,2>{3,1};
        std::array<unsigned char,16> input;input.fill(0xa5);
        auto* first=input.data()+(layout==2 ? 3:1);
        for(std::size_t i=0;i<6;++i)first[std::ptrdiff_t(i/3)*strides[0]+std::ptrdiff_t(i%3)*strides[1]]=logical[i];
        const auto before=input;
        ColocView view{first,input.data(),input.size(),{shape.data(),2},{strides.data(),2},type,false};
        require(coloc_binary_valid(view).value);
        std::array<unsigned char,8> out;out.fill(0xa5);
        ColocOrder c;c.rank=2;c.axes[0]=1;c.axes[1]=0;
        coloc_binary_into(view,c,{out.data()+1,6});
        require(std::memcmp(out.data()+1,logical.data(),6)==0);
        coloc_binary_into(view,coloc_keep_order(view),{out.data()+1,6});
        require(std::memcmp(out.data()+1,(layout==1 ? fortran_order:logical).data(),6)==0);
        require(out.front()==0xa5 && out.back()==0xa5 && input==before);
    }
    const std::int8_t invalid[]{0,1,0,1,0,-1};One signed_view(invalid,6,ColocType::int8,1);
    std::array<unsigned char,8> untouched;untouched.fill(0x6d);const auto before=untouched;
    require(!coloc_binary_valid(signed_view.value).value);
    bool refused=false;
    try{coloc_binary_into(signed_view.value,coloc_keep_order(signed_view.value),{untouched.data()+1,6});}
    catch(const std::domain_error&){refused=true;}
    require(refused && untouched==before); // late negative must refuse BEFORE any store
    const unsigned char noncanonical[]{2,0,255,128};One boolean(noncanonical,4,ColocType::boolean,1);
    std::array<unsigned char,4> normalized{};
    require(coloc_binary_valid(boolean.value).value);
    coloc_binary_into(boolean.value,coloc_keep_order(boolean.value),{normalized.data(),4});
    require((normalized==std::array<unsigned char,4>{1,0,1,1})); // bool is not raw-copy eligible
    for(const std::size_t n:{1,31,32,33,262151}) {
        std::vector<unsigned char> input(n),out(n+2,0xa5);
        for(std::size_t i=0;i<n;++i)input[i]=i%7==0;
        One view(input.data(),n,ColocType::uint8,1);
        require(coloc_binary_valid(view.value).value);
        const auto execution=coloc_binary_into(view.value,coloc_keep_order(view.value),{out.data()+1,n});
        require(std::memcmp(out.data()+1,input.data(),n)==0 && out.front()==0xa5 && out.back()==0xa5);
        require(execution.worker_limit==(n>=262144 ? get_num_threads():1));
        input.back()=2;out.assign(n+2,0xa5);refused=false;
        require(!coloc_binary_valid(view.value).value);
        try{coloc_binary_into(view.value,coloc_keep_order(view.value),{out.data()+1,n});}
        catch(const std::domain_error&){refused=true;}
        require(refused && out==std::vector<unsigned char>(n+2,0xa5));
    }
}
void finite_float32_bits() {
    const struct {std::uint32_t word;bool finite;} cases[]={
        {0,true},{0x80000000u,true},{1,true},{0x80000001u,true},
        {0x007fffffu,true},{0x807fffffu,true},{0x00800000u,true},{0x80800000u,true},
        {0x7f7fffffu,true},{0xff7fffffu,true},{0x7f800000u,false},{0xff800000u,false},
        {0x7fc12345u,false},{0xffc12345u,false},{0x7f800001u,false},{0xff800001u,false}};
    std::fenv_t saved;require(std::fegetenv(&saved)==0);
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD}) {
        require(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        for(int layout=0;layout<5;++layout)for(const std::size_t n:{0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,129})for(const auto c:cases) {
            // Aligned, unaligned, swapped, reverse, and step-two storage. The
            // last logical value exercises every tail; padding stays poisoned.
            std::vector<std::uint32_t> storage(2*n+4,0xa5a5a5a5u);
            auto* bytes=reinterpret_cast<unsigned char*>(storage.data());
            auto* first=bytes+(layout==0 ? 4:1)+(layout==3 && n ? 4*(n-1):0);
            const std::ptrdiff_t step=layout==3 ? -4:layout==4 ? 8:4;
            for(std::size_t i=0;i<n;++i) {
                const std::uint32_t word=i+1==n ? c.word:0x3f800000u;
                unsigned char encoded[4];std::memcpy(encoded,&word,4);
                for(std::size_t j=0;j<4;++j)first[std::ptrdiff_t(i)*step+std::ptrdiff_t(j)]=encoded[layout==2 ? 3-j:j];
            }
            const auto before=storage;
            One view(n ? first:nullptr,n,ColocType::float32,4,layout==2);view.stride=step;
            view.value.allocation=n ? storage.data():nullptr;view.value.allocation_bytes=n ? storage.size()*4:0;
            require(coloc_all_finite(view.value)==(!n || c.finite));
            require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO && std::fegetround()==rounding);
            require(storage==before);
            if(n && !c.finite && rounding==FE_TONEAREST) {
                std::vector<unsigned char> output(n*4+2,0x6d);const auto untouched=output;
                unsigned flags=99;bool refused=false;
                try{coloc_square_into(view.value,coloc_keep_order(view.value),{output.data()+1,n*4},flags);}
                catch(const ColocProviderRequired&){refused=true;}
                require(refused && output==untouched && flags==99 && storage==before);
                require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO && std::fegetround()==rounding);
            }
        }
    }
    require(std::fesetenv(&saved)==0);
}
void integer_products() {
    struct Kind{ColocType type;unsigned bytes;bool signed_value;};
    const Kind kinds[]={{ColocType::uint8,1,false},{ColocType::int8,1,true},
        {ColocType::uint16,2,false},{ColocType::int16,2,true},
        {ColocType::uint32,4,false},{ColocType::int32,4,true},
        {ColocType::uint64,8,false},{ColocType::int64,8,true}};
    for(const auto kind:kinds) {
        std::array<unsigned char,25> storage{};auto* input=storage.data()+1;
        // Three little/native raw representations: -1/max, 2, 0. The first
        // square wraps to1 in EVERY width; the promoted sum is signed1 or the
        // unsigned modular1. These expected values do not copy the algorithm.
        std::memset(input,0xff,kind.bytes);
        if(kind.bytes==1){const std::uint8_t x=2;std::memcpy(input+kind.bytes,&x,1);}
        else if(kind.bytes==2){const std::uint16_t x=2;std::memcpy(input+kind.bytes,&x,2);}
        else if(kind.bytes==4){const std::uint32_t x=2;std::memcpy(input+kind.bytes,&x,4);}
        else{const std::uint64_t x=2;std::memcpy(input+kind.bytes,&x,8);}
        One view(input,3,kind.type,kind.bytes);view.value.allocation=storage.data();view.value.allocation_bytes=storage.size();
        std::array<unsigned char,24> out{};unsigned status=99;
        coloc_square_into(view.value,coloc_keep_order(view.value),{out.data(),3*kind.bytes},status);
        require(status==0);
        for(unsigned i=0;i<kind.bytes;++i){require(out[i]==(i==0 ? 1:0));require(out[kind.bytes+i]==(i==0 ? 4:0));require(out[2*kind.bytes+i]==0);}
        auto sum=coloc_sum(view.value,{});
        require(sum.type==(kind.signed_value ? ColocType::int64:ColocType::uint64));
        const std::uint64_t maximum=kind.bytes==8 ? ~std::uint64_t(0):(std::uint64_t(1)<<(kind.bytes*8))-1;
        require(sum.bits==(kind.signed_value ? 1:maximum+2));
    }
    const unsigned char bools[]={2,255,0,128};One boolean(bools,4,ColocType::boolean,1);
    std::array<unsigned char,4> squared{};unsigned status=99;
    coloc_square_into(boolean.value,coloc_keep_order(boolean.value),{squared.data(),squared.size()},status);
    require(coloc_square_type(ColocType::boolean)==ColocType::int8);
    require((squared==std::array<unsigned char,4>{1,1,0,1}) && status==0);
}
void swapped_signed_boolean_products() {
    // Negative values survive a boolean product. Squares alone cannot detect
    // losing the sign bit, and byte-swapped 64-bit reads must not narrow.
    for(const auto type:{ColocType::int32,ColocType::int64}) {
        const std::size_t item=type==ColocType::int32 ? 4:8;
        std::array<unsigned char,26> storage;storage.fill(0xa5);
        const std::array<std::uint64_t,3> words{~std::uint64_t(2),2,0}; // -3,2,0
        for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<item;++j)
            storage[1+i*item+j]=static_cast<unsigned char>(words[i]>>(8*(item-1-j)));
        const auto before=storage;
        One a(storage.data()+1,3,type,item,true);
        a.value.allocation=storage.data();a.value.allocation_bytes=storage.size();
        const unsigned char masks[]{255,2,0};One mask_view(masks,3,ColocType::boolean,1);
        require(coloc_sum(a.value,{}).bits==~std::uint64_t(0));
        require(coloc_min_negative(a.value).value);
        std::array<unsigned char,26> output;output.fill(0xa5);unsigned status=99;
        coloc_multiply_into(a.value,mask_view.value,coloc_keep_order(a.value,mask_view.value),{output.data()+1,3*item},status);
        require(status==0);
        for(std::size_t i=0;i<3;++i) {
            std::uint64_t actual=0;std::memcpy(&actual,output.data()+1+i*item,item);
            const auto expected=item==8 ? words[i]:words[i]&0xffffffffULL;
            require(actual==expected);
        }
        require(output.front()==0xa5 && output[1+3*item]==0xa5 && storage==before);
    }
}
void gather_and_orders() {
    const std::uint16_t a[]={0x0102,0x0304,0x0506,0x0708};const unsigned char mask[]={0,2,255,0};
    One image(a,4,ColocType::uint16,2,true),selection(mask,4,ColocType::boolean,1);
    std::uint16_t out[2]={0,0};
    coloc_gather_into(image.value,selection.value,{out,sizeof(out)},2);
    require(std::memcmp(out,a+1,sizeof(out))==0); // raw endian bytes, no conversion
    std::array<std::size_t,2> shape{2,3};std::array<std::ptrdiff_t,2> first{4,8},second{12,4};
    std::array<float,6> values{1,2,3,4,5,6};
    ColocView f{values.data(),values.data(),sizeof(values),{shape.data(),2},{first.data(),2},ColocType::float32,false};
    ColocView c=f;c.strides.data=second.data();
    require(coloc_keep_order(f).axes[0]==0);
    require(coloc_keep_order(c).axes[0]==1);
    require(coloc_keep_order(f,c).axes[0]==1); // conflict -> C priority
    require(coloc_sum(f,{}).bits==raw(21.0f));require(coloc_sum(c,{}).bits==raw(21.0f));
    first={4,-8};f.data=values.data()+4;
    require(!coloc_sum_supported(f,{})); // no guessed cross-row association
}
void floating_association() {
    const struct {std::size_t length;float expected;} cases[]={{7,16777216.0f},{8,16777222.0f},{9,16777224.0f},{128,16777328.0f},{129,16777336.0f}};
    for(const auto c:cases) {
        std::vector<float> values(c.length,1);values[0]=16777216.0f;
        One view(values.data(),values.size(),ColocType::float32,4);
        for(const auto profile:{ColocProfile::numpy_1244,ColocProfile::numpy_254}) {
            const auto sum=coloc_sum(view.value,{profile,8192});require(sum.bits==raw(c.expected));require(sum.fp_status==0);
        }
    }
    std::array<double,8> d{9007199254740992.0,1,1,1,1,1,1,1};One dv(d.data(),8,ColocType::float64,8);
    require(coloc_sum(dv.value,{}).bits==raw(9007199254740998.0));
    std::array<float,8> x{4096,1,1,1,1,1,1,1},square{};One xv(x.data(),8,ColocType::float32,4);unsigned status=0;
    coloc_square_into(xv.value,coloc_keep_order(xv.value),{square.data(),sizeof(square)},status);
    One sv(square.data(),8,ColocType::float32,4);require(coloc_sum(sv.value,{}).bits==raw(16777222.0f));
    const std::array<std::uint16_t,8> h{0x6800,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00,0x3c00};One hv(h.data(),8,ColocType::float16,2);
    require(coloc_sum(hv.value,{}).bits==0x6804);
    const std::uint16_t tiny=1;One tv(&tiny,1,ColocType::float16,2);std::uint16_t result=42;
    coloc_square_into(tv.value,coloc_keep_order(tv.value),{&result,sizeof(result)},status);
    require(result==0 && status==coloc_underflow);
    const auto saved=std::fegetround();require(std::fesetround(FE_DOWNWARD)==0);
    require(!coloc_sum_supported(dv.value,{}));require(std::fegetround()==FE_DOWNWARD);require(std::fesetround(saved)==0);
    std::feraiseexcept(FE_DIVBYZERO);const auto before=std::fetestexcept(FE_ALL_EXCEPT);
    require(coloc_sum(dv.value,{}).fp_status==0);require(std::fetestexcept(FE_ALL_EXCEPT)==before);std::feclearexcept(FE_ALL_EXCEPT);
}
// The same twelve logical sites exercise address selection independently of
// physical order. Poisoned padding and output canaries expose accidental dense
// access; the second operand/mask always uses a different, logical C layout.
void multidimensional_addresses() {
    for(const std::size_t rank:{2,3})for(int layout=0;layout<6;++layout) {
        std::array<std::size_t,3> shape=rank==2 ? std::array<std::size_t,3>{3,4,1}:std::array<std::size_t,3>{2,2,3};
        std::array<std::ptrdiff_t,3> strides{};
        if(rank==2)strides={8,2,0};else strides={12,6,2};
        if(layout==1){if(rank==2)strides={2,6,0};else strides={2,4,8};} // F
        if(layout==2){if(rank==2)strides={14,2,0};else strides={24,8,2};} // padded
        if(layout==3)strides[rank-1]=-2; // only the last axis reverses
        if(layout==4)for(std::size_t d=0;d+1<rank;++d)strides[d]=0; // broadcast
        const bool swapped=layout==5;
        std::array<unsigned char,96> storage;storage.fill(0xa5);
        auto* origin=storage.data()+1+(layout==3 ? 2*(shape[rank-1]-1):0);
        std::array<std::uint16_t,12> rhs{},expected{};
        std::array<unsigned char,12> mask{};
        for(std::size_t i=0;i<12;++i) {
            const auto x=std::uint16_t(layout==4 ? i%shape[rank-1]+1:i+1);
            expected[i]=x;rhs[i]=std::uint16_t(i+2);mask[i]=i%3 ? (i%2 ? 2:255):0;
            std::size_t q=i;std::ptrdiff_t offset=0;
            for(std::size_t d=rank;d-->0;){offset+=std::ptrdiff_t(q%shape[d])*strides[d];q/=shape[d];}
            const auto word=std::uint16_t(swapped ? (x<<8)|(x>>8):x);
            std::memcpy(origin+offset,&word,2);
        }
        ColocView a{origin,storage.data(),storage.size(),{shape.data(),rank},{strides.data(),rank},ColocType::uint16,swapped};
        auto truth=a;truth.type=ColocType::boolean; // each low byte is nonzero (including swapped storage)
        // Swapped uint16 low bytes are zero, so bool truth uses the other byte.
        if(swapped)truth.data=origin+1;
        std::array<std::ptrdiff_t,3> c=rank==2 ? std::array<std::ptrdiff_t,3>{8,2,0}:std::array<std::ptrdiff_t,3>{12,6,2};
        auto b=a;b.data=rhs.data();b.allocation=rhs.data();b.allocation_bytes=sizeof(rhs);b.strides.data=c.data();b.swapped=false;
        auto selection=b;selection.data=mask.data();selection.allocation=mask.data();selection.allocation_bytes=sizeof(mask);selection.type=ColocType::boolean;
        auto mc=c;for(auto& s:mc)s/=2;selection.strides.data=mc.data();
        const auto before=storage;
        require(coloc_count_nonzero(truth)==12);require(coloc_joint_count(truth,selection)==8);
        std::array<std::uint16_t,10> gathered;gathered.fill(0xbeef);
        coloc_gather_into(a,selection,{gathered.data()+1,16},8);
        std::size_t selected=1;
        for(std::size_t i=0;i<12;++i)if(mask[i]) {
            const auto x=expected[i];require(gathered[selected++]==std::uint16_t(swapped ? (x<<8)|(x>>8):x));
        }
        require(gathered.front()==0xbeef && gathered.back()==0xbeef);
        ColocOrder order;order.rank=rank;for(std::size_t d=0;d<rank;++d)order.axes[d]=rank-1-d;
        std::array<std::uint16_t,14> product;product.fill(0xbeef);unsigned flags=99;
        coloc_multiply_into(a,b,order,{product.data()+1,24},flags);require(flags==0);
        for(std::size_t i=0;i<12;++i)require(product[i+1]==expected[i]*rhs[i]);
        require(product.front()==0xbeef && product.back()==0xbeef);
        coloc_square_into(a,order,{product.data()+1,24},flags);require(flags==0);
        for(std::size_t i=0;i<12;++i)require(product[i+1]==expected[i]*expected[i]);
        require(storage==before);
    }
}
void affine_reduction_edges() {
    // Exact dyadics isolate missing/repeated sites at iterator buffer boundaries;
    // the original 128/129 cancellation-sensitive literals remain above.
    for(const std::size_t n:{128,129,8191,8192,8193})for(bool fortran:{false,true}) {
        std::array<std::size_t,3> shape{1,1,n};
        if(n==128 || n==8192)shape={2,2,n/4};
        else if(n==129 || n==8193)shape={1,3,n/3};
        std::array<std::ptrdiff_t,3> fs=fortran ? std::array<std::ptrdiff_t,3>{4,4*std::ptrdiff_t(shape[0]),4*std::ptrdiff_t(shape[0]*shape[1])}:std::array<std::ptrdiff_t,3>{4*std::ptrdiff_t(shape[1]*shape[2]),4*std::ptrdiff_t(shape[2]),4};
        std::vector<float> f(n,0.5f);std::vector<double> d(n,0.5);
        ColocView fv{f.data(),f.data(),n*4,{shape.data(),3},{fs.data(),3},ColocType::float32,false};
        auto ds=fs;for(auto& s:ds)s*=2;
        auto dv=fv;dv.data=d.data();dv.allocation=d.data();dv.allocation_bytes=n*8;dv.strides.data=ds.data();dv.type=ColocType::float64;
        for(auto profile:{ColocProfile::numpy_1244,ColocProfile::numpy_254})for(std::size_t buffer:{128,8192}) {
            const auto a=coloc_sum(fv,{profile,buffer}),b=coloc_sum(dv,{profile,buffer});
            require(a.bits==raw(float(n)*0.5f) && a.fp_status==0);
            require(b.bits==raw(double(n)*0.5) && b.fp_status==0);
        }
        // Reverse every non-singleton axis: still affine, with a negative step.
        fv.data=f.data()+n-1;dv.data=d.data()+n-1;
        for(auto& s:fs)s=-s;for(auto& s:ds)s=-s;
        require(coloc_sum(fv,{}).bits==raw(float(n)*0.5f));
        require(coloc_sum(dv,{}).bits==raw(double(n)*0.5));
        for(auto& s:fs)s=-s;for(auto& s:ds)s=-s;fv.data=f.data();dv.data=d.data();
        if(n==129 || n==8193) {
            std::vector<unsigned char> bytes(n*4+2,0xa5);
            const std::uint32_t swapped_half=0x0000003f;
            for(std::size_t i=0;i<n;++i)std::memcpy(bytes.data()+1+4*i,&swapped_half,4);
            auto unaligned=fv;unaligned.data=bytes.data()+1;unaligned.allocation=bytes.data();unaligned.allocation_bytes=bytes.size();unaligned.swapped=true;
            for(auto profile:{ColocProfile::numpy_1244,ColocProfile::numpy_254}) {
                const auto sum=coloc_sum(unaligned,{profile,128});
                require(sum.bits==raw(float(n)*0.5f) && sum.fp_status==0);
            }
            require(bytes.front()==0xa5 && bytes.back()==0xa5);
        }
        // A padded row must not be silently admitted as an affine float sum.
        if(!fortran && n==128) {
            f.resize(n+4,0.5f);fv.data=f.data();fv.allocation=f.data();fv.allocation_bytes=f.size()*4;
            fs[1]+=4;fs[0]+=8;require(!coloc_sum_supported(fv,{}));
        }
    }
}
void reject_before_write() {
    const double one=1;One input(&one,1,ColocType::float64,8);const auto order=coloc_keep_order(input.value);
    std::array<unsigned char,sizeof(ColocView)> before{};std::memcpy(before.data(),&input.value,sizeof(input.value));
    reject([&]{unsigned flags=0;coloc_square_into(input.value,order,{&input.value.allocation_bytes,8},flags);});
    require(std::memcmp(&input.value,before.data(),before.size())==0);
    const auto shape_before=input.length;
    reject([&]{unsigned flags=0;coloc_square_into(input.value,order,{&input.length,8},flags);});require(input.length==shape_before);
    auto aliased_order=order;const auto order_before=aliased_order;
    reject([&]{unsigned flags=0;coloc_square_into(input.value,aliased_order,{aliased_order.axes.data(),8},flags);});
    require(std::memcmp(&aliased_order,&order_before,sizeof(order_before))==0);
    std::uint32_t integer=7,out=0xabcdef01;One iv(&integer,1,ColocType::uint32,4);
    reject([&]{coloc_square_into(iv.value,coloc_keep_order(iv.value),{&out,4},integer);});require(integer==7 && out==0xabcdef01);
    input.length=2;double untouched=123;
    reject([&]{unsigned flags=0;coloc_square_into(input.value,order,{&untouched,sizeof(untouched)},flags);});require(untouched==123);
    input.length=1;
    unsigned flags=99;coloc_square_into(input.value,order,{&untouched,sizeof(untouched)},flags);require(untouched==1 && flags==0);
}
std::uint64_t f64_cases=0;
void finite_float64_bits() {
    const struct {std::uint64_t word;bool finite;} cases[]={
        {0,true},{0x8000000000000000ULL,true},{1,true},{0x8000000000000001ULL,true},
        {0x000fffffffffffffULL,true},{0x800fffffffffffffULL,true},
        {0x0010000000000000ULL,true},{0x8010000000000000ULL,true},
        {0x7fefffffffffffffULL,true},{0xffefffffffffffffULL,true},
        {0x7ff0000000000000ULL,false},{0xfff0000000000000ULL,false},
        {0x7ff8123456789abcULL,false},{0xfff8123456789abcULL,false},
        {0x7ff0000000000001ULL,false},{0xfff0000000000001ULL,false}};
    std::fenv_t saved;require(std::fegetenv(&saved)==0);
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO})for(int seeded:{FE_DIVBYZERO,FE_DIVBYZERO|FE_INVALID}) {
        require(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(seeded);
        for(int layout=0;layout<6;++layout)for(const std::size_t n:{0,1,2,3,4,5,7,8,9,12,15,16,17,31,32,33,127,128,129,257})for(const auto c:cases) {
            std::vector<std::size_t> positions{n ? n-1:0};
            if(!c.finite)for(std::size_t lane=0;lane<4 && lane<n;++lane)
                if(lane!=positions[0])positions.push_back(lane);
            for(auto position:positions) {
                std::vector<std::uint64_t> storage(2*n+4,0xa5a5a5a5a5a5a5a5ULL);
                auto* bytes=reinterpret_cast<unsigned char*>(storage.data());
                auto* first=bytes+(layout==0 ? 8:1)+(layout==3 && n ? 8*(n-1):0);
                const std::ptrdiff_t step=layout==3 ? -8:layout==4 ? 16:8;
                for(std::size_t i=0;i<n;++i) {
                    const auto word=i==position ? c.word:0x3ff0000000000000ULL;
                    unsigned char encoded[8];std::memcpy(encoded,&word,8);
                    for(std::size_t j=0;j<8;++j)first[std::ptrdiff_t(i)*step+std::ptrdiff_t(j)]=encoded[layout==2 ? 7-j:j];
                }
                const auto before=storage;
                One view(n ? first:nullptr,n,ColocType::float64,8,layout==2);view.stride=step;
                view.value.allocation=n ? storage.data():nullptr;view.value.allocation_bytes=n ? storage.size()*8:0;
                // Nontrivial2D dense F order when divisible, singleton otherwise.
                std::array<std::size_t,2> shape{n && n%3==0 ? 3u:1u,n && n%3==0 ? n/3:n};
                std::array<std::ptrdiff_t,2> strides{8,8*std::ptrdiff_t(shape[0])};
                if(layout==5){view.value.shape={shape.data(),2};view.value.strides={strides.data(),2};}
                require(coloc_all_finite(view.value)==(!n || c.finite));++f64_cases;
                require(std::fetestexcept(FE_ALL_EXCEPT)==seeded && std::fegetround()==rounding);
                require(storage==before);
                if(n && !c.finite && rounding==FE_TONEAREST) {
                    std::vector<unsigned char> output(n*8+2,0x6d);const auto untouched=output;
                    unsigned flags=99;bool refused=false;
                    try{coloc_square_into(view.value,coloc_keep_order(view.value),{output.data()+1,n*8},flags);}
                    catch(const ColocProviderRequired&){refused=true;}
                    require(refused && output==untouched && flags==99 && storage==before);
                    require(std::fetestexcept(FE_ALL_EXCEPT)==seeded && std::fegetround()==rounding);
                }
            }
        }
    }
    require(std::fesetenv(&saved)==0);
}
}
int main() {
    using namespace skimagexx;
    require(get_num_threads()==1);
    for(int workers:{1,4}) {
#ifdef SKIMAGEXX_COLOC_PORTABLE
        if(workers==4){bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}require(refused);continue;}
#endif
        set_num_threads(workers);bool_counts();byte_binary_conversion();finite_float32_bits();integer_products();swapped_signed_boolean_products();gather_and_orders();floating_association();multidimensional_addresses();affine_reduction_edges();reject_before_write();
    }
    set_num_threads(1);
    std::cout<<"colocalization native controls="<<checks<<"; default worker=1; AVX2 available="<<has_avx2()<<'\n';
    const auto old_checks=checks;
    for(int workers:{1,4}) {
#ifdef SKIMAGEXX_COLOC_PORTABLE
        if(workers==4)continue;
#endif
        set_num_threads(workers);finite_float64_bits();
    }
    set_num_threads(1);
    std::cout<<"f64 finite cases="<<f64_cases<<"; additive checks="<<checks-old_checks<<"; default worker=1\n";
}
