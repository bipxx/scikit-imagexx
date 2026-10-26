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
}
int main() {
    using namespace skimagexx;
    require(get_num_threads()==1);
    for(int workers:{1,4}) {
#ifdef SKIMAGEXX_COLOC_PORTABLE
        if(workers==4){bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}require(refused);continue;}
#endif
        set_num_threads(workers);bool_counts();integer_products();gather_and_orders();floating_association();multidimensional_addresses();affine_reduction_edges();reject_before_write();
    }
    set_num_threads(1);
    std::cout<<"colocalization native controls="<<checks<<"; default worker=1; AVX2 available="<<has_avx2()<<'\n';
}
