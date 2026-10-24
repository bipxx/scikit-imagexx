#include "max_tree_stages.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace skimagexx;
namespace s=skimagexx::max_tree_stages;
namespace {
std::size_t checks=0;
void require(bool value,const char* message) {++checks;if(!value)throw std::runtime_error(message);}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b) {return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;}
template<class T> void check_type(MaxTreeType type) {
    const std::size_t n=20;
    std::vector<T> image(n),output(n),inverse(n);for(std::size_t i=0;i<n;++i)image[i]=T(i%4);
    const auto original=image;
    std::vector<std::int64_t> parent(n),order(n),roots(n),scratch(n),lo(n*2),hi(n*2);
    std::vector<double> attribute(n,1);
    MaxTreeImage input{image.data(),type,n,2,{{4,5,1}}};
    MaxTreeWorkspace workspace{{parent.data(),n},{order.data(),n},{roots.data(),n},{scratch.data(),n},
        {inverse.data(),n},{attribute.data(),n},{lo.data(),lo.size()},{hi.data(),hi.size()}};
    max_tree_into(input,{MaxTreeOperation::tree,1,0},workspace,{nullptr,0});
    auto expected_parent=parent,expected_order=order;
    std::vector<std::uint8_t> mask(n,0);
    for(std::size_t r=1;r<3;++r)for(std::size_t c=1;c<4;++c)mask[r*5+c]=1;
    const std::int32_t structure[]={-5,-1,1,5},shape[]={4,5},points[]={-1,0,0,-1,0,1,1,0};
    std::fill(parent.begin(),parent.end(),0);std::fill(roots.begin(),roots.end(),0);
    s::Image stage_image{image.data(),type,n};
    s::build(stage_image,{mask.data(),n},{structure,4},{shape,2},{points,8},{parent.data(),n},{order.data(),n},{roots.data(),n});
    require(same(parent,expected_parent) && same(order,expected_order),"staged ordered builder differs from original-qualified aggregate");
    s::Tree tree{{parent.data(),n},{order.data(),n}};
    for(bool diameter:{false,true}) {
        auto operation=diameter?MaxTreeOperation::diameter_opening:MaxTreeOperation::area_opening;
        max_tree_into(input,{operation,1,3},workspace,{output.data(),n});auto expected_output=output,expected_attribute=attribute;
        std::vector<double> observed(n,1),minimum(n*2),maximum(n*2);
        if(diameter) {
            for(std::size_t p=0;p<n;++p){minimum[p]=maximum[p]=double(p/5);minimum[p+n]=maximum[p+n]=double(p%5);}
            s::extension(stage_image,{shape,2},tree,{observed.data(),n},{minimum.data(),n*2,2,1,n},{maximum.data(),n*2,2,1,n});
        }else s::area(stage_image,tree,{observed.data(),n});
        require(same(observed,expected_attribute),"stage F64 attributes differ from aggregate/source order");
        std::memcpy(output.data(),image.data(),n*sizeof(T));
        s::direct_filter(stage_image,{output.data(),n},tree,{observed.data(),n},3);
        require(same(output,expected_output),"stage direct-filter differs from original-qualified aggregate");
        std::fill(observed.begin(),observed.end(),0);
        s::direct_filter(stage_image,{output.data(),n},tree,{observed.data(),n},3);
        for(auto value:output)require(value==T(0),"direct-filter ignored actual changed caller attributes");
    }
    std::vector<std::uint64_t> labels(n,1),expected_labels(n);
    max_tree_into(input,{MaxTreeOperation::local_maxima,1,0},workspace,{expected_labels.data(),n});
    s::local_maxima(stage_image,{labels.data(),n},tree);require(same(labels,expected_labels),"stage labels differ");
    require(same(image,original),"stages modified their source");
    auto before_parent=parent,before_roots=roots;
    auto bad_order=order;bad_order[0]=std::int64_t(n+1);bool rejected=false;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    try{s::build(stage_image,{mask.data(),n},{structure,4},{shape,2},{points,8},{parent.data(),n},{bad_order.data(),n},{roots.data(),n});}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected && same(parent,before_parent) && same(roots,before_roots),"builder guard wrote caller storage");
    require(std::fetestexcept(FE_ALL_EXCEPT)==flags,"stage guard did not restore FP flags");std::feclearexcept(FE_ALL_EXCEPT);
    auto old_attribute=attribute;rejected=false;
    try{s::extension(stage_image,{shape,2},tree,{attribute.data(),n},{attribute.data(),n*2,2,1,n},{attribute.data(),n*2,2,1,n});}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected && same(attribute,old_attribute),"coordinate alias guard wrote storage");
}
}
int main() {
    try {
        require(std::strcmp(s::interface,"max-tree-stages-v1")==0 && s::capabilities==31,"private stage interface");
        require(get_num_threads()==1,"default1");
        for(int workers:{1,4}) {
            bool available=true;try{set_num_threads(workers);}catch(const std::invalid_argument&){available=false;}
            if(!available){require(workers==4 && get_num_threads()==1,"portable4rejection");continue;}
#define RUN(code,T) check_type<T>(MaxTreeType::code);
            RUN(u8,std::uint8_t) RUN(u16,std::uint16_t) RUN(u32,std::uint32_t) RUN(u64,std::uint64_t)
            RUN(i8,std::int8_t) RUN(i16,std::int16_t) RUN(i32,std::int32_t) RUN(i64,std::int64_t) RUN(f32,float) RUN(f64,double)
#undef RUN
            require(get_num_threads()==workers,"one shared global worker limit");
        }
        set_num_threads(1);std::cout<<"PASS checks="<<checks<<" staged_traversals_serial=true\n";return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<'\n';return 1;}
}
