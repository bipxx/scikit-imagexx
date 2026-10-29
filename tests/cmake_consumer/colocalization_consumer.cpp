// Installed staged ABI/linkability checks; no Python or source/private includes.
#include <skimagexx/colocalization.hpp>
#include <skimagexx/image.hpp>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace skimagexx;
namespace { void check(bool ok){if(!ok)throw std::runtime_error("installed colocalization client mismatch");} }
int main(){
    try{
        check(get_num_threads()==1);
        const std::size_t shape[]={6};const std::ptrdiff_t strides[]={1};
        const std::uint8_t a[]={1,0,1,0,1,0},b[]={1,1,0,0,1,0};
        const ColocView first{a,a,sizeof(a),{shape,1},{strides,1},ColocType::uint8,false};
        const ColocView second{b,b,sizeof(b),{shape,1},{strides,1},ColocType::uint8,false};
        const ColocView mask{b,b,sizeof(b),{shape,1},{strides,1},ColocType::boolean,false};
        auto binary_first=first;binary_first.type=ColocType::boolean;
        for(int workers:{1,4}){
            set_num_threads(workers);check(get_num_threads()==workers);
            check(coloc_validate(first)==6&&coloc_all_finite(first)&&coloc_binary_valid(first).value);
            check(coloc_count_nonzero(binary_first)==3&&coloc_joint_count(binary_first,mask)==2);
            std::array<std::uint8_t,6> binary{},product{};std::array<std::uint8_t,3> gathered{};
            const auto order=coloc_keep_order(first);check(order.rank==1&&order.axes[0]==0);
            const auto execution=coloc_binary_into(first,order,{binary.data(),binary.size()});
            check(execution.worker_limit>=1&&execution.worker_limit<=workers&&(!execution.simd||has_avx2()));
            check(std::memcmp(binary.data(),a,sizeof(a))==0);
            coloc_gather_into(first,mask,{gathered.data(),gathered.size()},3);
            check((gathered==std::array<std::uint8_t,3>{1,0,1}));
            unsigned flags=0;coloc_multiply_into(first,second,coloc_keep_order(first,second),{product.data(),product.size()},flags);
            check(flags==0&&(product==std::array<std::uint8_t,6>{1,0,0,0,1,0}));
            const double values[]={1,2,3,4};const std::size_t four[]={4};const std::ptrdiff_t step[]={8};
            const ColocView real{values,values,sizeof(values),{four,1},{step,1},ColocType::float64,false};
            check(coloc_all_finite(real));
            const std::uint64_t exceptional[]={0,0x7ff0000000000001ULL,0,0};
            const ColocView nonfinite{exceptional,exceptional,sizeof(exceptional),{four,1},{step,1},ColocType::float64,false};
            check(!coloc_all_finite(nonfinite));
            for(auto profile:{ColocProfile::numpy_1244,ColocProfile::numpy_254}){
                const auto integer=coloc_sum(first,{profile,8192});check(integer.type==ColocType::uint64&&integer.bits==3&&integer.fp_status==0);
                const auto floating=coloc_sum(real,{profile,8192});double sum;std::memcpy(&sum,&floating.bits,sizeof(sum));
                check(floating.type==ColocType::float64&&sum==10&&floating.fp_status==0);
            }
        }
        set_num_threads(1);check(get_num_threads()==1);
        std::cout<<"installed colocalization stages PASS avx2="<<has_avx2()<<" workers=1\n";
    }catch(const std::exception& error){set_num_threads(1);std::cerr<<error.what()<<'\n';return 1;}
}
