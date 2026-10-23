#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("native rank check failed");}
template<class T> void check(bool volume,bool wide=false,bool uniform=false) {
    const std::array<std::size_t,3> shape=volume ? std::array<std::size_t,3>{7,193,199}:std::array<std::size_t,3>{1,513,519};
    const auto n=shape[0]*shape[1]*shape[2];std::vector<T> input(n);std::vector<std::uint8_t> mask(n);const auto columns=wide ? 65u:3u;std::vector<std::uint8_t> fp((volume ? 3:1)*3*columns,255);
    for(std::size_t i=0;i<n;++i){input[i]=uniform ? std::numeric_limits<T>::max():T((i*719+197)%(sizeof(T)==2 ? 65536:256));mask[i]=std::uint8_t(i%3!=0);}
    if(!wide)for(std::size_t i=0;i<fp.size();i+=4)fp[i]=0;
    std::vector<double> one(n),four(n);
    RankView view;view.image=input.data();view.footprint=fp.data();view.shape=shape;view.footprint_shape=volume ? std::array<std::size_t,3>{3,3,columns}:std::array<std::size_t,3>{1,3,columns};view.center=volume ? std::array<std::ptrdiff_t,3>{1,1,std::ptrdiff_t(columns/2)}:std::array<std::ptrdiff_t,3>{0,1,std::ptrdiff_t(columns/2)};view.output_strides={std::ptrdiff_t(shape[1]*shape[2]*8),std::ptrdiff_t(shape[2]*8),8};view.uint16=sizeof(T)==2;view.bins=sizeof(T)==2 ? 65536:256;view.output_type=PixelType::float64;
    for(bool masked:{false,true})for(auto op:{RankOperation::minimum,RankOperation::maximum,RankOperation::mean,RankOperation::sum,RankOperation::pop}) {
        view.mask=masked ? mask.data():nullptr;view.output=one.data();set_num_threads(1);rank_filter_into(view,op);
#ifdef _OPENMP
        set_num_threads(4);
#endif
        view.output=four.data();rank_filter_into(view,op);require(!std::memcmp(one.data(),four.data(),n*sizeof(double)));
        for(auto index:{std::size_t{0},n/2,n-1}) {
            const auto p=index/(shape[1]*shape[2]),r=index/shape[2]%shape[1],c=index%shape[2];std::vector<unsigned> values;
            for(std::size_t fp_p=0;fp_p<view.footprint_shape[0];++fp_p)for(std::size_t fp_r=0;fp_r<3;++fp_r)for(std::size_t fp_c=0;fp_c<columns;++fp_c) {
                const auto pp=std::ptrdiff_t(p)+std::ptrdiff_t(fp_p)-view.center[0],rr=std::ptrdiff_t(r)+std::ptrdiff_t(fp_r)-1,cc=std::ptrdiff_t(c)+std::ptrdiff_t(fp_c)-view.center[2];
                if(pp<0 || rr<0 || cc<0 || std::size_t(pp)>=shape[0] || std::size_t(rr)>=shape[1] || std::size_t(cc)>=shape[2] || !fp[(fp_p*3+fp_r)*columns+fp_c])continue;
                const auto sample=(std::size_t(pp)*shape[1]+std::size_t(rr))*shape[2]+std::size_t(cc);if(!masked || mask[sample])values.push_back(input[sample]);
            }
            double expected=0;
            if(op==RankOperation::pop)expected=double(values.size());
            else if(!values.empty()) {
                if(op==RankOperation::minimum)expected=*std::min_element(values.begin(),values.end());
                else if(op==RankOperation::maximum)expected=*std::max_element(values.begin(),values.end());
                else {for(auto value:values)expected+=value;if(op==RankOperation::mean)expected/=values.size();}
            }
            require(one[index]==expected);
        }
    }
    set_num_threads(1);
}
void half_conversion() {
    const auto n=std::size_t{262149};const std::uint16_t words[4]{1,0x3800,0xb800,0x3c00};const std::uint8_t expected[4]{0,128,0,255};
    std::vector<unsigned char> input(2*n+1);std::vector<std::uint8_t> one(n),four(n);
    for(std::size_t i=0;i<n;++i)std::memcpy(input.data()+1+2*i,&words[i%4],2);
    const auto* source=reinterpret_cast<const std::uint16_t*>(input.data()+1);
    set_num_threads(1);const auto a=rank_half_u8(source,one.data(),n);
#ifdef _OPENMP
    set_num_threads(4);
#endif
    const auto b=rank_half_u8(source,four.data(),n);require(one==four && !a.nan && !a.invalid_multiply && !a.overflow && !b.nan && !b.invalid_multiply && !b.overflow);
    for(std::size_t i=0;i<n;++i)require(one[i]==expected[i%4]);set_num_threads(1);
}
int main(){require(get_num_threads()==1);half_conversion();check<std::uint8_t>(false);check<std::uint8_t>(true);check<std::uint16_t>(false);check<std::uint16_t>(true);check<std::uint8_t>(false,true);check<std::uint16_t>(false,true);check<std::uint8_t>(false,true,true);check<std::uint16_t>(false,true,true);std::cout<<"Rank neighborhood oracle values and exact worker outputs passed\n";}
