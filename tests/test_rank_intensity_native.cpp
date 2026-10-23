#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> void check(bool volume,bool masked,bool large) {
    const std::size_t planes=volume ? (large ? 3:2):1,rows=large ? (volume ? 299:513):7,cols=large ? (volume ? 301:519):13;
    std::vector<T> image(planes*rows*cols);
    std::vector<std::uint8_t> mask(image.size()),fp((volume ? 3:1)*3*5,1);
    for(std::size_t i=0;i<image.size();++i){image[i]=T((i*41+i/7)%(sizeof(T)==2 ? 65536:256));mask[i]=i%4!=0;}
    fp[1]=fp[7]=0;
    for(int operation=9;operation<=12;++operation) {
        skimagexx::RankView v;v.image=image.data();v.mask=masked ? mask.data():nullptr;v.footprint=fp.data();v.shape={planes,rows,cols};v.footprint_shape={volume ? 3u:1u,3,5};v.center={volume ? 1:0,1,2};v.bins=sizeof(T)==2 ? 65536:256;v.uint16=sizeof(T)==2;v.output_type=skimagexx::PixelType::float64;v.output_strides={std::ptrdiff_t(rows*cols*8),std::ptrdiff_t(cols*8),8};
        std::vector<double> one(image.size(),99),four(image.size(),99);v.output=one.data();skimagexx::set_num_threads(1);skimagexx::rank_filter_into(v,skimagexx::RankOperation(operation));
        v.output=four.data();
#ifdef _OPENMP
        skimagexx::set_num_threads(4);
#endif
        skimagexx::rank_filter_into(v,skimagexx::RankOperation(operation));
        assert(std::memcmp(one.data(),four.data(),one.size()*8)==0);
        const auto step=large ? std::max(std::size_t{1},image.size()/29):1;
        for(std::size_t index=0;index<image.size();index+=step) {
            const auto p=index/(rows*cols),r=index/cols%rows,c=index%cols;
            std::vector<std::int64_t> values;
            for(std::size_t pp=0;pp<v.footprint_shape[0];++pp)for(std::size_t rr=0;rr<3;++rr)for(std::size_t cc=0;cc<5;++cc) {
                const auto a=std::ptrdiff_t(p)+std::ptrdiff_t(pp)-v.center[0],b=std::ptrdiff_t(r)+std::ptrdiff_t(rr)-1,d=std::ptrdiff_t(c)+std::ptrdiff_t(cc)-2;
                if(!fp[(pp*3+rr)*5+cc] || a<0 || b<0 || d<0 || std::size_t(a)>=planes || std::size_t(b)>=rows || std::size_t(d)>=cols)continue;
                const auto j=(std::size_t(a)*rows+std::size_t(b))*cols+std::size_t(d);
                if(!masked || mask[j])values.push_back(image[j]);
            }
            double expected=0;const auto g=std::int64_t(image[index]);
            if(!values.empty()) {
                std::sort(values.begin(),values.end());const auto low=values.front(),high=values.back();
                if(operation==9)expected=high>low ? std::int64_t(v.bins-1)*(g-low)/(high-low):0;
                else if(operation==10)expected=double(std::int64_t(v.bins-1)*std::count_if(values.begin(),values.end(),[&](auto x){return x<=g;}))/double(values.size());
                else if(operation==11)expected=high-g<g-low ? high:low;
                else {std::int64_t sum=0;for(auto x:values)sum+=x;expected=g>double(sum)/double(values.size());}
            }
            assert(one[index]==expected);
        }
    }
}

void wide_stores() {
    std::uint16_t image[]={65535,1,2},output[3]={};std::uint8_t mask[]={0,1,1},fp[]={1,1,1};
    skimagexx::RankView v;v.image=image;v.mask=mask;v.footprint=fp;v.output=output;v.shape={1,1,3};v.footprint_shape={1,1,3};v.center={0,0,0};v.output_strides={6,6,2};v.bins=65536;v.uint16=true;v.output_type=skimagexx::PixelType::uint16;
    skimagexx::rank_filter_into(v,skimagexx::RankOperation::autolevel);assert(output[0]==2);
    double floating[3]={};v.output=floating;v.output_type=skimagexx::PixelType::float64;v.output_strides={24,24,8};
    skimagexx::rank_filter_into(v,skimagexx::RankOperation::autolevel);assert(floating[0]==4294770690.);
    image[0]=0;image[1]=65533;image[2]=65534;v.bins=65535;
    skimagexx::rank_filter_into(v,skimagexx::RankOperation::autolevel);assert(floating[0]==-4294639622.);
    v.output=output;v.output_type=skimagexx::PixelType::uint16;v.output_strides={6,6,2};
    skimagexx::rank_filter_into(v,skimagexx::RankOperation::autolevel);assert(output[0]==65530);
    v.serial_alias=true;
    for(int op=9;op<=12;++op){bool rejected=false;try{skimagexx::rank_filter_into(v,skimagexx::RankOperation(op));}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
}

int main() {
    assert(skimagexx::rank_histogram_capabilities()==8191);
    for(bool volume:{false,true})for(bool masked:{false,true})for(bool large:{false,true}){check<std::uint8_t>(volume,masked,large);check<std::uint16_t>(volume,masked,large);}
    wide_stores();
    for(int invalid:{-1,13,31}){bool rejected=false;try{skimagexx::rank_filter_into({},skimagexx::RankOperation(invalid));}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
    std::cout<<"R3 private native: neighborhood, wide-store, invalid/alias and worker identity oracles passed\n";
}
