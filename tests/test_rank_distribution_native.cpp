#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

template<class T> void check(bool volume,bool masked,bool large) {
    const std::size_t planes=volume ? (large ? 3:2):1,rows=large ? (volume ? 299:513):7,cols=large ? (volume ? 301:519):13;
    std::vector<T> image(planes*rows*cols);
    std::vector<std::uint8_t> mask(image.size()),fp((volume ? 3:1)*3*5,1);
    for(std::size_t i=0;i<image.size();++i){image[i]=T((i*41+i/7)% (sizeof(T)==2 ? 65536:256));mask[i]=i%4!=0;}
    fp[1]=fp[7]=0;
    for(int operation=5;operation<=8;++operation) {
        skimagexx::RankView v;v.image=image.data();v.mask=masked ? mask.data():nullptr;v.footprint=fp.data();v.shape={planes,rows,cols};v.footprint_shape={volume ? 3u:1u,3,5};v.center={volume ? 1:0,1,2};v.bins=sizeof(T)==2 ? 65536:256;v.uint16=sizeof(T)==2;v.output_type=skimagexx::PixelType::float64;v.output_strides={std::ptrdiff_t(rows*cols*8),std::ptrdiff_t(cols*8),8};
        std::vector<double> one(image.size(),99),four(image.size(),99);v.output=one.data();skimagexx::set_num_threads(1);skimagexx::rank_filter_into(v,skimagexx::RankOperation(operation));
        v.output=four.data();
#ifdef _OPENMP
        skimagexx::set_num_threads(4);
#else
        bool unsupported=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){unsupported=true;}assert(unsupported);
        skimagexx::set_num_threads(1);
#endif
        skimagexx::rank_filter_into(v,skimagexx::RankOperation(operation));
        assert(std::memcmp(one.data(),four.data(),one.size()*8)==0);
        const auto step=large ? std::max(std::size_t{1},image.size()/29):1;
        for(std::size_t index=0;index<image.size();index+=step) {
            const auto p=index/(rows*cols),r=index/cols%rows,c=index%cols;
            std::vector<unsigned> values;std::map<unsigned,unsigned> counts;
            for(std::size_t pp=0;pp<v.footprint_shape[0];++pp)for(std::size_t rr=0;rr<3;++rr)for(std::size_t cc=0;cc<5;++cc) {
                const auto a=std::ptrdiff_t(p)+std::ptrdiff_t(pp)-v.center[0],b=std::ptrdiff_t(r)+std::ptrdiff_t(rr)-1,d=std::ptrdiff_t(c)+std::ptrdiff_t(cc)-2;
                if(!fp[(pp*3+rr)*5+cc] || a<0 || b<0 || d<0 || std::size_t(a)>=planes || std::size_t(b)>=rows || std::size_t(d)>=cols)continue;
                const auto j=(std::size_t(a)*rows+std::size_t(b))*cols+std::size_t(d);
                if(!masked || mask[j]){values.push_back(image[j]);++counts[image[j]];}
            }
            double expected=0;
            if(!values.empty()) {
                std::sort(values.begin(),values.end());
                if(operation==5)expected=values.back()-values.front();
                else if(operation==6)expected=values[values.size()/2];
                else if(operation==7){unsigned maximum=0;for(const auto& entry:counts)if(entry.second>maximum){maximum=entry.second;expected=entry.first;}}
                else for(const auto& entry:counts){const double probability=double(entry.second)/double(values.size());expected-=probability*std::log(probability)/0.6931471805599453;}
            }
            assert(std::memcmp(&one[index],&expected,sizeof(double))==0);
        }
    }
}
int main() {
    assert(skimagexx::rank_histogram_capabilities()==0x1ffu);
    for(bool volume:{false,true})for(bool masked:{false,true})for(bool large:{false,true}){check<std::uint8_t>(volume,masked,large);check<std::uint16_t>(volume,masked,large);}
    for(int invalid:{-1,9,31}){bool rejected=false;try{skimagexx::rank_filter_into({},skimagexx::RankOperation(invalid));}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
#ifdef _OPENMP
    std::cout<<"rank distribution native: exact neighborhood oracles and worker byte identity passed\n";
#else
    std::cout<<"rank distribution native: portable serial oracles passed; unsupported worker4 requests rejected\n";
#endif
}
