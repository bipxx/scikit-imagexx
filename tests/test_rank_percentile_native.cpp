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
void require(bool value){if(!value)throw std::runtime_error("rank percentile native check failed");}
template<class F> void rejected(F call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(caught);}
double replay(int op,const std::vector<std::int64_t>& counts,std::size_t bins,std::int64_t g,double p0,double p1){
 std::int64_t pop=0;for(auto c:counts)pop+=c;if(!pop)return 0;
 auto cross=[&](double target,bool strict,bool reverse){std::int64_t cumulative=0,selected=0;for(std::size_t j=0;j<bins;++j){const auto i=reverse ? bins-1-j:j;cumulative+=counts[i];if(strict ? double(cumulative)>target:double(cumulative)>=target){selected=std::int64_t(i);break;}}return selected;};
 if(op==25){if(p0==1){for(auto i=bins;i>0;--i)if(counts[i-1])return double(i-1);}return double(cross(p0*double(pop),true,false));}
 if(op==23)return double(bins-1)*(g>=cross(p0*double(pop),false,false));
 if(op==20 || op==21 || op==22 || op==24){std::int64_t cumulative=0,total=0,n=0;for(std::size_t i=0;i<bins;++i){cumulative+=counts[i];if(double(cumulative)>=p0*double(pop) && double(cumulative)<=p1*double(pop)){n+=counts[i];total+=counts[i]*std::int64_t(i);}}if(op==21)return double(n);if(!n)return 0.;if(op==20)return double(total/n);if(op==24)return double(total);return double(g-total/n)*.5+double(bins/2);}
 const auto low=cross(p0*double(pop),op!=19,false),high=cross((1.-p1)*double(pop),op!=19,true);
 if(op==19)return double(high-low);
 if(op==17)return double(high>low ? std::int64_t(bins-1)*(std::min(std::max(low,g),high)-low)/(high-low):high-low);
 return double(high-g<g-low ? high:low);
}
int main(){
 require(rank_histogram_capabilities()==8191 && rank_alias_capabilities()==480 && rank_statistics_capabilities()==0xe000 && rank_vector_capabilities()==0x10000 && rank_percentile_capabilities()==0x3fe0000);
 const std::vector<std::size_t> capacities{4,31,63,64,65,255,256,257,1023,1024,1025,4095,4096,4097,65536};
 for(const auto bins:capacities){
  const std::size_t h=bins>1025 ? 2:5,w=bins>1025 ? 3:7;std::vector<std::uint16_t> image(h*w);for(std::size_t i=0;i<image.size();++i)image[i]=std::uint16_t((i*4093+17)%bins);image[0]=std::uint16_t(bins-1);
  std::uint8_t fp[9]{1,0,1,1,1,0,0,1,1};std::vector<double> output(h*w);std::vector<std::uint8_t> mask(h*w);for(std::size_t i=0;i<mask.size();++i)mask[i]=std::uint8_t(i%3!=1);
  RankView v;v.image=image.data();v.mask=mask.data();v.footprint=fp;v.output=output.data();v.shape={1,h,w};v.footprint_shape={1,3,3};v.center={0,1,1};v.output_strides={0,std::ptrdiff_t(w*8),8};v.output_type=PixelType::float64;v.uint16=true;v.bins=bins;
  for(const auto interval:std::vector<std::pair<double,double>>{{0,1},{.25,.75},{.5,.5},{0,0},{1,1},{std::nextafter(0.,1.),std::nextafter(1.,0.)}})for(int op=17;op<=25;++op){
   rank_percentile_into(v,static_cast<RankOperation>(op),interval.first,interval.second);
   for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){std::vector<std::int64_t> counts(bins);for(int fy=0;fy<3;++fy)for(int fx=0;fx<3;++fx){auto yy=std::ptrdiff_t(y)+fy-1,xx=std::ptrdiff_t(x)+fx-1;if(fp[fy*3+fx] && yy>=0 && xx>=0 && yy<std::ptrdiff_t(h) && xx<std::ptrdiff_t(w) && mask[std::size_t(yy)*w+std::size_t(xx)])++counts[image[std::size_t(yy)*w+std::size_t(xx)]];}require(output[y*w+x]==replay(op,counts,bins,image[y*w+x],interval.first,interval.second));}
  }
  std::fill(mask.begin(),mask.end(),0);for(int op=17;op<=25;++op){rank_percentile_into(v,static_cast<RankOperation>(op),.25,.75);for(auto value:output)require(value==0);}
  const auto before=output;rejected([&]{rank_percentile_into(v,RankOperation::autolevel_percentile,.8,.2);});require(output==before);
  rejected([&]{rank_percentile_into(v,RankOperation::autolevel_percentile,std::numeric_limits<double>::quiet_NaN(),1.);});require(output==before);
  v.bins=1;rejected([&]{rank_percentile_into(v,RankOperation::percentile,.5,0.);});require(output==before);
 }
 // Channel gaps/unaligned stores across the shared worker threshold.
 const std::size_t h=513,w=519,n=h*w;std::vector<std::uint8_t> image(n);for(std::size_t i=0;i<n;++i)image[i]=std::uint8_t(i%71);std::uint8_t fp[9]{1,1,0,0,1,1,1,0,1};
 std::vector<unsigned char> a(n*16+3,0xa5),b(a.size(),0xa5);RankView v;v.image=image.data();v.footprint=fp;v.shape={1,h,w};v.footprint_shape={1,3,3};v.center={0,1,1};v.output_type=PixelType::float64;v.bins=256;v.output_strides={0,std::ptrdiff_t(w*16),16};
 for(int op=17;op<=25;++op){v.output=a.data()+1;set_num_threads(1);rank_percentile_into(v,static_cast<RankOperation>(op),.25,.75);v.output=b.data()+1;
#ifdef _OPENMP
  set_num_threads(4);
#else
  rejected([&]{set_num_threads(4);});require(get_num_threads()==1);
#endif
  rank_percentile_into(v,static_cast<RankOperation>(op),.25,.75);require(a==b);for(std::size_t i=0;i<n;++i)for(std::size_t c=8;c<16;++c)require(a[1+i*16+c]==0xa5);require(a.front()==0xa5 && a.back()==0xa5);}
 v.output=image.data();v.output_type=PixelType::uint8;v.output_strides={0,std::ptrdiff_t(w),1};const auto original=image;rejected([&]{rank_percentile_into(v,RankOperation::percentile,.5,0.);});require(image==original);
 set_num_threads(1);std::cout<<"rank percentile native exact scans, boundaries, tails, workers and guards passed\n";
}
