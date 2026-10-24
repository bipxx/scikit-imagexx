#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool value){if(!value)throw std::runtime_error("rank bilateral native check failed");}
template<class F> void rejected(F call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(caught);}

int main(){
 require(rank_bilateral_capabilities()==0x1c000000 && rank_histogram_capabilities()==8191 && rank_percentile_capabilities()==0x3fe0000);
 for(const auto bins:std::vector<std::size_t>{3,31,63,64,65,255,256,257,1023,1024,1025,4095,4096,65536}){
  const std::size_t h=5,w=7;std::vector<std::uint16_t> image(h*w);for(std::size_t i=0;i<image.size();++i)image[i]=std::uint16_t((i*4093+17)%bins);
  std::uint8_t fp[9]{1,0,1,1,1,0,0,1,1};std::vector<double> output(h*w);std::vector<std::uint8_t> mask(h*w);for(std::size_t i=0;i<mask.size();++i)mask[i]=std::uint8_t(i%3!=1);
  RankView v;v.image=image.data();v.mask=mask.data();v.footprint=fp;v.output=output.data();v.shape={1,h,w};v.footprint_shape={1,3,3};v.center={0,1,1};v.output_strides={0,std::ptrdiff_t(w*8),8};v.output_type=PixelType::float64;v.uint16=true;v.bins=bins;
  for(const auto radius:std::vector<std::pair<std::ptrdiff_t,std::ptrdiff_t>>{{0,0},{1,1},{1,5},{5,1},{-1,5},{300,300},{65536,65536},{std::numeric_limits<std::ptrdiff_t>::max(),10},{10,std::numeric_limits<std::ptrdiff_t>::min()}})for(int op=26;op<=28;++op){
   rank_bilateral_into(v,static_cast<RankOperation>(op),radius.first,radius.second);
   for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){std::int64_t count=0,sum=0;const auto g=std::int64_t(image[y*w+x]);for(int fy=0;fy<3;++fy)for(int fx=0;fx<3;++fx){auto yy=std::ptrdiff_t(y)+fy-1,xx=std::ptrdiff_t(x)+fx-1;if(fp[fy*3+fx] && yy>=0 && xx>=0 && yy<std::ptrdiff_t(h) && xx<std::ptrdiff_t(w) && mask[std::size_t(yy)*w+std::size_t(xx)]){const auto i=std::int64_t(image[std::size_t(yy)*w+std::size_t(xx)]);const bool upper=radius.first>65535 || i-g<radius.first;const bool lower=radius.second>65535 || g-i<radius.second;if(upper && lower){++count;sum+=i;}}}const auto expected=op==27 ? count:count ? (op==26 ? sum/count:sum):0;require(output[y*w+x]==double(expected));}
  }
  const auto before=output;rejected([&]{rank_bilateral_into(v,RankOperation::percentile,10,10);});require(output==before);
  rejected([&]{rank_bilateral_into(v,RankOperation::mean_bilateral,std::numeric_limits<std::ptrdiff_t>::min(),10);});require(output==before);
  std::fill(fp,fp+9,0);rank_bilateral_into(v,RankOperation::mean_bilateral,std::numeric_limits<std::ptrdiff_t>::min(),std::numeric_limits<std::ptrdiff_t>::max());for(auto value:output)require(value==0);
 }
 // Guard gaps, unaligned floating stores and actual global worker threshold.
 const std::size_t h=513,w=519,n=h*w;std::vector<std::uint8_t> image(n);for(std::size_t i=0;i<n;++i)image[i]=std::uint8_t(i%71);std::uint8_t fp[9]{1,1,0,0,1,1,1,0,1};
 std::vector<unsigned char> a(n*16+3,0xa5),b(a.size(),0xa5);RankView v;v.image=image.data();v.footprint=fp;v.shape={1,h,w};v.footprint_shape={1,3,3};v.center={0,1,1};v.output_type=PixelType::float64;v.bins=256;v.output_strides={0,std::ptrdiff_t(w*16),16};
 for(int op=26;op<=28;++op){v.output=a.data()+1;set_num_threads(1);rank_bilateral_into(v,static_cast<RankOperation>(op),17,5);v.output=b.data()+1;
#ifdef _OPENMP
  set_num_threads(4);
#else
  rejected([&]{set_num_threads(4);});require(get_num_threads()==1);
#endif
  rank_bilateral_into(v,static_cast<RankOperation>(op),17,5);require(a==b);for(std::size_t i=0;i<n;++i)for(std::size_t c=8;c<16;++c)require(a[1+i*16+c]==0xa5);require(a.front()==0xa5 && a.back()==0xa5);}
 set_num_threads(1);
 // All u8 stores are raw words, including bool destinations. Feedback uses
 // the same serial path regardless of requested workers and retains guards.
 std::vector<std::uint8_t> feedback(37);for(std::size_t i=0;i<feedback.size();++i)feedback[i]=std::uint8_t(i*7);
 auto initial=feedback;v.image=feedback.data();v.output=feedback.data()+1;v.shape={1,4,9};v.output_strides={0,9,1};v.output_type=PixelType::boolean;
 rank_bilateral_into(v,RankOperation::sum_bilateral,10,10);require(feedback.front()==initial.front());
 auto result=feedback;feedback=initial;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 rank_bilateral_into(v,RankOperation::sum_bilateral,10,10);require(feedback==result);
 set_num_threads(1);
 std::vector<std::uint16_t> words(40,17);v.image=words.data();v.output=words.data();v.uint16=true;v.bins=65536;v.output_type=PixelType::uint16;v.output_strides={0,19,2};
 const auto before=words;rejected([&]{rank_bilateral_into(v,RankOperation::mean_bilateral,10,10);});require(words==before);
 std::cout<<"rank bilateral native checks passed\n";
}
