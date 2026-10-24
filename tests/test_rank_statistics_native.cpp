#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("rank statistics native check failed");}
int main(){
 require(get_num_threads()==1);
 require(rank_histogram_capabilities()==8191 && rank_alias_capabilities()==480);
 require(rank_statistics_capabilities()==0xe000 && rank_vector_capabilities()==0x10000);
 std::array<std::uint8_t,6> image{0,1,3,7,15,0},fp{1,1,1,1,1,1};
 std::array<double,6> output{};RankView v;
 v.image=image.data();v.footprint=fp.data();v.output=output.data();v.shape={1,2,3};v.footprint_shape={1,2,3};v.center={0,1,1};v.output_strides={0,24,8};v.output_type=PixelType::float64;
 for(auto op:{RankOperation::geometric_mean,RankOperation::noise_filter,RankOperation::subtract_mean}){rank_statistics_into(v,op);for(auto value:output)require(std::isfinite(value));}
 std::vector<double> channels(6*256);v.output=channels.data();v.output_strides={0,3*256*8,256*8};rank_windowed_histogram_into(v,256);
 for(std::size_t p=0;p<6;++p){double sum=0;for(std::size_t b=0;b<256;++b)sum+=channels[p*256+b];require(std::abs(sum-1)<1e-15);}
 auto rejected=[&](auto call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(caught);};
 rejected([&]{rank_windowed_histogram_into(v,0);});rejected([&]{rank_windowed_histogram_into(v,257);});
 const auto before=channels;v.bins=4;rejected([&]{rank_windowed_histogram_into(v,4);});require(channels==before);v.bins=256;
 v.output=image.data();v.output_type=PixelType::uint8;v.output_strides={0,3,1};rejected([&]{rank_statistics_into(v,RankOperation::noise_filter);});
 // SIMD channel tails, byte-unaligned destination, and worker threshold.
 const std::size_t n=513*519;std::vector<std::uint8_t> large(n);for(std::size_t i=0;i<n;++i)large[i]=std::uint8_t(i%7);
 std::uint8_t one=1;std::vector<unsigned char> a(n*7*4+3,0xa5),b(a.size(),0xa5);
 v.image=large.data();v.footprint=&one;v.shape={1,513,519};v.footprint_shape={1,1,1};v.center={0,0,0};v.output_type=PixelType::float32;v.output_strides={0,519*7*4,7*4};
 v.output=a.data()+1;set_num_threads(1);rank_windowed_histogram_into(v,7);
 v.output=b.data()+1;
#ifdef _OPENMP
 set_num_threads(4);
#else
 rejected([&]{set_num_threads(4);});require(get_num_threads()==1);
#endif
 rank_windowed_histogram_into(v,7);require(a==b);require(a.front()==0xa5 && a.back()==0xa5);
 for(std::size_t p=0;p<n;++p)for(std::size_t c=0;c<7;++c){float value;std::memcpy(&value,a.data()+1+(p*7+c)*4,4);require(value==(c==large[p] ? 1.f:0.f));}
 set_num_threads(1);std::cout<<"rank statistics native checks passed\n";
}
