// Independent counts/population replay for the private opcode16 policy.
#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool condition){if(!condition)throw std::runtime_error("windowed counts/pop replay failed");}
template<class T> void put(unsigned char* out,T value){std::memcpy(out,&value,sizeof value);}
std::size_t checks=0;
template<class I,class O> void replay(PixelType type,std::size_t rows,std::size_t cols,std::size_t bins,std::size_t depth,int footprint_kind,int mask_kind,bool uniform,bool gaps){
 std::vector<I> image(rows*cols);for(std::size_t i=0;i<image.size();++i)image[i]=I(uniform ? bins-1:(i*17+i/7)%bins);
 const std::size_t fr=footprint_kind==4 ? 1:5,fc=footprint_kind==4 ? 65:7;
 std::vector<std::uint8_t> footprint(fr*fc,1),mask(image.size(),1);
 if(footprint_kind==1)for(std::size_t i=0;i<footprint.size();++i)footprint[i]=(i%3)!=0;
 if(footprint_kind==2)std::fill(footprint.begin(),footprint.end(),0);
 if(footprint_kind==3){std::fill(footprint.begin(),footprint.end(),0);footprint.front()=footprint.back()=1;}
 if(mask_kind)for(std::size_t i=0;i<mask.size();++i)mask[i]=mask_kind==2 ? 0:std::uint8_t(i%5!=0 ? (mask_kind==3 ? 255:1):0);
 const auto col_step=depth*sizeof(O)+(gaps ? 3:0),row_step=cols*col_step+(gaps ? 5:0),bytes=rows*row_step+2;
 std::vector<unsigned char> actual(bytes,0xa5),expected(actual),parallel(actual);
 RankView view;view.image=image.data();view.uint16=sizeof(I)==2;view.footprint=footprint.data();view.mask=mask_kind ? mask.data():nullptr;
 view.output=actual.data()+1;view.output_type=type;view.shape={1,rows,cols};view.footprint_shape={1,fr,fc};view.center={0,std::ptrdiff_t(fr/2),std::ptrdiff_t(fc/2)};
 view.output_strides={0,std::ptrdiff_t(row_step),std::ptrdiff_t(col_step)};view.bins=bins;
 for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){
  std::vector<std::int64_t> counts(depth);std::int64_t pop=0;
  for(std::size_t y=0;y<fr;++y)for(std::size_t x=0;x<fc;++x){
   auto rr=std::ptrdiff_t(r)+std::ptrdiff_t(y)-view.center[1],cc=std::ptrdiff_t(c)+std::ptrdiff_t(x)-view.center[2];
   if(!footprint[y*fc+x] || rr<0 || cc<0 || rr>=std::ptrdiff_t(rows) || cc>=std::ptrdiff_t(cols))continue;
   const auto index=std::size_t(rr)*cols+std::size_t(cc);if(mask_kind && !mask[index])continue;
   if(std::size_t(image[index])<depth)++counts[image[index]];++pop;
  }
  const double scale=pop ? 1./double(pop):0.;
  for(std::size_t b=0;b<depth;++b)put<O>(expected.data()+1+r*row_step+c*col_step+b*sizeof(O),O(double(counts[b])*scale));
 }
 const auto original=image;const auto fp_copy=footprint;const auto mask_copy=mask;
 set_num_threads(1);rank_windowed_histogram_into(view,depth);require(actual==expected);
#ifdef _OPENMP
 set_num_threads(4);view.output=parallel.data()+1;rank_windowed_histogram_into(view,depth);require(parallel==expected);set_num_threads(1);
#endif
 require(image==original && footprint==fp_copy && mask==mask_copy);++checks;
}
int main(){
 require(get_num_threads()==1);require(rank_vector_capabilities()==0x10000 && rank_statistics_capabilities()==0xe000 && rank_percentile_capabilities()==0x3fe0000 && rank_bilateral_capabilities()==0x1c000000);
 for(int fp=0;fp<5;++fp)for(int mask=0;mask<4;++mask)for(auto depth:{std::size_t(1),std::size_t(3),std::size_t(4),std::size_t(7),std::size_t(17),std::size_t(31),std::size_t(32),std::size_t(33),std::size_t(63),std::size_t(64),std::size_t(65)}){
  replay<std::uint8_t,double>(PixelType::float64,9,71,256,depth,fp,mask,false,true);
  replay<std::uint16_t,float>(PixelType::float32,9,71,1024,depth,fp,mask,true,false);
  replay<std::uint8_t,double>(PixelType::float64,9,71,256,depth,fp,mask,true,false);
  replay<std::uint8_t,std::uint8_t>(PixelType::uint8,7,11,256,depth,fp,mask,true,true);
  replay<std::uint16_t,std::uint16_t>(PixelType::uint16,7,11,65536,depth,fp,mask,false,false);
  replay<std::uint8_t,std::uint8_t>(PixelType::boolean,7,11,256,depth,fp,mask,false,false);
 }
 replay<std::uint16_t,float>(PixelType::float32,2,3,65536,65536,0,0,true,false);
 replay<std::uint8_t,double>(PixelType::float64,3,71,256,256,4,0,true,false);
 replay<std::uint16_t,float>(PixelType::float32,3,71,1024,1024,4,0,true,false);
 replay<std::uint16_t,std::uint16_t>(PixelType::uint16,1,1,65536,65536,0,0,true,false);
 replay<std::uint16_t,double>(PixelType::float64,1,1,65536,65536,0,0,true,false);
 // Cross the optional-worker threshold without a large depth allocation.
 replay<std::uint16_t,double>(PixelType::float64,257,259,17,7,3,1,false,true);
 std::array<std::uint8_t,4> image{0,1,2,255},fp{1,1,1,1};std::array<double,16> output;output.fill(99.);
 RankView v;v.image=image.data();v.footprint=fp.data();v.output=output.data();v.shape={1,2,2};v.footprint_shape={1,2,2};v.center={0,1,1};v.output_strides={0,64,32};v.output_type=PixelType::float64;v.bins=4;
 auto reject=[&](auto call){const auto old=output;bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(caught && output==old);++checks;};
 reject([&]{rank_windowed_histogram_into(v,4);});v.bins=256;
 reject([&]{rank_windowed_histogram_into(v,0);});reject([&]{rank_windowed_histogram_into(v,257);});
 std::array<std::uint16_t,4> wide{0,1,2,65535};v.image=wide.data();v.uint16=true;v.bins=256;reject([&]{rank_windowed_histogram_into(v,4);});
 v.image=image.data();v.uint16=false;v.output_strides[2]=1;reject([&]{rank_windowed_histogram_into(v,4);});
#ifndef _OPENMP
 bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected && get_num_threads()==1);
#endif
 std::cout<<"windowed counts/pop independent cases "<<checks<<" passed\n";
}
