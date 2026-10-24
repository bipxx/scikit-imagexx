#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool value){if(!value)throw std::runtime_error("rank decisions native check failed");}
template<class F> void rejected(F call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}require(caught);}

std::int64_t literal(int operation,const std::vector<std::int64_t>& counts){
 std::int64_t population=0,mean=0;for(std::size_t i=0;i<counts.size();++i){population+=counts[i];mean+=counts[i]*std::int64_t(i);}if(!population)return 0;
 if(operation==30){std::int64_t votes=counts[0],candidate=0;for(std::size_t i=1;i<counts.size();++i)if(counts[i]>votes){votes=counts[i];candidate=std::int64_t(i);}return candidate;}
 std::int64_t q=counts[0],moment=0,candidate=0;double best=0.;
 for(std::size_t i=1;i<counts.size();++i){const auto p=counts[i];if(!p)continue;q+=p;if(double(q)==double(population))break;moment+=std::int64_t(i)*p;const auto other=mean-moment;const double t=(double(population)-double(q))*double(moment)-double(other*q);const double score=(t*t)/(double(q)*(double(population)-double(q)));if(score>best){best=score;candidate=std::int64_t(i);}}
 return candidate;
}
int main(){
 require(rank_decisions_capabilities()==0x60000000 && rank_bilateral_capabilities()==0x1c000000 && rank_histogram_capabilities()==8191 && rank_alias_capabilities()==480 && rank_percentile_capabilities()==0x3fe0000);
#ifdef SKIMAGEXX_AVX2
 if(has_avx2()) {
  // Direct leaf arithmetic independently checks cross-word gathered bins,
  // high32 count bits and the first lane of a maximum tie. No FP regrouping.
  std::vector<std::int64_t> counts(65536);std::uint32_t indices[4]{1,2,4095,65535};
  counts[1]=counts[2]=(std::int64_t{1}<<32)+3;counts[4095]=7;counts[65535]=11;
  std::int64_t q[4],moment[4],population=17,total=81;
  simd::rank_decision_prefix4(counts.data(),indices,population,total,q,moment);
  for(int lane=0;lane<4;++lane){population+=counts[indices[lane]];total+=counts[indices[lane]]*indices[lane];require(q[lane]==population && moment[lane]==total);}
  std::int64_t votes=0,candidate=0;simd::rank_majority4(counts.data(),indices,votes,candidate);require(votes==counts[1] && candidate==1);
 }
#endif
 for(auto bins:std::vector<std::size_t>{3,31,63,64,65,255,256,257,1024,4096,65536})for(int dimensions: {2,3}){
  const std::size_t planes=dimensions==2 ? 1:3,rows=dimensions==2 ? 5:4,cols=dimensions==2 ? 7:5;
  std::vector<std::uint16_t> image(planes*rows*cols);for(std::size_t i=0;i<image.size();++i)image[i]=std::uint16_t((i*4093+17)%bins);
  std::vector<std::uint8_t> fp(dimensions==2 ? 9:27,1),mask(image.size(),1);for(std::size_t i=1;i<fp.size();i+=4)fp[i]=0;for(std::size_t i=0;i<mask.size();i+=3)mask[i]=0;
  std::vector<double> output(image.size(),99);RankView v;v.image=image.data();v.mask=mask.data();v.footprint=fp.data();v.output=output.data();v.uint16=true;v.bins=bins;v.output_type=PixelType::float64;v.shape={planes,rows,cols};v.footprint_shape={dimensions==2 ? 1u:3u,3,3};v.center={dimensions==2 ? 0:1,1,1};v.output_strides={std::ptrdiff_t(rows*cols*8),std::ptrdiff_t(cols*8),8};
  for(int operation=29;operation<=30;++operation){
   rank_decisions_into(v,static_cast<RankOperation>(operation));
   for(std::size_t p=0;p<planes;++p)for(std::size_t y=0;y<rows;++y)for(std::size_t x=0;x<cols;++x){
    std::vector<std::int64_t> counts(bins);
    for(std::size_t z=0;z<v.footprint_shape[0];++z)for(int fy=0;fy<3;++fy)for(int fx=0;fx<3;++fx){const auto pp=std::ptrdiff_t(p)+std::ptrdiff_t(z)-v.center[0],yy=std::ptrdiff_t(y)+fy-1,xx=std::ptrdiff_t(x)+fx-1;if(fp[(z*3+std::size_t(fy))*3+std::size_t(fx)] && pp>=0 && pp<std::ptrdiff_t(planes) && yy>=0 && yy<std::ptrdiff_t(rows) && xx>=0 && xx<std::ptrdiff_t(cols)){const auto i=(std::size_t(pp)*rows+std::size_t(yy))*cols+std::size_t(xx);if(mask[i])++counts[image[i]];}}
    require(output[(p*rows+y)*cols+x]==double(literal(operation,counts)));
   }
  }
  // Zero population cannot depend on a captured center-bin index.
  std::fill(fp.begin(),fp.end(),0);v.bins=3;std::fill(image.begin(),image.end(),2000);
  for(int operation=29;operation<=30;++operation){rank_decisions_into(v,static_cast<RankOperation>(operation));for(auto value:output)require(value==0);}
  std::fill(fp.begin(),fp.end(),1);std::fill(mask.begin(),mask.end(),0);
  for(int operation=29;operation<=30;++operation){rank_decisions_into(v,static_cast<RankOperation>(operation));for(auto value:output)require(value==0);}
  std::fill(mask.begin(),mask.end(),1);std::fill(output.begin(),output.end(),99);rejected([&]{rank_decisions_into(v,RankOperation::otsu);});for(auto value:output)require(value==99);
 }
 // Literal source endpoint/tie cases, including raw bool/narrow stores.
 for(auto values:std::vector<std::vector<std::uint16_t>>{{0,0,0,4,4,4,8,8,8},{1,1,1,9,9,9},{7,7,7,7},{0,1,2,3,32767,32768,65533,65534,65535}}){
  std::vector<std::uint8_t> fp(values.size(),1),out(values.size());RankView v;v.image=values.data();v.footprint=fp.data();v.output=out.data();v.uint16=true;v.bins=65536;v.output_type=PixelType::boolean;v.shape={1,1,values.size()};v.footprint_shape={1,1,values.size()};v.center={0,0,std::ptrdiff_t(values.size()/2)};v.output_strides={0,0,1};
  for(int operation=29;operation<=30;++operation){rank_decisions_into(v,static_cast<RankOperation>(operation));for(std::size_t x=0;x<values.size();++x){std::vector<std::int64_t> counts(65536);for(std::size_t f=0;f<values.size();++f){const auto xx=std::ptrdiff_t(x)+std::ptrdiff_t(f)-v.center[2];if(xx>=0 && xx<std::ptrdiff_t(values.size()))++counts[values[std::size_t(xx)]];}require(out[x]==std::uint8_t(literal(operation,counts)));}}
 }
 // Large independent stores use the existing sole global worker state.
 const std::size_t rows=513,cols=519,n=rows*cols;std::vector<std::uint8_t> image(n);for(std::size_t i=0;i<n;++i)image[i]=std::uint8_t((i*37)%61);std::uint8_t fp[9]{1,1,0,0,1,1,1,0,1};
 std::vector<unsigned char> a(n*16+3,0xa5),b(a);RankView v;v.image=image.data();v.footprint=fp;v.shape={1,rows,cols};v.footprint_shape={1,3,3};v.center={0,1,1};v.output_type=PixelType::float64;v.bins=256;v.output_strides={0,std::ptrdiff_t(cols*16),16};
 for(int operation=29;operation<=30;++operation){v.output=a.data()+1;set_num_threads(1);rank_decisions_into(v,static_cast<RankOperation>(operation));v.output=b.data()+1;
#ifdef _OPENMP
  set_num_threads(4);
#else
  rejected([&]{set_num_threads(4);});require(get_num_threads()==1);
#endif
  rank_decisions_into(v,static_cast<RankOperation>(operation));require(a==b);for(std::size_t i=0;i<n;++i)for(std::size_t c=8;c<16;++c)require(a[1+i*16+c]==0xa5);require(a.front()==0xa5 && a.back()==0xa5);}
 set_num_threads(1);v.output=image.data();v.output_type=PixelType::uint8;v.output_strides={0,std::ptrdiff_t(cols),1};const auto before=image;rejected([&]{rank_decisions_into(v,RankOperation::majority);});require(image==before);
 rejected([&]{rank_decisions_into(v,RankOperation::sum_bilateral);});require(image==before);
 std::cout<<"R7 native literal decisions, raw stores, guards and workers passed\n";
}
