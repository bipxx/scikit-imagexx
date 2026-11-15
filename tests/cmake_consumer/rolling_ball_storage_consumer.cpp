#include <skimagexx/rolling_ball.hpp>
#include <skimagexx/image.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using skimagexx::PixelType;
using skimagexx::RollingBallStorage;
namespace {
void check(bool value){if(!value)throw std::runtime_error("rolling_ball dtype SDK control failure");}
template<class T>T read(std::ifstream& file){T value{};file.read(reinterpret_cast<char*>(&value),sizeof(T));check(bool(file));return value;}
void saved(const char* path,int workers) {
 std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);check(std::string(magic,8)==std::string("ROLLDT1\0",8));
 const auto cases=read<std::uint64_t>(f);check(cases==11);
 for(std::size_t c=0;c<cases;++c) {
  const auto length=read<std::uint64_t>(f),code=read<std::uint64_t>(f),half=read<std::uint64_t>(f),swap=read<std::uint64_t>(f),rank=read<std::uint64_t>(f),backing=read<std::uint64_t>(f),origin=read<std::uint64_t>(f);
  const double radius=read<double>(f);const bool safe=read<std::uint64_t>(f)!=0;check(length<100&&rank==2&&backing<=128&&origin<=backing);
  std::string name(length,' ');f.read(name.data(),std::streamsize(length));skimagexx::RollingBallGeometry g;g.shape.resize(rank);g.strides.resize(rank);
  std::size_t n=1;for(auto& x:g.shape){x=std::ptrdiff_t(read<std::int64_t>(f));n*=std::size_t(x);}for(auto& x:g.strides)x=std::ptrdiff_t(read<std::int64_t>(f));
  const std::size_t widths[]={1,1,1,2,2,4,4,8,8,4,8};check(code<11);const auto size=widths[code];
  // Deliberately unaligned input: operation owns dtype conversion through memcpy.
  std::vector<unsigned char> bytes(backing+1,0xa5),expected(n*size);f.read(reinterpret_cast<char*>(bytes.data()+1),std::streamsize(backing));f.read(reinterpret_cast<char*>(expected.data()),std::streamsize(expected.size()));check(bool(f));const auto before=bytes;
  RollingBallStorage view{bytes.data()+1+origin,PixelType(code),bool(half),bool(swap)};
  const auto result=skimagexx::rolling_ball(view,g,{},radius,safe,workers);
  if(result.bytes!=expected){std::cerr<<name<<" mismatch workers="<<workers<<'\n';throw std::runtime_error("saved output words differ");}
  check(result.type==view.type&&result.half==view.half&&result.swap==view.swap&&result.shape==g.shape);
  check(result.captured_workers==(workers?workers:skimagexx::get_num_threads())&&bytes==before);
 }
 check(f.peek()==std::char_traits<char>::eof());
}
void independent_controls() {
 // Nonsaved ordinary finite integers with negative/zero strides and custom
 // float32 kernel prove a complete conversion+kernel+min-plus+return operation.
 const std::int16_t image[]={-3,-2,-1,0,1,2};const float point=7.f;
 skimagexx::RollingBallGeometry g{{2,3},{-6,2},{1,1}};
 auto r=skimagexx::rolling_ball(RollingBallStorage{image+3,PixelType::int16},g,RollingBallStorage{&point,PixelType::float32},999.,false,4);
 const std::int16_t expected[]={0,1,2,-3,-2,-1};check(r.bytes.size()==sizeof(expected)&&std::memcmp(r.bytes.data(),expected,sizeof(expected))==0);
 g={{2,3},{0,0},{1,1}};r=skimagexx::rolling_ball(RollingBallStorage{image,PixelType::int16},g,RollingBallStorage{&point,PixelType::float32});
 for(std::size_t i=0;i<6;++i){std::int16_t v;std::memcpy(&v,r.bytes.data()+2*i,2);check(v==-3);}
 // Binary16 uses the established portable decoder/nearest encoder, never F16C.
 const std::uint16_t half[]={0x8000,0x3c00,0x4000};g={{3},{2},{1}};
 r=skimagexx::rolling_ball(RollingBallStorage{half,PixelType::uint16,true},g,RollingBallStorage{&point,PixelType::float32});
 const std::uint16_t half_expected[]={0,0x3c00,0x4000};
 check(r.bytes.size()==sizeof(half_expected)&&std::memcmp(r.bytes.data(),half_expected,sizeof(half_expected))==0);
 // Swapped uint32 words with a swapped custom double kernel, no arithmetic scaling.
 const unsigned char big[]={0,0,0,9,0,0,0,2,0,0,0,7};const unsigned char kernel[]={0x40,0x1c,0,0,0,0,0,0};g={{3},{4},{1}};
 r=skimagexx::rolling_ball(RollingBallStorage{big,PixelType::uint32,false,true},g,RollingBallStorage{kernel,PixelType::float64,false,true});check(r.bytes==std::vector<unsigned char>(big,big+12));
 // Large radius0 identity crosses the existing independent-output worker span.
 std::vector<std::uint16_t> large(512*512);for(std::size_t i=0;i<large.size();++i)large[i]=std::uint16_t(i%1000);
 g={{512,512},{1024,2},{}};r=skimagexx::rolling_ball(RollingBallStorage{large.data(),PixelType::uint16},g,{},0.,false,4);check(r.captured_workers==4&&r.bytes.size()==large.size()*2&&std::memcmp(r.bytes.data(),large.data(),r.bytes.size())==0);
 // Unsafe integer cast domain and invalid half metadata remain explicit refusals.
 const double nan=std::numeric_limits<double>::quiet_NaN();g={{3},{4},{1}};bool refused=false;
 try{skimagexx::rolling_ball(RollingBallStorage{big,PixelType::uint32,false,true},g,RollingBallStorage{&nan,PixelType::float64},1.,true);}catch(const std::domain_error&){refused=true;}check(refused);
 refused=false;try{skimagexx::rolling_ball(RollingBallStorage{big,PixelType::float32,true},g);}catch(const std::invalid_argument&){refused=true;}check(refused);
 refused=false;auto bad=g;bad.shape={3};bad.strides={std::numeric_limits<std::ptrdiff_t>::max()};try{skimagexx::rolling_ball(RollingBallStorage{big,PixelType::uint32},bad);}catch(const std::overflow_error&){refused=true;}check(refused);
}
}
int rolling_ball_storage_client(int argc,char** argv) {
 try{
  check(argc==2&&skimagexx::get_num_threads()==1);
  for(int budget:{1,4}){skimagexx::set_num_threads(budget);for(int workers:{0,1,4})saved(argv[1],workers);}
  skimagexx::set_num_threads(1);independent_controls();check(skimagexx::get_num_threads()==1);
  std::cout<<"new11saved xglobal1/4 xworkers0/1/4=66 complete owning operations +8dtype/layout/refusal controls PASS originals=0 timing=0\n";
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
