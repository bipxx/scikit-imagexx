// Mirror cubic recurrence and accumulation derived from SciPy ni_splines.c /
// ni_interpolation.c1.15.3/1.18.1; BSD-3-Clause, see existing
// third_party/scipy-ndimage-LICENSE.txt. No original/provider is invoked.
#include "skimagexx/mark_boundaries.hpp"
#include "skimagexx/image.hpp"
#include "spline_detail.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
constexpr std::size_t limit=16*1024*1024;
std::size_t count(const std::vector<std::size_t>& shape){std::size_t n=1;for(auto s:shape){if(s&&n>limit/s)throw std::overflow_error("mark_boundaries element budget exceeded");n*=s;}return n;}
std::vector<std::ptrdiff_t> cstrides(const std::vector<std::size_t>& s){std::vector<std::ptrdiff_t> v(s.size());std::ptrdiff_t k=1;for(auto d=s.size();d-->0;){v[d]=k;k*=s[d];}return v;}
int budget(int workers){if(!workers)workers=get_num_threads();if(workers<1||workers>1024)throw std::invalid_argument("mark worker snapshot outside [1,1024]");return workers;}
std::size_t offset(std::size_t i,const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides){std::ptrdiff_t out=0;for(auto d=shape.size();d-->0;){out+=std::ptrdiff_t(i%shape[d])*strides[d];i/=shape[d];}return std::size_t(out);}
std::ptrdiff_t source_offset(std::size_t i,const ComponentsView& v){std::ptrdiff_t out=0;for(auto d=v.shape.size();d-->0;){out+=std::ptrdiff_t(i%v.shape[d])*v.strides[d];i/=v.shape[d];}return out;}
std::ptrdiff_t mirror(std::ptrdiff_t i,std::size_t n){if(n<=1)return 0;const auto period=std::ptrdiff_t(2*n-2);i%=period;if(i<0)i+=period;return i>=std::ptrdiff_t(n)?period-i:i;}
// Exact mirror recurrence; each line and axis retain upstream scalar order.
void filter(std::vector<double>& data,const std::vector<std::size_t>& shape){
    constexpr double z=-0.267949192431122706472553658494127633;
    double gain=1.;gain*=(1.-z)*(1.-1./z);
    const auto n=data.size();std::size_t inner=n;
    for(auto length:shape){inner/=length;if(length<=1)continue;std::vector<double> line(length);
        const auto blocks=n/(length*inner);
        for(std::size_t b=0;b<blocks;++b)for(std::size_t k=0;k<inner;++k){
            const auto base=b*length*inner+k;for(std::size_t j=0;j<length;++j)line[j]=data[base+j*inner];
            for(auto& v:line)v*=gain;
            double zi=z;const double zn=std::pow(z,double(length-1));line[0]=line[0]+zn*line[length-1];
            for(std::size_t j=1;j<length-1;++j){line[0]+=zi*(line[j]+zn*line[length-1-j]);zi*=z;}
            line[0]/=1-zn*zn;
            for(std::size_t j=1;j<length;++j)line[j]+=z*line[j-1];
            line[length-1]=(z*line[length-2]+line[length-1])*z/(z*z-1);
            for(auto j=length-1;j-->0;)line[j]=z*(line[j+1]-line[j]);
            for(std::size_t j=0;j<length;++j)data[base+j*inner]=line[j];
        }
    }
}
template<class T>std::vector<T> zoom(const std::vector<T>& in,const std::vector<std::size_t>& shape,const std::vector<std::size_t>& target,MarkCubicProfile profile){
    if(profile==MarkCubicProfile::scipy118&&shape==target)return in;
    std::vector<double> coefficients(in.begin(),in.end());filter(coefficients,shape);
    const auto stride=cstrides(shape);const auto rank=shape.size();std::size_t corners=1;for(std::size_t d=0;d<rank;++d)corners*=4;
    struct Point{std::array<double,6> weights;std::array<std::ptrdiff_t,4> offsets;};std::vector<std::vector<Point>> axes(rank);
    for(std::size_t d=0;d<rank;++d){axes[d].resize(target[d]);const double scale=target[d]>1?double(shape[d]-1)/double(target[d]-1):1.;
        for(std::size_t j=0;j<target[d];++j){const double coordinate=double(j)*scale;auto& p=axes[d][j];p.weights=detail::spline_weights(coordinate,3);const auto start=std::ptrdiff_t(std::floor(coordinate))-1;for(int t=0;t<4;++t)p.offsets[t]=mirror(start+t,shape[d])*stride[d];}}
    std::vector<T> out(count(target));std::vector<const Point*> points(rank);
    for(std::size_t i=0;i<out.size();++i){auto rest=i;for(auto d=rank;d-->0;){points[d]=&axes[d][rest%target[d]];rest/=target[d];}double total=0.;
        for(std::size_t corner=0;corner<corners;++corner){std::size_t rest=corner;std::array<unsigned,4> taps{};std::ptrdiff_t index=0;for(auto d=rank;d-->0;){taps[d]=unsigned(rest%4);rest/=4;index+=points[d]->offsets[taps[d]];}double coefficient=coefficients[std::size_t(index)];for(std::size_t d=0;d<rank;++d)coefficient*=points[d]->weights[taps[d]];total+=coefficient;}out[i]=static_cast<T>(total);
    }return out;
}
std::vector<std::ptrdiff_t> ordinary_strides(const ComponentsView& input,bool gray){
    std::size_t step_bytes=input.half?2:pixel_size(input.type);bool contiguous=true;
    for(auto d=input.shape.size();d-->0;){const auto stride=input.strides[d];if(input.shape[d]>1&&std::size_t(stride<0?-stride:stride)!=step_bytes)contiguous=false;step_bytes*=input.shape[d];}
    if(contiguous){auto shape=input.shape;if(gray)shape.push_back(3);return cstrides(shape);}
    std::vector<std::size_t> axes(input.shape.size());std::iota(axes.begin(),axes.end(),0);
    std::stable_sort(axes.begin(),axes.end(),[&](auto a,auto b){auto x=input.strides[a],y=input.strides[b];return (x<0?-x:x)<(y<0?-y:y);});
    std::vector<std::ptrdiff_t> out(input.shape.size()+gray);std::ptrdiff_t step=gray?3:1;if(gray)out.back()=1;
    for(auto axis:axes){out[axis]=step;step*=input.shape[axis];}return out;
}
template<class T>void layout(std::vector<T>& data,const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides){if(strides==cstrides(shape))return;auto old=data;for(std::size_t i=0;i<data.size();++i)data[offset(i,shape,strides)]=old[i];}
template<class T>std::vector<T> palette(const MarkColor& c,std::size_t pixels,std::size_t channels){
    const auto size=count(c.shape);if(c.values.size()!=size)throw std::invalid_argument("mark color storage/shape mismatch");
    if(c.shape.size()>2||(c.shape.size()&&c.shape.back()!=1&&c.shape.back()!=channels)||(c.shape.size()==2&&c.shape[0]!=1&&c.shape[0]!=pixels))throw std::invalid_argument("mark color cannot broadcast to selected pixels");
    std::vector<T> out(c.values.size());for(std::size_t i=0;i<out.size();++i)out[i]=static_cast<T>(c.values[i]);return out;
}
template<class T>void paint(T* out,const MarkImage& image,const std::vector<std::uint8_t>& mask,const MarkColor& color){
    const auto channels=image.shape.back();const auto selected=std::size_t(std::count(mask.begin(),mask.end(),std::uint8_t(1)));const auto values=palette<T>(color,selected,channels);std::size_t pixel=0;
    for(std::size_t i=0;i<mask.size();++i)if(mask[i]){for(std::size_t c=0;c<channels;++c){const auto column=color.shape.empty()||color.shape.back()==1?0:c;const auto row=color.shape.size()==2&&color.shape[0]!=1?pixel:0;out[offset(i*channels+c,image.shape,image.strides)]=values[row*(color.shape.empty()?1:color.shape.back())+column];}++pixel;}
}
}
MarkImage mark_boundaries_prepare(const ComponentsView& input,bool subpixel,MarkCubicProfile profile,int workers){
    workers=budget(workers);(void)workers;
    if(input.shape.size()>4||input.strides.size()!=input.shape.size())throw std::invalid_argument("mark image rank frontier is0through4");
    std::size_t span=0;for(std::size_t d=0;d<input.shape.size();++d){const auto stride=input.strides[d];if(stride==std::numeric_limits<std::ptrdiff_t>::min())throw std::invalid_argument("mark source stride overflow");const auto magnitude=std::size_t(stride<0?-stride:stride);const auto repeat=input.shape[d]?input.shape[d]-1:0;if(repeat&&magnitude>(std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-span)/repeat)throw std::invalid_argument("mark source extent overflow");span+=repeat*magnitude;}
    const auto n=count(input.shape);if(n&&!input.data)throw std::invalid_argument("mark requires nonempty image storage");
    const bool gray=input.shape.size()==2;MarkImage out;out.shape=input.shape;if(gray)out.shape.push_back(3);count(out.shape);
    out.type=input.half||input.type==PixelType::float32?PixelType::float32:PixelType::float64;
    const auto bytes=input.half?2:pixel_size(input.type);std::vector<std::uint64_t> packed((n*bytes+7)/8);auto* raw=reinterpret_cast<unsigned char*>(packed.data());
    for(std::size_t i=0;i<n;++i){const auto* p=static_cast<const unsigned char*>(input.data)+source_offset(i,input);for(std::size_t b=0;b<bytes;++b)raw[i*bytes+b]=p[input.swap?bytes-1-b:b];}
    if(out.type==PixelType::float32){out.f32.resize(gray?n*3:n);std::vector<float> scalar(n);
        if(input.half)for(std::size_t i=0;i<n;++i){std::uint16_t bits;std::memcpy(&bits,raw+2*i,2);scalar[i]=detail::decode_half(bits);}
        else if(n)std::memcpy(scalar.data(),raw,n*sizeof(float));
        for(std::size_t i=0;i<out.f32.size();++i)out.f32[i]=scalar[gray?i/3:i];
    }else{std::vector<double> scalar(n);for(std::size_t i=0;i<n;){const auto span=std::min(std::size_t(65536),n-i);convert_into(raw+i*bytes,scalar.data()+i,span,input.type,PixelType::float64);i+=span;}out.f64.resize(gray?n*3:n);for(std::size_t i=0;i<out.f64.size();++i)out.f64[i]=scalar[gray?i/3:i];}
    if(subpixel){if(!n)throw std::invalid_argument("mark subpixel requires nonempty spatial axes");auto target=out.shape;for(std::size_t d=0;d+1<target.size();++d)target[d]=2*target[d]-1;count(target);if(out.type==PixelType::float32)out.f32=zoom(out.f32,out.shape,target,profile);else out.f64=zoom(out.f64,out.shape,target,profile);out.shape=target;out.strides=cstrides(target);}
    else{out.strides=ordinary_strides(input,gray);if(out.type==PixelType::float32)layout(out.f32,out.shape,out.strides);else layout(out.f64,out.shape,out.strides);}
    return out;
}
MarkMask mark_boundaries_mask(const ComponentsView& labels,const detail::FindBoundaryParameters& p,int workers){
    workers=budget(workers);MarkMask out;
    if(labels.type==PixelType::float32||labels.type==PixelType::float64){
        if(labels.swap||labels.half||labels.shape.empty()||labels.shape.size()>3||(p.mode!=detail::FindBoundaryMode::thick&&p.mode!=detail::FindBoundaryMode::inner))throw std::invalid_argument("mark floating label frontier");
        out.shape=labels.shape;out.values.resize(count(out.shape));std::vector<double> values(out.values.size());for(std::size_t i=0;i<values.size();++i){const auto* address=static_cast<const unsigned char*>(labels.data)+source_offset(i,labels);if(labels.type==PixelType::float32){float value;std::memcpy(&value,address,4);values[i]=value;}else std::memcpy(&values[i],address,8);if(!std::isfinite(values[i]))throw std::invalid_argument("mark nonfinite label frontier");}
        const auto logical=cstrides(out.shape);std::int64_t signed_background;std::memcpy(&signed_background,&p.background_bits,8);const auto background=double(signed_background);
        for(std::size_t i=0;i<values.size();++i){bool different=false;for(std::size_t d=0;d<out.shape.size();++d){const auto coordinate=(i/std::size_t(logical[d]))%out.shape[d];if(coordinate)different|=values[i]!=values[i-std::size_t(logical[d])];if(coordinate+1<out.shape[d])different|=values[i]!=values[i+std::size_t(logical[d])];}out.values[i]=std::uint8_t(different&&(p.mode!=detail::FindBoundaryMode::inner||!p.background_representable||values[i]!=background));}return out;
    }
    if(!detail::find_boundaries_shape(labels,p.mode,out.shape))throw std::invalid_argument("mark label dtype/shape frontier");out.values.resize(count(out.shape));auto strides=cstrides(out.shape);
    if(!detail::find_boundaries_into(labels,p,out.values.data(),strides,workers))throw std::invalid_argument("mark label exact-promotion frontier");return out;
}
MarkMask mark_boundaries_outline(const MarkMask& mask){
    if(mask.shape.size()!=2)throw std::invalid_argument("outline footprint rank requires2D labels");const auto h=mask.shape[0],w=mask.shape[1];MarkMask out{mask.shape,std::vector<std::uint8_t>(mask.values.size())};
    for(std::size_t r=0;r<h;++r)for(std::size_t c=0;c<w;++c)for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){const auto rr=std::clamp(std::ptrdiff_t(r)+dr,std::ptrdiff_t(0),std::ptrdiff_t(h-1)),cc=std::clamp(std::ptrdiff_t(c)+dc,std::ptrdiff_t(0),std::ptrdiff_t(w-1));out.values[r*w+c]|=mask.values[std::size_t(rr)*w+std::size_t(cc)];}return out;
}
void mark_boundaries_paint_into(void* output,PixelType type,const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,std::size_t capacity,const MarkMask& mask,const MarkColor& color){
    if(capacity>limit||shape.empty()||shape.size()!=mask.shape.size()+1||!std::equal(mask.shape.begin(),mask.shape.end(),shape.begin())||strides.size()!=shape.size()||count(shape)>capacity)throw std::invalid_argument("mark image/label shapes differ");
    std::size_t extent=0;for(std::size_t d=0;d<shape.size();++d){if(strides[d]<0||std::size_t(strides[d])>capacity/std::max(std::size_t(1),shape[d]))throw std::invalid_argument("mark output strides outside capacity");extent+=(shape[d]-1)*std::size_t(strides[d]);}if(extent>=capacity||!output)throw std::invalid_argument("mark output outside capacity");
    MarkImage metadata;metadata.type=type;metadata.shape=shape;metadata.strides=strides;
    if(type==PixelType::float32)paint(static_cast<float*>(output),metadata,mask.values,color);else if(type==PixelType::float64)paint(static_cast<double*>(output),metadata,mask.values,color);else throw std::invalid_argument("mark output must be floating");
}
void mark_boundaries_paint(MarkImage& image,const MarkMask& mask,const MarkColor& color){mark_boundaries_paint_into(image.data(),image.type,image.shape,image.strides,image.type==PixelType::float32?image.f32.size():image.f64.size(),mask,color);}
void mark_boundaries_finish(MarkImage& image,const ComponentsView& labels,const MarkBoundariesParameters& p,int workers){
    workers=budget(workers);auto mask=mark_boundaries_mask(labels,p.boundary,workers);
    if(p.outline)mark_boundaries_paint(image,mark_boundaries_outline(mask),p.outline_color);
    mark_boundaries_paint(image,mask,p.color);
}
MarkImage mark_boundaries(const ComponentsView& image,const ComponentsView& labels,const MarkBoundariesParameters& p,int workers){workers=budget(workers);auto out=mark_boundaries_prepare(image,p.boundary.mode==detail::FindBoundaryMode::subpixel,p.cubic_profile,workers);mark_boundaries_finish(out,labels,p,workers);return out;}
}
