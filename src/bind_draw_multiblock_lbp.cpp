#include <skimagexx/draw_multiblock_lbp.hpp>
#include <skimagexx/image.hpp>
#include "execution.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstring>
namespace py=pybind11;
namespace {
skimagexx::MultiblockLbpFloatView floating(const py::array& a){
    if(a.dtype().kind()!='f')throw py::type_error("LBP phase requires floating storage");
    skimagexx::CropView v;v.data=a.data();v.itemsize=std::size_t(a.itemsize());
    for(py::ssize_t k=0;k<a.ndim();++k){v.shape.push_back(a.shape(k));v.strides.push_back(a.strides(k));}
    return {v,!a.dtype().attr("isnative").cast<bool>()};
}
}
void bind_draw_multiblock_lbp(py::module_& m){
    m.def("_lbp_prepare",[](const py::array& a){
        using P=skimagexx::PixelType;P type;bool half=false;const auto k=a.dtype().kind();const auto n=a.itemsize();
        if(k=='b')type=P::boolean;
        else if(k=='f'){if(n==2){type=P::uint16;half=true;}else if(n==4)type=P::float32;else if(n==8)type=P::float64;else throw py::type_error("LBP floating width");}
        else if(k=='u'){if(n==1)type=P::uint8;else if(n==2)type=P::uint16;else if(n==4)type=P::uint32;else type=P::uint64;}
        else if(k=='i'){if(n==1)type=P::int8;else if(n==2)type=P::int16;else if(n==4)type=P::int32;else type=P::int64;}
        else throw py::type_error("LBP conversion dtype");
        if(type==P::uint8&&a.ndim()==2&&(a.flags()&py::array::c_style)){
            const auto size=std::size_t(a.size());
            if(size>std::size_t(PTRDIFF_MAX)/3)throw std::overflow_error("LBP shape overflow");
            // The ordinary NumPy allocation owns its storage. Normalize into
            // its first n slots, then replicate backwards without a second full
            // RGB vector/allocation/copy. The existing converter and its strict
            // arithmetic are unchanged; each chunk stays below a worker read.
            (void)skimagexx::get_num_threads();
            py::array result(py::dtype::of<double>(),std::vector<py::ssize_t>{a.shape(0),a.shape(1),3});
            auto* target=static_cast<double*>(result.mutable_data());
            const auto* source=static_cast<const std::uint8_t*>(a.data());
            constexpr std::size_t chunk=skimagexx::detail::parallel_min-1;
            for(std::size_t at=0;at<size;){const auto amount=std::min(chunk,size-at);skimagexx::convert_into(source+at,target+at,amount,P::uint8,P::float64);at+=amount;}
            for(std::size_t at=size;at-->0;){std::uint64_t word;std::memcpy(&word,target+at,8);std::memcpy(target+3*at,&word,8);std::memcpy(target+3*at+1,&word,8);std::memcpy(target+3*at+2,&word,8);}
            return result;
        }
        skimagexx::CropView view;view.data=a.data();view.itemsize=std::size_t(n);for(py::ssize_t d=0;d<a.ndim();++d){view.shape.push_back(a.shape(d));view.strides.push_back(a.strides(d));}
        skimagexx::MultiblockLbpParameters p;p.prepare_only=true;
        auto out=skimagexx::draw_multiblock_lbp({view,type,!a.dtype().attr("isnative").cast<bool>(),half},p);
        // Source gray2rgb stacks three same-type planes before img_as_float:
        // stacking resolves their canonical scalar dtype (metadata/byte order
        // are not carried through). RGB's copied descriptor stays unchanged.
        const auto dtype=k=='f'?(a.ndim()<3?py::dtype::from_args(a.dtype().attr("type")):a.dtype()):py::dtype::of<double>();py::array result(dtype,out.shape);
        auto dst=floating(result);
        skimagexx::CropView source;source.data=out.is_float32?static_cast<const void*>(out.float32.data()):static_cast<const void*>(out.float64.data());source.itemsize=out.is_float32?4:8;source.shape=out.shape;std::ptrdiff_t stride=std::ptrdiff_t(source.itemsize);source.strides.resize(source.shape.size());for(std::size_t d=source.shape.size();d-->0;){source.strides[d]=stride;stride*=source.shape[d];}
        skimagexx::multiblock_lbp_assign(dst,{source,false});return result;
    },py::arg("image").noconvert());
    m.def("_lbp_binary",[](const py::array& a,const py::array& b,const py::tuple& shape,const py::dtype& dtype,bool plus){
        std::vector<py::ssize_t> dims;for(auto d:shape)dims.push_back(py::cast<py::ssize_t>(d));py::array out(dtype,dims);
        const auto flags=skimagexx::multiblock_lbp_binary(floating(a),floating(b),floating(out),plus);return py::make_tuple(out,flags);
    });
    m.def("_lbp_assign",[](py::array destination,const py::array& source){if(!destination.writeable())throw py::value_error("assignment destination is read-only");skimagexx::multiblock_lbp_assign(floating(destination),floating(source));});
    m.def("_lbp_apply",[](py::array image,std::ptrdiff_t r,std::ptrdiff_t c,std::ptrdiff_t width,std::ptrdiff_t height,std::uint8_t code,const py::array& greater,const py::array& less,double alpha,double complement,bool first_float32,bool capture_workers,const py::function& deliver){
        if(!image.writeable())throw py::value_error("assignment destination is read-only");
        skimagexx::MultiblockLbpParameters p;p.r=r;p.c=c;p.width=width;p.height=height;p.code=code;p.alpha=alpha;
        // The GIL and owning Python arguments stay live for the complete call.
        // The callback is used only for a real arithmetic exception, in order.
        skimagexx::multiblock_lbp_apply(floating(image),p,floating(greater),floating(less),complement,first_float32,capture_workers,[&](unsigned flags,bool plus){deliver(flags,plus?"add":"multiply");});
    });
    m.def("draw_multiblock_lbp",[](const py::array& a,std::ptrdiff_t r,std::ptrdiff_t c,std::ptrdiff_t width,std::ptrdiff_t height,std::int64_t code,const py::array_t<double,py::array::c_style>& greater,const py::array_t<double,py::array::c_style>& less,double alpha,bool first_f32){
        using P=skimagexx::PixelType;P type;const auto kind=a.dtype().kind();const auto bytes=a.itemsize();
        if(kind=='b')type=P::boolean;
        else if(kind=='f'&&bytes==4)type=P::float32;else if(kind=='f'&&bytes==8)type=P::float64;
        else if(kind=='u'){switch(bytes){case 1:type=P::uint8;break;case 2:type=P::uint16;break;case 4:type=P::uint32;break;case 8:type=P::uint64;break;default:throw py::type_error("LBP integer width unsupported");}}
        else if(kind=='i'){switch(bytes){case 1:type=P::int8;break;case 2:type=P::int16;break;case 4:type=P::int32;break;case 8:type=P::int64;break;default:throw py::type_error("LBP integer width unsupported");}}
        else throw py::type_error("LBP numeric dtype pending");
        auto color=[](const py::array_t<double,py::array::c_style>& v){skimagexx::MultiblockLbpColor out;out.values.resize(std::size_t(v.size()));if(v.size())std::memcpy(out.values.data(),v.data(),std::size_t(v.size())*sizeof(double));for(py::ssize_t k=0;k<v.ndim();++k)out.shape.push_back(v.shape(k));return out;};
        std::vector<std::ptrdiff_t> shape,strides;for(py::ssize_t k=0;k<a.ndim();++k){shape.push_back(a.shape(k));strides.push_back(a.strides(k));}
        skimagexx::MultiblockLbpInput input{{a.data(),shape,strides,std::size_t(a.itemsize())},type,!a.dtype().attr("isnative").cast<bool>()};
        skimagexx::MultiblockLbpParameters p;p.r=r;p.c=c;p.width=width;p.height=height;p.code=code;p.alpha=alpha;p.greater=color(greater);p.less=color(less);p.first_product_float32=first_f32;
        skimagexx::MultiblockLbpImage out;{py::gil_scoped_release release;out=skimagexx::draw_multiblock_lbp(input,p);}
        py::array result(out.is_float32?py::dtype::of<float>():py::dtype::of<double>(),out.shape);
        if(out.elements())std::memcpy(result.mutable_data(),out.is_float32?static_cast<const void*>(out.float32.data()):static_cast<const void*>(out.float64.data()),out.elements()*(out.is_float32?4:8));
        return result;
    },py::arg("image").noconvert(),py::arg("r"),py::arg("c"),py::arg("width"),py::arg("height"),py::arg("code"),py::arg("greater").noconvert(),py::arg("less").noconvert(),py::arg("alpha"),py::arg("first_product_float32"));
}
