#include "resample_detail.hpp"
#include "bind_numeric.hpp"
#include "execution.hpp"
#include "resize_gaussian.hpp"
#include "resize_clip.hpp"
#include "resize_range.hpp"
#include "resize_half.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace {
template<class T> void clip_python(py::array& image,const py::array& lower,const py::array& upper,
                                  skimagexx::PixelType type,int policy) {
    T low,high;std::memcpy(&low,lower.data(),sizeof(T));std::memcpy(&high,upper.data(),sizeof(T));
    py::gil_scoped_release release;
    skimagexx::detail::resize_clip_into(image.mutable_data(),std::size_t(image.size()),type,&low,&high,static_cast<skimagexx::detail::ResizeClipPolicy>(policy));
}
template<std::size_t Bytes> void copy_layout(const void* source,void* destination,std::size_t count,
                 const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,bool swap) {
    auto* target=static_cast<unsigned char*>(destination);const auto* input=static_cast<const unsigned char*>(source);
    std::vector<std::size_t> steps(shape.size());std::size_t step=1;
    for(auto axis=shape.size();axis-->0;){steps[axis]=step;step*=shape[axis];}
    const auto threads=skimagexx::detail::active_workers(count);
    const auto width=shape.empty() ? std::size_t{1}:shape.back();
    const auto inner=shape.empty() ? std::ptrdiff_t{0}:strides.back();
    // Calculate coordinates once per inner-axis span. The pixel loop only
    // loads/stores raw words; contiguous and broadcast spans use bulk copies.
    skimagexx::detail::partition(count,threads,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;) {
            std::ptrdiff_t offset=0;
            for(std::size_t axis=0;axis<shape.size();++axis)
                offset+=std::ptrdiff_t((i/steps[axis])%shape[axis])*strides[axis];
            const auto span=std::min(end-i,width-i%width);
            auto* output=target+i*Bytes;
            if(inner==std::ptrdiff_t(Bytes) && !swap)
                std::memcpy(output,input+offset,span*Bytes);
            else if(inner==0) {
                unsigned char word[Bytes];
                for(std::size_t byte=0;byte<Bytes;++byte)word[byte]=input[offset+std::ptrdiff_t(swap ? Bytes-1-byte:byte)];
                if constexpr(Bytes==1)std::memset(output,word[0],span);
                else {
                    std::memcpy(output,word,Bytes);
                    for(std::size_t filled=1;filled<span;) {
                        const auto chunk=std::min(filled,span-filled);
                        std::memcpy(output+filled*Bytes,output,chunk*Bytes);filled+=chunk;
                    }
                }
            } else {
                for(std::size_t column=0;column<span;++column) {
                    const auto* value=input+offset+std::ptrdiff_t(column)*inner;
                    auto* destination=output+column*Bytes;
                    if(swap)for(std::size_t byte=0;byte<Bytes;++byte)destination[byte]=value[Bytes-1-byte];
                    else std::memcpy(destination,value,Bytes);
                }
            }
            i+=span;
        }
    });
}
void copy_native(const void* source,void* destination,std::size_t count,std::size_t bytes,
                 const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,
                 bool swap,bool contiguous) {
    if(!count)return;
    if(contiguous && !swap){std::memcpy(destination,source,count*bytes);return;}
    switch(bytes) {
        case 1:copy_layout<1>(source,destination,count,shape,strides,swap);break;
        case 2:copy_layout<2>(source,destination,count,shape,strides,swap);break;
        case 4:copy_layout<4>(source,destination,count,shape,strides,swap);break;
        case 8:copy_layout<8>(source,destination,count,shape,strides,swap);break;
        default:throw std::invalid_argument("unsupported zoom word size");
    }
}
}
void bind_resample(py::module_& module) {
    // These are bound NumPy C methods, not Python image implementations.
    // Only ambiguous zero ties/all-NaN payloads need their runtime ordering.
    const auto numpy=py::module_::import("numpy");
    const py::object minimum=numpy.attr("minimum").attr("reduce"),maximum=numpy.attr("maximum").attr("reduce"),
                     fmin=numpy.attr("fmin").attr("reduce"),fmax=numpy.attr("fmax").attr("reduce");
    module.def("resize_half_identity",[minimum,maximum,fmin,fmax](py::array image,bool clip,int policy)->py::tuple {
        if(image.dtype().kind()!='f' || image.itemsize()!=2)throw py::type_error("binary16 identity input required");
        if(policy<0 || policy>1)throw py::value_error("invalid half clip policy");
        std::vector<std::size_t> shape;std::vector<py::ssize_t> dimensions;std::vector<std::ptrdiff_t> strides;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            shape.push_back(std::size_t(image.shape(axis)));dimensions.push_back(image.shape(axis));strides.push_back(std::ptrdiff_t(image.strides(axis)));
        }
        const bool swap=!image.dtype().attr("isnative").cast<bool>(),contiguous=bool(image.flags()&py::array::c_style);
        const py::dtype dtype("float16");py::array result(dtype,dimensions);
        const auto count=std::size_t(image.size());
        {
            py::gil_scoped_release release;
            if(contiguous)skimagexx::detail::half_copy_into(image.data(),static_cast<std::uint16_t*>(result.mutable_data()),count,swap);
            else copy_native(image.data(),result.mutable_data(),count,2,shape,strides,swap,false);
        }
        if(!clip)return py::make_tuple(result,py::none(),py::none(),false,false,false);
        if(!count)throw py::value_error("zero-size array to reduction operation minimum which has no identity");
        py::array lower(dtype,std::vector<py::ssize_t>{}),upper(dtype,std::vector<py::ssize_t>{});
        skimagexx::detail::ResizeRangeInfo info;
        {
            py::gil_scoped_release release;
            info=skimagexx::detail::half_range_into(static_cast<const std::uint16_t*>(result.data()),count,
                static_cast<std::uint16_t*>(lower.mutable_data()),static_cast<std::uint16_t*>(upper.mutable_data()));
        }
        std::uint16_t low,high;std::memcpy(&low,lower.data(),2);std::memcpy(&high,upper.data(),2);
        const bool runtime_order=!info.has_value || (info.positive_zero && info.negative_zero &&
            (skimagexx::detail::half_zero(low)||skimagexx::detail::half_zero(high)));
        if(runtime_order) {
            lower=py::array::ensure((info.has_nan ? fmin : minimum)(image,py::arg("axis")=py::none()));
            upper=py::array::ensure((info.has_nan ? fmax : maximum)(image,py::arg("axis")=py::none()));
            std::memcpy(&low,lower.data(),2);std::memcpy(&high,upper.data(),2);
        }
        {
            py::gil_scoped_release release;
            skimagexx::detail::half_clip_into(static_cast<std::uint16_t*>(result.mutable_data()),count,low,high,policy);
        }
        return py::make_tuple(result,lower,upper,info.has_nan,info.has_value,runtime_order);
    });
    module.def("resize_range",[minimum,maximum,fmin,fmax](py::array image) {
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        if(!image.size())throw py::value_error("zero-size array to reduction operation minimum which has no identity");
        const auto dtype=py::dtype::from_args(image.dtype().attr("type"));
        std::vector<std::size_t> shape;std::vector<py::ssize_t> dimensions;std::vector<std::ptrdiff_t> strides;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            shape.push_back(std::size_t(image.shape(axis)));dimensions.push_back(image.shape(axis));strides.push_back(std::ptrdiff_t(image.strides(axis)));
        }
        const auto itemsize=std::size_t(image.itemsize());
        const bool swap=!image.dtype().attr("isnative").cast<bool>() && itemsize>1;
        const bool contiguous=bool(image.flags()&(py::array::c_style|py::array::f_style));
        const bool copy=swap || !contiguous || !image.attr("flags").attr("aligned").cast<bool>();
        py::array normalized=copy ? py::array(dtype,dimensions) : image;
        py::array lower(dtype,std::vector<py::ssize_t>{}),upper(dtype,std::vector<py::ssize_t>{});
        skimagexx::detail::ResizeRangeInfo info;
        {
            py::gil_scoped_release release;
            if(copy)copy_native(image.data(),normalized.mutable_data(),std::size_t(image.size()),itemsize,shape,strides,swap,contiguous);
            info=skimagexx::detail::resize_range_into(normalized.data(),std::size_t(image.size()),type,lower.mutable_data(),upper.mutable_data());
        }
        bool ambiguous=!info.has_value;
        if(info.positive_zero && info.negative_zero) {
            if(type==skimagexx::PixelType::float32)ambiguous|=*static_cast<const float*>(lower.data())==0 || *static_cast<const float*>(upper.data())==0;
            else if(type==skimagexx::PixelType::float64)ambiguous|=*static_cast<const double*>(lower.data())==0 || *static_cast<const double*>(upper.data())==0;
        }
        if(ambiguous) {
            lower=py::array::ensure((info.has_nan ? fmin : minimum)(image,py::arg("axis")=py::none()));
            upper=py::array::ensure((info.has_nan ? fmax : maximum)(image,py::arg("axis")=py::none()));
        }
        return py::make_tuple(lower,upper,info.has_nan,info.has_value,ambiguous);
    });
    module.def("resize_clip",[](py::array image,py::array lower,py::array upper,int policy) {
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        if(!(image.flags()&py::array::c_style) || !image.writeable() || !image.dtype().attr("isnative").cast<bool>() ||
           !image.attr("flags").attr("aligned").cast<bool>())throw py::value_error("writable native aligned C-order clip image required");
        for(const auto& bound:{lower,upper})
            if(bound.size()!=1 || !bound.dtype().attr("isnative").cast<bool>() ||
               skimagexx::bindings::real_pixel_type(bound.dtype())!=type)throw py::type_error("one same-dtype native scalar per clip bound required");
        switch(type) {
            case skimagexx::PixelType::boolean:clip_python<std::uint8_t>(image,lower,upper,type,policy);break;
            #define CLIP(tag,T) case skimagexx::PixelType::tag:clip_python<T>(image,lower,upper,type,policy);break;
            CLIP(uint8,std::uint8_t) CLIP(int8,std::int8_t) CLIP(uint16,std::uint16_t) CLIP(int16,std::int16_t)
            CLIP(uint32,std::uint32_t) CLIP(int32,std::int32_t) CLIP(uint64,std::uint64_t) CLIP(int64,std::int64_t)
            CLIP(float32,float) CLIP(float64,double)
            #undef CLIP
            default:throw py::type_error("unsupported clip dtype");
        }
        return image;
    });
    module.def("resize_gaussian_raw",[](py::array image,py::list kernels,
        const std::vector<std::size_t>& axes,const std::vector<int>& modes,double cval) {
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        if(axes.size()!=std::size_t(kernels.size()) || modes.size()!=axes.size())throw py::value_error("one Gaussian kernel and mode per axis required");
        const auto dtype=py::dtype::from_args(image.dtype().attr("type"));
        std::vector<std::size_t> shape;std::vector<py::ssize_t> dimensions;std::vector<std::ptrdiff_t> strides;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            shape.push_back(std::size_t(image.shape(axis)));dimensions.push_back(image.shape(axis));strides.push_back(std::ptrdiff_t(image.strides(axis)));
        }
        std::vector<skimagexx::GaussianPass> passes;
        for(std::size_t j=0;j<axes.size();++j) {
            auto weights=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(kernels[j]);
            if(!weights || weights.ndim()!=1)throw py::type_error("one-dimensional Gaussian coefficients required");
            skimagexx::GaussianPass pass;pass.axis=axes[j];pass.border=static_cast<skimagexx::GaussianBorder>(modes[j]);pass.cval=cval;
            pass.weights.assign(weights.data(),weights.data()+weights.size());passes.push_back(std::move(pass));
        }
        const auto itemsize=std::size_t(image.itemsize());
        const bool swap=!image.dtype().attr("isnative").cast<bool>() && itemsize>1;
        const bool contiguous=bool(image.flags()&py::array::c_style);
        const bool copy=swap || !contiguous || !image.attr("flags").attr("aligned").cast<bool>();
        py::array normalized=copy ? py::array(dtype,dimensions) : image;
        py::array result(dtype,dimensions);
        {
            py::gil_scoped_release release;
            if(copy)copy_native(image.data(),normalized.mutable_data(),std::size_t(image.size()),itemsize,shape,strides,swap,contiguous);
            skimagexx::detail::gaussian_raw_into(normalized.data(),result.mutable_data(),type,shape,passes);
        }
        return result;
    });
    module.def("resize_zoom",[](py::array image,const std::vector<py::ssize_t>& dimensions,
                               int order,int border,double cval,bool legacy,bool identity,bool simd) {
        if(image.dtype().kind()!='f' || image.itemsize()!=8 || image.ndim()<1)
            throw py::type_error("float64 zoom input required");
        if(dimensions.size()!=std::size_t(image.ndim()))throw py::value_error("matching zoom ranks required");
        auto input=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(image);
        if(!input)throw py::type_error("float64 zoom input required");
        if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<decltype(input)>();
        std::vector<std::size_t> shape,target;
        for(py::ssize_t axis=0;axis<input.ndim();++axis) {
            if(dimensions[std::size_t(axis)]<0)throw py::value_error("negative zoom dimension");
            shape.push_back(std::size_t(input.shape(axis)));target.push_back(std::size_t(dimensions[std::size_t(axis)]));
        }
        py::array_t<double> result(dimensions);
        skimagexx::detail::ZoomOptions options;
        options.order=order;options.border=static_cast<skimagexx::GaussianBorder>(border);options.cval=cval;
        options.legacy_reflect=legacy;options.identity_copy=identity;options.allow_simd=simd;
        {py::gil_scoped_release release;skimagexx::detail::zoom_into(input.data(),result.mutable_data(),shape,target,options);}
        return result;
    });
    module.def("resize_zoom_typed",[](py::array image,const std::vector<py::ssize_t>& dimensions,
                                     int order,int border,double cval,bool legacy,bool identity,bool simd) {
        const auto type=skimagexx::bindings::real_pixel_type(image.dtype());
        if(image.ndim()<1 || dimensions.size()!=std::size_t(image.ndim()))throw py::value_error("matching nonzero zoom ranks required");
        const auto dtype=py::dtype::from_args(image.dtype().attr("type"));
        std::vector<std::size_t> shape,target;
        std::vector<py::ssize_t> source_shape;std::vector<std::ptrdiff_t> strides;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            if(dimensions[std::size_t(axis)]<0)throw py::value_error("negative zoom dimension");
            shape.push_back(std::size_t(image.shape(axis)));target.push_back(std::size_t(dimensions[std::size_t(axis)]));
            source_shape.push_back(image.shape(axis));strides.push_back(std::ptrdiff_t(image.strides(axis)));
        }
        const auto itemsize=std::size_t(image.itemsize());
        const bool swap=!image.dtype().attr("isnative").cast<bool>() && itemsize>1;
        const bool contiguous=bool(image.flags()&py::array::c_style);
        const bool copy=swap || !contiguous || !image.attr("flags").attr("aligned").cast<bool>();
        py::array normalized=copy ? py::array(dtype,source_shape) : image;
        py::array result(dtype,dimensions);
        skimagexx::detail::ZoomOptions options;
        options.order=order;options.border=static_cast<skimagexx::GaussianBorder>(border);options.cval=cval;
        options.legacy_reflect=legacy;options.identity_copy=identity;options.allow_simd=simd;
        {
            py::gil_scoped_release release;
            if(copy)copy_native(image.data(),normalized.mutable_data(),std::size_t(image.size()),itemsize,shape,strides,swap,contiguous);
            skimagexx::detail::zoom_typed_into(normalized.data(),result.mutable_data(),type,shape,target,options);
        }
        return result;
    });
}
