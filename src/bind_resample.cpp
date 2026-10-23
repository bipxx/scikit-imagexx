#include "resample_detail.hpp"
#include "bind_numeric.hpp"
#include "execution.hpp"
#include "resize_gaussian.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace {
template<std::size_t Bytes> void copy_layout(const void* source,void* destination,std::size_t count,
                 const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,bool swap) {
    auto* target=static_cast<unsigned char*>(destination);const auto* input=static_cast<const unsigned char*>(source);
    std::vector<std::size_t> steps(shape.size());std::size_t step=1;
    for(auto axis=shape.size();axis-->0;){steps[axis]=step;step*=shape[axis];}
    const auto threads=skimagexx::detail::active_workers(count);
    // Each worker updates these counters per pixel. Align their payloads to
    // separate cache lines rather than sharing tiny vector allocations.
    std::vector<std::vector<std::size_t>> coordinates(std::size_t(threads),
        std::vector<std::size_t>(shape.size()+64/sizeof(std::size_t)));
    skimagexx::detail::partition(count,threads,[&](std::size_t begin,std::size_t end,std::size_t worker) {
        auto* raw=coordinates[worker].data();
        auto* index=raw+((64-std::uintptr_t(raw)%64)%64)/sizeof(std::size_t);
        std::ptrdiff_t offset=0;
        for(std::size_t axis=0;axis<shape.size();++axis) {
            index[axis]=(begin/steps[axis])%shape[axis];offset+=std::ptrdiff_t(index[axis])*strides[axis];
        }
        for(auto i=begin;i<end;++i) {
            auto* output=target+i*Bytes;const auto* value=input+offset;
            if(swap)for(std::size_t byte=0;byte<Bytes;++byte)output[byte]=value[Bytes-1-byte];
            else std::memcpy(output,value,Bytes);
            // Advance a logical C-order coordinate without division per pixel.
            for(auto axis=shape.size();axis-->0;) {
                if(index[axis]+1<shape[axis]){++index[axis];offset+=strides[axis];break;}
                offset-=std::ptrdiff_t(shape[axis]-1)*strides[axis];index[axis]=0;
            }
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
        const bool swap=!image.dtype().attr("isnative").cast<bool>() && image.itemsize()>1;
        const bool contiguous=bool(image.flags()&py::array::c_style);
        const bool copy=swap || !contiguous || !image.attr("flags").attr("aligned").cast<bool>();
        py::array normalized=copy ? py::array(dtype,dimensions) : image;
        py::array result(dtype,dimensions);
        {
            py::gil_scoped_release release;
            if(copy)copy_native(image.data(),normalized.mutable_data(),std::size_t(image.size()),std::size_t(image.itemsize()),shape,strides,swap,contiguous);
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
        const bool swap=!image.dtype().attr("isnative").cast<bool>() && image.itemsize()>1;
        const bool contiguous=bool(image.flags()&py::array::c_style);
        const bool copy=swap || !contiguous || !image.attr("flags").attr("aligned").cast<bool>();
        py::array normalized=copy ? py::array(dtype,source_shape) : image;
        py::array result(dtype,dimensions);
        skimagexx::detail::ZoomOptions options;
        options.order=order;options.border=static_cast<skimagexx::GaussianBorder>(border);options.cval=cval;
        options.legacy_reflect=legacy;options.identity_copy=identity;options.allow_simd=simd;
        {
            py::gil_scoped_release release;
            if(copy)copy_native(image.data(),normalized.mutable_data(),std::size_t(image.size()),std::size_t(image.itemsize()),shape,strides,swap,contiguous);
            skimagexx::detail::zoom_typed_into(normalized.data(),result.mutable_data(),type,shape,target,options);
        }
        return result;
    });
}
