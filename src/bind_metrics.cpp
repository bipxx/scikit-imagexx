#include "skimagexx/metrics.hpp"
#include "bind_numeric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstring>
#include <numeric>
namespace py=pybind11;
namespace {
skimagexx::MetricImageView view(const py::array& image,const std::vector<std::size_t>& axes) {
    skimagexx::MetricImageView result;result.data=image.data();
    result.half=image.dtype().kind()=='f' && image.itemsize()==2;
    result.type=result.half ? skimagexx::PixelType::uint16:skimagexx::bindings::real_pixel_type(image.dtype());
    result.swap=!image.dtype().attr("isnative").cast<bool>() && image.itemsize()>1;
    for(auto axis:axes){result.shape.push_back(std::size_t(image.shape(py::ssize_t(axis))));result.strides.push_back(std::ptrdiff_t(image.strides(py::ssize_t(axis))));}
    return result;
}
std::vector<std::size_t> order(const py::array& image,bool keep) {
    std::vector<std::size_t> axes(std::size_t(image.ndim()));std::iota(axes.begin(),axes.end(),std::size_t{0});
    // Broadcast axes carry no physical ordering information. NumPy retains
    // logical C order rather than moving a zero-stride axis innermost.
    if(keep)for(auto axis:axes)if(image.strides(py::ssize_t(axis))==0)return axes;
    const auto magnitude=[](py::ssize_t value){return value<0 ? std::uint64_t(-(value+1))+1:std::uint64_t(value);};
    if(keep)std::stable_sort(axes.begin(),axes.end(),[&](auto a,auto b){return magnitude(image.strides(py::ssize_t(a)))>magnitude(image.strides(py::ssize_t(b)));});
    return axes;
}
}
void bind_metrics(py::module_& module) {
    const auto numpy=py::module_::import("numpy");
    const py::object minimum=numpy.attr("minimum").attr("reduce"),maximum=numpy.attr("maximum").attr("reduce");
    const bool buffered=numpy.attr("lib").attr("NumpyVersion")(numpy.attr("__version__")).attr("__lt__")("2.3.0").cast<bool>();
    module.def("metrics_reduce",[numpy,minimum,maximum,buffered](py::array first,py::array second,bool single,int normalization,bool capture_underflow,bool scalar_power_double) {
        if(first.ndim()!=second.ndim())throw py::value_error("Input images must have the same dimensions.");
        for(py::ssize_t axis=0;axis<first.ndim();++axis)if(first.shape(axis)!=second.shape(axis))throw py::value_error("Input images must have the same dimensions.");
        const bool common_f=bool(first.flags()&py::array::f_style) && bool(second.flags()&py::array::f_style);
        auto axes=order(first,common_f);auto a=view(first,axes),b=view(second,axes);
        const auto mode=static_cast<skimagexx::MetricNormalization>(normalization);
        const auto true_axes=order(first,true);const bool separate=normalization!=0 && true_axes!=axes;
        const auto normalized=view(first,true_axes);
        const auto block=buffered ? numpy.attr("getbufsize")().cast<std::size_t>():std::size_t{0};
        skimagexx::MetricSums result;
        {
            py::gil_scoped_release release;
            result=skimagexx::metric_sums(a,b,single,mode,capture_underflow,scalar_power_double,block);
            if(separate) {
                const auto stats=skimagexx::metric_sums(normalized,normalized,single,mode,capture_underflow,false,block);
                result.energy=stats.energy;result.mean=stats.mean;result.low=stats.low;result.high=stats.high;
                result.energy_flags=stats.energy_flags;result.energy_reduce_flags=stats.energy_reduce_flags;result.mean_reduce_flags=stats.mean_reduce_flags;
                result.valid=stats.valid;result.nan=stats.nan;
                result.range_signaling=stats.range_signaling;
            }
        }
        py::dict out;
        out["error"]=result.error;out["energy"]=result.energy;out["mean"]=result.mean;out["low"]=result.low;out["high"]=result.high;
        out["subtract_flags"]=result.subtract_flags;out["square_flags"]=result.square_flags;out["error_reduce_flags"]=result.error_reduce_flags;
        out["energy_flags"]=result.energy_flags;out["energy_reduce_flags"]=result.energy_reduce_flags;out["mean_reduce_flags"]=result.mean_reduce_flags;
        out["cast0_flags"]=result.cast0_flags;out["cast1_flags"]=result.cast1_flags;
        out["valid"]=result.valid;out["nan"]=result.nan;out["range_signaling"]=result.range_signaling;
        if(normalization==3 && result.nan) {
            // Runtime SIMD reduction order decides whether signaling NaNs
            // survive max/min. Preserve that rare scalar-warning contract
            // through the original compiled C loops; finite data stays fused.
            const auto policy=numpy.attr("errstate")(py::arg("all")="ignore");policy.attr("__enter__")();
            py::array prepared;
            try {
                prepared=single ? py::array(py::array_t<float,py::array::forcecast>::ensure(first)):
                                  py::array(py::array_t<double,py::array::forcecast>::ensure(first));
            } catch(...) {policy.attr("__exit__")(py::none(),py::none(),py::none());throw;}
            policy.attr("__exit__")(py::none(),py::none(),py::none());
            out["minmax_scalars"]=py::make_tuple(minimum(prepared,py::arg("axis")=py::none()),maximum(prepared,py::arg("axis")=py::none()));
        }
        return out;
    });
    module.def("metrics_stage",[numpy](py::array first,py::array second,bool single,int operation) {
        const auto dtype=single ? py::dtype::of<float>():py::dtype::of<double>();
        py::array output;
        if(operation==1) {
            // The compiled iterator chooses the same common K-order output
            // layout as the binary ufunc; it performs no pixel operations.
            const auto iterator=numpy.attr("nditer")(py::make_tuple(first,second,py::none()),
                py::arg("flags")=py::make_tuple("zerosize_ok"),
                py::arg("op_flags")=py::make_tuple(py::make_tuple("readonly"),py::make_tuple("readonly"),py::make_tuple("writeonly","allocate")),
                py::arg("op_dtypes")=py::make_tuple(first.dtype(),second.dtype(),dtype));
            output=iterator.attr("operands")[py::int_(2)].cast<py::array>();
        } else output=numpy.attr("empty_like")(first,py::arg("dtype")=dtype,py::arg("order")="K",py::arg("subok")=false).cast<py::array>();
        const auto axes=order(output,true);const auto out=view(output,axes);
        const auto broadcast=[&](const py::array& input) {
            skimagexx::MetricImageView result;result.data=input.data();
            result.half=input.dtype().kind()=='f' && input.itemsize()==2;
            result.type=result.half ? skimagexx::PixelType::uint16:skimagexx::bindings::real_pixel_type(input.dtype());
            result.swap=!input.dtype().attr("isnative").cast<bool>() && input.itemsize()>1;
            for(auto axis:axes) {
                const auto logical=py::ssize_t(axis)-(output.ndim()-input.ndim());
                result.shape.push_back(std::size_t(output.shape(py::ssize_t(axis))));
                result.strides.push_back(logical<0 || input.shape(logical)==1 ? 0:input.strides(logical));
            }
            return result;
        };
        const auto a=broadcast(first),b=broadcast(second);
        unsigned flags;
        {py::gil_scoped_release release;flags=skimagexx::metric_stage_into(a,b,out,operation,single);}
        return py::make_tuple(output,flags);
    });
    module.def("metrics_stage_sum",[numpy,buffered](py::array first,bool single) {
        const auto input=view(first,order(first,true));
        const auto block=buffered ? numpy.attr("getbufsize")().cast<std::size_t>():std::size_t{0};
        skimagexx::MetricSums result;
        {py::gil_scoped_release release;result=skimagexx::metric_sums(input,input,single,skimagexx::MetricNormalization::mean,true,false,block);}
        return py::make_tuple(result.mean,result.mean_reduce_flags);
    });
    module.def("metrics_stage_extremum",[minimum,maximum](py::array first,bool largest) {
        return (largest ? maximum:minimum)(first,py::arg("axis")=py::none());
    });
    module.def("metrics_raw_range",[](py::array image) {
        if(!image.size())throw py::value_error("zero-size array to reduction operation minimum which has no identity");
        const auto input=view(image,order(image,false));
        skimagexx::MetricRawRange result;
        {py::gil_scoped_release release;result=skimagexx::metric_raw_range(input);}
        const auto dtype=py::dtype::from_args(image.dtype().attr("type"));
        py::array low(dtype,std::vector<py::ssize_t>{}),high(dtype,std::vector<py::ssize_t>{});
        std::memcpy(low.mutable_data(),&result.low,std::size_t(image.itemsize()));std::memcpy(high.mutable_data(),&result.high,std::size_t(image.itemsize()));
        return py::make_tuple(low,high,result.has_nan);
    });
}
