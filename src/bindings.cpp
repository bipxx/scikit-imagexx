#include "skimagexx/integrate.hpp"
#include "skimagexx/downscale.hpp"
#include "skimagexx/contrast.hpp"
#include "skimagexx/label_color.hpp"
#include "skimagexx/histogram_matching.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_23_API_VERSION
#include <numpy/arrayobject.h>
#include "skimagexx/integral.hpp"
#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/farid.hpp"
#include "skimagexx/small_edge.hpp"
#include "skimagexx/alpha.hpp"
#include <cstring>
#include "skimagexx/local_threshold.hpp"
#include "skimagexx/threshold.hpp"
#include "skimagexx/image.hpp"
#include "skimagexx/convert.hpp"
#include "skimagexx/color.hpp"
#include "skimagexx/perceptual.hpp"
#include "skimagexx/morphology.hpp"
#include "skimagexx/stain.hpp"
#include "skimagexx/distance.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <type_traits>

namespace py = pybind11;
namespace {
bool dtype_matches(const py::dtype& a,const py::dtype& b) {return a.is(b) || (a.itemsize()==b.itemsize() && a.kind()==b.kind() && a.equal(b));}
struct RawRows {
    std::size_t rows=0,columns=1;
    std::ptrdiff_t image_step=0,alpha_step=0;
    std::vector<std::ptrdiff_t> image,alpha;
};
RawRows raw_rows(const py::array& a,const py::array* b,int dimensions) {
    RawRows result;std::size_t count=1;
    for(int axis=0;axis<dimensions;++axis)count*=std::size_t(a.shape(axis));
    if(!count)return result;
    int axis=dimensions-1;
    while(axis>=0 && a.shape(axis)==1)--axis;
    if(axis<0) {result.rows=1;result.image={0};if(b)result.alpha={0};return result;}
    result.columns=std::size_t(a.shape(axis));result.image_step=a.strides(axis);result.alpha_step=b ? b->strides(axis) : 0;
    int prefix=axis;
    for(int outer=axis-1;outer>=0;--outer) {
        const auto span=std::ptrdiff_t(result.columns),step=a.strides(outer),other=b ? b->strides(outer) : 0;
        if(a.shape(outer)>1 && (step%span || step/span!=result.image_step || other%span || other/span!=result.alpha_step))break;
        result.columns*=std::size_t(a.shape(outer));prefix=outer;
    }
    result.rows=count/result.columns;result.image.resize(result.rows);if(b)result.alpha.resize(result.rows);
    for(std::size_t row=0;row<result.rows;++row) {
        auto remainder=row;std::ptrdiff_t first=0,second=0;
        for(int outer=prefix-1;outer>=0;--outer) {
            const auto dim=std::size_t(a.shape(outer)),index=remainder%dim;remainder/=dim;
            first+=std::ptrdiff_t(index)*a.strides(outer);if(b)second+=std::ptrdiff_t(index)*b->strides(outer);
        }
        result.image[row]=first;if(b)result.alpha[row]=second;
    }
    return result;
}
std::uint64_t raw_component(const unsigned char* p,std::size_t bytes,bool swap) {
    std::uint64_t result=0;
    if(bytes==2) {std::uint16_t value;std::memcpy(&value,p,2);result=value;}
    else if(bytes==4) {std::uint32_t value;std::memcpy(&value,p,4);result=value;}
    else std::memcpy(&result,p,8);
    if(swap) {
        auto value=result;result=0;
        for(std::size_t i=0;i<bytes;++i) {result=(result<<8)|(value&255);value>>=8;}
    }
    return result;
}
unsigned alpha_array_flags(const py::array& a) {
    const char kind=a.dtype().kind();
    if(kind!='f' && kind!='c' && kind!='m' && kind!='M')return 0;
    const auto bytes=std::size_t(a.itemsize())/(kind=='c' ? 2 : 1);
    if(bytes!=2 && bytes!=4 && bytes!=8)return 4;
    const bool swap=!a.dtype().attr("isnative").cast<bool>();
    const auto layout=raw_rows(a,nullptr,int(a.ndim()));
    const auto* data=static_cast<const unsigned char*>(a.data());unsigned flags=0;
    for(std::size_t row=0;row<layout.rows;++row)for(std::size_t col=0;col<layout.columns;++col) {
        const auto* pixel=data+layout.image[row]+std::ptrdiff_t(col)*layout.image_step;
        for(int component=0;component<(kind=='c' ? 2 : 1);++component) {
            const auto bits=raw_component(pixel+std::size_t(component)*bytes,bytes,swap);
            if(kind=='m' || kind=='M') {if(bits==0x8000000000000000ULL)flags|=1;continue;}
            const std::uint64_t exponent=bytes==2 ? 0x7c00ULL : bytes==4 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
            const std::uint64_t mantissa=bytes==2 ? 0x3ffULL : bytes==4 ? 0x7fffffULL : 0xfffffffffffffULL;
            const std::uint64_t quiet=bytes==2 ? 0x200ULL : bytes==4 ? 0x400000ULL : 0x8000000000000ULL;
            if((bits&exponent)==exponent && (bits&mantissa)) {flags|=1;if(!(bits&quiet))flags|=2;}
        }
    }
    return flags;
}

bool linear_color_view(const py::array& a, std::ptrdiff_t& pixel_stride, std::ptrdiff_t& channel_stride) {
    if (!a.size()) return false;
    const auto bytes = a.itemsize();
    if (reinterpret_cast<std::uintptr_t>(a.data()) % static_cast<std::size_t>(bytes)) return false;
    py::ssize_t step = 3*bytes, span = 1;
    bool found = false;
    for (int axis = static_cast<int>(a.ndim())-2; axis >= 0; --axis) {
        if (a.shape(axis) <= 1) continue;
        if (!found) { step = a.strides(axis); found = true; }
        else if (a.strides(axis)%span != 0 || a.strides(axis)/span != step) return false;
        span *= a.shape(axis); // NumPy's element count bounds this product.
    }
    const auto channels = a.strides(a.ndim()-1);
    if (step%bytes || channels%bytes) return false;
    pixel_stride = static_cast<std::ptrdiff_t>(step/bytes);
    channel_stride = static_cast<std::ptrdiff_t>(channels/bytes);
    return true;
}
py::array aligned_distance_view(py::array a) {
    const auto bytes=a.itemsize();
    bool aligned=reinterpret_cast<std::uintptr_t>(a.data())%std::size_t(bytes)==0;
    for(int axis=0;axis<a.ndim();++axis) aligned=aligned && a.strides(axis)%bytes==0;
    return aligned ? a : a.attr("copy")().cast<py::array>();
}
bool linear_histogram_input(const py::array& a,bool multichannel,std::ptrdiff_t& pixels,std::ptrdiff_t& channels) {
    pixels=channels=0;
    if(!a.size())return true;
    const auto bytes=a.itemsize();
    if(reinterpret_cast<std::uintptr_t>(a.data())%std::size_t(bytes))return false;
    py::ssize_t span=1,step=0;bool found=false;
    for(int axis=int(a.ndim())-(multichannel ? 2 : 1);axis>=0;--axis) {
        if(a.shape(axis)<=1)continue;
        const auto stride=a.strides(axis);
        if(stride%bytes)return false;
        if(!found){step=stride;found=true;}
        else if(stride%span || stride/span!=step)return false;
        span*=a.shape(axis);
    }
    pixels=found ? step/bytes : 0;
    if(multichannel && a.shape(a.ndim()-1)>1) {
        const auto stride=a.strides(a.ndim()-1);
        if(stride%bytes)return false;
        channels=stride/bytes;
    }
    return true;
}
struct DistanceRowLayout {
    std::size_t columns=1;
    std::ptrdiff_t ap=3,ac=1,bp=3,bc=1;
    std::vector<std::ptrdiff_t> first,second;
};
DistanceRowLayout distance_rows(const py::array& a,const py::array& b) {
    DistanceRowLayout result;
    const auto ab=a.itemsize(),bb=b.itemsize();
    result.ac=a.strides(a.ndim()-1)/ab;result.bc=b.strides(b.ndim()-1)/bb;
    int axis=int(a.ndim())-2;
    while(axis>=0 && a.shape(axis)==1) --axis;
    if(axis<0) {result.first={0};result.second={0};return result;}
    result.columns=std::size_t(a.shape(axis));
    result.ap=a.strides(axis)/ab;result.bp=b.strides(axis)/bb;
    int prefix=axis;
    for(int outer=axis-1;outer>=0;--outer) {
        const auto span=std::ptrdiff_t(result.columns);
        const auto astep=a.strides(outer)/ab,bstep=b.strides(outer)/bb;
        if(a.shape(outer)>1 && (astep%span || bstep%span || astep/span!=result.ap || bstep/span!=result.bp)) break;
        result.columns*=std::size_t(a.shape(outer));prefix=outer;
    }
    const auto rows=std::size_t(a.size()/3)/result.columns;
    result.first.resize(rows);result.second.resize(rows);
    for(std::size_t row=0;row<rows;++row) {
        auto remainder=row;
        std::ptrdiff_t first=0,second=0;
        for(int outer=prefix-1;outer>=0;--outer) {
            const auto dim=std::size_t(a.shape(outer)),index=remainder%dim;remainder/=dim;
            first+=std::ptrdiff_t(index)*(a.strides(outer)/ab);
            second+=std::ptrdiff_t(index)*(b.strides(outer)/bb);
        }
        result.first[row]=first;result.second[row]=second;
    }
    return result;
}
skimagexx::PixelType pixel_type(const py::dtype& dtype) {
    using Type = skimagexx::PixelType;
    const auto kind = dtype.kind();
    const auto size = dtype.itemsize();
    if (size > 1 && !dtype.attr("isnative").cast<bool>()) throw py::type_error("native byte order is required");
    if (kind == 'b' && size == 1) return Type::boolean;
    if (kind == 'u') {
        if (size == 1) return Type::uint8;
        if (size == 2) return Type::uint16;
        if (size == 4) return Type::uint32;
        if (size == 8) return Type::uint64;
    }
    if (kind == 'i') {
        if (size == 1) return Type::int8;
        if (size == 2) return Type::int16;
        if (size == 4) return Type::int32;
        if (size == 8) return Type::int64;
    }
    if (kind == 'f') {
        if (size == 4) return Type::float32;
        if (size == 8) return Type::float64;
    }
    throw py::type_error("unsupported dtype for native conversion");
}
void shape(const py::array& a, bool rgb) {
    if (a.ndim() != (rgb ? 3 : 2) || (rgb && a.shape(2) != 3))
        throw py::value_error(rgb ? "expected an HxWx3 RGB image" : "expected a 2D grayscale image");
    if (a.shape(0) == 0 || a.shape(1) == 0) throw py::value_error("image must be nonempty");
}
template<class T, bool preserve = false, bool flat = false, class F> py::array execute(const py::array& image, F function) {
    // NumPy owns the uninitialized output, avoiding vector zero-fill and capsules.
    // Ensure aligned input as well as contiguous input before typed C++ access.
    auto input = py::array_t<T,py::array::c_style>::ensure(image);
    if (!input) throw py::type_error("unable to make input contiguous");
    if ((reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T)) != 0) {
        input = py::array_t<T,py::array::c_style>::ensure(input.attr("copy")());
    }
    using Output = std::conditional_t<std::is_same_v<T,std::uint8_t> && !preserve,double,T>;
    const auto h = static_cast<std::size_t>(flat ? input.size() : input.shape(0));
    const auto w = flat ? std::size_t{1} : static_cast<std::size_t>(input.shape(1));
    const auto dimensions = flat ? std::vector<py::ssize_t>(input.shape(),input.shape()+input.ndim()) :
                                  std::vector<py::ssize_t>{input.shape(0),input.shape(1)};
    py::array_t<Output> output(dimensions);
    const auto* source = input.data();
    auto* destination = output.mutable_data();
    { py::gil_scoped_release release; function(source,destination,h,w); }
    return output;
}
template<class F> py::array dispatch(const py::array& image, F function) {
    if (dtype_matches(image.dtype(),py::dtype::of<float>())) return execute<float>(image,function);
    if (dtype_matches(image.dtype(),py::dtype::of<double>())) return execute<double>(image,function);
    if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return execute<std::uint8_t>(image,function);
    throw py::type_error("supported dtypes are uint8, float32, and float64");
}
}

PYBIND11_MODULE(_core, m) {
    m.def("match_histograms",[](py::array first,py::array reference,py::dtype output_dtype,
        bool multichannel,bool unsigned_source,bool half_round) {
        if(multichannel && (!first.ndim() || !reference.ndim()))
            throw py::value_error("histogram-matching channel axis requires non-scalar inputs");
        const auto channels=multichannel ? std::size_t(first.shape(first.ndim()-1)) : 1;
        if(multichannel && (!reference.ndim() || std::size_t(reference.shape(reference.ndim()-1))!=channels))
            throw py::value_error("histogram-matching channel counts must match");
        skimagexx::HistogramMatchingStrides strides{};
        auto prepare=[&](py::array a,std::ptrdiff_t& pixels,std::ptrdiff_t& component) {
            if(!linear_histogram_input(a,multichannel,pixels,component)) {
                a=a.attr("copy")().cast<py::array>();
                pixels=multichannel ? std::ptrdiff_t(channels) : 1;component=1;
            }
            return a;
        };
        first=prepare(first,strides.source_pixels,strides.source_channels);
        reference=prepare(reference,strides.reference_pixels,strides.reference_channels);
        std::vector<py::ssize_t> dims(first.shape(),first.shape()+first.ndim());
        py::array out(output_dtype,dims);
        const auto st=pixel_type(first.dtype()),rt=pixel_type(reference.dtype()),
            ot=pixel_type(output_dtype.attr("newbyteorder")("=").cast<py::dtype>());
        std::vector<unsigned> flags(channels,0);
        const bool swap=!output_dtype.attr("isnative").cast<bool>();
        {py::gil_scoped_release release;
         skimagexx::match_histograms_strided_into(first.data(),reference.data(),out.mutable_data(),
            channels ? std::size_t(first.size())/channels : 0,channels ? std::size_t(reference.size())/channels : 0,
            channels,st,rt,ot,strides,unsigned_source,half_round,swap,flags.data());}
        return py::make_tuple(out,flags);
    });
    if(_import_array()<0)throw py::error_already_set();
    m.def("label_ranks",[](py::array labels,std::uint64_t background,bool overlay)->py::object {
        auto input=py::array::ensure(labels,py::array::c_style);
        const auto native=py::dtype::from_args(input.dtype().attr("newbyteorder")("="));
        const auto type=pixel_type(native);
        const bool swap=!input.dtype().attr("isnative").cast<bool>();
        std::vector<py::ssize_t> shape(input.shape(),input.shape()+input.ndim());
        py::array_t<std::uint64_t> ranks(shape);skimagexx::LabelSummary info;bool ok;
        {py::gil_scoped_release release;ok=skimagexx::label_ranks(input.data(),std::size_t(input.size()),type,swap,background,overlay,ranks.mutable_data(),info);}
        if(!ok)return py::none();
        return py::make_tuple(ranks,info.regions,info.background,info.has_background);
    });
    m.def("label_image_valid",[](py::array image,bool overlay) {
        auto input=py::array::ensure(image,py::array::c_style);const auto type=pixel_type(input.dtype());
        py::gil_scoped_release release;return skimagexx::label_image_valid(input.data(),std::size_t(input.size()),type,overlay);
    });
    m.def("label_saturation",[](py::array hsv,double value) {
        auto input=py::array::ensure(hsv,py::array::c_style);const auto type=pixel_type(input.dtype());
        if(input.size()%3)throw py::value_error("three HSV channels required");
        {py::gil_scoped_release release;skimagexx::label_saturation(input.mutable_data(),std::size_t(input.size()/3),type,value);}
        return input;
    });
    m.def("label_overlay",[](py::array_t<std::uint64_t,py::array::c_style> ranks,py::object image,
        py::array_t<double,py::array::c_style|py::array::forcecast> palette,bool rgb,double alpha,double image_bg,double beta,
        bool remove,std::uint64_t background,py::dtype dtype) {
        if(reinterpret_cast<std::uintptr_t>(ranks.data())%alignof(std::uint64_t))ranks=decltype(ranks)::ensure(ranks.attr("copy")());
        if(reinterpret_cast<std::uintptr_t>(palette.data())%alignof(double))palette=decltype(palette)::ensure(palette.attr("copy")());
        if(palette.ndim()!=2 || palette.shape(1)!=3 || palette.shape(0)<2)throw py::value_error("RGB palette required");
        py::array input;const void* data=nullptr;auto type=skimagexx::PixelType::float64;
        if(!image.is_none()) {
            input=py::array::ensure(image,py::array::c_style);type=pixel_type(input.dtype());data=input.data();
            if(input.size()!=ranks.size()*(rgb ? 3 : 1))throw py::value_error("matching underlay storage required");
        }
        std::vector<py::ssize_t> shape(ranks.shape(),ranks.shape()+ranks.ndim());shape.push_back(3);
        py::array output(dtype,shape);const bool single=pixel_type(dtype)==skimagexx::PixelType::float32;
        {py::gil_scoped_release release;skimagexx::label_overlay(ranks.data(),std::size_t(ranks.size()),data,type,rgb,palette.data(),std::size_t(palette.shape(0)),alpha,image_bg,beta,remove,background,output.mutable_data(),single);}
        return output;
    });
    m.def("label_average",[](py::array_t<std::uint64_t,py::array::c_style> ranks,std::size_t regions,py::array image,
        bool rgb,py::array background,bool has_background,std::size_t reduction_block) {
        if(reinterpret_cast<std::uintptr_t>(ranks.data())%alignof(std::uint64_t))ranks=decltype(ranks)::ensure(ranks.attr("copy")());
        auto input=py::array::ensure(image,py::array::c_style),bg=py::array::ensure(background,py::array::c_style);
        if(input.size()!=ranks.size()*(rgb ? 3 : 1) || bg.size()!=3 || pixel_type(bg.dtype())!=pixel_type(input.dtype()))throw py::value_error("matching average storage required");
        const auto type=pixel_type(input.dtype());
        std::vector<py::ssize_t> shape(ranks.shape(),ranks.shape()+ranks.ndim());shape.push_back(3);
        py::array output(input.dtype(),shape);
        {py::gil_scoped_release release;skimagexx::label_average(ranks.data(),std::size_t(ranks.size()),regions,input.data(),type,rgb,bg.data(),has_background,output.mutable_data(),reduction_block);}
        return output;
    });
    m.def("contrast",[](py::array image,py::object ranks) -> py::object {
        skimagexx::ContrastView view;view.data=image.data();
        const auto descriptor=image.dtype();view.half=descriptor.kind()=='f' && descriptor.itemsize()==2;
        view.swap=!PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject*>(image.ptr()));
        const auto native_descriptor=view.swap ? py::dtype::from_args(descriptor.attr("type")) : descriptor;
        view.type=view.half ? skimagexx::PixelType::uint16 : pixel_type(native_descriptor);
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            view.shape.push_back(std::size_t(image.shape(axis)));view.strides.push_back(image.strides(axis));
        }
        if(ranks.is_none()) {
            bool constant;
            {py::gil_scoped_release release;constant=skimagexx::contrast_boolean_constant(view);}
            return py::bool_(constant);
        }
        const auto positions=ranks.cast<std::array<double,2>>();std::array<double,2> output;
        bool supported;
        {py::gil_scoped_release release;supported=skimagexx::contrast_percentiles(view,positions,output);}
        return supported ? py::cast(output) : py::none();
    });
    m.def("downscale_local_mean",[](py::array image,py::array result,
        const std::vector<std::size_t>& factors,py::object constant) {
        skimagexx::LocalMeanView view;view.data=image.data();
        const auto descriptor=image.dtype();view.half=descriptor.kind()=='f' && descriptor.itemsize()==2;
        view.swap=!PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject*>(image.ptr()));
        const auto native_descriptor=view.swap ? py::dtype::from_args(descriptor.attr("type")) : descriptor;
        view.type=view.half ? skimagexx::PixelType::uint16 : pixel_type(native_descriptor);
        if(result.ndim()!=image.ndim() || factors.size()!=std::size_t(image.ndim()) || !result.writeable())
            throw py::value_error("invalid local-mean output dimensions");
        const auto output_descriptor=result.dtype();
        const bool valid_output=view.half ? output_descriptor.kind()=='f' && output_descriptor.itemsize()==2
            : pixel_type(output_descriptor)==(view.type==skimagexx::PixelType::float32 ? skimagexx::PixelType::float32 : skimagexx::PixelType::float64);
        if(!valid_output || !PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject*>(result.ptr())))
            throw py::type_error("native local-mean output dtype required");
        std::vector<std::size_t> output_strides;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            const auto length=std::size_t(image.shape(axis)),factor=factors[std::size_t(axis)];
            if(!factor || std::size_t(result.shape(axis))!=(length ? (length-1)/factor+1 : 0))
                throw py::value_error("invalid local-mean output shape");
            view.shape.push_back(length);view.strides.push_back(image.strides(axis));
            if(result.strides(axis)<0)throw py::value_error("positive local-mean output strides required");
            output_strides.push_back(std::size_t(result.strides(axis)));
        }
        const void* constant_data=nullptr;
        if(!constant.is_none()) {
            const auto constant_storage=constant.cast<py::array>();
            if(constant_storage.size()!=1 || !dtype_matches(constant_storage.dtype(),descriptor))
                throw py::type_error("one input-storage padding constant required");
            constant_data=constant_storage.data();
        }
        auto* output_data=result.mutable_data();
        py::gil_scoped_release release;
        return skimagexx::downscale_local_mean_into(view,factors,constant_data,output_data,output_strides);
    });
    m.def("distance_metric",[](py::array first,py::array second,int operation,std::array<double,5> parameters,
        bool wide_trig,bool promoted_cmc,bool legacy94,unsigned scalar_parameter_mask,bool legacy_ciede2000) -> py::array {
        if(first.ndim()<1 || second.ndim()<1 || first.shape(first.ndim()-1)!=3 || second.shape(second.ndim()-1)!=3)
            throw py::value_error("distance inputs must have three channels");
        if(first.ndim()!=second.ndim() || !std::equal(first.shape(),first.shape()+first.ndim(),second.shape()))
            throw py::value_error("distance inputs must have matching broadcasted shapes");
        const auto at=pixel_type(first.dtype()),bt=pixel_type(second.dtype());
        if(legacy94 && (operation!=0 || first.ndim()!=1 || at!=skimagexx::PixelType::float32 || bt!=at))
            throw py::value_error("legacy CIE94 requires two scalar float32 colors");
        auto prepare=[](py::array a,std::ptrdiff_t& pixels,std::ptrdiff_t& channels) {
            if(!linear_color_view(a,pixels,channels)) {a=a.attr("copy")().cast<py::array>();pixels=3;channels=1;}
            return a;
        };
        std::ptrdiff_t ap=3,ac=1,bp=3,bc=1;
        first=prepare(first,ap,ac);second=prepare(second,bp,bc);
        std::vector<py::ssize_t> dims(first.shape(),first.shape()+first.ndim()-1);
        const auto dtype=at==skimagexx::PixelType::float32 && bt==skimagexx::PixelType::float32 && !promoted_cmc && !legacy94 ?
            py::dtype::of<float>() : py::dtype::of<double>();
        py::array out(dtype,dims);
        const skimagexx::DistanceParameters p{parameters[0],parameters[1],parameters[2],parameters[3],parameters[4]};
        if(legacy94) {
            *static_cast<double*>(out.mutable_data())=skimagexx::delta_e94_scalar32(
                static_cast<const float*>(first.data()),static_cast<const float*>(second.data()),p,
                bool(scalar_parameter_mask&1),bool(scalar_parameter_mask&2),ac,bc);
            return out;
        }
        {py::gil_scoped_release release;
         skimagexx::delta_e_into(first.data(),second.data(),out.mutable_data(),std::size_t(out.size()),at,bt,
            static_cast<skimagexx::DistanceMetric>(operation),p,wide_trig,promoted_cmc,ap,ac,bp,bc,legacy_ciede2000);}
        return out;
    },py::arg("first"),py::arg("second"),py::arg("operation"),py::arg("parameters"),py::arg("wide_trigonometry"),py::arg("promoted_cmc"),py::arg("legacy94")=false,py::arg("scalar_parameter_mask")=0,py::arg("legacy_ciede2000")=false);
    m.def("delta_e76",[](py::array first,py::array second,bool legacy_scalar) -> py::array {
        if(first.ndim()<1 || second.ndim()<1 || first.shape(first.ndim()-1)!=3 || second.shape(second.ndim()-1)!=3)
            throw py::value_error("distance inputs must have three channels");
        if(first.ndim()!=second.ndim() || !std::equal(first.shape(),first.shape()+first.ndim(),second.shape()))
            throw py::value_error("distance inputs must have matching broadcasted shapes");
        const auto at=pixel_type(first.dtype()),bt=pixel_type(second.dtype());
        if(legacy_scalar && (first.ndim()!=1 || at!=skimagexx::PixelType::float32 || bt!=at))
            throw py::value_error("legacy CIE76 requires two scalar float32 colors");
        std::ptrdiff_t ap=3,ac=1,bp=3,bc=1;
        first=aligned_distance_view(first);second=aligned_distance_view(second);
        const bool linear=!first.size() || (linear_color_view(first,ap,ac) && linear_color_view(second,bp,bc));
        std::vector<py::ssize_t> dims(first.shape(),first.shape()+first.ndim()-1);
        const auto dtype=at==skimagexx::PixelType::float32 && bt==skimagexx::PixelType::float32 && !legacy_scalar ?
            py::dtype::of<float>() : py::dtype::of<double>();
        py::array out(dtype,dims);
        if(legacy_scalar) {
            *static_cast<double*>(out.mutable_data())=skimagexx::delta_e76_scalar32(
                static_cast<const float*>(first.data()),static_cast<const float*>(second.data()),ac,bc);
            return out;
        }
        if(linear) {
            py::gil_scoped_release release;
            skimagexx::delta_e76_into(first.data(),second.data(),out.mutable_data(),std::size_t(out.size()),at,bt,ap,ac,bp,bc);
        } else {
            const auto layout=distance_rows(first,second);
            py::gil_scoped_release release;
            skimagexx::delta_e76_rows_into(first.data(),second.data(),out.mutable_data(),layout.first.size(),layout.columns,
                at,bt,layout.first.data(),layout.second.data(),layout.ap,layout.ac,layout.bp,layout.bc);
        }
        if(dims.empty()) return out.attr("reshape")(py::tuple()).cast<py::array>();
        return out;
    },py::arg("first"),py::arg("second"),py::arg("legacy_scalar")=false);
    m.def("stain_transform",[](py::array image,py::array_t<double,py::array::c_style|py::array::forcecast> weights,
        bool combine,bool intermediate32,bool output32,double adjust) {
        if (!image.ndim() || image.shape(image.ndim()-1)!=3) throw py::value_error("expected three input channels");
        if (weights.ndim()!=2 || weights.shape(1)!=3) throw py::value_error("expected an output-major stain matrix");
        if (!weights.attr("flags").attr("aligned").cast<bool>()) weights=weights.attr("copy")().cast<py::array_t<double>>();
        const auto type=pixel_type(image.dtype());
        const auto count=static_cast<std::size_t>(image.size()/3),channels=static_cast<std::size_t>(weights.shape(0));
        auto dims=std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()); dims.back()=weights.shape(0);
        py::array output(output32 ? py::dtype::of<float>() : py::dtype::of<double>(),dims);
        std::ptrdiff_t pixels=3,stride=1;
        py::array input=image;
        if ((type!=skimagexx::PixelType::float32 && type!=skimagexx::PixelType::float64) || !linear_color_view(image,pixels,stride)) {
            input=py::array::ensure(image,py::array::c_style);
            if (!input) throw py::type_error("unable to make input contiguous");
            if (reinterpret_cast<std::uintptr_t>(input.data())%skimagexx::pixel_size(type)) input=input.attr("copy")().cast<py::array>();
            pixels=3; stride=1;
        }
        { py::gil_scoped_release release;
          skimagexx::stain_transform_into(input.data(),output.mutable_data(),count,type,
            combine ? skimagexx::StainTransform::combine : skimagexx::StainTransform::separate,weights.data(),channels,
            intermediate32 ? skimagexx::PixelType::float32 : skimagexx::PixelType::float64,
            output32 ? skimagexx::PixelType::float32 : skimagexx::PixelType::float64,pixels,stride,adjust); }
        return output;
    });
    m.def("morphology", [](py::array image,py::list footprints,int operation,int mode,double cval,py::object out) {
        const auto in_type=pixel_type(image.dtype());
        std::vector<py::ssize_t> dimensions(image.shape(),image.shape()+image.ndim());
        std::vector<std::size_t> sizes(dimensions.begin(),dimensions.end());
        auto input=py::array::ensure(image,py::array::c_style);
        if (!input.attr("flags").attr("aligned").cast<bool>()) input=input.attr("copy")().cast<py::array>();
        py::array target=out.is_none() ? py::array(operation>=6 ? py::dtype::of<bool>() : image.dtype(),dimensions) : out.cast<py::array>();
        const auto out_type=pixel_type(target.dtype());
        if (!target.writeable()) throw py::value_error("output array is read-only");
        if (target.ndim()!=image.ndim() || !std::equal(dimensions.begin(),dimensions.end(),target.shape())) throw py::value_error("output shape mismatch");
        const bool direct=(target.flags() & py::array::c_style) && target.attr("flags").attr("aligned").cast<bool>();
        py::array buffer=direct ? target : py::array(target.dtype(),dimensions);
        std::vector<skimagexx::MorphologyFootprint> plans;
        for (auto item:footprints) {
            auto tuple=py::cast<py::tuple>(item);
            auto offsets=py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>::ensure(tuple[0]);
            if (!offsets || offsets.ndim()!=2 || offsets.shape(1)!=image.ndim()) throw py::value_error("footprint rank mismatch");
            skimagexx::MorphologyFootprint fp;
            fp.offsets.assign(offsets.data(),offsets.data()+offsets.size());
            fp.iterations=tuple[1].cast<std::int64_t>(); fp.scipy_separable=tuple[2].cast<bool>(); plans.push_back(std::move(fp));
        }
        skimagexx::MorphologyInfo info;
        { py::gil_scoped_release release; info=skimagexx::morphology_into(input.data(),buffer.mutable_data(),in_type,out_type,sizes,plans,
            static_cast<skimagexx::MorphologyOp>(operation),static_cast<skimagexx::MorphologyMode>(mode),cval); }
        if (!info.unsupported_nan && !direct) target.attr("__setitem__")(py::ellipsis(),buffer);
        return py::make_tuple(target,info.unsupported_nan,info.invalid_subtract,info.overflow_subtract);
    });
    m.def("perceptual_transform", [](py::array image, int operation, std::array<double,3> white,py::object uv) {
        if (!image.ndim()) throw py::value_error("expected a color array");
        const auto nch = image.shape(image.ndim()-1);
        if (nch < 3 || ((operation!=6 && operation!=7) && nch != 3)) throw py::value_error("invalid color channel count");
        const auto type = pixel_type(image.dtype());
        const bool floating = type == skimagexx::PixelType::float32 || type == skimagexx::PixelType::float64;
        const auto dtype = type == skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        py::array output(dtype,std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        const auto n = static_cast<std::size_t>(image.size()/nch);
        std::ptrdiff_t pixels = nch, channels = 1;
        py::array input = image;
        if (!floating || !linear_color_view(image,pixels,channels)) {
            input = py::array::ensure(image,py::array::c_style);
            if (!input) throw py::type_error("unable to make input contiguous");
            if (reinterpret_cast<std::uintptr_t>(input.data()) % skimagexx::pixel_size(type))
                input = py::array::ensure(input.attr("copy")(),py::array::c_style);
            pixels = nch; channels = 1;
        }
        const auto* source = input.data(); auto* destination = output.mutable_data();
        skimagexx::PerceptualInfo info;
        std::optional<std::array<double,2>> uv_reference;
        if (!uv.is_none()) uv_reference=uv.cast<std::array<double,2>>();
        { py::gil_scoped_release release; info = skimagexx::perceptual_into(source,destination,n,type,
            static_cast<skimagexx::PerceptualTransform>(operation),white,static_cast<std::size_t>(nch),pixels,channels,uv_reference); }
        if (image.ndim() == 1 && type == skimagexx::PixelType::float32 && info.negative_z && (operation == 3 || operation == 5)) {
            py::array_t<double> promoted(3);
            auto* target = promoted.mutable_data();
            { py::gil_scoped_release release; skimagexx::lab_scalar_negative_z_into(static_cast<const float*>(source),target,
                static_cast<skimagexx::PerceptualTransform>(operation),white,channels); }
            output = std::move(promoted);
        }
        return py::make_tuple(output,info.negative_z,info.unsupported);
    },py::arg("image"),py::arg("operation"),py::arg("white"),py::arg("uv_reference")=py::none());
    m.def("color_transform", [](py::array image, int operation) {
        if (!image.ndim() || image.shape(image.ndim()-1) != 3) throw py::value_error("expected an array of color triples");
        const auto input_type = pixel_type(image.dtype());
        const auto output_type = input_type == skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        py::array output(output_type,std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        auto* destination = output.mutable_data();
        const auto count = static_cast<std::size_t>(image.size()/3);
        std::ptrdiff_t pixels, channels;
        if ((input_type == skimagexx::PixelType::float32 || input_type == skimagexx::PixelType::float64) && linear_color_view(image,pixels,channels)) {
            const auto* source = image.data();
            { py::gil_scoped_release release;
              if (input_type == skimagexx::PixelType::float32)
                  skimagexx::color_transform_strided_into(static_cast<const float*>(source),static_cast<float*>(destination),count,pixels,channels,static_cast<skimagexx::ColorTransform>(operation));
              else skimagexx::color_transform_strided_into(static_cast<const double*>(source),static_cast<double*>(destination),count,pixels,channels,static_cast<skimagexx::ColorTransform>(operation)); }
            return output;
        }
        auto input = py::array::ensure(image,py::array::c_style);
        if (!input) throw py::type_error("unable to make input contiguous");
        if ((reinterpret_cast<std::uintptr_t>(input.data()) % skimagexx::pixel_size(input_type)) != 0)
            input = py::array::ensure(input.attr("copy")(),py::array::c_style);
        const auto* source = input.data();
        { py::gil_scoped_release release;
          skimagexx::color_transform_into(source,destination,count,input_type,static_cast<skimagexx::ColorTransform>(operation)); }
        return output;
    },py::arg("image"),py::arg("operation"));
    m.def("convert", [](py::array image, py::dtype dtype) {
        const auto source_type = pixel_type(image.dtype()), target_type = pixel_type(dtype);
        const bool physical_fortran = (image.flags()&py::array::f_style) && !(image.flags()&py::array::c_style)
            && (target_type==skimagexx::PixelType::boolean ||
                (source_type==skimagexx::PixelType::uint16 && target_type==skimagexx::PixelType::int16));
        auto input = physical_fortran ? image : py::array::ensure(image,py::array::c_style);
        if (!input) throw py::type_error("unable to make input contiguous");
        if ((reinterpret_cast<std::uintptr_t>(input.data()) % skimagexx::pixel_size(source_type)) != 0)
            input = input.attr("copy")(py::arg("order")=physical_fortran ? "F" : "C").cast<py::array>();
        // Allocate in the input's physical order without reading any pixels.
        // Boolean comparison and unscaled narrowing preserve upstream F order.
        py::array output = physical_fortran
            ? py::module_::import("numpy").attr("empty_like")(input,py::arg("dtype")=dtype,
                py::arg("order")="F",py::arg("subok")=false).cast<py::array>()
            : py::array(dtype,std::vector<py::ssize_t>(input.shape(),input.shape()+input.ndim()));
        const auto* source = input.data(); auto* destination = output.mutable_data();
        const auto n = static_cast<std::size_t>(input.size());
        skimagexx::ConversionInfo info;
        { py::gil_scoped_release release; info = skimagexx::convert_into(source,destination,n,source_type,target_type); }
        // Upstream's scaled uint16 -> int16 branch explicitly allocates C order.
        if(physical_fortran && target_type==skimagexx::PixelType::int16 && !info.downcast_without_scaling)
            output=output.attr("copy")(py::arg("order")="C").cast<py::array>();
        const bool unsigned_input = input.dtype().kind() == 'u';
        py::object maximum = unsigned_input ? py::object(py::int_(info.maximum_unsigned)) : py::object(py::int_(info.maximum_signed));
        return py::make_tuple(output,info.downcast_without_scaling,info.nan_to_integer,info.floating_overflow,info.floating_underflow,info.invalid_float_cast,maximum);
    },py::arg("image"),py::arg("dtype"));
    m.def("histogram_u8", [](py::array image, bool full_range, bool normalize, bool cumulative) -> py::tuple {
        if (!image.size()) throw py::value_error("image must be nonempty");
        if (!dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) throw py::type_error("histogram requires uint8");
        auto input = py::array_t<std::uint8_t,py::array::c_style>::ensure(image);
        const auto* p = input.data(); const auto n = static_cast<std::size_t>(input.size());
        std::array<std::uint64_t,256> counts;
        { py::gil_scoped_release release; counts = skimagexx::histogram_u8(p,n,1); }
        std::size_t first = 0, last = 255;
        if (!full_range) {
            while (counts[first] == 0) ++first;
            while (counts[last] == 0) --last;
        }
        const auto bins = static_cast<py::ssize_t>(last-first+1);
        py::array_t<std::int64_t> centers(bins);
        for (std::size_t i = first; i <= last; ++i) centers.mutable_data()[i-first] = static_cast<std::int64_t>(i);
        if (normalize || cumulative) {
            py::array_t<double> values(bins);
            std::uint64_t total = 0;
            for (std::size_t i = first; i <= last; ++i) {
                total += counts[i];
                values.mutable_data()[i-first] = double(cumulative ? total : counts[i])/double(n);
            }
            return py::make_tuple(values,centers);
        }
        py::array_t<std::int64_t> values(bins);
        for (std::size_t i = first; i <= last; ++i) values.mutable_data()[i-first] = static_cast<std::int64_t>(counts[i]);
        return py::make_tuple(values,centers);
    },py::arg("image"),py::arg("full_range")=false,py::arg("normalize")=false,py::arg("cumulative")=false);
    m.def("equalize_hist", [](py::array image) {
        if (!image.size()) throw py::value_error("image must be nonempty");
        if (!dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) throw py::type_error("equalize_hist requires uint8");
        return execute<std::uint8_t,false,true>(image,[](const auto* p, auto* out, auto h, auto w) {
            skimagexx::equalize_hist_into(p,out,h,w);
        });
    },py::arg("image"));
    m.def("intensity_bounds", [](py::array image) {
        if (!image.size()) throw py::value_error("image must be nonempty");
        auto run = [&](auto dummy) {
            using T = decltype(dummy);
            auto input = py::array_t<T,py::array::c_style>::ensure(image);
            if ((reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T)) != 0)
                input = py::array_t<T,py::array::c_style>::ensure(input.attr("copy")());
            const auto* p = input.data(); const auto n = static_cast<std::size_t>(input.size());
            std::pair<double,double> range;
            { py::gil_scoped_release release; range = skimagexx::intensity_bounds(p,n,1); }
            return py::make_tuple(range.first,range.second);
        };
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return run(std::uint8_t{});
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return run(float{});
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return run(double{});
        throw py::type_error("unsupported dtype");
    });
    m.def("rescale", [](py::array image, double imin, double imax, double omin, double omax, bool float_output) {
        if (!image.size()) throw py::value_error("image must be nonempty");
        auto fn = [=](const auto* p, auto* out, auto h, auto w) { skimagexx::rescale_into(p,out,h,w,imin,imax,omin,omax); };
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) {
            if (float_output) return execute<std::uint8_t,false,true>(image,fn);
            return execute<std::uint8_t,true,true>(image,fn);
        }
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return execute<float,true,true>(image,fn);
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return execute<double,true,true>(image,fn);
        throw py::type_error("unsupported dtype");
    },py::arg("image"),py::arg("imin"),py::arg("imax"),py::arg("omin"),py::arg("omax"),py::arg("float_output")=false);
    m.def("exposure", [](py::array image, const std::string& name, double parameter, double gain, bool inverse) {
        if (!image.size()) throw py::value_error("image must be nonempty");
        skimagexx::ExposureTransform transform;
        if (name == "gamma") transform = skimagexx::ExposureTransform::gamma;
        else if (name == "log") transform = skimagexx::ExposureTransform::log;
        else if (name == "sigmoid") transform = skimagexx::ExposureTransform::sigmoid;
        else throw py::value_error("unknown exposure transform");
        auto fn = [=](const auto* p, auto* out, auto h, auto w) { skimagexx::exposure_into(p,out,h,w,transform,parameter,gain,inverse); };
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return execute<std::uint8_t,true,true>(image,fn);
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return execute<float,true,true>(image,fn);
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return execute<double,true,true>(image,fn);
        throw py::type_error("supported dtypes are uint8, float32, and float64");
    },py::arg("image"),py::arg("name"),py::arg("parameter"),py::arg("gain"),py::arg("inverse")=false);
    m.def("histogram_edges", [](py::array image,py::array edges) -> py::array {
        auto run=[&](auto dummy) -> py::array {
            using T=decltype(dummy);
            auto input=py::array_t<T,py::array::c_style>::ensure(image);
            if(reinterpret_cast<std::uintptr_t>(input.data())%alignof(T))
                input=py::array_t<T,py::array::c_style>::ensure(input.attr("copy")());
            auto e=py::array_t<T,py::array::c_style|py::array::forcecast>::ensure(edges);
            if(reinterpret_cast<std::uintptr_t>(e.data())%alignof(T))e=decltype(e)::ensure(e.attr("copy")());
            if(e.ndim()!=1 || e.size()<2)throw py::value_error("at least two edges required");
            std::vector<std::uint64_t> counts;
            {py::gil_scoped_release release;counts=skimagexx::histogram_edges(input.data(),std::size_t(input.size()),e.data(),std::size_t(e.size()-1));}
            py::array_t<std::int64_t> output(py::ssize_t(counts.size()));
            std::copy(counts.begin(),counts.end(),output.mutable_data());return output;
        };
        if(dtype_matches(image.dtype(),py::dtype::of<float>()))return run(float{});
        if(dtype_matches(image.dtype(),py::dtype::of<double>()))return run(double{});
        throw py::type_error("floating image required");
    });
    m.def("threshold_logarithms", [](py::array_t<float,py::array::c_style|py::array::forcecast> values,bool numpy_log) {
        py::array_t<float> output(values.size());
        if(reinterpret_cast<std::uintptr_t>(values.data())%alignof(float))values=decltype(values)::ensure(values.attr("copy")());
        skimagexx::threshold_logs(values.data(),output.mutable_data(),std::size_t(values.size()),numpy_log);
        return output;
    });
    m.def("threshold_minimum", [](py::array_t<float,py::array::c_style|py::array::forcecast> counts,std::size_t iterations) {
        if(counts.ndim()!=1)throw py::value_error("one-dimensional histogram required");
        if(reinterpret_cast<std::uintptr_t>(counts.data())%alignof(float))counts=decltype(counts)::ensure(counts.attr("copy")());
        py::gil_scoped_release release;
        return skimagexx::threshold_minimum(counts.data(),std::size_t(counts.size()),iterations);
    });
    m.def("threshold_indices", [](py::array_t<double,py::array::c_style|py::array::forcecast> counts,
            py::array_t<double,py::array::c_style|py::array::forcecast> centers,
            const std::string& method,bool float_centers,bool numpy_log,double bin_width) {
        if(reinterpret_cast<std::uintptr_t>(counts.data())%alignof(double))counts=decltype(counts)::ensure(counts.attr("copy")());
        if(reinterpret_cast<std::uintptr_t>(centers.data())%alignof(double))centers=decltype(centers)::ensure(centers.attr("copy")());
        if(counts.ndim()!=1 || centers.ndim()!=1 || counts.size()!=centers.size())throw py::value_error("matching one-dimensional histogram required");
        skimagexx::HistogramThreshold kind;
        if(method=="yen")kind=skimagexx::HistogramThreshold::yen;
        else if(method=="triangle")kind=skimagexx::HistogramThreshold::triangle;
        else if(method=="isodata")kind=skimagexx::HistogramThreshold::isodata;
        else throw py::value_error("unknown threshold method");
        std::vector<std::size_t> indices;unsigned warnings=0;
        {py::gil_scoped_release release;indices=skimagexx::threshold_indices(counts.data(),centers.data(),std::size_t(counts.size()),kind,float_centers,&warnings,numpy_log,bin_width);}
        py::array_t<py::ssize_t> result(py::ssize_t(indices.size()));
        std::copy(indices.begin(),indices.end(),result.mutable_data());return py::make_tuple(result,warnings);
    });
    m.def("local_threshold", [](py::array image,const std::vector<std::size_t>& window,
            const std::string& method,double k,double r,bool divide_double,bool r_double,bool k_double) -> py::array {
        const auto type=pixel_type(image.dtype());
        auto input=py::array::ensure(image,py::array::c_style);
        if(!input)throw py::type_error("array storage required");
        if(reinterpret_cast<std::uintptr_t>(input.data())%std::size_t(input.itemsize()))input=input.attr("copy")().cast<py::array>();
        std::vector<std::size_t> dimensions;
        std::vector<py::ssize_t> output_shape;
        for(py::ssize_t axis=0;axis<input.ndim();++axis) {
            dimensions.push_back(std::size_t(input.shape(axis)));output_shape.push_back(input.shape(axis));
        }
        skimagexx::LocalThreshold operation;
        if(method=="niblack")operation=skimagexx::LocalThreshold::niblack;
        else if(method=="sauvola")operation=skimagexx::LocalThreshold::sauvola;
        else throw py::value_error("invalid local threshold method");
        const bool wide=type!=skimagexx::PixelType::float32 || k_double || (operation==skimagexx::LocalThreshold::sauvola && r_double);
        py::array result(wide ? py::dtype::of<double>() : py::dtype::of<float>(),output_shape);
        const skimagexx::LocalThresholdParameters parameters{k,r,divide_double,r_double,k_double};
        {py::gil_scoped_release release;skimagexx::local_threshold_into(input.data(),result.mutable_data(),type,dimensions,window,operation,parameters);}
        return result;
    });
    m.def("edge_native", [](py::array image) {
        const auto bytes=std::size_t(image.itemsize());
        if(image.dtype().attr("isnative").cast<bool>() || image.dtype().attr("hasobject").cast<bool>() || (bytes!=2 && bytes!=4 && bytes!=8) || std::string("iuf").find(image.dtype().kind())==std::string::npos)throw py::type_error("nonnative real multibyte storage required");
        auto input=py::array::ensure(image,py::array::c_style);
        py::dtype dtype=py::dtype::from_args(image.dtype().attr("type"));
        py::array output(dtype,std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        {py::gil_scoped_release release;skimagexx::edge_swap_into(input.data(),output.mutable_data(),std::size_t(input.size()),bytes);}
        return output;
    });
    m.def("edge_half", [](py::array image) {
        if(image.dtype().kind()!='f' || image.itemsize()!=2 || !image.dtype().attr("isnative").cast<bool>())throw py::type_error("native binary16 input required");
        auto input=py::array::ensure(image,py::array::c_style);
        if(reinterpret_cast<std::uintptr_t>(input.data())%2)input=input.attr("copy")().cast<py::array>();
        py::array_t<float> output(std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        {py::gil_scoped_release release;skimagexx::edge_half_to_float(static_cast<const std::uint16_t*>(input.data()),output.mutable_data(),std::size_t(input.size()));}
        return output;
    });
    m.def("farid", [](py::array image,const std::vector<std::ptrdiff_t>& axes,int border,double cval,bool directional) -> py::array {
        const auto type=pixel_type(image.dtype());
        if(type!=skimagexx::PixelType::uint8 && type!=skimagexx::PixelType::float32 && type!=skimagexx::PixelType::float64)throw py::type_error("prepared real image required");
        auto input=py::array::ensure(image,py::array::c_style);
        if(reinterpret_cast<std::uintptr_t>(input.data())%std::size_t(input.itemsize()))input=input.attr("copy")().cast<py::array>();
        const std::vector<py::ssize_t> dimensions(image.shape(),image.shape()+image.ndim());
        const std::vector<std::size_t> shape_values(dimensions.begin(),dimensions.end());
        const auto output_dtype=type==skimagexx::PixelType::uint8 ? py::dtype::of<double>() :
            !directional ? image.dtype() : type==skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>();
        py::array output(output_dtype,dimensions);
        {py::gil_scoped_release release;skimagexx::farid_into(input.data(),output.mutable_data(),type,shape_values,axes,static_cast<skimagexx::FaridBorder>(border),cval,directional);}
        return output;
    });
    m.def("laplace_nd", [](py::array image) -> py::array {
        if(image.ndim()<1)throw py::value_error("at least one image dimension required");
        const std::vector<py::ssize_t> dimensions(image.shape(),image.shape()+image.ndim());
        const std::vector<std::size_t> shape_values(dimensions.begin(),dimensions.end());
        auto run=[&](auto dummy) -> py::array {
            using T=decltype(dummy);using F=std::conditional_t<std::is_same_v<T,std::uint8_t>,double,T>;
            auto input=py::array_t<T,py::array::c_style>::ensure(image);
            if(reinterpret_cast<std::uintptr_t>(input.data())%alignof(T))input=input.attr("copy")().template cast<decltype(input)>();
            py::array_t<F> output(dimensions);
            {py::gil_scoped_release release;skimagexx::laplace_nd_into(input.data(),output.mutable_data(),shape_values);}
            return output;
        };
        if(dtype_matches(image.dtype(),py::dtype::of<float>()))return run(float{});
        if(dtype_matches(image.dtype(),py::dtype::of<double>()))return run(double{});
        if(dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>()))return run(std::uint8_t{});
        throw py::type_error("prepared real image required");
    });
    m.def("edge_apply_mask", [](py::array image,py::array mask) {
        const bool single=dtype_matches(image.dtype(),py::dtype::of<float>());
        if((!single && !dtype_matches(image.dtype(),py::dtype::of<double>())) || !(image.flags()&py::array::c_style) || !image.attr("flags").attr("aligned").cast<bool>() || !image.writeable())throw py::type_error("contiguous writable floating output required");
        if(!dtype_matches(mask.dtype(),py::dtype::of<bool>()) || mask.ndim()!=image.ndim())throw py::type_error("broadcast boolean mask required");
        for(py::ssize_t axis=0;axis<image.ndim();++axis)if(image.shape(axis)!=mask.shape(axis))throw py::value_error("mask shape mismatch");
        const auto layout=raw_rows(image,&mask,int(image.ndim()));
        {py::gil_scoped_release release;skimagexx::edge_mask_rows_into(image.mutable_data(),single,static_cast<const unsigned char*>(mask.data()),layout.rows,layout.columns,layout.alpha.data(),layout.alpha_step);}
    });
    m.def("small_edge", [](py::array image,int operation) {
        shape(image,false);
        return dispatch(image,[=](const auto* input,auto* output,auto h,auto w) {
            skimagexx::small_edge_into(input,output,h,w,static_cast<skimagexx::SmallEdge>(operation));
        });
    });
    m.def("alpha_array_flags", [](py::array a) {return alpha_array_flags(a);});
    m.def("gray_expand", [](py::array image,py::object alpha_object,int channels,py::dtype target,py::object swap_object) -> py::array {
        if(image.dtype().attr("hasobject").cast<bool>() || target.attr("hasobject").cast<bool>() || target.itemsize()!=image.itemsize())throw py::type_error("matching fixed-size object-free storage required");
        if(channels!=3 && channels!=4)throw py::value_error("three or four channels required");
        py::array alpha;
        if(channels==4) {
            alpha=alpha_object.cast<py::array>();
            if(!image.dtype().attr("__eq__")(alpha.dtype()).cast<bool>() || image.ndim()!=alpha.ndim())throw py::type_error("matching alpha dtype and shape required");
            for(py::ssize_t axis=0;axis<image.ndim();++axis)if(image.shape(axis)!=alpha.shape(axis))throw py::value_error("alpha must be broadcast to image.shape");
        }
        std::size_t swap_component=0;std::vector<std::size_t> byte_order;
        if(py::isinstance<py::int_>(swap_object))swap_component=swap_object.cast<std::size_t>();
        else {
            byte_order=swap_object.cast<std::vector<std::size_t>>();
            if(byte_order.size()!=std::size_t(image.itemsize()))throw py::value_error("one byte index per item byte required");
        }
        std::vector<py::ssize_t> shape;
        for(py::ssize_t axis=0;axis<image.ndim();++axis)shape.push_back(image.shape(axis));
        shape.push_back(channels);py::array result(target,shape);
        const auto layout=raw_rows(image,channels==4 ? &alpha : nullptr,int(image.ndim()));
        {py::gil_scoped_release release;skimagexx::gray_expand_rows_into(image.data(),channels==4 ? alpha.data() : nullptr,
            result.mutable_data(),layout.rows,layout.columns,std::size_t(image.itemsize()),channels,layout.image.data(),layout.alpha.data(),layout.image_step,layout.alpha_step,swap_component,byte_order.empty() ? nullptr : byte_order.data());}
        return result;
    });
    m.def("rgba_composite", [](py::array image,py::array_t<double,py::array::c_style|py::array::forcecast> background) -> py::array {
        if(image.ndim()<1 || image.shape(image.ndim()-1)!=4 || background.ndim()!=1 || background.size()!=3)
            throw py::value_error("RGBA channels and three background values required");
        if(reinterpret_cast<std::uintptr_t>(background.data())%alignof(double))background=decltype(background)::ensure(background.attr("copy")());
        const auto type=pixel_type(image.dtype());
        std::vector<py::ssize_t> shape;
        for(py::ssize_t axis=0;axis<image.ndim()-1;++axis)shape.push_back(image.shape(axis));
        shape.push_back(3);
        py::array result(type==skimagexx::PixelType::float32 ? py::dtype::of<float>() : py::dtype::of<double>(),shape);
        const auto layout=raw_rows(image,nullptr,int(image.ndim()-1));
        {py::gil_scoped_release release;skimagexx::rgba2rgb_rows_into(image.data(),result.mutable_data(),layout.rows,layout.columns,type,
            background.data(),layout.image.data(),layout.image_step,image.strides(image.ndim()-1));}
        return result;
    });
    m.def("threshold_otsu_result", [](py::array image) {
        shape(image,false);
        if (!dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) throw py::type_error("threshold_otsu requires uint8");
        auto a = py::array_t<std::uint8_t,py::array::c_style>::ensure(image);
        bool constant = false;
        std::uint8_t result;
        { py::gil_scoped_release release; result = skimagexx::threshold_otsu(a.data(),
            static_cast<std::size_t>(a.shape(0)),static_cast<std::size_t>(a.shape(1)),&constant); }
        return py::make_tuple(result,constant);
    });
    m.def("gradient", [](py::array image, const std::string& name, int axis) {
        shape(image,false);
        skimagexx::EdgeFilter filter;
        if (name == "sobel") filter = skimagexx::EdgeFilter::sobel;
        else if (name == "prewitt") filter = skimagexx::EdgeFilter::prewitt;
        else if (name == "scharr") filter = skimagexx::EdgeFilter::scharr;
        else throw py::value_error("unknown gradient filter");
        return dispatch(image,[=](const auto* p, auto* out, auto h, auto w) { skimagexx::gradient_into(p,out,h,w,filter,axis); });
    },py::arg("image"),py::arg("name"),py::arg("axis")=-1);
    m.def("invert", [](py::array image, bool signed_float) {
        shape(image,false);
        auto fn = [=](const auto* p, auto* out, auto h, auto w) { skimagexx::invert_into(p,out,h,w,signed_float); };
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return execute<std::uint8_t,true>(image,fn);
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return execute<float,true>(image,fn);
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return execute<double,true>(image,fn);
        throw py::type_error("supported dtypes are uint8, float32, and float64");
    },py::arg("image"),py::arg("signed_float")=false);
    m.def("gray2rgb", [](py::array image) -> py::array {
        shape(image,false);
        auto make = [&](auto dummy) -> py::array {
            using T = decltype(dummy);
            auto input = py::array_t<T,py::array::c_style>::ensure(image);
            if ((reinterpret_cast<std::uintptr_t>(input.data()) % alignof(T)) != 0)
                input = py::array_t<T,py::array::c_style>::ensure(input.attr("copy")());
            py::array_t<T> output({input.shape(0),input.shape(1),py::ssize_t{3}});
            const auto* p = input.data(); auto* out = output.mutable_data();
            const auto h = static_cast<std::size_t>(input.shape(0)), w = static_cast<std::size_t>(input.shape(1));
            { py::gil_scoped_release release; skimagexx::gray2rgb_into(p,out,h,w); }
            return output;
        };
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return make(float{});
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return make(double{});
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return make(std::uint8_t{});
        throw py::type_error("supported dtypes are uint8, float32, and float64");
    },py::arg("image"));
    m.def("threshold_mean", [](py::array image) {
        shape(image,false);
        auto mean = [&](auto dummy) {
            using T = decltype(dummy);
            auto a = py::array_t<T,py::array::c_style>::ensure(image);
            if ((reinterpret_cast<std::uintptr_t>(a.data()) % alignof(T)) != 0)
                a = py::array_t<T,py::array::c_style>::ensure(a.attr("copy")());
            const auto* p = a.data(); const auto h = static_cast<std::size_t>(a.shape(0)), w = static_cast<std::size_t>(a.shape(1));
            py::gil_scoped_release release;
            return skimagexx::threshold_mean(p,h,w);
        };
        if (dtype_matches(image.dtype(),py::dtype::of<float>())) return mean(float{});
        if (dtype_matches(image.dtype(),py::dtype::of<double>())) return mean(double{});
        if (dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) return mean(std::uint8_t{});
        throw py::type_error("supported dtypes are uint8, float32, and float64");
    },py::arg("image"));
    m.def("set_num_threads", &skimagexx::set_num_threads, py::arg("count"),
          "Set the process-wide worker limit for all algorithms. Default is 1 (serial); use >1 to enable parallel work. SIMD is independent.");
    m.def("get_num_threads", &skimagexx::get_num_threads, "Return the process-wide configured worker limit.");
    m.def("has_avx2", &skimagexx::has_avx2);
    m.def("rgb2gray", [](py::array image) {
        shape(image,true);
        return dispatch(image,[](const auto* p, auto* out, auto h, auto w) { skimagexx::rgb2gray_into(p,out,h,w); });
    },py::arg("image"));
    m.def("integrate", [](py::array table,py::array_t<std::int64_t,py::array::c_style> lower,
            py::array_t<std::int64_t,py::array::c_style> upper,py::array output) {
        if(lower.ndim()!=2 || upper.ndim()!=2 || lower.shape(0)!=upper.shape(0) || lower.shape(1)!=table.ndim() || upper.shape(1)!=table.ndim())throw py::value_error("one lower/upper coordinate per integral table axis required");
        if(output.ndim()!=1 || output.shape(0)!=lower.shape(0) || pixel_type(output.dtype())!=skimagexx::PixelType::float64 || !(output.flags()&py::array::c_style) || !output.writeable() || !output.attr("flags").attr("aligned").cast<bool>())throw py::type_error("contiguous aligned writable double integral output required");
        skimagexx::IntegralTableView view;const auto dtype=table.dtype();
        view.half=dtype.kind()=='f' && dtype.itemsize()==2;
        view.type=view.half ? skimagexx::PixelType::uint16 : pixel_type(dtype.attr("newbyteorder")("=").cast<py::dtype>());
        view.swap=dtype.itemsize()>1 && !dtype.attr("isnative").cast<bool>();view.data=table.data();
        for(py::ssize_t axis=0;axis<table.ndim();++axis) {view.shape.push_back(std::size_t(table.shape(axis)));view.strides.push_back(table.strides(axis));}
        bool supported;
        {py::gil_scoped_release release;supported=skimagexx::integrate_windows_into(view,lower.data(),upper.data(),static_cast<double*>(output.mutable_data()),std::size_t(lower.shape(0)));}
        return supported;
    });
    m.def("integral_line", [](py::array image,py::array result) {
        const auto source=image.dtype();
        if(source.itemsize()!=8 || (source.kind()!='i' && source.kind()!='u'))throw py::type_error("64-bit integer integral line required");
        const auto target=pixel_type(result.dtype());
        if(target!=skimagexx::PixelType::int64 && target!=skimagexx::PixelType::uint64)throw py::type_error("64-bit integer integral output required");
        if(image.ndim()!=result.ndim())throw py::value_error("integral line shape mismatch");
        py::ssize_t stride=0;bool axis_found=false;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {
            if(image.shape(axis)!=result.shape(axis))throw py::value_error("integral line shape mismatch");
            if(image.size() && image.shape(axis)>1) {
                if(axis_found)throw py::value_error("only one nonsingleton integral axis allowed");
                axis_found=true;stride=image.strides(axis);
            }
        }
        if(!result.writeable() || !(result.flags()&py::array::c_style) || !result.attr("flags").attr("aligned").cast<bool>())throw py::type_error("contiguous aligned writable integral line output required");
        const bool swap=!source.attr("isnative").cast<bool>();
        const auto* input=image.data();auto* output=static_cast<std::uint64_t*>(result.mutable_data());
        const auto count=std::size_t(image.size());
        py::gil_scoped_release release;
        skimagexx::integral_line_u64_into(input,output,count,stride,swap);
    });
    m.def("integral_image", [](py::array image,py::dtype output_dtype) -> py::object {
        const auto target=pixel_type(output_dtype);
        const auto descriptor=image.dtype();
        const bool half=descriptor.kind()=='f' && descriptor.itemsize()==2;
        const bool swap=!descriptor.attr("isnative").cast<bool>();
        const auto native_dtype=swap ? py::dtype::from_args(descriptor.attr("type")) : descriptor;
        const auto source=half ? skimagexx::PixelType::float32 : pixel_type(native_dtype);
        auto dense=[](const py::array& array) {
            if(!array.size() || (array.flags()&(py::array::c_style|py::array::f_style)))return true;
            std::vector<std::pair<py::ssize_t,py::ssize_t>> axes;
            for(py::ssize_t axis=0;axis<array.ndim();++axis)if(array.shape(axis)>1)axes.emplace_back(array.strides(axis),array.shape(axis));
            std::sort(axes.begin(),axes.end());auto extent=array.itemsize();
            for(const auto& axis:axes) {if(axis.first!=extent)return false;extent*=axis.second;}
            return extent==array.nbytes();
        };
        const auto count=std::size_t(image.size());
        bool raw_line=(source==skimagexx::PixelType::int64 || source==skimagexx::PixelType::uint64)
            && (target==skimagexx::PixelType::int64 || target==skimagexx::PixelType::uint64);
        py::ssize_t line_stride=0;bool axis_found=false;
        if(raw_line && count)for(py::ssize_t axis=0;axis<image.ndim();++axis)if(image.shape(axis)>1) {
            if(axis_found) {raw_line=false;break;}
            axis_found=true;line_stride=image.strides(axis);
        }
        auto input=image;
        if(!raw_line) {
            if(!dense(input) || !input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")("K").cast<py::array>();
            if(swap || half) {
                std::vector<py::ssize_t> dimensions(input.shape(),input.shape()+input.ndim()),steps(std::size_t(input.ndim()));
                for(py::ssize_t axis=0;axis<input.ndim();++axis)
                    steps[std::size_t(axis)]=count && input.shape(axis)>1 ? input.strides(axis)/input.itemsize() : 0;
                auto physical=input.attr("ravel")("K").cast<py::array>();
                // The flattened view owns any needed dense copy. Release the
                // extra reference before replacing conversion intermediates.
                input=image;
                if(swap) {
                    py::array converted(native_dtype,std::vector<py::ssize_t>{py::ssize_t(count)});
                    {py::gil_scoped_release release;skimagexx::edge_swap_into(physical.data(),converted.mutable_data(),count,std::size_t(descriptor.itemsize()));}
                    physical=std::move(converted);
                }
                if(half) {
                    py::array_t<float> converted(count);
                    {py::gil_scoped_release release;skimagexx::edge_half_to_float(static_cast<const std::uint16_t*>(physical.data()),converted.mutable_data(),count);}
                    physical=std::move(converted);
                }
                for(auto& step:steps)step*=physical.itemsize();
                input=py::array(physical.dtype(),dimensions,steps,physical.data(),physical);
            }
        }
        // This is the same keep-order allocation used by NumPy cumulative
        // sums. The C API steals the descriptor reference passed as arg3.
        auto* allocated=PyArray_NewLikeArray(reinterpret_cast<PyArrayObject*>(image.ptr()),NPY_KEEPORDER,
            reinterpret_cast<PyArray_Descr*>(output_dtype.inc_ref().ptr()),0);
        if(!allocated)throw py::error_already_set();
        auto result=py::reinterpret_steal<py::array>(allocated);
        if(raw_line) {
            const auto* bytes=image.data();auto* values=static_cast<std::uint64_t*>(result.mutable_data());
            {py::gil_scoped_release release;skimagexx::integral_line_u64_into(bytes,values,count,line_stride,swap);}
            return result;
        }
        const bool c_order=(result.flags()&py::array::c_style)!=0;
        std::vector<std::size_t> shape,strides;shape.reserve(std::size_t(input.ndim()));
        if(!c_order)strides.reserve(std::size_t(input.ndim()));
        for(py::ssize_t axis=0;axis<input.ndim();++axis) {
            if(input.shape(axis)!=result.shape(axis))throw py::value_error("integral output shape mismatch");
            if(input.size() && input.shape(axis)>1 && input.strides(axis)/input.itemsize()!=result.strides(axis)/result.itemsize())throw py::value_error("matching integral physical axis order required");
            shape.push_back(std::size_t(input.shape(axis)));
            if(!c_order)strides.push_back(std::size_t(result.strides(axis)/result.itemsize()));
        }
        bool supported=true;
        {
            py::gil_scoped_release release;
            if(count && (source==skimagexx::PixelType::float32 || source==skimagexx::PixelType::float64)) {
                const auto range=source==skimagexx::PixelType::float32
                    ? skimagexx::intensity_bounds(static_cast<const float*>(input.data()),count,1)
                    : skimagexx::intensity_bounds(static_cast<const double*>(input.data()),count,1);
                const double maximum=target==skimagexx::PixelType::float32 ? double(std::numeric_limits<float>::max()) : std::numeric_limits<double>::max();
                const auto limit=std::ldexp(maximum/double(count),-int(shape.size()));
                supported=std::isfinite(range.first) && std::isfinite(range.second)
                    && std::max(std::abs(range.first),std::abs(range.second))<=limit;
            }
            if(supported)skimagexx::integral_image_into(input.data(),result.mutable_data(),source,target,shape,strides);
        }
        if(!supported)return py::none();
        return result;
    });
    m.def("gaussian_difference", [](py::array first,py::array second) {
        const auto type=pixel_type(first.dtype());
        if(!first.dtype().equal(second.dtype()) || first.ndim()!=second.ndim() || !(first.flags()&py::array::c_style) || !(second.flags()&py::array::c_style) || !first.writeable() || !first.attr("flags").attr("aligned").cast<bool>() || !second.attr("flags").attr("aligned").cast<bool>())throw py::type_error("matching contiguous writable difference buffers required");
        for(py::ssize_t axis=0;axis<first.ndim();++axis)if(first.shape(axis)!=second.shape(axis))throw py::value_error("difference shape mismatch");
        {py::gil_scoped_release release;skimagexx::gaussian_difference_into(first.data(),second.data(),first.mutable_data(),std::size_t(first.size()),type);}
        return first;
    });
    m.def("gaussian_unsharp", [](py::array image,py::array blur,double amount,bool wide,bool clip,double lower,py::dtype dtype) {
        const auto source=pixel_type(image.dtype()),target=pixel_type(dtype);
        if(!image.dtype().equal(blur.dtype()) || image.ndim()!=blur.ndim() || !(image.flags()&py::array::c_style) || !(blur.flags()&py::array::c_style) || !image.attr("flags").attr("aligned").cast<bool>() || !blur.attr("flags").attr("aligned").cast<bool>())throw py::type_error("matching contiguous unsharp buffers required");
        std::vector<py::ssize_t> dimensions;
        for(py::ssize_t axis=0;axis<image.ndim();++axis) {if(image.shape(axis)!=blur.shape(axis))throw py::value_error("blur shape mismatch");dimensions.push_back(image.shape(axis));}
        py::array result(dtype,dimensions);
        {py::gil_scoped_release release;skimagexx::gaussian_unsharp_into(image.data(),blur.data(),result.mutable_data(),std::size_t(image.size()),source,target,amount,wide,clip,lower);}
        return result;
    });
    m.def("gaussian_nd", [](py::array image,py::list kernels,const std::vector<std::size_t>& axes,
            const std::vector<int>& modes,double cval,bool preserve,py::dtype output_dtype) {
        const auto source=pixel_type(image.dtype()),target=pixel_type(output_dtype);
        if(target!=skimagexx::PixelType::float32 && target!=skimagexx::PixelType::float64)throw py::type_error("floating Gaussian output required");
        if(axes.size()!=std::size_t(kernels.size()) || modes.size()!=axes.size())throw py::value_error("one Gaussian kernel and mode per axis required");
        auto input=py::array::ensure(image,py::array::c_style);
        if(!input.attr("flags").attr("aligned").cast<bool>())input=input.attr("copy")().cast<py::array>();
        std::vector<py::ssize_t> dimensions;std::vector<std::size_t> shape;
        for(py::ssize_t axis=0;axis<input.ndim();++axis) {dimensions.push_back(input.shape(axis));shape.push_back(std::size_t(input.shape(axis)));}
        std::vector<skimagexx::GaussianPass> passes;
        for(std::size_t j=0;j<axes.size();++j) {
            auto weights=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(kernels[j]);
            if(!weights || weights.ndim()!=1)throw py::type_error("one-dimensional Gaussian coefficients required");
            skimagexx::GaussianPass pass;pass.axis=axes[j];pass.border=static_cast<skimagexx::GaussianBorder>(modes[j]);pass.cval=cval;
            pass.weights.resize(std::size_t(weights.size()));
            if(weights.size())std::memcpy(pass.weights.data(),weights.data(),std::size_t(weights.size())*sizeof(double));
            passes.push_back(std::move(pass));
        }
        py::array result(output_dtype,dimensions);skimagexx::ConversionInfo info;
        {py::gil_scoped_release release;info=skimagexx::gaussian_nd_into(input.data(),result.mutable_data(),source,target,shape,passes,preserve);}
        const unsigned flags=unsigned(info.floating_overflow)|(unsigned(info.floating_underflow)<<1)|(unsigned(info.invalid_float_cast)<<2);
        return py::make_tuple(result,flags);
    });
    m.def("gaussian", [](py::array image, double sigma, const std::string& mode, bool preserve_range, double truncate) {
        shape(image,false);
        skimagexx::Border border;
        if (mode == "nearest") border = skimagexx::Border::nearest;
        else if (mode == "reflect") border = skimagexx::Border::reflect;
        else throw py::value_error("supported modes are nearest and reflect");
        return dispatch(image,[=](const auto* p, auto* out, auto h, auto w) {
            using Input = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            if constexpr (std::is_same_v<Input,std::uint8_t>)
                skimagexx::gaussian_into(p,out,h,w,sigma,border,truncate,preserve_range);
            else skimagexx::gaussian_into(p,out,h,w,sigma,border,truncate);
        });
    },py::arg("image"),py::arg("sigma")=1.0,py::kw_only(),py::arg("mode")="nearest",
       py::arg("preserve_range")=false,py::arg("truncate")=4.0);
    m.def("sobel", [](py::array image) {
        shape(image,false);
        return dispatch(image,[](const auto* p, auto* out, auto h, auto w) { skimagexx::sobel_into(p,out,h,w); });
    },py::arg("image"));
    m.def("threshold_otsu", [](py::array image) {
        shape(image,false);
        if (!dtype_matches(image.dtype(),py::dtype::of<std::uint8_t>())) throw py::type_error("threshold_otsu requires uint8");
        auto a = py::array_t<std::uint8_t,py::array::c_style>::ensure(image);
        const auto* data = a.data();
        const auto h = static_cast<std::size_t>(a.shape(0)), w = static_cast<std::size_t>(a.shape(1));
        py::gil_scoped_release release;
        return skimagexx::threshold_otsu(data,h,w);
    },py::arg("image"));
}
