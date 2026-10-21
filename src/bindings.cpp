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
    if (image.dtype().is(py::dtype::of<float>())) return execute<float>(image,function);
    if (image.dtype().is(py::dtype::of<double>())) return execute<double>(image,function);
    if (image.dtype().is(py::dtype::of<std::uint8_t>())) return execute<std::uint8_t>(image,function);
    throw py::type_error("supported dtypes are uint8, float32, and float64");
}
}

PYBIND11_MODULE(_core, m) {
    m.def("distance_metric",[](py::array first,py::array second,int operation,std::array<double,5> parameters,
        bool wide_trig,bool promoted_cmc) -> py::array {
        if(first.ndim()<1 || second.ndim()<1 || first.shape(first.ndim()-1)!=3 || second.shape(second.ndim()-1)!=3)
            throw py::value_error("distance inputs must have three channels");
        if(first.ndim()!=second.ndim() || !std::equal(first.shape(),first.shape()+first.ndim(),second.shape()))
            throw py::value_error("distance inputs must have matching broadcasted shapes");
        const auto at=pixel_type(first.dtype()),bt=pixel_type(second.dtype());
        auto prepare=[](py::array a,std::ptrdiff_t& pixels,std::ptrdiff_t& channels) {
            if(!linear_color_view(a,pixels,channels)) {a=a.attr("copy")().cast<py::array>();pixels=3;channels=1;}
            return a;
        };
        std::ptrdiff_t ap=3,ac=1,bp=3,bc=1;
        first=prepare(first,ap,ac);second=prepare(second,bp,bc);
        std::vector<py::ssize_t> dims(first.shape(),first.shape()+first.ndim()-1);
        const auto dtype=at==skimagexx::PixelType::float32 && bt==skimagexx::PixelType::float32 && !promoted_cmc ?
            py::dtype::of<float>() : py::dtype::of<double>();
        py::array out(dtype,dims);
        const skimagexx::DistanceParameters p{parameters[0],parameters[1],parameters[2],parameters[3],parameters[4]};
        {py::gil_scoped_release release;
         skimagexx::delta_e_into(first.data(),second.data(),out.mutable_data(),std::size_t(out.size()),at,bt,
            static_cast<skimagexx::DistanceMetric>(operation),p,wide_trig,promoted_cmc,ap,ac,bp,bc);}
        return out;
    },py::arg("first"),py::arg("second"),py::arg("operation"),py::arg("parameters"),py::arg("wide_trigonometry"),py::arg("promoted_cmc"));
    m.def("delta_e76",[](py::array first,py::array second) -> py::array {
        if(first.ndim()<1 || second.ndim()<1 || first.shape(first.ndim()-1)!=3 || second.shape(second.ndim()-1)!=3)
            throw py::value_error("distance inputs must have three channels");
        if(first.ndim()!=second.ndim() || !std::equal(first.shape(),first.shape()+first.ndim(),second.shape()))
            throw py::value_error("distance inputs must have matching broadcasted shapes");
        const auto at=pixel_type(first.dtype()),bt=pixel_type(second.dtype());
        std::ptrdiff_t ap=3,ac=1,bp=3,bc=1;
        first=aligned_distance_view(first);second=aligned_distance_view(second);
        const bool linear=!first.size() || (linear_color_view(first,ap,ac) && linear_color_view(second,bp,bc));
        std::vector<py::ssize_t> dims(first.shape(),first.shape()+first.ndim()-1);
        const auto dtype=at==skimagexx::PixelType::float32 && bt==skimagexx::PixelType::float32 ?
            py::dtype::of<float>() : py::dtype::of<double>();
        py::array out(dtype,dims);
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
    },py::arg("first"),py::arg("second"));
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
        auto input = py::array::ensure(image,py::array::c_style);
        if (!input) throw py::type_error("unable to make input contiguous");
        if ((reinterpret_cast<std::uintptr_t>(input.data()) % skimagexx::pixel_size(source_type)) != 0)
            input = py::array::ensure(input.attr("copy")(),py::array::c_style);
        py::array output(dtype,std::vector<py::ssize_t>(input.shape(),input.shape()+input.ndim()));
        const auto* source = input.data(); auto* destination = output.mutable_data();
        const auto n = static_cast<std::size_t>(input.size());
        skimagexx::ConversionInfo info;
        { py::gil_scoped_release release; info = skimagexx::convert_into(source,destination,n,source_type,target_type); }
        const bool unsigned_input = input.dtype().kind() == 'u';
        py::object maximum = unsigned_input ? py::object(py::int_(info.maximum_unsigned)) : py::object(py::int_(info.maximum_signed));
        return py::make_tuple(output,info.downcast_without_scaling,info.nan_to_integer,info.floating_overflow,info.floating_underflow,info.invalid_float_cast,maximum);
    },py::arg("image"),py::arg("dtype"));
    m.def("histogram_u8", [](py::array image, bool full_range, bool normalize, bool cumulative) -> py::tuple {
        if (!image.size()) throw py::value_error("image must be nonempty");
        if (!image.dtype().is(py::dtype::of<std::uint8_t>())) throw py::type_error("histogram requires uint8");
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
        if (!image.dtype().is(py::dtype::of<std::uint8_t>())) throw py::type_error("equalize_hist requires uint8");
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
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) return run(std::uint8_t{});
        if (image.dtype().is(py::dtype::of<float>())) return run(float{});
        if (image.dtype().is(py::dtype::of<double>())) return run(double{});
        throw py::type_error("unsupported dtype");
    });
    m.def("rescale", [](py::array image, double imin, double imax, double omin, double omax, bool float_output) {
        if (!image.size()) throw py::value_error("image must be nonempty");
        auto fn = [=](const auto* p, auto* out, auto h, auto w) { skimagexx::rescale_into(p,out,h,w,imin,imax,omin,omax); };
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) {
            if (float_output) return execute<std::uint8_t,false,true>(image,fn);
            return execute<std::uint8_t,true,true>(image,fn);
        }
        if (image.dtype().is(py::dtype::of<float>())) return execute<float,true,true>(image,fn);
        if (image.dtype().is(py::dtype::of<double>())) return execute<double,true,true>(image,fn);
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
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) return execute<std::uint8_t,true,true>(image,fn);
        if (image.dtype().is(py::dtype::of<float>())) return execute<float,true,true>(image,fn);
        if (image.dtype().is(py::dtype::of<double>())) return execute<double,true,true>(image,fn);
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
        if(image.dtype().is(py::dtype::of<float>()))return run(float{});
        if(image.dtype().is(py::dtype::of<double>()))return run(double{});
        throw py::type_error("floating image required");
    });
    m.def("threshold_logarithms", [](py::array_t<float,py::array::c_style|py::array::forcecast> values,bool numpy_log) {
        py::array_t<float> output(values.size());
        if(reinterpret_cast<std::uintptr_t>(values.data())%alignof(float))values=decltype(values)::ensure(values.attr("copy")());
        skimagexx::threshold_logs(values.data(),output.mutable_data(),std::size_t(values.size()),numpy_log);
        return output;
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
    m.def("threshold_otsu_result", [](py::array image) {
        shape(image,false);
        if (!image.dtype().is(py::dtype::of<std::uint8_t>())) throw py::type_error("threshold_otsu requires uint8");
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
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) return execute<std::uint8_t,true>(image,fn);
        if (image.dtype().is(py::dtype::of<float>())) return execute<float,true>(image,fn);
        if (image.dtype().is(py::dtype::of<double>())) return execute<double,true>(image,fn);
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
        if (image.dtype().is(py::dtype::of<float>())) return make(float{});
        if (image.dtype().is(py::dtype::of<double>())) return make(double{});
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) return make(std::uint8_t{});
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
        if (image.dtype().is(py::dtype::of<float>())) return mean(float{});
        if (image.dtype().is(py::dtype::of<double>())) return mean(double{});
        if (image.dtype().is(py::dtype::of<std::uint8_t>())) return mean(std::uint8_t{});
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
        if (!image.dtype().is(py::dtype::of<std::uint8_t>())) throw py::type_error("threshold_otsu requires uint8");
        auto a = py::array_t<std::uint8_t,py::array::c_style>::ensure(image);
        const auto* data = a.data();
        const auto h = static_cast<std::size_t>(a.shape(0)), w = static_cast<std::size_t>(a.shape(1));
        py::gil_scoped_release release;
        return skimagexx::threshold_otsu(data,h,w);
    },py::arg("image"));
}
