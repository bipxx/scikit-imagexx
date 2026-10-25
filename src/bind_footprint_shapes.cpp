#include "skimagexx/footprint_shapes.hpp"
#include "skimagexx/footprint_decomposition.hpp"
#include "skimagexx/footprint_dense.hpp"
#include "skimagexx/morphology.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
namespace py=pybind11;
namespace {
// The admitted stages use ordinary ndarray-backed owned allocations only.
// Check direct Cslots before any provider copy. No Python .base dispatch and
// no forged/as_strided unknown owner. No callbacks occur between final check
// and GIL release; external concurrent resize/write remains a caller contract.
void check(const py::array& input){
    auto current=input;std::vector<py::array> chain;
    for(unsigned i=0;i<64;++i){
        if(std::any_of(chain.begin(),chain.end(),[&](auto x){return x.ptr()==current.ptr();}))throw py::value_error("footprint owner cycle");
        chain.push_back(current);auto base=current.base();
        if(base.is_none())break;
        if(!py::isinstance<py::array>(base))throw py::value_error("footprint requires ndarray-backed storage");
        current=py::reinterpret_borrow<py::array>(base);
    }
    if(!current.base().is_none()||!current.owndata()||!(current.flags()&(py::array::c_style|py::array::f_style)))throw py::value_error("footprint requires bounded contiguous root");
    const auto limit=std::numeric_limits<std::uintptr_t>::max();
    const auto root=reinterpret_cast<std::uintptr_t>(current.data());
    if(current.nbytes()<0||std::size_t(current.nbytes())>limit-root)throw py::value_error("footprint root overflows");
    for(const auto& a:chain){
        if(a.size()==0)continue;
        std::uintptr_t low=0,high=0;
        for(py::ssize_t d=0;d<a.ndim();++d){
            if(a.shape(d)<0)throw py::value_error("negative footprint extent");
            const auto stride=a.strides(d),mag=stride<0?std::uintptr_t(-(stride+1))+1:std::uintptr_t(stride);
            auto& extent=stride<0?low:high;const auto count=std::uintptr_t(a.shape(d)-1);
            if(count&&mag>(limit-extent)/count)throw py::value_error("footprint extent overflows");extent+=count*mag;
        }
        const auto data=reinterpret_cast<std::uintptr_t>(a.data());
        if(low>data||high>limit-data||std::size_t(a.itemsize())>limit-data-high||data-low<root||data+high+a.itemsize()>root+current.nbytes())throw py::value_error("footprint escapes owner allocation");
    }
}
void dense(const py::array& a){check(a);if(!(a.flags()&py::array::c_style))throw py::value_error("prepared footprint must be C-contiguous");}
void u8(const py::array& a){dense(a);if(a.dtype().kind()!='u'||a.itemsize()!=1)throw py::type_error("prepared mask requires uint8");}
skimagexx::FootprintConstBytes source(const py::array& a){return {a.data(),std::size_t(a.nbytes())};}
skimagexx::FootprintBytes target(py::array& a){if(!a.writeable())throw py::value_error("footprint output read-only");return {a.mutable_data(),std::size_t(a.nbytes())};}
}
void bind_footprint_shapes(py::module_& m){
    m.attr("footprint_shapes_interface")=skimagexx::footprint_shapes_interface;
    m.def("footprint_shapes_check",&check,py::arg("array").noconvert());
    m.def("footprint_shapes_full_mask",[](std::vector<py::ssize_t> shape){
        for(auto size:shape)if(size<0)throw py::value_error("negative mask dimension");
        py::array_t<std::uint8_t> result(shape);const unsigned char zero=0,one=1;
        const auto count=std::size_t(result.size());auto* data=result.mutable_data();
        {py::gil_scoped_release release;skimagexx::footprint_dense_into(data,count,1,&zero,&one,nullptr,0,skimagexx::DenseFootprint::rectangle,0);}return result;
    });
    m.def("footprint_shapes_cross",[](std::size_t vertical,std::size_t horizontal){
        if(vertical>std::size_t(PY_SSIZE_T_MAX/2)||horizontal>std::size_t(PY_SSIZE_T_MAX/2))throw py::value_error("cross shape overflows");
        py::array_t<std::uint8_t> result({py::ssize_t(2*vertical+1),py::ssize_t(2*horizontal+1)});
        const unsigned char zero=0,one=1;auto* data=result.mutable_data();
        py::gil_scoped_release release;skimagexx::footprint_element_into(data,1,&zero,&one,2,2,0,0,vertical,horizontal);return result;
    });
    m.def("footprint_shapes_ellipse",[](std::size_t rows,std::size_t cols,double h,double w){
        py::array_t<std::uint8_t> result({py::ssize_t(rows),py::ssize_t(cols)});auto out=target(result);
        {py::gil_scoped_release release;skimagexx::footprint_shapes_ellipse(out,rows,cols,h,w);}return result;
    });
    m.def("footprint_shapes_hull",[](py::array input){
        u8(input);if(input.ndim()!=2)throw py::value_error("hull requires2D");
        py::array_t<std::uint8_t> result({input.shape(0),input.shape(1)});u8(input);
        auto out=target(result);auto in=source(input);
        {py::gil_scoped_release release;skimagexx::footprint_shapes_hull(in,out,std::size_t(input.shape(0)),std::size_t(input.shape(1)));}return result;
    },py::arg("input").noconvert());
    m.def("footprint_shapes_l1",[](py::array x,py::array y,py::array z,double radius){
        for(const auto& a:{x,y,z}){dense(a);if(a.dtype().kind()!='f'||a.itemsize()!=8||!a.dtype().is(py::dtype::of<double>()))throw py::type_error("mgrid requires native float64");}
        if(x.ndim()!=3||y.ndim()!=3||z.ndim()!=3)throw py::value_error("mgrid shapes differ");
        for(py::ssize_t d=0;d<3;++d)if(x.shape(d)!=y.shape(d)||x.shape(d)!=z.shape(d))throw py::value_error("mgrid shapes differ");
        py::array_t<std::uint8_t> result({x.shape(0),x.shape(1),x.shape(2)});
        for(const auto& a:{x,y,z})dense(a);const auto a=source(x),b=source(y),c=source(z);auto out=target(result);
        {py::gil_scoped_release release;skimagexx::footprint_shapes_l1(a,b,c,out,std::size_t(x.size()),radius);}return result;
    },py::arg("x").noconvert(),py::arg("y").noconvert(),py::arg("z").noconvert(),py::arg("radius"));
    m.def("footprint_shapes_atoms",[](py::array mask,py::array output,py::array zero,py::array one){
        u8(mask);for(const auto& a:{output,zero,one}){dense(a);if(a.dtype().has_fields()||a.dtype().attr("hasobject").cast<bool>())throw py::type_error("reference-free atoms required");}
        if(mask.size()!=output.size()||zero.size()!=1||one.size()!=1||!zero.dtype().is(output.dtype())||!one.dtype().is(output.dtype()))throw py::value_error("footprint atoms mismatch");
        // Dtype metadata lookups above can dispatch Python. Recheck afterward.
        for(const auto& a:{mask,output,zero,one})dense(a);
        auto in=source(mask),a=source(zero),b=source(one);auto out=target(output);const auto n=std::size_t(mask.size()),item=std::size_t(output.itemsize());
        py::gil_scoped_release release;skimagexx::footprint_shapes_atoms(in,out,a,b,n,item);
    },py::arg("mask").noconvert(),py::arg("output").noconvert(),py::arg("zero").noconvert(),py::arg("one").noconvert());
    m.def("footprint_shapes_cross_plan",[](py::array_t<std::ptrdiff_t,py::array::c_style> sums,std::size_t height){
        dense(sums);if(sums.ndim()!=1)throw py::value_error("cross sums must be1D");
        // C-contiguous does not imply alignment. The inherited typed stage
        // reads ptrdiff_t directly; ordinary owned SUM preparation is aligned.
        if(!(sums.flags()&py::detail::npy_api::NPY_ARRAY_ALIGNED_))
            throw py::value_error("cross sums must be aligned");
        std::vector<skimagexx::FootprintCross> plan;
        {py::gil_scoped_release release;plan=skimagexx::footprint_cross_sums(sums.data(),std::size_t(sums.size()),height);}
        py::list output;for(auto p:plan)output.append(py::make_tuple(p.vertical,p.horizontal,p.repetitions));return output;
    },py::arg("sums").noconvert(),py::arg("height"));
    m.def("footprint_shapes_dilate",[](py::array image,py::array footprint,py::array output){
        for(const auto& a:{image,footprint,output}){dense(a);if(a.dtype().kind()!='b'||a.itemsize()!=1)throw py::type_error("prepared dilation requires bool");}
        if(image.ndim()<1||image.ndim()!=footprint.ndim()||image.ndim()!=output.ndim())throw py::value_error("dilation ranks differ");
        std::vector<std::size_t> shape;for(py::ssize_t d=0;d<image.ndim();++d){if(image.shape(d)!=output.shape(d))throw py::value_error("dilation shapes differ");shape.push_back(std::size_t(image.shape(d)));}
        skimagexx::MorphologyFootprint fp;fp.scipy_separable=true;
        const auto* bytes=static_cast<const unsigned char*>(footprint.data());
        for(std::size_t i=0;i<std::size_t(footprint.size());++i){
            if(!bytes[i]){fp.scipy_separable=false;continue;}
            std::vector<std::ptrdiff_t> point(std::size_t(footprint.ndim()));auto index=i;
            for(auto d=footprint.ndim();d-->0;){const auto n=std::size_t(footprint.shape(d));point[std::size_t(d)]=std::ptrdiff_t(index%n)-std::ptrdiff_t(n/2);index/=n;}
            fp.offsets.insert(fp.offsets.end(),point.begin(),point.end());
        }
        if(fp.offsets.empty())throw py::value_error("All-zero footprint is not supported.");
        auto out=target(output);const auto* in=image.data();
        py::gil_scoped_release release;skimagexx::morphology_into(in,out.data,skimagexx::PixelType::boolean,skimagexx::PixelType::boolean,shape,{fp},skimagexx::MorphologyOp::dilation,skimagexx::MorphologyMode::reflect,0);
    },py::arg("image").noconvert(),py::arg("footprint").noconvert(),py::arg("output").noconvert());
}
