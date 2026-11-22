#include <skimagexx/draw_haar_like_feature.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cfenv>
#include <cstring>
#include <limits>
#include <utility>

namespace py = pybind11;
namespace {
using skimagexx::HaarFeature;
using skimagexx::HaarFeatures;
using skimagexx::PixelType;

constexpr std::size_t max_pixels = 1 << 22;
constexpr std::size_t max_features = 4096;
constexpr std::size_t max_rectangles_per_feature = 16;
constexpr std::int64_t coordinate_limit = 1LL << 30;

bool sequence(py::handle value) {
    return PyList_CheckExact(value.ptr()) || PyTuple_CheckExact(value.ptr());
}

py::ssize_t sequence_size(py::handle value) {
    return PyList_CheckExact(value.ptr()) ? PyList_GET_SIZE(value.ptr()) : PyTuple_GET_SIZE(value.ptr());
}

py::handle item(py::handle value, py::ssize_t index) {
    return PyList_CheckExact(value.ptr()) ? PyList_GET_ITEM(value.ptr(), index)
                                        : PyTuple_GET_ITEM(value.ptr(), index);
}

bool integer(py::handle value, std::int64_t& out) {
    if (!PyLong_CheckExact(value.ptr())) return false;
    int overflow=0;
    const auto number=PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) throw py::error_already_set();
    if (overflow || number < -coordinate_limit || number > coordinate_limit) return false;
    out=number;
    return true;
}

bool endpoint(py::handle object, std::int64_t& r, std::int64_t& c) {
    return sequence(object) && sequence_size(object)==2
        && integer(item(object,0),r) && integer(item(object,1),c);
}

bool feature(py::handle value, std::int64_t r, std::int64_t c, HaarFeature& output) {
    if (!sequence(value) || std::size_t(sequence_size(value)) > max_rectangles_per_feature) return false;
    for (py::ssize_t i=0; i<sequence_size(value); ++i) {
        const auto rect=item(value,i);
        skimagexx::HaarRectangle coordinates{};
        if (!sequence(rect) || sequence_size(rect)!=2
            || !endpoint(item(rect,0),coordinates.row_start,coordinates.column_start)
            || !endpoint(item(rect,1),coordinates.row_end,coordinates.column_end))
            return false;
        coordinates.row_start+=r; coordinates.row_end+=r;
        coordinates.column_start+=c; coordinates.column_end+=c;
        output.push_back(coordinates);
    }
    return true;
}

// Validate the borrowed C-contiguous span before reading any words. Only a
// direct owning array, a bounded view of it, or an immutable bytes owner is admitted.
bool bounded_contiguous(const py::array& array) {
    if (!(array.flags() & py::array::c_style)) return false;
    if (array.owndata()) return true;
    const auto base=array.base();
    if (!base) return false;
    std::uintptr_t begin;
    std::size_t bytes;
    if (PyBytes_CheckExact(base.ptr())) {
        begin=reinterpret_cast<std::uintptr_t>(PyBytes_AS_STRING(base.ptr()));
        bytes=std::size_t(PyBytes_GET_SIZE(base.ptr()));
    } else if (Py_TYPE(base.ptr())==Py_TYPE(array.ptr())) {
        // NumPy 2 returns a compact selection over a directly owning array.
        const auto owner=py::reinterpret_borrow<py::array>(base);
        if (!owner.owndata() || !(owner.flags()&py::array::c_style)) return false;
        begin=reinterpret_cast<std::uintptr_t>(owner.data());
        bytes=std::size_t(owner.nbytes());
    } else return false;
    const auto data=reinterpret_cast<std::uintptr_t>(array.data());
    return data>=begin && data-begin<=bytes && std::size_t(array.nbytes())<=bytes-(data-begin);
}

bool parse_features(py::handle value, std::int64_t row, std::int64_t column, HaarFeatures& output) {
    if (sequence(value)) {
        const auto n=sequence_size(value);
        if (std::size_t(n)>max_features) return false;
        output.resize(n);
        for (py::ssize_t i=0; i<n; ++i)
            if (!feature(item(value,i),row,column,output[i])) return false;
        return true;
    }
    if (!py::isinstance<py::array>(value)) return false;
    const auto array=py::reinterpret_borrow<py::array>(value);
    if (!array.owndata() || !bounded_contiguous(array) || !array.ndim()
        || std::size_t(array.shape(0))>max_features) return false;
    if (array.ndim()==1 && array.dtype().kind()=='O') {
        const auto* words=static_cast<PyObject* const*>(array.data());
        output.resize(array.shape(0));
        for (py::ssize_t i=0; i<array.shape(0); ++i)
            if (!words[i] || !feature(py::handle(words[i]),row,column,output[i])) return false;
        return true;
    }
    if (array.ndim()!=4 || !array.dtype().is(py::dtype::of<std::int64_t>())
        || std::size_t(array.shape(1))>max_rectangles_per_feature
        || array.shape(2)!=2 || array.shape(3)!=2)
        return false;
    const auto* words=static_cast<const std::int64_t*>(array.data());
    output.resize(array.shape(0));
    for (auto& dst : output) for (py::ssize_t j=0; j<array.shape(1); ++j, words+=4) {
        for (int i=0; i<4; ++i) if (words[i] < -coordinate_limit || words[i] > coordinate_limit) return false;
        dst.push_back({row+words[0],column+words[1],row+words[2],column+words[3]});
    }
    return true;
}

template<class Word>
bool finite_image(const void* data, std::size_t count, Word low, Word high) {
    const auto* bytes=static_cast<const unsigned char*>(data);
    for (std::size_t i=0; i<count; ++i) {
        Word bits;
        std::memcpy(&bits,bytes+i*sizeof(bits),sizeof(bits));
        const auto magnitude=bits & (std::numeric_limits<Word>::max() >> 1);
        if (magnitude && (magnitude<low || magnitude>high)) return false;
    }
    return true;
}

struct Drawing {
    py::array source;
    py::dtype dtype;
    const void* source_data;
    PixelType type;
    std::size_t rows,columns,channels;
    HaarFeatures features;
};

template<class T>
bool matches_image_dtype(const py::dtype& dtype) {
    const auto canonical=py::dtype::of<T>();
    return dtype.is(canonical) || (dtype.num()==py::dtype::num_of<T>()
        && dtype.itemsize()==sizeof(T) && dtype.attr("isnative").cast<bool>()
        && dtype.equal(canonical));
}

py::object prepare(const py::array& image,py::handle r,py::handle c,py::handle coordinates) {
    if (std::fegetround()!=FE_TONEAREST || (image.ndim()!=2 && image.ndim()!=3)) return py::none();
    const auto rows=std::size_t(image.shape(0)), columns=std::size_t(image.shape(1));
    if (!rows || !columns || rows>max_pixels/columns || (image.ndim()==3 && image.shape(2)!=3)
        || !bounded_contiguous(image)) return py::none();
    const auto channels=image.ndim()==2 ? 1u : 3u;
    const auto count=rows*columns*channels;
    const auto alignment=reinterpret_cast<std::uintptr_t>(image.data());
    const auto dtype=image.dtype();
    PixelType type;
    if (matches_image_dtype<std::uint8_t>(dtype)) type=PixelType::uint8;
    else if (matches_image_dtype<bool>(dtype)) type=PixelType::boolean;
    else if (matches_image_dtype<float>(dtype) && !(alignment%alignof(float))
             && finite_image(image.data(),count,0x33800000u,0x49742400u)) type=PixelType::float32;
    else if (matches_image_dtype<double>(dtype) && !(alignment%alignof(double))
             && finite_image(image.data(),count,UINT64_C(0x3e70000000000000),UINT64_C(0x412e848000000000))) type=PixelType::float64;
    else return py::none();
    std::int64_t row,column;
    HaarFeatures features;
    if (!integer(r,row) || !integer(c,column) || !parse_features(coordinates,row,column,features)
        || !skimagexx::detail::haar_coordinates_in_bounds(rows,columns,features))
        return py::none();
    std::size_t rectangles=0;
    for (const auto& group : features) rectangles+=group.size();
    if (rectangles>(type==PixelType::float32 ? 16u : 128u)) return py::none();
    return py::cast(Drawing{image,dtype,image.data(),type,rows,columns,channels,std::move(features)});
}

template<class Output>
py::array render(const Drawing& drawing, const std::vector<std::size_t>* selected,
                 const std::array<double,3>& positive, const std::array<double,3>& negative, double alpha) {
    const auto& image=drawing.source;
    if (!bounded_contiguous(image) || !image.dtype().is(drawing.dtype) || image.data()!=drawing.source_data
        || image.ndim()!=(drawing.channels==1 ? 2 : 3)
        || std::size_t(image.shape(0))!=drawing.rows || std::size_t(image.shape(1))!=drawing.columns
        || (drawing.channels==3 && image.shape(2)!=3))
        throw py::value_error("prepared Haar input changed");
    const py::array::ShapeContainer shape{py::ssize_t(drawing.rows),py::ssize_t(drawing.columns),py::ssize_t(3)};
    // The stock RGB float path copies the input dtype through img_as_float.
    const bool preserve_dtype=drawing.channels==3
        && (drawing.type==PixelType::float32 || drawing.type==PixelType::float64);
    py::array output(preserve_dtype ? drawing.dtype : py::dtype::of<Output>(), shape);
    if (!skimagexx::draw_haar_like_feature_into(
            drawing.source.data(),drawing.source.size(),drawing.type,drawing.rows,drawing.columns,
            drawing.channels,output.mutable_data(),output.size(),drawing.features,selected,positive,negative,alpha))
        throw py::value_error("invalid prepared Haar drawing");
    return output;
}

}

void bind_draw_haar_like_feature(py::module_& module) {
    py::class_<Drawing>(module,"HaarDrawing")
        .def_property_readonly("size",[](const Drawing& drawing){return drawing.features.size();});
    module.def("_prepare_haar_drawing",&prepare,py::arg("image").noconvert(),py::arg("r"),
               py::arg("c"),py::arg("coordinates"));
    module.def("_draw_haar_prepared",[](const Drawing& drawing, const py::object& selection,
                const std::array<double,3>& positive,const std::array<double,3>& negative,double alpha)->py::array {
        std::vector<std::size_t> indices;
        if (!selection.is_none()) {
            if (!py::isinstance<py::array_t<py::ssize_t>>(selection))
                throw py::value_error("invalid prepared Haar selection dtype");
            const auto array=py::reinterpret_borrow<py::array_t<py::ssize_t>>(selection);
            if (array.ndim()!=1 || !bounded_contiguous(array)
                || reinterpret_cast<std::uintptr_t>(array.data())%alignof(py::ssize_t))
                throw py::value_error("invalid prepared Haar selection layout");
            for (py::ssize_t i=0; i<array.size(); ++i) {
                const auto index=array.data()[i];
                if (index<0 || std::size_t(index)>=drawing.features.size())
                    throw py::value_error("invalid prepared Haar selection index");
                indices.push_back(static_cast<std::size_t>(index));
            }
        }
        const auto* selected=selection.is_none() ? nullptr : &indices;
        return drawing.type==PixelType::float32 ? render<float>(drawing,selected,positive,negative,alpha)
                                                : render<double>(drawing,selected,positive,negative,alpha);
    });
}
