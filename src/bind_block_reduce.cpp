#include "block_reduce_internal.hpp"
#include "bind_array_storage.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
using Shape = std::vector<py::ssize_t>;
Shape shape(const py::array& a) { return Shape(a.shape(), a.shape()+a.ndim()); }

// NumPy's reduction iterator orders all block-view axes before allocating
// its surviving axes. Zero strides (including unit axes) are ambiguous and
// keep C priority; empty_like(order="K") instead sorts broadcast zeros first.
Shape reduction_output_strides(const py::array& input,
        const std::vector<std::size_t>& blocks, const Shape& output_shape,
        py::ssize_t itemsize) {
    const auto rank=output_shape.size();
    Shape result(rank,0), order(2*rank);
    if(input.size()==0)return result; // NumPy zero-size owning allocations.
    std::vector<std::size_t> magnitude(2*rank);
    for(std::size_t i=0;i<rank;++i) {
        const auto stride=input.strides(i);
        const auto absolute=stride<0?std::size_t(-(stride+1))+1:std::size_t(stride);
        magnitude[i]=output_shape[i]==1?0:absolute*blocks[i];
        magnitude[rank+i]=blocks[i]==1?0:absolute;
    }
    for(std::size_t i=0;i<order.size();++i)order[i]=order.size()-1-i;
    for(std::size_t i=1;i<order.size();++i) {
        const auto axis=order[i];auto position=i;
        for(auto j=py::ssize_t(i)-1;j>=0;--j) {
            const auto left=magnitude[axis],right=magnitude[order[j]];
            if(left && right) {
                if(right<=left)break;
                position=std::size_t(j);
            }
        }
        for(auto j=i;j>position;--j)order[j]=order[j-1];
        order[position]=axis;
    }
    auto stride=itemsize;
    for(const auto axis:order)if(std::size_t(axis)<rank) {
        result[axis]=stride;
        if(output_shape[axis]>std::numeric_limits<py::ssize_t>::max()/stride)
            throw py::value_error("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
        stride*=output_shape[axis];
    }
    return result;
}

bool finite_output(const py::array& output, char code) {
    if(code!='e' && code!='f' && code!='d')return true;
    const auto* p = static_cast<const unsigned char*>(output.data());
    for (py::ssize_t i=0; i<output.size(); ++i) {
        if (code=='e') {std::uint16_t word;std::memcpy(&word,p+2*i,2);if((word&0x7c00)==0x7c00)return false;}
        if (code=='f') {std::uint32_t word;std::memcpy(&word,p+4*i,4);if((word&0x7f800000)==0x7f800000)return false;}
        if (code=='d') {std::uint64_t word;std::memcpy(&word,p+8*i,8);if((word&0x7ff0000000000000ULL)==0x7ff0000000000000ULL)return false;}
    }
    return true;
}
}

void bind_block_reduce(py::module_& module) {
    module.def("block_reduce_sum", [](const py::array& input, const py::sequence& factors)->py::object {
        using P = skimagexx::PixelType;
        const char code=input.dtype().char_(); const auto bytes=input.itemsize();
        P type; bool half=false, signed_result=false, unsigned_result=false;
        if(code=='?'){type=P::boolean;signed_result=true;}
        else if(code=='e'){type=P::uint16;half=true;}
        else if(code=='f')type=P::float32;
        else if(code=='d')type=P::float64;
        else {
            const char kind=py::str(input.dtype().attr("kind")).cast<std::string>()[0];
            if(kind!='i' && kind!='u')return py::none();
            signed_result=kind=='i';unsigned_result=kind=='u';
            if(bytes==1)type=signed_result?P::int8:P::uint8;
            else if(bytes==2)type=signed_result?P::int16:P::uint16;
            else if(bytes==4)type=signed_result?P::int32:P::uint32;
            else if(bytes==8)type=signed_result?P::int64:P::uint64;
            else return py::none();
        }
        // Metadata-bearing integer promotion remains a literal NumPy provider.
        if((signed_result||unsigned_result) && !input.dtype().attr("metadata").is_none())return py::none();
        const auto dimensions=shape(input);
        if(py::len(factors)!=dimensions.size())throw py::value_error("one block factor per input axis required");
        std::vector<std::size_t> blocks, native_shape;Shape output_shape;
        for(std::size_t i=0;i<dimensions.size();++i) {
            const auto factor=py::cast<py::ssize_t>(factors[i]);
            if(factor<=0 || dimensions[i]%factor)throw py::value_error("positive block factors must divide padded dimensions");
            blocks.push_back(std::size_t(factor));native_shape.push_back(std::size_t(dimensions[i]));output_shape.push_back(dimensions[i]/factor);
        }
        skimagexx::bindings::check_array_storage(input);
        auto np=py::module_::import("numpy");
        // Half reductions retain source iterator boundaries only for this
        // unbuffered C-layout domain. Do not normalize unsupported layouts
        // into the admitted domain: the caller replays its literal reducer.
        if(half) {
            if(!(input.flags()&py::array::c_style) || !input.dtype().attr("isnative").cast<bool>() ||
               reinterpret_cast<std::uintptr_t>(input.data())%std::size_t(bytes))return py::none();
            const auto limit=py::cast<std::size_t>(np.attr("getbufsize")());
            std::size_t volume=1;
            for(const auto factor:blocks) {
                if(volume>limit/factor)return py::none();
                volume*=factor;
            }
        }
        const auto native=py::reinterpret_borrow<py::dtype>(input.dtype().attr("newbyteorder")("="));
        py::array prepared=np.attr("ascontiguousarray")(input,py::arg("dtype")=native).cast<py::array>();
        // ascontiguousarray promotes rank-zero input to rank one; keep the
        // established logical shape without changing the typed pixel buffer.
        if(dimensions.empty())prepared=prepared.attr("reshape")(py::tuple()).cast<py::array>();
        if(reinterpret_cast<std::uintptr_t>(prepared.data())%std::size_t(bytes))
            prepared=np.attr("require")(prepared,py::arg("requirements")=py::make_tuple("C","A")).cast<py::array>();
        // NumPy 1 uses C-long promotion (32 bits on Windows); NumPy 2 uses
        // its platform default integer. Native accumulation stays modulo 64.
        const auto promoted_bytes=py::dtype::from_args(np.attr("int_")).itemsize();
        const auto integer_bytes=bytes>promoted_bytes?bytes:promoted_bytes;
        const py::dtype output_dtype=signed_result?py::dtype("int64"):unsigned_result?py::dtype("uint64"):native;
        py::array output(output_dtype, output_shape,
                         reduction_output_strides(input, blocks, output_shape, output_dtype.itemsize()));
        if(!(output.flags()&py::array::c_style) && !(output.flags()&py::array::f_style))return py::none();
        // Conversion and allocation can invoke callbacks. This is the final
        // established-buffer check before native reads, with no Python inside.
        skimagexx::bindings::check_array_storage(prepared);
        if(shape(prepared)!=dimensions || prepared.dtype().char_()!=code || prepared.itemsize()!=bytes ||
           !(prepared.flags()&py::array::c_style) || !prepared.dtype().attr("isnative").cast<bool>() ||
           reinterpret_cast<std::uintptr_t>(prepared.data())%std::size_t(bytes))
            throw py::value_error("block sum input changed during allocation");
        std::vector<std::size_t> strides(output.strides(),output.strides()+output.ndim());
        const skimagexx::detail::BlockReduceInput view{prepared.data(),type,half,native_shape};
        void* target=output.mutable_data();
        unsigned status;
        {py::gil_scoped_release release;status=skimagexx::detail::block_reduce_sum(view,blocks,target,strides);}
        if(status || !finite_output(output,code))return py::none();
        if((signed_result||unsigned_result) && integer_bytes<8) {
            const py::dtype narrowed(std::string(signed_result?"i":"u")+std::to_string(integer_bytes));
            py::array result(narrowed, output_shape,
                             reduction_output_strides(input, blocks, output_shape, narrowed.itemsize()));
            np.attr("copyto")(result, output, py::arg("casting")="unsafe");
            return result;
        }
        return output;
    },py::arg("image").noconvert(),py::arg("factors"));
}
