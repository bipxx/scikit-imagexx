#include <skimagexx/draw_haar_like_feature.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace skimagexx {
namespace {

struct Interval {
    std::size_t start, stop;
};

// The draw rectangle's source bounds are inclusive and order independent.
// NumPy gathers all values of one rectangle before scattering. A wrapped
// index that occurs twice must therefore receive one blend, not two.
std::array<Interval,2> intervals(std::int64_t first, std::int64_t last, std::size_t dimension) {
    if (first > last) std::swap(first, last);
    const auto size = static_cast<std::int64_t>(dimension);
    if (first >= 0)
        return {{{static_cast<std::size_t>(first),static_cast<std::size_t>(last+1)}, {0,0}}};
    if (last < 0)
        return {{{static_cast<std::size_t>(first+size),static_cast<std::size_t>(last+size+1)}, {0,0}}};
    const auto wrapped = static_cast<std::size_t>(first+size);
    const auto stop = static_cast<std::size_t>(last+1);
    if (wrapped <= stop) return {{{0,dimension}, {0,0}}};
    return {{{0,stop}, {wrapped,dimension}}};
}

template<class Out>
void draw_feature(Out* output, std::size_t rows, std::size_t columns, const HaarFeature& feature,
                  const std::array<double,3>& positive, const std::array<double,3>& negative,
                  double alpha) {
    const auto fade = static_cast<Out>(1.0 - alpha);
    bool positive_block = false;
    for (const auto& rect : feature) {
        const auto& color = positive_block ? positive : negative;
        positive_block = !positive_block;
        const std::array<double,3> blend{alpha*color[0], alpha*color[1], alpha*color[2]};
        const auto rows_in_rect = intervals(rect.row_start,rect.row_end,rows);
        const auto columns_in_rect = intervals(rect.column_start,rect.column_end,columns);
        for (auto rr : rows_in_rect) for (std::size_t r=rr.start; r<rr.stop; ++r) {
            for (auto cc : columns_in_rect) for (std::size_t c=cc.start; c<cc.stop; ++c) {
                auto* pixel = output + (r*columns+c)*3;
                for (std::size_t channel=0; channel<3; ++channel) {
                    // NumPy rounds the float32 image product before promoting
                    // it to float64 for the addition and final assignment.
                    const Out background = fade * pixel[channel];
                    pixel[channel] = static_cast<Out>(static_cast<double>(background)+blend[channel]);
                }
            }
        }
    }
}

template<class In, class Out>
void render(const In* input, Out* output, std::size_t rows, std::size_t columns, std::size_t channels,
            double multiplier, bool boolean, const HaarFeatures& features,
            const std::vector<std::size_t>* selection, const std::array<double,3>& positive,
            const std::array<double,3>& negative, double alpha) {
    const auto count=rows*columns;
    if constexpr (std::is_same_v<In,Out>) {
        if (channels==3) std::copy_n(input,count*3,output);
        else for (std::size_t i=0; i<count; ++i)
            output[3*i]=output[3*i+1]=output[3*i+2]=input[i];
    } else {
        for (std::size_t i=0; i<count; ++i) for (std::size_t c=0; c<3; ++c) {
            const auto value=input[i*channels+(channels==1 ? 0 : c)];
            output[3*i+c]=static_cast<Out>((boolean ? double(value!=0) : double(value))*multiplier);
        }
    }
    if (selection) {
        for (auto index : *selection)
            draw_feature(output,rows,columns,features[index],positive,negative,alpha);
    } else {
        for (const auto& feature : features)
            draw_feature(output,rows,columns,feature,positive,negative,alpha);
    }
}

}

namespace detail {
bool haar_coordinates_in_bounds(std::size_t rows, std::size_t columns, const HaarFeatures& features) {
    if (rows>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
        || columns>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        return false;
    const auto height=static_cast<std::int64_t>(rows);
    const auto width=static_cast<std::int64_t>(columns);
    for (const auto& feature : features) for (const auto& rect : feature)
        if (rect.row_start < -height || rect.row_start >= height
            || rect.row_end < -height || rect.row_end >= height
            || rect.column_start < -width || rect.column_start >= width
            || rect.column_end < -width || rect.column_end >= width)
            return false;
    return true;
}
}

bool draw_haar_like_feature_into(const void* input, std::size_t input_elements, PixelType type,
                                std::size_t rows, std::size_t columns, std::size_t channels,
                                void* output, std::size_t output_elements,
                                const HaarFeatures& features, const std::vector<std::size_t>* selection,
                                const std::array<double,3>& positive, const std::array<double,3>& negative,
                                double alpha) {
    if (!input || !output || !rows || !columns || (channels!=1 && channels!=3)
        || rows>std::numeric_limits<std::size_t>::max()/3/columns
        || input_elements!=rows*columns*channels || output_elements!=rows*columns*3
        || !detail::haar_coordinates_in_bounds(rows,columns,features))
        return false;
    if (selection && std::any_of(selection->begin(),selection->end(),
                                [&](std::size_t index){return index>=features.size();}))
        return false;
    switch (type) {
    case PixelType::boolean:
    case PixelType::uint8:
        render(static_cast<const std::uint8_t*>(input),static_cast<double*>(output),rows,columns,channels,
               type==PixelType::boolean ? 1.0 : 1.0/255.0,type==PixelType::boolean,
               features,selection,positive,negative,alpha);
        return true;
    case PixelType::float32:
        render(static_cast<const float*>(input),static_cast<float*>(output),rows,columns,channels,1,false,
               features,selection,positive,negative,alpha);
        return true;
    case PixelType::float64:
        render(static_cast<const double*>(input),static_cast<double*>(output),rows,columns,channels,1,false,
               features,selection,positive,negative,alpha);
        return true;
    default:
        return false;
    }
}
}
