#include "skimagexx/integral.hpp"
#include "skimagexx/integrate.hpp"
#include "../src/execution.hpp"
#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/farid.hpp"
#include "skimagexx/small_edge.hpp"
#include "skimagexx/alpha.hpp"
#include "skimagexx/local_threshold.hpp"
#include "skimagexx/threshold.hpp"
#include "skimagexx/image.hpp"
#include "skimagexx/convert.hpp"
#include "skimagexx/color.hpp"
#include "skimagexx/perceptual.hpp"
#include "skimagexx/stain.hpp"
#include "skimagexx/distance.hpp"
#include "skimagexx/morphology.hpp"
#include <cmath>
#include <cstring>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>

void require(bool condition) {
    if (!condition) throw std::runtime_error("native check failed");
}
int main() {
    using namespace skimagexx;
    require(get_num_threads() == 1);
    {
        const auto length=std::size_t(std::numeric_limits<int>::max())+17;
        bool visited=false;
        detail::partition(length,1,[&](auto begin,auto end,auto worker) {
            static_assert(std::is_same_v<decltype(begin),std::size_t>);
            static_assert(std::is_same_v<decltype(end),std::size_t>);
            static_assert(std::is_same_v<decltype(worker),std::size_t>);
            require(begin==0 && end==length && worker==0);visited=true;
        });
        require(visited);
    }

#ifdef _OPENMP
    const int thread_counts[] = {1, 4};
#else
    const int thread_counts[] = {1};
#endif
    for (int threads : thread_counts) {
        set_num_threads(threads);
        require(get_num_threads() == threads);
        {
            // A strided modular prefix is exact across arbitrary chunk splits.
            constexpr std::size_t count=262145;
            std::vector<std::uint64_t> input(count*3),output(count);
            for(std::size_t i=0;i<count;++i)input[3*i]=std::uint64_t(i)*UINT64_C(0x9e3779b97f4a7c15);
            for(int direction:{1,-1}) {
                const auto* first=direction>0 ? input.data() : input.data()+3*(count-1);
                integral_line_u64_into(first,output.data(),count,direction*std::ptrdiff_t(24));
                std::uint64_t sum=0;
                for(std::size_t i=0;i<count;++i) {sum+=input[3*(direction>0 ? i : count-1-i)];require(output[i]==sum);}
            }
            const std::uint64_t swapped[]={UINT64_C(0x0102030405060708),1};
            integral_line_u64_into(swapped,output.data(),2,8,true);
            require(output[0]==UINT64_C(0x0807060504030201) && output[1]==UINT64_C(0x0907060504030201));
            const std::uint64_t repeating=std::numeric_limits<std::uint64_t>::max();
            integral_line_u64_into(&repeating,output.data(),33,0);
            for(std::size_t i=0;i<33;++i)require(output[i]==std::uint64_t(0)-std::uint64_t(i+1));
            for(std::size_t length:{1,2,3,4,7,8,15,16,17,31,32,33,65}) {
                auto inplace=input;
                integral_line_u64_into(inplace.data(),inplace.data(),length,8);
                std::uint64_t sum=0;
                for(std::size_t i=0;i<length;++i) {sum+=input[i];require(inplace[i]==sum);}
            }
            integral_line_u64_into(nullptr,nullptr,0,0);
            bool rejected=false;
            try {integral_line_u64_into(nullptr,output.data(),1,8);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected);rejected=false;
            try {integral_line_u64_into(input.data(),output.data(),2,std::numeric_limits<std::ptrdiff_t>::min());}catch(const std::overflow_error&){rejected=true;}
            require(rejected);rejected=false;
            try {integral_line_u64_into(input.data(),output.data(),std::numeric_limits<std::size_t>::max(),8);}catch(const std::overflow_error&){rejected=true;}
            require(rejected);
        }
        {
            const double table[]={1,3,6,5,12,21};
            IntegralTableView view{table,PixelType::float64,false,false,{2,3},{24,8}};
            const std::int64_t lower[]={-1,-1,0,0,-1,1},upper[]={1,2,1,2,1,2};
            double result[3];
            require(integrate_windows_into(view,lower,upper,result,3));
            require(result[0]==21 && result[1]==11 && result[2]==9);
            // A reversed view reads raw corners through signed byte strides.
            view.data=table+5;view.strides={-24,-8};
            require(integrate_windows_into(view,lower,upper,result,3));
            require(result[0]==1 && result[1]==11 && result[2]==-2);
            const std::uint16_t half[]={0x3c00,0xc000,0x8000,0x0001};
            view={half,PixelType::uint16,true,false,{4},{2}};
            const std::int64_t half_lower[]={-1,-1,-1,-1},half_upper[]={0,1,2,3};
            double decoded[4];
            require(integrate_windows_into(view,half_lower,half_upper,decoded,4));
            require(decoded[0]==1 && decoded[1]==-2 && decoded[2]==0 && !std::signbit(decoded[2]) && decoded[3]==std::ldexp(1.,-24));
            const unsigned char swapped[]={0,1,1,0};
            view={swapped,PixelType::uint16,false,true,{2},{2}};
            const std::int64_t lo[]={-1,-1},hi[]={0,1};
            require(integrate_windows_into(view,lo,hi,result,2));
            const std::uint16_t endian=1;
            if(*reinterpret_cast<const unsigned char*>(&endian))require(result[0]==1 && result[1]==256);
            else require(result[0]==256 && result[1]==1);
            const double infinity=std::numeric_limits<double>::infinity();
            view={&infinity,PixelType::float64,false,false,{}, {}};
            require(!integrate_windows_into(view,nullptr,nullptr,result,1));
            const double finite=-0.;view.data=&finite;
            require(integrate_windows_into(view,nullptr,nullptr,result,1));
            require(result[0]==0 && !std::signbit(result[0]));
            view={table,PixelType::float64,false,false,{2,3},{24,8}};
            constexpr std::size_t windows=65539;
            std::vector<std::int64_t> many_lower(windows*2),many_upper(windows*2);
            std::vector<double> many_result(windows);
            for(std::size_t i=0;i<windows;++i) {many_upper[2*i]=1;many_upper[2*i+1]=2;}
            require(integrate_windows_into(view,many_lower.data(),many_upper.data(),many_result.data(),windows));
            for(auto value:many_result)require(value==11);
            bool rejected=false;
            const std::int64_t outside[]={2,2};
            try {integrate_windows_into(view,lower,outside,result,1);}catch(const std::out_of_range&){rejected=true;}
            require(rejected);
            view.strides[0]=std::numeric_limits<std::ptrdiff_t>::min();rejected=false;
            try {integrate_windows_into(view,lower,upper,result,1);}catch(const std::overflow_error&){rejected=true;}
            require(rejected);
            view={nullptr,PixelType::float64,false,false,{0,3},{24,8}};
            require(integrate_windows_into(view,nullptr,nullptr,nullptr,0));
        }
        {
            // Reference impulse responses pin derivative sign and border rules.
            const double impulse[]={0,0,1,0,0};
            const double expected[][5]={
                {.109603762960254,.276690988455557,0,-.276690988455557,-.109603762960254},
                {.013030075106301248,.2492900477154935,0,-.2492900477154935,-.013030075106301244},
                {.109603762960254,.276690988455557,0,-.276690988455557,-.109603762960254},
                {0,.276690988455557,0,-.276690988455557,0},
                {.109603762960254,.276690988455557,0,-.276690988455557,-.109603762960254}};
            double result[5];
            for(int border=0;border<5;++border) {
                farid_into(impulse,result,PixelType::float64,{5},{0},FaridBorder(border),.25);
                for(int j=0;j<5;++j)require(std::abs(result[j]-expected[border][j])<1e-16);
            }
            const std::uint8_t bytes[]={0,0,255,0,0};
            farid_into(bytes,result,PixelType::uint8,{5},{0});
            for(int j=0;j<5;++j)require(result[j]==expected[0][j]);
            const std::uint16_t half[]={0,0,0x3c00,0,0};float single[5];
            farid_half_into(half,single,{5},{0});
            for(int j=0;j<5;++j)require(single[j]==float(expected[0][j]));
            const std::int16_t signed_pixels[]={0,0,32767,0,0};
            farid_into(signed_pixels,result,PixelType::int16,{5},{0});
            for(int j=0;j<5;++j)require(result[j]==expected[0][j]);
            // Enough interior pixels for SIMD and enough rows for worker splits.
            constexpr std::size_t height=513,width=517;
            std::vector<float> ramp(height*width),gradient(height*width);
            for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x)ramp[y*width+x]=float(y+2*x);
            farid_into(ramp.data(),gradient.data(),PixelType::float32,{height,width},{0,1});
            const double slope=2*(.276690988455557+2*.109603762960254);
            const double smoothing=2*(.0376593171958126+.249153396177344)+.426374573253687;
            const double magnitude=slope*smoothing*std::sqrt(2.5);
            for(std::size_t y=2;y+2<height;++y)for(std::size_t x=2;x+2<width;++x)
                require(std::abs(gradient[y*width+x]-magnitude)<1e-6);
            farid_into(nullptr,result,PixelType::float64,{5},{});
            for(auto value:result)require(value==0 && !std::signbit(value));
            farid_into(nullptr,nullptr,PixelType::float32,{0,7},{0,1});
            bool bad_axis=false;
            try {farid_into(impulse,result,PixelType::float64,{5},{1});}
            catch(const std::invalid_argument&){bad_axis=true;}require(bad_axis);
            bool overflow=false;
            try {farid_into(impulse,result,PixelType::float64,{std::numeric_limits<std::size_t>::max()},{0});}
            catch(const std::overflow_error&){overflow=true;}require(overflow);
        }
        {
            const unsigned char image[]={0x11,0x22,0x33,0x44,0x55,0x66};
            const unsigned char alpha[]={0xaa,0xbb,0xcc,0xdd,0xee,0xff};
            const unsigned char expected[]={0x22,0x11,0x66,0x55,0x44,0x33};
            const unsigned char expected_alpha[]={0xbb,0xaa,0xff,0xee,0xdd,0xcc};
            const std::size_t order[]={1,0,5,4,3,2};
            const std::ptrdiff_t offsets[]={0};
            for(int channels:{3,4}) {
                constexpr std::size_t count=262149;
                std::vector<unsigned char> output(count*6*channels);
                gray_expand_rows_into(image,alpha,output.data(),1,count,6,channels,offsets,offsets,0,0,0,order);
                for(std::size_t i=0;i<count;++i)for(int channel=0;channel<channels;++channel)
                    require(std::equal(output.data()+(i*channels+channel)*6,output.data()+(i*channels+channel+1)*6,channel==3 ? expected_alpha : expected));
            }
            const std::size_t bad[]={0,1,2,3,4,6};unsigned char output[18];bool rejected=false;
            try {gray_expand_rows_into(image,nullptr,output,1,1,6,3,offsets,nullptr,6,0,0,bad);}
            catch(const std::invalid_argument&){rejected=true;}require(rejected);
            gray_expand_rows_into(nullptr,nullptr,nullptr,0,17,6,3,nullptr,nullptr,6,0,0,order);
        }
        {
            const std::uint8_t input[]={1,2,3,4,5,6};
            const std::uint64_t expected[]={1,3,6,5,12,21};std::uint64_t sums[6];
            integral_image_into(input,sums,PixelType::uint8,PixelType::uint64,{2,3});
            require(std::equal(sums,sums+6,expected));
            const std::uint8_t fortran_input[]={1,4,2,5,3,6};
            const std::uint64_t fortran_expected[]={1,5,3,12,6,21};
            integral_image_into(fortran_input,sums,PixelType::uint8,PixelType::uint64,{2,3},{1,2});
            require(std::equal(sums,sums+6,fortran_expected));
            const auto maximum=std::numeric_limits<std::uint64_t>::max();
            std::uint64_t repeated[66],scanned[66];std::fill(repeated,repeated+66,maximum);
            integral_image_into(repeated,scanned,PixelType::uint64,PixelType::uint64,{33});
            for(std::size_t j=0;j<33;++j)require(scanned[j]==maximum-j);
            integral_image_into(repeated,scanned,PixelType::uint64,PixelType::uint64,{2,33});
            for(std::size_t j=0;j<33;++j) {require(scanned[j]==maximum-j);require(scanned[33+j]==maximum-(2*j+1));}
            const std::uint64_t wrapping[]={maximum,1,maximum,1,maximum},wrapped[]={maximum,0,maximum,0,maximum};
            integral_image_into(wrapping,sums,PixelType::uint64,PixelType::uint64,{5});
            require(std::equal(sums,sums+5,wrapped));
            const auto high=std::numeric_limits<std::int64_t>::max(),low=std::numeric_limits<std::int64_t>::min();
            const std::int64_t signed_input[]={high,1,-1,low},signed_expected[]={high,low,high-1,-1};std::int64_t signed_sums[4];
            integral_image_into(signed_input,signed_sums,PixelType::int64,PixelType::int64,{2,2});
            require(std::equal(signed_sums,signed_sums+4,signed_expected));
            const double cancellation[]={1e16,1,-1e16,2},ordered[]={1e16,1e16,0,2};double wide[4];
            integral_image_into(cancellation,wide,PixelType::float64,PixelType::float64,{4});
            require(std::equal(wide,wide+4,ordered));
            const float narrow[]={16777216.f,1.f,-16777216.f};float single[3];
            integral_image_into(narrow,wide,PixelType::float32,PixelType::float64,{3});require(wide[2]==1.);
            integral_image_into(narrow,single,PixelType::float32,PixelType::float32,{3});require(single[2]==0.f);
            const double negative_zero[]={-0.,-0.,-0.,-0.};
            integral_image_into(negative_zero,wide,PixelType::float64,PixelType::float64,{2,2});
            for(auto value:wide)require(value==0. && std::signbit(value));
            integral_image_into(nullptr,nullptr,PixelType::uint8,PixelType::uint64,{3,0,5});
        }
        {
            const GaussianPass one{0,{.25,.5,.25},GaussianBorder::reflect,0};
            const double impulse[]={0,0,1,0,0},expected[]={0,.25,.5,.25,0};double result[5];
            gaussian_nd_into(impulse,result,PixelType::float64,PixelType::float64,{5},{one});
            require(std::equal(result,result+5,expected));
            const double precise[]={16777217.,0.,-16777216.};float single[3];
            gaussian_nd_into(precise,single,PixelType::float64,PixelType::float32,{3},{one});
            require(single[1]==.25f); // No premature float32 cast before the first axis.
            const std::uint8_t bytes[]={0,0,255,0,0};
            gaussian_nd_into(bytes,result,PixelType::uint8,PixelType::float64,{5},{one});
            require(std::equal(result,result+5,expected));
            gaussian_nd_into(bytes,result,PixelType::uint8,PixelType::float64,{5},{one},true);
            for(int j=0;j<5;++j)require(result[j]==255*expected[j]);
            std::uint32_t signaling=0x7f800001,bits;float unchanged;
            gaussian_nd_into(&signaling,&unchanged,PixelType::float32,PixelType::float32,{},{});
            std::memcpy(&bits,&unchanged,4);require(bits==signaling);
            gaussian_nd_into(nullptr,nullptr,PixelType::float32,PixelType::float64,{0,7},{});
        }
        const float log_inputs[]={.001f,.1f,.2f,.5f,.7f,.9f,1.f,1.1f,2.f,3.f,7.f,10.973364f,10.973365f,13.f,100.f,1e10f,0.f,-1.f,std::numeric_limits<float>::denorm_min()};
        float log_results[19];
        for(bool numpy_log : {false,true}) {
            threshold_logs(log_inputs,log_results,19,numpy_log);
            for(std::size_t j=0;j<19;++j) {
                const auto expected=threshold_log(log_inputs[j],numpy_log);
                require(log_results[j]==expected || (std::isnan(log_results[j]) && std::isnan(expected)));
            }
        }
        threshold_logs(nullptr,nullptr,0);
        std::vector<std::uint8_t> local_input(17*19,10);
        std::vector<double> local_output(local_input.size());
        local_threshold_into(local_input.data(),local_output.data(),PixelType::uint8,{17,19},{15,15},LocalThreshold::niblack);
        require(std::all_of(local_output.begin(),local_output.end(),[](double v){return v==10;}));
        local_threshold_into(local_input.data(),local_output.data(),PixelType::uint8,{17,19},{15,15},LocalThreshold::sauvola);
        require(std::all_of(local_output.begin(),local_output.end(),[](double v){return v==8;}));
        const float local32[]={.5f};float local32_output[1];
        local_threshold_into(local32,local32_output,PixelType::float32,{1},{1},LocalThreshold::sauvola);
        require(local32_output[0]==.4f);
        LocalThresholdParameters local_promoted;local_promoted.k_double=true;
        local_threshold_into(local32,local_output.data(),PixelType::float32,{1},{1},LocalThreshold::sauvola,local_promoted);
        require(local_output[0]==.4);
        const std::ptrdiff_t alpha_offsets[]={0};
        std::vector<std::uint64_t> gray_words(19),expanded_words(19*4);
        for(std::size_t j=0;j<19;++j)gray_words[j]=UINT64_MAX-j;
        const std::uint64_t opacity_word=0x123456789abcdef0ULL;
        gray_expand_rows_into(gray_words.data(),&opacity_word,expanded_words.data(),1,19,8,4,alpha_offsets,alpha_offsets,8,0);
        for(std::size_t j=0;j<19;++j) {
            for(std::size_t c=0;c<3;++c)require(expanded_words[4*j+c]==gray_words[j]);
            require(expanded_words[4*j+3]==opacity_word);
        }
        const std::uint8_t rgba_pixels[]={255,0,0,255,0,0,255,0,0,255,0,128,255,0,0,0,0,0,0,255};
        const double alpha_background[]={1,1,1};double rgb_composed[15],rgb_reversed[15];
        rgba2rgb_rows_into(rgba_pixels,rgb_composed,1,5,PixelType::uint8,alpha_background,alpha_offsets,4,1);
        require(rgb_composed[0]==1 && rgb_composed[1]==0 && rgb_composed[2]==0);
        require(rgb_composed[3]==1 && rgb_composed[4]==1 && rgb_composed[5]==1);
        require(rgb_composed[7]==1 && std::abs(rgb_composed[6]-(1-128*(1./255)))<1e-15);
        require(rgb_composed[12]==0 && rgb_composed[13]==0 && rgb_composed[14]==0);
        rgba2rgb_rows_into(rgba_pixels+16,rgb_reversed,1,5,PixelType::uint8,alpha_background,alpha_offsets,-4,1);
        for(std::size_t j=0;j<5;++j)for(std::size_t c=0;c<3;++c)require(rgb_reversed[3*j+c]==rgb_composed[3*(4-j)+c]);
        gray_expand_rows_into(nullptr,nullptr,nullptr,0,1,4,3,nullptr,nullptr,4,0);
        std::uint8_t impulse[3*19]={};impulse[19+9]=255;
        double small_edge_output[3*19];
        small_edge_into(impulse,small_edge_output,3,19,SmallEdge::laplace);
        for(std::size_t j=0;j<3*19;++j) {
            const double expected=j==28 ? 4. : (j==9 || j==27 || j==29 || j==47) ? -1. : 0.;
            require(small_edge_output[j]==expected);
        }
        small_edge_into(impulse,small_edge_output,3,19,SmallEdge::roberts_positive);
        require(small_edge_output[8]==1 && small_edge_output[28]==-1);
        small_edge_into(impulse,small_edge_output,3,19,SmallEdge::roberts_negative);
        require(small_edge_output[9]==1 && small_edge_output[27]==-1);
        small_edge_into(impulse,small_edge_output,3,19,SmallEdge::roberts);
        for(std::size_t j=0;j<3*19;++j) {
            const double expected=(j==8 || j==9 || j==27 || j==28) ? 1/std::sqrt(2.) : 0.;
            require(small_edge_output[j]==expected);
        }
        const float singleton_edge[]={.7f};float singleton_result[1];
        small_edge_into(singleton_edge,singleton_result,1,1,SmallEdge::roberts);
        require(singleton_result[0]==0);
        float line[17],line_laplace[17];for(int j=0;j<17;++j)line[j]=float(j);
        for(const auto& dimensions:std::vector<std::vector<std::size_t>>{{17},{17,1,1},{1,17,1}}) {
            laplace_nd_into(line,line_laplace,dimensions);
            require(line_laplace[0]==-1 && line_laplace[16]==1);
            for(int j=1;j<16;++j)require(line_laplace[j]==0);
        }
        const std::uint16_t halves[]={0,0x8000,1,0x03ff,0x0400,0x3c00,0x7bff,0xfbff,0x7c00,0xfc00,0x7e00};
        float decoded[11];edge_half_to_float(halves,decoded,11);
        require(decoded[0]==0 && std::signbit(decoded[1]));
        require(decoded[2]==std::ldexp(1.f,-24) && decoded[3]==1023*std::ldexp(1.f,-24));
        require(decoded[4]==std::ldexp(1.f,-14) && decoded[5]==1 && decoded[6]==65504 && decoded[7]==-65504);
        require(std::isinf(decoded[8]) && decoded[8]>0 && std::isinf(decoded[9]) && decoded[9]<0 && std::isnan(decoded[10]));
        const unsigned char mask_bytes[]={1,0,1,0,1,0,1,0};const std::ptrdiff_t mask_offsets[]={0};
        float masked32[]={1,-1,2,-2,3,-3,4,-4};double masked64[]={1,-1,2,-2,3,-3,4,-4};
        edge_mask_rows_into(masked32,true,mask_bytes,1,8,mask_offsets,1);
        edge_mask_rows_into(masked64,false,mask_bytes,1,8,mask_offsets,1);
        for(int j=0;j<8;++j) {
            require(masked32[j]==(j%2 ? 0 : j/2+1) && masked64[j]==masked32[j]);
            if(j%2)require(std::signbit(masked32[j]) && std::signbit(masked64[j]));
        }
        std::uint16_t impulse16[3*19]={};impulse16[28]=65535;
        small_edge_numeric_into(impulse16,small_edge_output,{3,19},PixelType::uint16,SmallEdge::laplace);
        require(small_edge_output[28]==4 && small_edge_output[9]==-1 && small_edge_output[27]==-1);
        impulse16[28]=0x3c00;float half_edges[3*19];
        small_edge_half_into(impulse16,half_edges,{3,19},SmallEdge::laplace);
        require(half_edges[28]==4 && half_edges[9]==-1 && half_edges[27]==-1);
        const std::uint16_t source_swap[]={0x1234,0xabcd};std::uint16_t target_swap[2];
        edge_swap_into(source_swap,target_swap,2,2);
        require(target_swap[0]==0x3412 && target_swap[1]==0xcdab);
        edge_swap_into(nullptr,nullptr,0,2);edge_half_to_float(nullptr,nullptr,0);
        laplace_nd_into(static_cast<const float*>(nullptr),static_cast<float*>(nullptr),{0,17});
        const double hist_counts[]={1,2,3,4,3,2,1};
        const double hist_centers[]={0,1,2,3,4,5,6};
        require(threshold_indices(hist_counts,hist_centers,7,HistogramThreshold::yen)[0]==2);
        require(threshold_indices(hist_counts,hist_centers,7,HistogramThreshold::isodata)==std::vector<std::size_t>({2,3}));
        require(threshold_indices(hist_counts,hist_centers,7,HistogramThreshold::triangle)[0]==2);
        const float histogram_input[]={0,.25f,.5f,.75f,1};
        const float histogram_bins[]={0,.5f,1};
        require(histogram_edges(histogram_input,5,histogram_bins,2)==std::vector<std::uint64_t>({2,3}));
        require(histogram_edges(static_cast<const float*>(nullptr),0,histogram_bins,2)==std::vector<std::uint64_t>({0,0}));
        bool invalid_histogram=false;
        try {histogram_edges(histogram_input,5,histogram_bins,0);}catch(const std::invalid_argument&) {invalid_histogram=true;}
        require(invalid_histogram);
        const double identity_stain[]={1,0,0,0,1,0,0,0,1};
        const std::uint8_t white_stain[]={255,255,255};
        double separated[3],combined[3];
        stain_transform_into(white_stain,separated,1,PixelType::uint8,StainTransform::separate,identity_stain);
        require(separated[0]==0 && separated[1]==0 && separated[2]==0);
        stain_transform_into(separated,combined,1,PixelType::float64,StainTransform::combine,identity_stain);
        require(combined[0]==1 && combined[1]==1 && combined[2]==1);
        const float zero_stain[]={0,0,0};
        float combined32[3];
        stain_transform_into(zero_stain,combined32,1,PixelType::float32,StainTransform::combine,identity_stain,3,PixelType::float32,PixelType::float32);
        require(combined32[0]==1 && combined32[1]==1 && combined32[2]==1);
        MorphologyFootprint rectangle;
        rectangle.scipy_separable=true;
        for (int y=-15;y<=15;++y) for (int x=-15;x<=15;++x) {
            rectangle.offsets.push_back(y); rectangle.offsets.push_back(x);
        }
        std::vector<std::uint8_t> bool_source(33*67,1),bool_output(33*67+2,255);
        morphology_into(bool_source.data(),bool_output.data()+1,PixelType::boolean,PixelType::boolean,
            {33,67},{rectangle},MorphologyOp::erosion,MorphologyMode::constant,0);
        require(std::count(bool_output.begin()+1,bool_output.end()-1,std::uint8_t(1))==3*37);
        require(bool_output.front()==255 && bool_output.back()==255);
        bool_source[0]=2;
        morphology_into(bool_source.data(),bool_output.data()+1,PixelType::boolean,PixelType::boolean,
            {33,67},{rectangle},MorphologyOp::dilation,MorphologyMode::reflect);
        require(bool_output[1]==2 && bool_output[33*67]==1);
        const std::uint8_t morph_source[]={4,1,7,2,8};
        std::uint8_t morph_result[5];
        const std::vector<MorphologyFootprint> footprint={{{-1,0,1},1,true}};
        const std::uint64_t wide_source[]={1,2,3};
        std::uint64_t wide_output[3];
        morphology_into(wide_source,wide_output,PixelType::uint64,PixelType::uint64,{3},footprint,
            MorphologyOp::erosion,MorphologyMode::constant,-std::numeric_limits<double>::infinity());
        require(wide_output[0]==(std::uint64_t(1)<<63) && wide_output[1]==1 && wide_output[2]==(std::uint64_t(1)<<63));
        morphology_into(morph_source,morph_result,PixelType::uint8,PixelType::uint8,{5},footprint,MorphologyOp::erosion,MorphologyMode::reflect);
        require(std::equal(morph_result,morph_result+5,std::vector<std::uint8_t>{1,1,1,2,2}.begin()));
        morphology_into(morph_source,morph_result,PixelType::uint8,PixelType::uint8,{5},footprint,MorphologyOp::dilation,MorphologyMode::reflect);
        require(std::equal(morph_result,morph_result+5,std::vector<std::uint8_t>{4,7,7,8,8}.begin()));
        morphology_into(morph_source,morph_result,PixelType::uint8,PixelType::uint8,{5},footprint,MorphologyOp::binary_erosion,MorphologyMode::ignore);
        require(std::all_of(morph_result,morph_result+5,[](auto v){return v==1;}));
        const double rgb[] = {1, 0, 0, 0, 1, 0};
        const auto gray = rgb2gray(rgb, 1, 2);
        require(gray[0] == 0.2125 && gray[1] == 0.7154);
        const std::uint8_t rgb8[] = {255,0,0,0,255,0};
        require(rgb2gray(rgb8,1,2) == gray);
        std::vector<double> direct(2);
        rgb2gray_into(rgb,direct.data(),1,2);
        require(direct == gray);
        std::vector<double> a(513*515, 0.5);
        const auto blur = gaussian(a.data(),513,515);
        const auto edges = sobel(a.data(),513,515);
        direct.assign(a.size()+2,-999.0);
        gaussian_into(a.data(),direct.data()+1,513,515);
        require(direct.front() == -999.0 && direct.back() == -999.0);
        require(std::equal(blur.begin(),blur.end(),direct.begin()+1));
        sobel_into(a.data(),direct.data()+1,513,515);
        require(direct.front() == -999.0 && direct.back() == -999.0);
        require(std::equal(edges.begin(),edges.end(),direct.begin()+1));
        for (std::size_t i = 0; i < a.size(); ++i) {
            require(std::abs(blur[i]-0.5) < 1e-14);
            require(edges[i] == 0);
        }
        const std::uint8_t binary[] = {10, 10, 200, 200};
        require(threshold_otsu(binary,2,2) == 10);
        require(threshold_mean(binary,2,2) == 105);
        std::vector<std::uint8_t> inverted(4), tripled(12);
        invert_into(binary,inverted.data(),2,2);
        gray2rgb_into(binary,tripled.data(),2,2);
        for (std::size_t i = 0; i < 4; ++i) {
            require(inverted[i] == 255-binary[i]);
            for (std::size_t c = 0; c < 3; ++c) require(tripled[3*i+c] == binary[i]);
        }
        for (auto filter : {EdgeFilter::sobel,EdgeFilter::prewitt,EdgeFilter::scharr}) {
            gradient_into(a.data(),direct.data()+1,513,515,filter,0);
            for (std::size_t i = 1; i <= a.size(); ++i) require(direct[i] == 0);
        }
        const auto counts = histogram_u8(binary,2,2);
        require(counts[10] == 2 && counts[200] == 2);
        std::vector<double> equalized(4);
        equalize_hist_into(binary,equalized.data(),2,2);
        require(equalized[0] == 0.5 && equalized[3] == 1.0);
        const auto range = intensity_bounds(binary,2,2);
        require(range.first == 10 && range.second == 200);
        rescale_into(binary,inverted.data(),2,2,10,200,0,255);
        require(inverted[0] == 0 && inverted[3] == 255);
        exposure_into(binary,inverted.data(),2,2,ExposureTransform::gamma,1,1,false);
        require(std::equal(inverted.begin(),inverted.end(),binary));
        const double values[] = {0.0,0.25,1.0};
        std::vector<double> corrected(3);
        exposure_into(values,corrected.data(),1,3,ExposureTransform::gamma,0.5,1,false);
        require(corrected[0] == 0 && corrected[1] == 0.5 && corrected[2] == 1);
        const std::uint8_t bytes[] = {0,1,127,128,255};
        float normalized[5];
        convert_into(bytes,normalized,5,PixelType::uint8,PixelType::float32);
        require(normalized[0] == 0 && normalized[4] == 1);
        std::vector<double> color_rgb(3*19), yuv(color_rgb.size()), roundtrip(color_rgb.size());
        for (std::size_t j = 0; j < color_rgb.size(); ++j) color_rgb[j] = double(j)/double(color_rgb.size());
        color_transform_into(color_rgb.data(),yuv.data(),19,PixelType::float64,ColorTransform::rgb2yuv);
        color_transform_into(yuv.data(),roundtrip.data(),19,PixelType::float64,ColorTransform::yuv2rgb);
        for (std::size_t j = 0; j < color_rgb.size(); ++j) require(std::abs(roundtrip[j]-color_rgb[j]) < 2e-15);
        std::vector<double> planar(color_rgb.size()), strided(yuv.size());
        for (std::size_t j = 0; j < 19; ++j)
            for (std::size_t channel = 0; channel < 3; ++channel) planar[19*channel+j] = color_rgb[3*j+channel];
        color_transform_strided_into(planar.data(),strided.data(),19,1,19,ColorTransform::rgb2yuv);
        for (std::size_t j = 0; j < yuv.size(); ++j) require(strided[j] == yuv[j]);
        color_transform_strided_into(planar.data()+18,strided.data(),19,-1,19,ColorTransform::rgb2yuv);
        for (std::size_t j = 0; j < 19; ++j)
            for (std::size_t channel = 0; channel < 3; ++channel) require(strided[3*j+channel] == yuv[3*(18-j)+channel]);
        const std::uint8_t white[] = {255,255,255};
        double xyz[3];
        color_transform_into(white,xyz,1,PixelType::uint8,ColorTransform::rgb2xyz);
        require(std::abs(xyz[0]-0.950456) < 1e-14 && std::abs(xyz[1]-1) < 1e-14 && std::abs(xyz[2]-1.088754) < 1e-14);
        color_transform_into(nullptr,nullptr,0,PixelType::float32,ColorTransform::rgb2yuv);
        const double hsv_red[] = {1,0,0}, hsv_gray[] = {0.4,0.4,0.4};
        double hsv[3];
        auto perception = perceptual_into(hsv_red,hsv,1,PixelType::float64,PerceptualTransform::rgb2hsv);
        require(!perception.unsupported && hsv[0] == 0 && hsv[1] == 1 && hsv[2] == 1);
        perceptual_into(hsv_gray,hsv,1,PixelType::float64,PerceptualTransform::rgb2hsv);
        require(hsv[0] == 0 && hsv[1] == 0 && hsv[2] == 0.4);
        const double xyz_source[] = {0.1,0.4,0.7};
        double lab[3], xyz_back[3];
        double luv[3];
        perceptual_into(xyz_source,luv,1,PixelType::float64,PerceptualTransform::xyz2luv);
        perceptual_into(luv,xyz_back,1,PixelType::float64,PerceptualTransform::luv2xyz);
        for (int j=0;j<3;++j) require(std::abs(xyz_source[j]-xyz_back[j])<2e-14);
        perceptual_into(xyz_source,lab,1,PixelType::float64,PerceptualTransform::xyz2lab);
        perceptual_into(lab,xyz_back,1,PixelType::float64,PerceptualTransform::lab2xyz);
        for (int j = 0; j < 3; ++j) require(std::abs(xyz_source[j]-xyz_back[j]) < 2e-15);
        const float invalid_lab[] = {0,0,100};
        float xyz32[3];
        perception = perceptual_into(invalid_lab,xyz32,1,PixelType::float32,PerceptualTransform::lab2xyz);
        require(perception.negative_z == 1);
        lab_scalar_negative_z_into(invalid_lab,xyz_back,PerceptualTransform::lab2xyz,{0.95047,1,1.08883});
        require(xyz_back[0] < 0 && xyz_back[2] < 0);
        std::uint16_t expanded[5];
        convert_into(bytes,expanded,5,PixelType::uint8,PixelType::uint16);
        require(expanded[1] == 257 && expanded[4] == 65535);
        std::uint8_t restored[5];
        convert_into(expanded,restored,5,PixelType::uint16,PixelType::uint8);
        require(std::equal(restored,restored+5,bytes));
        const std::int64_t small[] = {-65535,0,127};
        const auto info = convert_into(small,restored,3,PixelType::int64,PixelType::uint8);
        require(info.downcast_without_scaling && info.maximum_signed == 127);
        require(restored[0] == 1 && restored[2] == 127);
    }
    set_num_threads(1);
    require(get_num_threads() == 1);
    {
        const std::int16_t a[]={-100,20,30,0,0,0};
        const float b[]={-97,24,30,0,0,12};
        double distances[2];
        delta_e76_into(a,b,distances,2,PixelType::int16,PixelType::float32);
        require(distances[0]==5 && distances[1]==12);
        float single[1];
        delta_e76_into(b,b,single,1,PixelType::float32,PixelType::float32);
        require(single[0]==0);
        delta_e76_into(nullptr,nullptr,nullptr,0,PixelType::float32,PixelType::float64);
        {
            const float origin[]={0,0,0};
            float pixels[54]={},distances[18];
            for(int i=0;i<18;++i)pixels[3*i]=float(i+1);
            const std::ptrdiff_t first_rows[]={0,0},second_rows[]={27,0};
            delta_e76_rows_into(origin,pixels,distances,2,9,PixelType::float32,PixelType::float32,
                first_rows,second_rows,0,1,3,1);
            for(int i=0;i<18;++i)require(distances[i]==float((i+9)%18+1));
            delta_e76_rows_into(nullptr,nullptr,nullptr,0,9,PixelType::float32,PixelType::float32,nullptr,nullptr);
            const std::ptrdiff_t overflow[]={std::numeric_limits<std::ptrdiff_t>::max()};
            bool caught=false;
            try {delta_e76_rows_into(origin,pixels,distances,1,1,PixelType::float32,PixelType::float32,
                overflow,second_rows);}catch(const std::overflow_error&){caught=true;}
            require(caught);
        }
        const double gray1[]={50,0,0},gray2[]={55,0,0};
        double weighted;
        delta_e_into(gray1,gray2,&weighted,1,PixelType::float64,PixelType::float64,DistanceMetric::cie94);
        require(weighted==5);
        delta_e_into(gray1,gray2,&weighted,1,PixelType::float64,PixelType::float64,DistanceMetric::ciede2000);
        require(std::abs(weighted-5/(1+.015*6.25/std::sqrt(26.25)))<1e-12);
        delta_e_into(gray1,gray2,&weighted,1,PixelType::float64,PixelType::float64,DistanceMetric::cmc);
        require(std::abs(weighted-5/(.040975*50/(1+.01765*50)))<1e-12);
        delta_e_into(nullptr,nullptr,nullptr,0,PixelType::float32,PixelType::float32,DistanceMetric::cmc,{},true,true);
    }
    bool rejected = false;
    try { sobel<double>(nullptr,2,2); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    rejected = false;
    try { convert_into(nullptr,nullptr,1,PixelType::uint8,PixelType::float32); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    convert_into(nullptr,nullptr,0,PixelType::uint8,PixelType::float32);
    rejected = false;
    const float pixel = 1;
    try { sobel_into(&pixel,static_cast<float*>(nullptr),1,1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    rejected = false;
    const double value = 0;
    try { sobel(&value,std::numeric_limits<std::size_t>::max(),2); }
    catch (const std::overflow_error&) { rejected = true; }
    require(rejected);
    std::cout << "Native C++ checks passed; AVX2=" << has_avx2() << '\n';
}
