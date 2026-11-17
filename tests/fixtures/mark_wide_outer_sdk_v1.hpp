#pragma once
#include "../cmake_consumer/mark_boundaries_consumer.hpp"
#include <cstdint>
#include <utility>
namespace skimagexx_test {
struct WideFP {std::uint16_t control,status;std::uint32_t mxcsr;};
struct WideSaved {MarkSaved data;bool mark;skimagexx::PixelType result_type;WideFP before,after;};
inline std::vector<WideSaved> mark_wide_saved_cases(){std::vector<WideSaved> out;
{WideSaved w;auto& c=w.data;
c.id="minimum/wide-outer/find-int64-small-3x4";c.modern=false;w.mark=false;
c.labels.type=skimagexx::PixelType::int64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
w.result_type=skimagexx::PixelType::boolean;c.result_shape={3,4};c.result_strides={4,1};
c.expected=mark_bytes("010101010001010101010101");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,32,8099};
w.after={895,32,8098};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="minimum/wide-outer/find-uint64-fullwidth-3x4";c.modern=false;w.mark=false;
c.labels.type=skimagexx::PixelType::uint64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("0000000000000000fffffffffffffffffeffffffffffffff0100000000000000ffffffffffffff7f0000000000000080010000000000008000000000000000000000000000002000010000000000200002000000000020000200000000000000");
w.result_type=skimagexx::PixelType::boolean;c.result_shape={3,4};c.result_strides={4,1};
c.expected=mark_bytes("000000000101000101010101");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,32,8099};
w.after={895,32,8098};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="minimum/wide-outer/mark-gray-u8-int64-small-3x4";c.modern=false;w.mark=true;
c.labels.type=skimagexx::PixelType::int64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
c.image.type=skimagexx::PixelType::uint8;c.image.half=false;c.image.swap=false;c.image.shape={3,4};c.image.strides={4,1};
c.image_bytes=mark_bytes("000102030405060708090a0b");
w.result_type=skimagexx::PixelType::float64;c.result_shape={3,4,3};c.result_strides={12,3,1};
c.expected=mark_bytes("000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000101010101010903f101010101010903f101010101010903f000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,32,8099};
w.after={895,32,8098};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="minimum/wide-outer/mark-RGB-f32-uint64-small-3x4";c.modern=false;w.mark=true;
c.labels.type=skimagexx::PixelType::uint64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
c.image.type=skimagexx::PixelType::float32;c.image.half=false;c.image.swap=false;c.image.shape={3,4,3};c.image.strides={48,12,4};
c.image_bytes=mark_bytes("000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f");
w.result_type=skimagexx::PixelType::float32;c.result_shape={3,4,3};c.result_strides={12,3,1};
c.expected=mark_bytes("0000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f00000000000000000000803e0000003f0000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f00000000");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,32,8099};
w.after={895,32,8098};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="modern/wide-outer/find-int64-small-3x4";c.modern=true;w.mark=false;
c.labels.type=skimagexx::PixelType::int64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
w.result_type=skimagexx::PixelType::boolean;c.result_shape={3,4};c.result_strides={4,1};
c.expected=mark_bytes("010101010001010101010101");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,0,8097};
w.after={895,0,8096};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="modern/wide-outer/find-uint64-fullwidth-3x4";c.modern=true;w.mark=false;
c.labels.type=skimagexx::PixelType::uint64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("0000000000000000fffffffffffffffffeffffffffffffff0100000000000000ffffffffffffff7f0000000000000080010000000000008000000000000000000000000000002000010000000000200002000000000020000200000000000000");
w.result_type=skimagexx::PixelType::boolean;c.result_shape={3,4};c.result_strides={4,1};
c.expected=mark_bytes("000000000101000101010101");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,0,8097};
w.after={895,0,8096};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="modern/wide-outer/mark-gray-u8-int64-small-3x4";c.modern=true;w.mark=true;
c.labels.type=skimagexx::PixelType::int64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
c.image.type=skimagexx::PixelType::uint8;c.image.half=false;c.image.swap=false;c.image.shape={3,4};c.image.strides={4,1};
c.image_bytes=mark_bytes("000102030405060708090a0b");
w.result_type=skimagexx::PixelType::float64;c.result_shape={3,4,3};c.result_strides={12,3,1};
c.expected=mark_bytes("000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000101010101010903f101010101010903f101010101010903f000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000000000000000f03f000000000000f03f0000000000000000");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,0,8097};
w.after={895,0,8096};
out.push_back(std::move(w));}
{WideSaved w;auto& c=w.data;
c.id="modern/wide-outer/mark-RGB-f32-uint64-small-3x4";c.modern=true;w.mark=true;
c.labels.type=skimagexx::PixelType::uint64;c.labels.half=false;c.labels.swap=false;c.labels.shape={3,4};c.labels.strides={32,8};
c.label_bytes=mark_bytes("000000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000030000000000000003000000000000000000000000000000000000000000000001000000000000000100000000000000");
c.image.type=skimagexx::PixelType::float32;c.image.half=false;c.image.swap=false;c.image.shape={3,4,3};c.image.strides={48,12,4};
c.image_bytes=mark_bytes("000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f000000000000803e0000003f0000403f");
w.result_type=skimagexx::PixelType::float32;c.result_shape={3,4,3};c.result_strides={12,3,1};
c.expected=mark_bytes("0000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f00000000000000000000803e0000003f0000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f000000000000803f0000803f00000000");
c.parameters.boundary={skimagexx::detail::FindBoundaryMode::outer,1,0,true};c.parameters.outline=false;
c.parameters.color={{1,1,0},{3}};c.parameters.cubic_profile=c.modern?skimagexx::MarkCubicProfile::scipy118:skimagexx::MarkCubicProfile::scipy115;
w.before={895,0,8097};
w.after={895,0,8096};
out.push_back(std::move(w));}
return out;}
}
