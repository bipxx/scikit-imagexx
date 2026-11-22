#include <skimagexx/draw_haar_like_feature.hpp>
#include <skimagexx/haar_like_features.hpp>
#include <type_traits>
static_assert(!std::is_same_v<skimagexx::HaarRectangle, skimagexx::HaarLikeRectangle>);
static_assert(!std::is_same_v<skimagexx::HaarFeature, skimagexx::HaarLikeFeature>);
