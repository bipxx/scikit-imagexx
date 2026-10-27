#pragma once
#include "skimagexx/colocalization.hpp"

namespace skimagexx::detail {
// Private prepared input: aligned native-C float32/64, shape(F,K,3), K>=3.
// NumPy has already gathered ALL face indices. Only the first three vertices
// enter arithmetic; the caller owns and pins the complete readable allocation.
struct MeshAreaInput {
    const void* data;
    ColocType type;
    std::size_t faces,vertices_per_face;
};
// Raw typed sum BEFORE the source's final scalar /2.0 provider. The existing
// ColocProviderRequired exception means no scalar was published: replay the
// remaining NumPy primitives on the same gathered snapshot. Other exceptions
// propagate. Refuse nonfinite/subnormal arithmetic and raised NumPy FP flags;
// restore each participating thread's complete floating environment.
ColocScalar mesh_surface_area(const MeshAreaInput&,ColocReductionPlan,int workers);
}

namespace skimagexx::simd {
// Consume a packet prefix; stride is elements between gathered faces.
std::size_t mesh_face_norms(const void*,ColocType,std::size_t count,
    std::size_t stride,void* output,bool& accepted);
}
