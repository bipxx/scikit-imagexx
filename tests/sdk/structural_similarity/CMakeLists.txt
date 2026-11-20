cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(StructuralSimilarityClient LANGUAGES CXX)
    find_package(skimagexx CONFIG REQUIRED)
    enable_testing()
endif()

add_executable(structural_similarity_client ../../cmake_consumer/structural_similarity_consumer.cpp)
target_compile_features(structural_similarity_client PRIVATE cxx_std_17)
target_link_libraries(structural_similarity_client PRIVATE skimagexx::skimagexx)
if(MSVC)
    target_compile_options(structural_similarity_client PRIVATE /fp:strict)
else()
    target_compile_options(structural_similarity_client PRIVATE -ffp-contract=off -fno-fast-math -frounding-math)
endif()
# scikit-image 0.26.0 single-center scores from structural_similarity_reference.py.
add_test(NAME installed_structural_similarity_consumer
    COMMAND structural_similarity_client 0x1.88dacc0000000p-2 0x1.88dacb8cd48dbp-2)
