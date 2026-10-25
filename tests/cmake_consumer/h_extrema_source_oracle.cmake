# Sixteen exact source-derived integer masks; all3original runtimes identical.
set(SKIMAGEXX_H_EXTREMA_SOURCE_ORACLE "${CMAKE_CURRENT_LIST_DIR}/h_extrema_source_oracle.bin")
if(NOT EXISTS "${SKIMAGEXX_H_EXTREMA_SOURCE_ORACLE}")
    message(FATAL_ERROR "The H-extrema source-derived SDK oracle is missing")
endif()
file(SHA256 "${SKIMAGEXX_H_EXTREMA_SOURCE_ORACLE}" _h_extrema_oracle_sha256)
if(NOT _h_extrema_oracle_sha256 STREQUAL "091de3e03a728ada6e181a9ed22f7ceca071d66a5c8f1df968c09c39905716a5")
    message(FATAL_ERROR "The H-extrema source-derived SDK oracle differs")
endif()
