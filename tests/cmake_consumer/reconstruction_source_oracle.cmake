# Exact259 source-safe0.26 cases; all three recorded runtime payloads identical.
set(SKIMAGEXX_RECONSTRUCTION_SOURCE_ORACLE "${CMAKE_CURRENT_LIST_DIR}/reconstruction_source_oracle.bin")
if(NOT EXISTS "${SKIMAGEXX_RECONSTRUCTION_SOURCE_ORACLE}")
    message(FATAL_ERROR "The source-derived reconstruction SDK oracle is missing")
endif()
file(SHA256 "${SKIMAGEXX_RECONSTRUCTION_SOURCE_ORACLE}" _reconstruction_oracle_sha256)
if(NOT _reconstruction_oracle_sha256 STREQUAL "b3c8f265e92a431ba282ad715856d9c7f4d24834b878876c63a90dcf703e860e")
    message(FATAL_ERROR "The reconstruction SDK oracle differs from frozen source observations")
endif()
