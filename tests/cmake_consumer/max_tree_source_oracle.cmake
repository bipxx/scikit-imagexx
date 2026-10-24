# Immutable233 safe observations from scikit-image0.26.0, shared byte-for-byte
# by Windows, modern NumPy and minimum NumPy. No Python is needed by clients.
set(SKIMAGEXX_MAX_TREE_SOURCE_ORACLE "${CMAKE_CURRENT_LIST_DIR}/max_tree_source_oracle.bin")
if(NOT EXISTS "${SKIMAGEXX_MAX_TREE_SOURCE_ORACLE}")
    message(FATAL_ERROR "The authenticated max-tree source oracle is missing")
endif()
file(SHA256 "${SKIMAGEXX_MAX_TREE_SOURCE_ORACLE}" _max_tree_oracle_sha256)
if(NOT _max_tree_oracle_sha256 STREQUAL "e7162c6aa8a43434e0c6db0f835912dfc48b64eedca7e79192cab61bc39d93b5")
    message(FATAL_ERROR "The max-tree source oracle does not match the frozen original observations")
endif()
