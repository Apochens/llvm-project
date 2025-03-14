include_directories(include)

set(TOSA_SMITH_MAIN_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TOSA_SMITH_INCLUDE_DIR ${TOSA_SMITH_MAIN_DIR}/include)

set(LLVM_LINK_COMPONENTS
    Support
)

add_mlir_tool(tosa-smith
    tosa-smith.cpp
    lib/TOSASmith.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(tosa-smith

    PRIVATE
    ${dialect_libs}
)