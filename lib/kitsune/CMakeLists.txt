add_custom_target(kitsune)

set(KITSUNE_RTL_CFLAGS -std=c++11)
append_rtti_flag(OFF KITSUNE_RTL_CFLAGS)

include_directories(../../include ${CUDA_INCLUDE_DIRS})

set(KITSUNE_SOURCES
    CUDARuntime.cpp
    GPURuntime.cpp)

foreach (arch ${KITSUNE_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.kitsune
    STATIC
    ARCHS ${arch}
    SOURCES ${KITSUNE_SOURCES}
    CFLAGS ${KITSUNE_RTL_CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0
    PARENT_TARGET kitsune)
  add_dependencies(kitsune
    clang_rt.kitsune-${arch})
endforeach()

add_dependencies(compiler-rt kitsune)
