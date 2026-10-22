#------------------------------------------------------------------------------
# Microstructure generator (namespace exaconstit::microstructure)
#
# Builds one static library per module (options, mesh, io, geometry, seeding, labeling,
# morphology), plus the driver library exaconstit_microstructure and the standalone tool
# microstructure_generator.
# Splitting by module means a change in one module never recompiles another.
#
# Each module "globs" its own subdirectory: CMake collects every <dir>/*.cpp and *.hpp itself
# (file(GLOB ... CONFIGURE_DEPENDS) re-checks the directory at build time), so adding a source
# file needs no edit here, only a re-run of cmake.
#
# The libraries do not depend on exaconstit_static: the generator needs only MFEM, Axom and MPI.
# This file is only added when ENABLE_MICROSTRUCTURE is ON (top-level CMakeLists.txt), which in
# turn requires Axom.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Dependencies (mirrors src/CMakeLists.txt)
#
# MICROSTRUCTURE_DEPENDS is the third-party list every ms_* library, the tool and the tests link.
# exaconstit_fill_depends_list adds each package only if it was found/enabled.
#------------------------------------------------------------------------------
set(MICROSTRUCTURE_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  MICROSTRUCTURE_DEPENDS
                             DEPENDS_ON mfem axom RAJA camp umpire mpi)

# GPU and OpenMP runtimes. BLT >= 0.6 namespaces its targets (blt::cuda), older BLT does not.
# CUDA::cublas (CUDA) and hipblas/rocsparse/rocrand (HIP) are linked explicitly, exactly as the
# mechanics executable (src/CMakeLists.txt, EXACONSTIT_DRIVER) and the ExaConstit tests
# (test/CMakeLists.txt, EXACONSTIT_TEST_DEPENDS) do; without them a GPU build of the tool or the
# tests can fail to link against the GPU-enabled third-party libraries.
if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND MICROSTRUCTURE_DEPENDS blt::cuda_runtime blt::cuda CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND MICROSTRUCTURE_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND MICROSTRUCTURE_DEPENDS cuda cuda_runtime CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND MICROSTRUCTURE_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND MICROSTRUCTURE_DEPENDS blt::hip blt::hip_runtime hipblas rocsparse rocrand)
endif()

# Some Linux toolchains need pthreads named explicitly (same switch as src/CMakeLists.txt)
if(UNIX AND NOT APPLE AND TARGET Threads::Threads)
    if(EXACONSTIT_THREADS_EXPLICIT_LINK)
        list(APPEND MICROSTRUCTURE_DEPENDS Threads::Threads)
    endif()
endif()

message(STATUS "MICROSTRUCTURE_DEPENDS: ${MICROSTRUCTURE_DEPENDS}")
# Exported to the parent scope because test/microstructure passes the list straight to its test
# executables: BLT compiles a source file as CUDA/HIP only when blt::cuda/blt::hip is a direct
# dependency of its target, and GPU kernels in a test must be compiled as device code. For the
# same reason, because MICROSTRUCTURE_DEPENDS contains blt::cuda / blt::hip, every ms_* source is
# compiled as device code (nvcc/hipcc) in GPU builds.
set(MICROSTRUCTURE_DEPENDS ${MICROSTRUCTURE_DEPENDS} PARENT_SCOPE)

# src/ is on the include path so headers are written "microstructure/..." and
# "utilities/auto_mesh.hpp", as in the rest of ExaConstit
set(MICROSTRUCTURE_INCLUDES ${HEADER_INCLUDE_DIR}
                            ${PROJECT_SOURCE_DIR}/src)

# MICROSTRUCTURE_STRICT_FP: forbid the compiler from fusing a*b+c into one FMA instruction
# (fused multiply-add, rounded once instead of twice). Compilers fuse differently on different
# hardware (nvcc fuses by default), so turning fusion off removes one source of CPU/GPU round-off
# differences, e.g. in which seed wins on an element that is almost equidistant from two seeds.
# Applied PRIVATE to the compiled ms_* sources only, never propagated to consumers such as
# mechanics.
set(MICROSTRUCTURE_COMPILE_OPTIONS)
if(MICROSTRUCTURE_STRICT_FP)
    list(APPEND MICROSTRUCTURE_COMPILE_OPTIONS
         $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
         $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
         $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>
         $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
endif()

#------------------------------------------------------------------------------
# ms_core: header-only (types, exec dispatch, options struct, util/)
#
# With HEADERS only, blt_add_library creates an INTERFACE library: nothing is compiled, but the
# include paths, dependencies and the define below propagate to everything that links ms_core.
#------------------------------------------------------------------------------
file(GLOB MS_CORE_HEADERS CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/util/*.hpp)

blt_add_library(NAME       ms_core
                HEADERS    ${MS_CORE_HEADERS}
                INCLUDES   ${MICROSTRUCTURE_INCLUDES}
                DEPENDS_ON ${MICROSTRUCTURE_DEPENDS})

# BLT applies DEFINES with PUBLIC, which an INTERFACE library rejects. Every other module
# depends on ms_core, so the define reaches all of them and their consumers.
target_compile_definitions(ms_core INTERFACE EXACONSTIT_USE_MICROSTRUCTURE)

#------------------------------------------------------------------------------
# ms_add_module(NAME <target> GLOBS <glob...> DEPENDS_ON <targets...>)
#
# Creates the static library <target> from every <glob>.cpp / <glob>.hpp (paths relative to this
# directory, e.g. "mesh/*"). Every module links ms_core and MICROSTRUCTURE_DEPENDS; DEPENDS_ON
# lists the other ms_* modules it uses. Fails at configure time if a glob matches no .cpp.
#------------------------------------------------------------------------------
function(ms_add_module)
    # Not the usual arg_ prefix: the BLT macros called below parse into arg_* in this scope
    cmake_parse_arguments(_ms "" "NAME" "GLOBS;DEPENDS_ON" ${ARGN})

    # Expand each glob into the matching source and header files
    set(_sources)
    set(_headers)
    foreach(_glob ${_ms_GLOBS})
        file(GLOB _src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_glob}.cpp)
        file(GLOB _hdr CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_glob}.hpp)
        list(APPEND _sources ${_src})
        list(APPEND _headers ${_hdr})
    endforeach()

    if(NOT _sources)
        message(FATAL_ERROR "ms_add_module(${_ms_NAME}): no sources match ${_ms_GLOBS}")
    endif()

    blt_add_library(NAME       ${_ms_NAME}
                    SOURCES    ${_sources}
                    HEADERS    ${_headers}
                    INCLUDES   ${MICROSTRUCTURE_INCLUDES}
                    DEPENDS_ON ${_ms_DEPENDS_ON} ms_core ${MICROSTRUCTURE_DEPENDS}
                    SHARED     FALSE)

    # Strict floating-point flags, private to this library's own sources
    if(MICROSTRUCTURE_COMPILE_OPTIONS)
        target_compile_options(${_ms_NAME} PRIVATE ${MICROSTRUCTURE_COMPILE_OPTIONS})
    endif()
endfunction()

#------------------------------------------------------------------------------
# Module libraries (the dependency order follows the pipeline: seeds need the mesh sampler,
# labeling needs the mesh and the seed search structure in geometry/)
#------------------------------------------------------------------------------
ms_add_module(NAME ms_options  GLOBS microstructure_options)
ms_add_module(NAME ms_mesh     GLOBS mesh/*)
ms_add_module(NAME ms_io       GLOBS io/*)
ms_add_module(NAME ms_geometry GLOBS geometry/*)
ms_add_module(NAME ms_seeding  GLOBS seeding/*
              DEPENDS_ON ms_mesh)
# orientation/ (random orientation sampler and the per-grain orientation assigner) and regions/
# (grain -> material region assignment) are part of ms_labeling. ms_io provides the voxel map
# reader used by the voxel labeler, and ms_seeding the point-in-element locator used to find the
# element that holds each seed (connected and geodesic labeling).
ms_add_module(NAME ms_labeling GLOBS labeling/* orientation/* regions/*
              DEPENDS_ON ms_mesh ms_geometry ms_io ms_seeding)
# morphology/ (size control): target grain volumes, Laguerre weight fitting and Lloyd relaxation.
# Uses the seed search (geometry), the cell measures (labeling) and the point-in-material test
# (seeding).
ms_add_module(NAME ms_morphology GLOBS morphology/*
              DEPENDS_ON ms_seeding ms_labeling ms_geometry ms_mesh)

# Driver library: everything a consumer (tool, tests, mechanics hook) links against: the
# pipeline (microstructure_driver) and the record tables and log phrases both consumers write
# (microstructure_report)
ms_add_module(NAME exaconstit_microstructure GLOBS microstructure_driver microstructure_report
              DEPENDS_ON ms_options ms_io ms_morphology ms_seeding ms_labeling ms_geometry
                         ms_mesh)

#------------------------------------------------------------------------------
# Standalone tool: bin/microstructure_generator (options file and/or -m -n -s -o --parts)
#------------------------------------------------------------------------------
blt_add_executable(NAME       microstructure_generator
                   SOURCES    tools/microstructure_generator.cpp
                   OUTPUT_DIR ${BINARY_DIR}
                   DEPENDS_ON exaconstit_microstructure ${MICROSTRUCTURE_DEPENDS})
