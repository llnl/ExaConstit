#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
# The header of user classes (exaconstit_user.hpp) names the compiler ExaConstit was built with
# (EXACONSTIT_USER_CXX, set in the top-level CMakeLists.txt)
configure_file(${PROJECT_SOURCE_DIR}/src/exaconstit_user.hpp.in
               ${HEADER_INCLUDE_DIR}/exaconstit_user.hpp @ONLY)

set(EXACONSTIT_HEADERS
    ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
    ${HEADER_INCLUDE_DIR}/exaconstit_user.hpp
    system_driver.hpp
    boundary_conditions/BCData.hpp
    boundary_conditions/BCManager.hpp
    fem_operators/mechanics_integrators.hpp
    fem_operators/mechanics_operator_ext.hpp
    fem_operators/mechanics_operator.hpp
    mfem_expt/partial_qspace.hpp
    mfem_expt/partial_qfunc.hpp
    models/mechanics_model.hpp
    models/mechanics_ecmech.hpp
    models/mechanics_multi_model.hpp
    models/mechanics_umat.hpp
    options/option_parser_v2.hpp
    postprocessing/projection_class.hpp
    postprocessing/postprocessing_driver.hpp
    postprocessing/mechanics_lightup.hpp
    postprocessing/surface_quantities.hpp
    postprocessing/viz_point_map.hpp
    postprocessing/volume_statistics.hpp
    postprocessing/postprocessing_hdf5_file.hpp
    sim_state/simulation_state.hpp
    sim_state/exa_sidre_parmesh.hpp
    sim_state/restart_checkpoint.hpp
    solvers/mechanics_solver.hpp
    solvers/trust_region_solver.hpp
    utilities/dynamic_function_loader.hpp
    utilities/kinematics_kernels.hpp
    utilities/mechanics_kernels.hpp
    utilities/mechanics_log.hpp
    utilities/assembly_ops.hpp
    utilities/auto_mesh.hpp
    utilities/rotations.hpp
    utilities/strain_measures.hpp
    utilities/unified_logger.hpp
    utilities/user_classes.hpp
    utilities/user_function_loader.hpp
    umats/userumat.h
    umats/unified_umat_loader.hpp
    TOML_Reader/toml.hpp
    )

set(EXACONSTIT_SOURCES
    system_driver.cpp
    boundary_conditions/BCData.cpp
    boundary_conditions/BCManager.cpp
    fem_operators/mechanics_integrators.cpp
    fem_operators/mechanics_operator_ext.cpp
    fem_operators/mechanics_operator.cpp
    mfem_expt/partial_qspace.cpp
    mfem_expt/partial_qfunc.cpp
    models/mechanics_model.cpp
    models/mechanics_ecmech.cpp
    models/mechanics_umat.cpp
    models/mechanics_multi_model.cpp
    options/option_parser_v2.cpp
    options/option_boundary_conditions.cpp
    options/option_enum.cpp
    options/option_material.cpp
    options/option_mesh.cpp
    options/option_post_processing.cpp
    options/option_restart.cpp
    options/option_solvers.cpp
    options/option_time.cpp
    postprocessing/postprocessing_driver.cpp
    postprocessing/projection_class.cpp
    postprocessing/mechanics_lightup.cpp
    postprocessing/surface_quantities.cpp
    postprocessing/viz_point_map.cpp
    postprocessing/volume_statistics.cpp
    postprocessing/postprocessing_hdf5_file.cpp
    sim_state/simulation_state.cpp
    sim_state/exa_sidre_parmesh.cpp
    sim_state/restart_checkpoint.cpp
    solvers/mechanics_solver.cpp
    solvers/trust_region_solver.cpp
    utilities/mechanics_kernels.cpp
    utilities/unified_logger.cpp
    utilities/user_classes.cpp
    )

# Built-in UMAT (used when a UMAT material sets enable_dynamic_loading = false). It is an
# object library added directly to the executables below rather than part of
# exaconstit_static: nothing in ExaConstit calls it by name (it is found by symbol lookup at
# run time), so from a static library the linker would leave it out.
if (ENABLE_FORTRAN)
    add_library(exaconstit_builtin_umat OBJECT umats/umat.f)
else()
    add_library(exaconstit_builtin_umat OBJECT umats/umat.cxx)
endif()
set_target_properties(exaconstit_builtin_umat PROPERTIES POSITION_INDEPENDENT_CODE ON)


set(DYNAMIC_LOADING_LIBS)

# Windows uses kernel32 automatically
if(UNIX)
    list(APPEND DYNAMIC_LOADING_LIBS ${CMAKE_DL_LIBS}) 
endif()
#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(EXACONSTIT_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_DEPENDS
                             DEPENDS_ON  mfem ecmech snls RAJA camp mpi)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DEPENDS blt::cuda_runtime blt::cuda)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DEPENDS cuda cuda_runtime)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_DEPENDS blt::hip blt::hip_runtime)
endif()

if (SNLS_USE_RAJA_PORT_SUITE)
    list(APPEND EXACONSTIT_DEPENDS chai umpire fmt::fmt)
endif()

if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEPENDS caliper)
endif()

if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_DEPENDS axom)
    # Conduit relay HDF5 / MPI for the HDF5 post-processing output (Axom already brings them
    # with sidre; listed so the dependency is explicit)
    if(TARGET conduit::conduit_mpi)
        list(APPEND EXACONSTIT_DEPENDS conduit::conduit_mpi)
    endif()
endif()

if(UNIX AND NOT APPLE AND TARGET Threads::Threads)
    # Check if we determined explicit linking is needed
    if(EXACONSTIT_THREADS_EXPLICIT_LINK)
        list(APPEND EXACONSTIT_DEPENDS Threads::Threads)
    endif()
endif()

list(APPEND EXACONSTIT_DEPENDS ${DYNAMIC_LOADING_LIBS})

message("-- EXACONSTIT_DEPENDS: ${EXACONSTIT_DEPENDS}")

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------
set(EXACONSTIT_DEFINES HAVE_EXACONSTIT)
if(ENABLE_CALIPER)
    list(APPEND EXACONSTIT_DEFINES HAVE_CALIPER)
endif()
if(ENABLE_AXOM)
    list(APPEND EXACONSTIT_DEFINES EXACONSTIT_USE_AXOM)
endif()
#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(EXACONSTIT_INCLUDES ${HEADER_INCLUDE_DIR}
                        ${CMAKE_SOURCE_DIR}/src/)
#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------

blt_add_library(NAME    exaconstit_static
                        OUTPUT_NAME exaconstit
                        DEFINES     ${EXACONSTIT_DEFINES}
                        INCLUDES    ${EXACONSTIT_INCLUDES}
                        DEPENDS_ON  ${EXACONSTIT_DEPENDS}
                        HEADERS     ${EXACONSTIT_HEADERS}
                        SOURCES     ${EXACONSTIT_SOURCES}
                        SHARED      FALSE)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

set(EXACONSTIT_DRIVER)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DRIVER blt::cuda_runtime blt::cuda CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DRIVER blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND EXACONSTIT_DRIVER cuda cuda_runtime CUDA::cublas)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND EXACONSTIT_DRIVER openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND EXACONSTIT_DRIVER blt::hip blt::hip_runtime hipblas rocsparse rocrand)
endif()

blt_add_executable(NAME       mechanics
                   SOURCES    mechanics_driver.cpp
                   OUTPUT_DIR ${BINARY_DIR}
                   DEPENDS_ON exaconstit_static ${EXACONSTIT_DRIVER})
target_sources(mechanics PRIVATE $<TARGET_OBJECTS:exaconstit_builtin_umat>)
# User functions loaded from shared libraries (UMATs, [Materials.State_Vars] and
# [[PostProcessing.user_functions]] functions) call back into ExaConstit, e.g.
# SimulationState::GetQuadratureFunction, so the executable exports its symbols (-rdynamic).
# This also lets the built-in UMAT be found by its symbol name.
set_target_properties(mechanics PROPERTIES ENABLE_EXPORTS ON)

#------------------------------------------------------------------------------
# User function libraries (see cmake/ExaConstitUserLibrary.cmake)
#------------------------------------------------------------------------------
# ExaConstit::user_function carries what a user library needs to compile against the
# ExaConstit headers: ExaConstit's include directories and compile definitions together
# with those of every library it uses (MFEM, RAJA, ExaCMech, SNLS, Axom, ...), taken from
# exaconstit_static's interface. The library is not linked with ExaConstit; it resolves the
# ExaConstit functions it calls from the mechanics executable when loaded.
add_library(exaconstit_user_function INTERFACE)
target_include_directories(exaconstit_user_function SYSTEM INTERFACE
    $<TARGET_PROPERTY:exaconstit_static,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(exaconstit_user_function INTERFACE
    $<TARGET_PROPERTY:exaconstit_static,INTERFACE_COMPILE_DEFINITIONS>)
if(APPLE)
    # Linux allows undefined symbols in shared libraries by default; the macOS linker must be
    # told that they are resolved when the library is loaded
    target_link_options(exaconstit_user_function INTERFACE "LINKER:-undefined,dynamic_lookup")
endif()
set_target_properties(exaconstit_user_function PROPERTIES
                      EXACONSTIT_ENABLE_CUDA "${ENABLE_CUDA}"
                      EXACONSTIT_ENABLE_HIP "${ENABLE_HIP}"
                      EXACONSTIT_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                      EXACONSTIT_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_library(ExaConstit::user_function ALIAS exaconstit_user_function)
include(${PROJECT_SOURCE_DIR}/cmake/ExaConstitUserLibrary.cmake)

#------------------------------------------------------------------------------
# CMake package: find_package(ExaConstit) for building user function libraries
#------------------------------------------------------------------------------
# The include directories, definitions and flags of this build, written at generate time
# (they come from generator expressions). The package config removes the build-tree header
# directories and adds the installed ones.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _exaconstit_build_type)
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/ExaConstitBuildFlags.cmake CONTENT
"# Generated by ExaConstit's CMake: settings ExaConstit was built with
set(EXACONSTIT_BUILD_INCLUDE_DIRS \"$<JOIN:$<TARGET_PROPERTY:exaconstit_static,INTERFACE_INCLUDE_DIRECTORIES>,;>\")
set(EXACONSTIT_BUILD_COMPILE_DEFINITIONS \"$<JOIN:$<TARGET_PROPERTY:exaconstit_static,INTERFACE_COMPILE_DEFINITIONS>,;>\")
set(EXACONSTIT_BUILD_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_exaconstit_build_type}}\")
")

string(REPLACE "\"" "" _exaconstit_patch "${EXACONSTIT_VERSION_PATCH}")
set(EXACONSTIT_PACKAGE_VERSION
    "${EXACONSTIT_VERSION_MAJOR}.${EXACONSTIT_VERSION_MINOR}.${_exaconstit_patch}")
set(EXACONSTIT_BUILD_TREE_HEADER_DIRS "${PROJECT_SOURCE_DIR}/src;${HEADER_INCLUDE_DIR}")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${PROJECT_BINARY_DIR}/ExaConstitConfigVersion.cmake
                                 VERSION ${EXACONSTIT_PACKAGE_VERSION}
                                 COMPATIBILITY SameMinorVersion)

# Build tree: lets tests and examples use the package before installing
set(EXACONSTIT_CONFIG_HEADER_DIRS "${EXACONSTIT_BUILD_TREE_HEADER_DIRS}")
set(EXACONSTIT_CONFIG_USER_LIBRARY_MODULE "${PROJECT_SOURCE_DIR}/cmake/ExaConstitUserLibrary.cmake")
set(EXACONSTIT_CONFIG_FLAGS_FILE "${PROJECT_BINARY_DIR}/ExaConstitBuildFlags.cmake")
configure_file(${PROJECT_SOURCE_DIR}/cmake/ExaConstitConfig.cmake.in
               ${PROJECT_BINARY_DIR}/ExaConstitConfig.cmake @ONLY)

# Install tree
set(EXACONSTIT_CONFIG_HEADER_DIRS "\${_exaconstit_prefix}/include")
set(EXACONSTIT_CONFIG_USER_LIBRARY_MODULE "\${CMAKE_CURRENT_LIST_DIR}/ExaConstitUserLibrary.cmake")
set(EXACONSTIT_CONFIG_FLAGS_FILE "\${CMAKE_CURRENT_LIST_DIR}/ExaConstitBuildFlags.cmake")
configure_file(${PROJECT_SOURCE_DIR}/cmake/ExaConstitConfig.cmake.in
               ${PROJECT_BINARY_DIR}/install_config/ExaConstitConfig.cmake @ONLY)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------
# Headers keep their directory layout (sim_state/, options/, ...) so their includes work
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(FILES ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
              ${HEADER_INCLUDE_DIR}/exaconstit_user.hpp
        DESTINATION include)
install(TARGETS exaconstit_static DESTINATION lib)
install(TARGETS mechanics DESTINATION bin)
install(FILES ${PROJECT_BINARY_DIR}/install_config/ExaConstitConfig.cmake
              ${PROJECT_BINARY_DIR}/ExaConstitConfigVersion.cmake
              ${PROJECT_BINARY_DIR}/ExaConstitBuildFlags.cmake
              ${PROJECT_SOURCE_DIR}/cmake/ExaConstitUserLibrary.cmake
        DESTINATION lib/cmake/exaconstit)
