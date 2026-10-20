#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(EXACONSTIT_HEADERS
    ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
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
    sim_state/simulation_state.hpp
    solvers/mechanics_solver.hpp
    solvers/trust_region_solver.hpp
    utilities/dynamic_function_loader.hpp
    utilities/mechanics_kernels.hpp
    utilities/mechanics_log.hpp
    utilities/assembly_ops.hpp
    utilities/rotations.hpp
    utilities/strain_measures.hpp
    utilities/unified_logger.hpp
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
    options/option_solvers.cpp
    options/option_time.cpp
    postprocessing/postprocessing_driver.cpp
    postprocessing/projection_class.cpp
    postprocessing/mechanics_lightup.cpp
    sim_state/simulation_state.cpp
    solvers/mechanics_solver.cpp
    solvers/trust_region_solver.cpp
    utilities/mechanics_kernels.cpp
    utilities/unified_logger.cpp
    )

if (ENABLE_FORTRAN)
    list(APPEND EXACONSTIT_SOURCES ./umats/umat.f)
else()
    list(APPEND EXACONSTIT_SOURCES ./umats/umat.cxx)
endif()


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

set(EXACONSTIT_INSTALLED_HEADERS ${EXACONSTIT_HEADERS})
install(FILES ${EXACONSTIT_INSTALLED_HEADERS} DESTINATION include)
install(TARGETS exaconstit_static DESTINATION lib)
install(TARGETS mechanics DESTINATION bin)
