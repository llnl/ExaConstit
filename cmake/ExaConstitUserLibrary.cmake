#------------------------------------------------------------------------------
# exaconstit_add_user_library
#
# Builds a shared library of ExaConstit user functions (state variable register /
# initialize functions, post-processing user functions, UMATs) with everything it
# needs to compile against ExaConstit's headers: the include directories, compile
# definitions and flags ExaConstit and its libraries (MFEM, RAJA, ExaCMech, SNLS,
# Axom, ...) were built with, C++17, and position independent code.
#
#   exaconstit_add_user_library(NAME my_functions SOURCES my_functions.cpp)
#
# The library does not link ExaConstit itself: the functions are loaded by the
# `mechanics` executable at run time and use the ExaConstit code already in it
# (on macOS the link option -undefined dynamic_lookup allows that; Linux allows it
# by default). Point library_path in the options file at the built library.
#
# Used both inside the ExaConstit build (tests) and from an installed ExaConstit
# through find_package(ExaConstit); in both cases the settings come from the
# ExaConstit::user_function target.
#------------------------------------------------------------------------------
function(exaconstit_add_user_library)
    set(options)
    set(single_value_args NAME)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(arg "${options}" "${single_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT arg_NAME)
        message(FATAL_ERROR "exaconstit_add_user_library requires NAME")
    endif()
    if(NOT arg_SOURCES)
        message(FATAL_ERROR "exaconstit_add_user_library requires SOURCES")
    endif()
    if(NOT TARGET ExaConstit::user_function)
        message(FATAL_ERROR "exaconstit_add_user_library: ExaConstit::user_function is not "
                            "defined; call find_package(ExaConstit) first")
    endif()

    add_library(${arg_NAME} SHARED ${arg_SOURCES})
    target_link_libraries(${arg_NAME} PRIVATE ExaConstit::user_function)
    set_target_properties(${arg_NAME} PROPERTIES
                          CXX_STANDARD 17
                          CXX_STANDARD_REQUIRED ON
                          CXX_EXTENSIONS OFF
                          POSITION_INDEPENDENT_CODE ON)

    # GPU builds of ExaConstit: user kernels (mfem::forall) must be compiled for the same device
    # (the settings are stored on the ExaConstit::user_function target)
    get_target_property(_exaconstit_cuda ExaConstit::user_function EXACONSTIT_ENABLE_CUDA)
    get_target_property(_exaconstit_hip ExaConstit::user_function EXACONSTIT_ENABLE_HIP)
    if(_exaconstit_cuda)
        get_target_property(_exaconstit_arch ExaConstit::user_function EXACONSTIT_CUDA_ARCHITECTURES)
        enable_language(CUDA)
        set_source_files_properties(${arg_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_target_properties(${arg_NAME} PROPERTIES CUDA_ARCHITECTURES "${_exaconstit_arch}")
        target_compile_options(${arg_NAME} PRIVATE
                               $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    elseif(_exaconstit_hip)
        get_target_property(_exaconstit_arch ExaConstit::user_function EXACONSTIT_HIP_ARCHITECTURES)
        enable_language(HIP)
        set_source_files_properties(${arg_SOURCES} PROPERTIES LANGUAGE HIP)
        set_target_properties(${arg_NAME} PROPERTIES HIP_ARCHITECTURES "${_exaconstit_arch}")
    endif()
endfunction()
