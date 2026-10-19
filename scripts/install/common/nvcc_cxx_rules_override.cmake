# Passed as CMAKE_USER_MAKE_RULES_OVERRIDE_CXX when nvcc is used as the CXX
# compiler (MFEM and ExaConstit GPU builds).
#
# With NVCC_CCBIN/-ccbin pointing at g++, CMake identifies nvcc as a GNU CXX
# compiler and emits gcc-style "-Wl,..." link options (e.g. rpaths for MPI and
# CUDA libraries outside the default linker search path). nvcc rejects those,
# so switch them to nvcc's "-Xlinker=..." form. No-op for any other compiler.
if (CMAKE_CXX_COMPILER MATCHES "nvcc$")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker=-rpath,")
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG "-Xlinker=-rpath-link,")
  set(CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker=-rpath,")
  set(CMAKE_EXECUTABLE_RPATH_LINK_CXX_FLAG "-Xlinker=-rpath-link,")
  set(CMAKE_CXX_LINKER_WRAPPER_FLAG "-Xlinker=")
  set(CMAKE_CXX_LINKER_WRAPPER_FLAG_SEP ",")
  # nvcc has no -std=gnu++NN; the compiler checks inside project() use the
  # extension flags since CMAKE_CXX_EXTENSIONS defaults to ON there
  foreach(std 11 14 17 20)
    set(CMAKE_CXX${std}_EXTENSION_COMPILE_OPTION "-std=c++${std}")
  endforeach()
  # Position-independent code flags must be forwarded to the host compiler
  set(CMAKE_CXX_COMPILE_OPTIONS_PIC "-Xcompiler=-fPIC")
  set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-Xcompiler=-fPIE")
  set(CMAKE_CXX_LINK_OPTIONS_PIE "-Xcompiler=-fPIE" "-Xcompiler=-pie")
  set(CMAKE_CXX_LINK_OPTIONS_NO_PIE "-Xcompiler=-no-pie")
  # Anything else gcc-specific (e.g. -Wall that BLT or googletest add for GNU)
  # is passed through to the host compiler instead of being an nvcc error.
  # These are CMake's default CXX rules with that one option added; they must
  # be set in full since this file is read before CMake fills in the defaults.
  set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> -forward-unknown-to-host-compiler <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
  set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> -forward-unknown-to-host-compiler <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY
    "<CMAKE_CXX_COMPILER> -forward-unknown-to-host-compiler <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()
