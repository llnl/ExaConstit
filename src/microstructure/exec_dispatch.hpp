#pragma once

#include "mfem.hpp"
#include "RAJA/config.hpp"

#include "axom/config.hpp"
#include "axom/core/execution/execution_space.hpp"
#include "axom/core/execution/runtime_policy.hpp"
#include "axom/core/memory_management.hpp"

#include <stdexcept>
#include <string>
#include <utility>

/**
 * @file exec_dispatch.hpp
 * @brief Selects where the microstructure kernels run, from the user-facing `exec` option
 * ("cpu", "openmp", "gpu").
 *
 * **Design.** The generator follows ExaConstit itself: the backend is an mfem::Device, configured
 * once at start-up exactly like the mechanics driver does for its RTModel option
 * (src/mechanics_driver.cpp, device configuration phase). All ordinary kernels are written with
 * mfem::forall and mfem::Vector / mfem::Array, and so automatically run on whatever backend the
 * device selects. Axom is used only for its bounding volume hierarchy (BVH: a tree of nested
 * bounding boxes that lets a search skip seeds that are provably too far away). Axom's BVH code is
 * templated on an Axom *execution space*, a small tag type (axom::SEQ_EXEC, axom::OMP_EXEC,
 * axom::CUDA_EXEC<256>, axom::HIP_EXEC<256>) that says where a loop runs. DispatchAxomExec picks
 * the execution space that matches the configured mfem::Device, so BVH work follows the same
 * backend as everything else.
 *
 * Terms used in this file:
 * - *Backend*: where loops run: one host thread (cpu), all host threads through RAJA's OpenMP
 *   support (MFEM device string "raja-omp"), or a GPU through RAJA ("raja-cuda" / "raja-hip").
 * - *Allocator id*: Umpire's integer handle for a memory pool (host memory, GPU device memory).
 *   Arrays a BVH kernel touches must live in memory that its execution space can reach.
 * - *Translation unit* (TU): one .cpp file after its #includes are expanded; each is compiled
 *   separately (some by the host compiler, some by nvcc/hipcc in GPU builds) and the linker joins
 *   them.
 *
 * **Same answer in every translation unit.** An `inline` function must have the same body in
 * every TU that uses it (the C++ "one definition rule", ODR); otherwise the linker keeps one
 * arbitrary copy. Axom defines AXOM_RUNTIME_POLICY_USE_CUDA/HIP, and with them the GPU execution
 * spaces, only while a file is compiled by nvcc/hipcc (axom/core/execution/runtime_policy.hpp:
 * 51-56, execution_space.hpp:117-125). Therefore:
 * - EXEC_*_AVAILABLE, IsExecAvailable, ConfigureMfemDevice and AxomAllocatorIdForCurrentDevice
 *   depend only on macros from generated configuration headers (mfem/config/config.hpp,
 *   RAJA/config.hpp, axom/config.hpp), which are identical in every TU.
 * - DispatchAxomExec must name CUDA_EXEC/HIP_EXEC, which only exist in device-compiled TUs, so its
 *   GPU branches stay guarded by AXOM_RUNTIME_POLICY_USE_*. It is a template that must only be
 *   instantiated with lambdas (see its documentation).
 */

namespace exaconstit::microstructure {

// -------------------------------------------------------------------------------------------------
// Build-wide availability of each backend. Decided only from configuration headers, so every TU
// agrees.
// -------------------------------------------------------------------------------------------------

#if defined(MFEM_USE_RAJA) && defined(MFEM_USE_OPENMP)
/** @brief True when MFEM was built with RAJA and OpenMP (exec = "openmp", device "raja-omp"). */
inline constexpr bool EXEC_OPENMP_AVAILABLE = true;
#else
inline constexpr bool EXEC_OPENMP_AVAILABLE = false;
#endif

#if (defined(MFEM_USE_CUDA) && defined(AXOM_USE_CUDA)) || \
    (defined(MFEM_USE_HIP) && defined(AXOM_USE_HIP))
/**
 * @brief True when both MFEM and Axom were built for the same GPU platform (CUDA or HIP), so the
 * mfem::forall kernels and the Axom BVH can both run on the GPU (exec = "gpu").
 */
inline constexpr bool EXEC_GPU_AVAILABLE = true;
#else
inline constexpr bool EXEC_GPU_AVAILABLE = false;
#endif

/**
 * @brief Whether this build can run the `exec` option value `exec`.
 *
 * "cpu" is always available; "openmp" needs EXEC_OPENMP_AVAILABLE and "gpu" needs
 * EXEC_GPU_AVAILABLE. Any other string returns false. Host only; not collective (every rank gets
 * the same answer); never throws. Used by MicrostructureOptions::validate().
 */
inline bool IsExecAvailable(const std::string& exec) {
    if (exec == "cpu") {
        return true;
    }
    if (exec == "openmp") {
        return EXEC_OPENMP_AVAILABLE;
    }
    if (exec == "gpu") {
        return EXEC_GPU_AVAILABLE;
    }
    return false;
}

/**
 * @brief Configures the MFEM device for `exec`, exactly as src/mechanics_driver.cpp does for
 * ExaConstit's RTModel option.
 *
 * Mapping: "cpu" -> "cpu", "openmp" -> "raja-omp", "gpu" -> "raja-cuda" (RAJA built with CUDA)
 * or "raja-hip" (RAJA built with HIP). Memory types: HOST_UMPIRE / DEVICE_UMPIRE when MFEM uses
 * Umpire, otherwise HOST_64 / DEVICE (mechanics_driver.cpp chooses the same pair for every
 * RTModel; the structure is mirrored so the two stay easy to compare).
 *
 * Host only. Call once, early, on every rank (mfem::Device is a process-wide singleton and MFEM
 * allows configuring it only once); not collective in the MPI sense.
 *
 * @param device the application's mfem::Device object (normally a local in main()).
 * @param exec   "cpu", "openmp" or "gpu".
 * @throws std::invalid_argument if IsExecAvailable(exec) is false. (The mechanics driver silently
 *         falls back to "cpu" for an unavailable GPU; the generator refuses instead, so a user
 *         never gets a CPU run when asking for a GPU one.)
 */
inline void ConfigureMfemDevice(mfem::Device& device, const std::string& exec) {
    if (!IsExecAvailable(exec)) {
        throw std::invalid_argument("exec = \"" + exec +
                                    "\" is unknown or not available in this build "
                                    "(expected \"cpu\", \"openmp\" or \"gpu\")");
    }

    // Step 1: the MFEM device string, as in mechanics_driver.cpp
    std::string device_config = "cpu";
    if (exec == "openmp") {
        device_config = "raja-omp";
    } else if (exec == "gpu") {
#if defined(RAJA_ENABLE_CUDA)
        device_config = "raja-cuda";
#elif defined(RAJA_ENABLE_HIP)
        device_config = "raja-hip";
#else
        throw std::invalid_argument(
            "exec = \"gpu\": MFEM/Axom have GPU support but RAJA was built without CUDA or HIP");
#endif
    }

    // Step 2: memory types (where MFEM allocates host and device arrays)
    if (exec == "gpu") {
#if defined(MFEM_USE_UMPIRE)
        device.SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);
#else
        device.SetMemoryTypes(mfem::MemoryType::HOST_64, mfem::MemoryType::DEVICE);
#endif
    } else {
#if defined(MFEM_USE_UMPIRE)
        device.SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);
#else
        device.SetMemoryTypes(mfem::MemoryType::HOST_64, mfem::MemoryType::DEVICE);
#endif
    }

    // Step 3: select the backend
    device.Configure(device_config);
}

namespace detail {

/** @brief True if the configured mfem::Device runs kernels on a CUDA GPU (device.hpp:91). */
inline bool MfemDeviceIsCuda() {
    return mfem::Device::Allows(mfem::Backend::CUDA_MASK);
}

/** @brief True if the configured mfem::Device runs kernels on a HIP GPU (device.hpp:93). */
inline bool MfemDeviceIsHip() {
    return mfem::Device::Allows(mfem::Backend::HIP_MASK);
}

/** @brief True if the configured mfem::Device uses OpenMP threads, e.g. "raja-omp"
 * (device.hpp:95). */
inline bool MfemDeviceIsOpenMP() {
    return mfem::Device::Allows(mfem::Backend::OMP_MASK);
}

} // namespace detail

/**
 * @brief Calls `f(ExecSpace{})` with the Axom execution space that matches the currently
 * configured mfem::Device. For BVH (axom::spin) code only; everything else uses mfem::forall.
 *
 * Mapping (checked with mfem::Device::Allows and the Backend masks of mfem/general/device.hpp):
 * - CUDA backend   -> axom::CUDA_EXEC<256> (256 GPU threads per block);
 * - HIP backend    -> axom::HIP_EXEC<256>;
 * - OpenMP backend -> axom::OMP_EXEC if Axom has OpenMP, otherwise axom::SEQ_EXEC (correct, just
 *                     serial for the BVH part);
 * - otherwise      -> axom::SEQ_EXEC.
 *
 * Typical use:
 * @code
 *   DispatchAxomExec([&](auto exec) {
 *       using E = decltype(exec);
 *       BuildSeedIndex<E>(...);
 *   });
 * @endcode
 *
 * Always pass a lambda, or a functor type local to one .cpp file. The body of
 * DispatchAxomExec<F> depends on the AXOM_RUNTIME_POLICY_USE_* macros, which differ between
 * host- and device-compiled files; a named functor type instantiated in both kinds of file would
 * give two different definitions of the same function (an ODR violation, see the file comment).
 * A lambda has a distinct type in every file, so its instantiation can never be shared.
 *
 * Runs on the host (the kernels `f` launches may run on the device). Not collective by itself.
 *
 * @param f callable invoked exactly once.
 * @throws std::invalid_argument if the device is a GPU but this file was not compiled as CUDA/HIP
 *         (the GPU execution spaces do not exist here). Call BVH code with a GPU device only from
 *         sources that BLT compiles as device code.
 */
template <typename Func>
void DispatchAxomExec(Func&& f) {
    // GPU backends: only possible in device-compiled TUs
    if (detail::MfemDeviceIsCuda()) {
#if defined(AXOM_RUNTIME_POLICY_USE_CUDA)
        std::forward<Func>(f)(axom::CUDA_EXEC<256>{});
        return;
#else
        throw std::invalid_argument("DispatchAxomExec: the MFEM device is CUDA, but this file is "
                                    "not compiled as CUDA (no axom::CUDA_EXEC here)");
#endif
    }
    if (detail::MfemDeviceIsHip()) {
#if defined(AXOM_RUNTIME_POLICY_USE_HIP)
        std::forward<Func>(f)(axom::HIP_EXEC<256>{});
        return;
#else
        throw std::invalid_argument("DispatchAxomExec: the MFEM device is HIP, but this file is "
                                    "not compiled as HIP (no axom::HIP_EXEC here)");
#endif
    }

    // OpenMP backend: use Axom's OpenMP space when Axom has one, else fall through to serial
#if defined(AXOM_RUNTIME_POLICY_USE_OPENMP)
    if (detail::MfemDeviceIsOpenMP()) {
        std::forward<Func>(f)(axom::OMP_EXEC{});
        return;
    }
#endif

    // Host serial (also the fallback for an OpenMP device when Axom lacks OpenMP)
    std::forward<Func>(f)(axom::SEQ_EXEC{});
}

/**
 * @brief Umpire allocator id for arrays used inside Axom BVH kernels under the currently
 * configured mfem::Device.
 *
 * Host backends (cpu, OpenMP): Axom's host allocator (the Umpire "Host" resource, or Axom's
 * default allocator without Umpire; seq_exec.hpp:65-71, omp_exec.hpp:56-62 use the same).
 * GPU backends: the Umpire "Device" resource, i.e. GPU-only memory, not unified memory, the same
 * call CUDA_EXEC/HIP_EXEC::allocatorID() make (memory_management.hpp:102, cuda_exec.hpp:65-67).
 * Written only with TU-independent macros (see the file comment); axom::policyToDefaultAllocatorID
 * is not used because its body depends on the compiler.
 *
 * Host only. The GPU branch calls into Umpire, which initializes itself on first use.
 *
 * @return the allocator id, or axom::INVALID_ALLOCATOR_ID if the device is a GPU but Axom was
 *         built without Umpire (cannot happen when EXEC_GPU_AVAILABLE is true and Axom's GPU
 *         support is enabled, since Axom requires Umpire for GPU execution).
 */
inline int AxomAllocatorIdForCurrentDevice() {
    const bool device_is_gpu = detail::MfemDeviceIsCuda() || detail::MfemDeviceIsHip();
    if (!device_is_gpu) {
        return axom::execution_space<axom::SEQ_EXEC>::allocatorID();
    }
#if defined(AXOM_USE_UMPIRE)
    return axom::getUmpireResourceAllocatorID(umpire::resource::MemoryResourceType::Device);
#else
    return axom::INVALID_ALLOCATOR_ID;
#endif
}

} // namespace exaconstit::microstructure
