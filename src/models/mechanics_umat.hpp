#ifndef MECHANICS_UMAT
#define MECHANICS_UMAT

#include "models/mechanics_model.hpp"
#include "umats/unified_umat_loader.hpp"

#include "mfem.hpp"

#include <array>
#include <filesystem>

/**
 * @brief Abaqus-style UMAT material model, with an optional batched (all points at once) form
 *
 * @details Runs a user material subroutine (UMAT) for one material region. Everything
 * ExaConstit prepares for the UMAT and everything it does with the UMAT's results runs in
 * mfem::forall loops, so on GPU runs the data stays on the GPU:
 * 1. Kinematics: the deformation gradients at the beginning (F_beg) and end (F_end) of the step
 *    are computed exactly from the stored nodal positions (reference, beginning-of-step and
 *    current nodes) with the shared GradCalc kernel and reference Jacobians, so nothing is
 *    accumulated from step to step. With the B-bar integrator both get the F-bar treatment
 *    (exaconstit::kernel::FBarElement).
 * 2. Packing: the UMAT arguments are filled for every point in the Abaqus conventions: stress
 *    in (11, 22, 33, 12, 13, 23) order, DROT = exact rotation of F_incr = F_end F_beg^-1
 *    (polar decomposition), STRAN / DSTRAN = logarithmic (ln V, default) or Eulerian strain of
 *    F_beg / F_incr, STRESS and STRAN rotated by DROT unless rotate_stress = false,
 *    DFGRD0 = F_beg, DFGRD1 = F_end, COORDS = current point positions, CELENT = cube root of
 *    the element volume near the point; see the UMAT section of src/options.toml.
 * 3. The UMAT call, in one of two forms ([Materials.Model.UMAT] interface):
 *    - "standard": the Abaqus interface (UmatFunction), once per point, on the host (arrays are
 *      copied from the GPU on GPU runs). With thread_safe = true and an OpenMP build the points
 *      run in an OpenMP loop.
 *    - "batched": one call with all of the region's points on this rank (UmatBatchFunction in
 *      umats/userumat.h), with the arrays in the run's memory space (GPU arrays on GPU runs).
 * 4. Unpacking: the stress goes back to ExaConstit's order, and the tangent DDSDDE is reordered
 *    and turned into the rate form the integrators use (DDSDDE dt).
 *
 * UMAT libraries are loaded dynamically (or the built-in UMAT is used), with the persistent,
 * load-on-setup or lazy loading strategies.
 */
class AbaqusUmatModel : public ExaModel {
protected:
    // ---------------------------------------------------------------------------------------
    // UMAT function and how it is called
    // ---------------------------------------------------------------------------------------

    /** @brief Path to UMAT shared library (empty: built-in UMAT) */
    std::filesystem::path umat_library_path;

    /** @brief Loaded standard UMAT (interface = "standard") */
    UmatFunction umat_function;

    /** @brief Loaded batched UMAT (interface = "batched") */
    UmatBatchFunction umat_batch_function;

    /** @brief Loading strategy for the library */
    exaconstit::LoadStrategy load_strategy;

    /** @brief Whether the UMAT comes from a shared library (false: built-in UMAT) */
    bool use_dynamic_loading;

    /** @brief UMAT function name given in the options */
    const std::string umat_function_name;

    /** @brief Batched interface: one call for all points of the region */
    const bool m_batched;

    /** @brief Standard interface: the per-point calls may run in an OpenMP loop */
    const bool m_thread_safe;

    /** @brief Temperature passed to the UMAT (TEMP), from the material's temperature option */
    const double m_temperature;

    /** @brief STRAN / DSTRAN as logarithmic strains (true) or Eulerian strains (false) */
    const bool m_log_strain;

    /** @brief Rotate STRESS and STRAN by DROT before the call */
    const bool m_rotate_stress;

    // ---------------------------------------------------------------------------------------
    // Kinematics (region-sized, in the run's memory space)
    // ---------------------------------------------------------------------------------------

    /** @brief Quadrature points per element */
    int m_nqpts;

    /** @brief Nodes (dofs per component) per element */
    int m_nnodes;

    /** @brief Region elements on this rank */
    int m_nelems;

    /** @brief Reference Jacobians and volume weights have been computed (first ModelSetup) */
    bool m_reference_ready;

    /**
     * @brief Reference Jacobians d X / d xi of the region's elements, layout (3, 3, nqpts,
     * region elements), column-major (the layout GradCalc reads)
     */
    mfem::Vector m_ref_jacobians;

    /** @brief Quadrature weight times det(d X / d xi), one value per region point (F-bar) */
    mfem::Vector m_ref_wdetj;

    /** @brief Nodal coordinates of all elements of the rank, NATIVE dof order (scratch) */
    mfem::Vector m_elem_coords;

    /** @brief Nodal coordinates of the region's elements, layout (nnodes, 3, region elements) */
    mfem::Vector m_region_coords;

    /** @brief Deformation gradient at the beginning of the step, 9 values per region point */
    mfem::Vector m_def_grad_beg;

    /** @brief Deformation gradient at the end of the step (current iterate), 9 per point */
    mfem::Vector m_def_grad_end;

    // ---------------------------------------------------------------------------------------
    // UMAT arguments, point-major (point i's values start at i * size), Abaqus conventions.
    // STATEV and DDSDDE use the region's state_var_end and tangent_stiffness directly, and
    // DFGRD0 / DFGRD1 are m_def_grad_beg / m_def_grad_end.
    // ---------------------------------------------------------------------------------------

    mfem::Vector m_stress; ///< STRESS, 6 per point, Abaqus order
    mfem::Vector m_stran;  ///< STRAN, 6 per point
    mfem::Vector m_dstran; ///< DSTRAN, 6 per point
    mfem::Vector m_drot;   ///< DROT, 9 per point
    mfem::Vector m_coords; ///< COORDS, 3 per point
    mfem::Vector m_celent; ///< CELENT, 1 per point
    mfem::Vector m_pnewdt; ///< PNEWDT, 1 per point
    mfem::Vector m_temp;   ///< TEMP, 1 per point
    mfem::Vector m_dtemp;  ///< DTEMP, 1 per point
    mfem::Vector m_predef; ///< PREDEF, 1 per point (no predefined fields)
    mfem::Vector m_dpred;  ///< DPRED, 1 per point
    mfem::Vector m_sse;    ///< SSE, 1 per point
    mfem::Vector m_spd;    ///< SPD, 1 per point
    mfem::Vector m_scd;    ///< SCD, 1 per point
    mfem::Vector m_rpl;    ///< RPL, 1 per point
    mfem::Vector m_drpldt; ///< DRPLDT, 1 per point
    mfem::Vector m_ddsdt;  ///< DDSDDT, 6 per point
    mfem::Vector m_drplde; ///< DRPLDE, 6 per point
    mfem::Vector m_props;  ///< PROPS, the region's material properties
    mfem::Array<int> m_noel; ///< NOEL: the point's global element number (1-based)
    mfem::Array<int> m_npt;  ///< NPT: the point's number within its element (1-based)
    std::array<char, 81> m_cmname; ///< CMNAME: material name, blank-padded to 80 characters

public:
    /**
     * @brief Constructor
     *
     * @param region Region identifier (0-based)
     * @param n_state_vars Number of state variables
     * @param sim_state Simulation state
     * @param umat_library_path_ Path to UMAT shared library (empty for the built-in UMAT)
     * @param load_strategy_ Strategy for loading/unloading the library
     * @param umat_function_name_ UMAT function name to load
     * @param batched Use the batched interface (UmatBatchFunction) instead of the standard one
     * @param thread_safe Standard interface: run the per-point calls in an OpenMP loop
     * @param temperature Temperature passed to the UMAT (K)
     * @param material_name Material name passed to the UMAT as CMNAME
     * @param global_elem_offset Global number of this rank's first element, for NOEL
     *        (ParMesh::GetGlobalElementNum(0); computed by the caller, since it is collective)
     * @param log_strain STRAN / DSTRAN as logarithmic (true) or Eulerian (false) strains
     * @param rotate_stress Rotate STRESS and STRAN by DROT before the call
     */
    AbaqusUmatModel(
        const int region,
        int n_state_vars,
        std::shared_ptr<SimulationState> sim_state,
        const std::filesystem::path& umat_library_path_ = "",
        const exaconstit::LoadStrategy& load_strategy_ = exaconstit::LoadStrategy::PERSISTENT,
        const std::string umat_function_name_ = "",
        const bool batched = false,
        const bool thread_safe = false,
        const double temperature = 298.0,
        const std::string& material_name = "",
        const long long global_elem_offset = 0,
        const bool log_strain = true,
        const bool rotate_stress = true);

    /**
     * @brief Destructor - unloads the library if it is not persistent
     */
    virtual ~AbaqusUmatModel();

    /**
     * @brief The region's beginning-of-step deformation gradient quadrature function
     *
     * @details Holds the converged deformation gradient after each step (written by
     * UpdateModelVars) for output and restart files; the kinematics are recomputed from the
     * nodes every step and don't read it.
     */
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> GetDefGrad0();

    /**
     * @brief Store the converged end-of-step deformation gradient in "def_grad_beg"
     */
    virtual void UpdateModelVars() override;

    /**
     * @brief Run the UMAT for every point of the region
     *
     * @param nqpts Number of quadrature points per element
     * @param nelems Number of elements on this rank (all regions)
     * @param space_dim Spatial dimension (3)
     * @param nnodes Number of nodes per element
     * @param jacobian Current Jacobians d x / d xi, layout (3, 3, nqpts, nelems)
     * @param loc_grad Reference shape derivatives, layout (nnodes, 3, nqpts), NATIVE dof order
     * @param vel Element nodal velocities (unused: the kinematics come from the nodal positions)
     *
     * @details Kinematics, packing, UMAT call(s) and unpacking as described in the class
     * documentation; fills the region's cauchy_stress_end, state_var_end and tangent_stiffness
     * and copies the stress and tangent into the global quadrature functions. Throws if any
     * point returns PNEWDT < 1.
     */
    virtual void ModelSetup(const int nqpts,
                            const int nelems,
                            const int space_dim,
                            const int nnodes,
                            const mfem::Vector& jacobian,
                            const mfem::Vector& loc_grad,
                            const mfem::Vector& vel) override;

    /**
     * @brief Configure dynamic loading of a UMAT library
     *
     * @param library_path Path to the UMAT shared library
     * @param strategy Loading strategy to use
     * @return True if library setup succeeded, false otherwise
     */
    bool SetUmatLibrary(const std::filesystem::path& library_path,
                        exaconstit::LoadStrategy strategy = exaconstit::LoadStrategy::PERSISTENT);

    /**
     * @brief Get the current UMAT library path
     */
    const std::filesystem::path& GetUmatLibraryPath() const {
        return umat_library_path;
    }

    /**
     * @brief Check if using dynamic loading
     */
    bool UsingDynamicLoading() const {
        return use_dynamic_loading;
    }

    /**
     * @brief Force reload of the current UMAT library
     * @return True if reload succeeded, false otherwise
     */
    bool ReloadUmatLibrary();

    /**
     * @brief Load the UMAT shared library and get the UMAT function (standard or batched)
     * @return True if loading succeeded, false otherwise
     */
    bool LoadUmatLibrary();

    /**
     * @brief Unload the currently loaded UMAT library and reset the function pointer
     */
    void UnloadUmatLibrary();

protected:
    /** @brief Whether the UMAT function for the selected interface is loaded */
    bool UmatLoaded() const {
        return m_batched ? (umat_batch_function != nullptr) : (umat_function != nullptr);
    }

    /**
     * @brief Reference Jacobians and volume weights of the region's points (first ModelSetup)
     *
     * @param loc_grad Reference shape derivatives from the operator (only checked against the
     *        quadrature rule size here)
     *
     * @details MFEM's QuadratureInterpolator turns the reference nodes into d X / d xi at the
     * points (NATIVE dof order, the order of loc_grad), then the region's elements are copied out.
     * Done once: the reference configuration never changes, and computing it from the stored
     * reference nodes is also correct after a restart.
     */
    void SetupReferenceGeometry(const mfem::Vector& loc_grad);

    /**
     * @brief Deformation gradient F = d x / d X at the region's points for given nodal positions
     *
     * @param nodes Nodal positions (beginning-of-step or current), a grid function on the mesh
     *        space
     * @param loc_grad Reference shape derivatives (NATIVE dof order)
     * @param[out] def_grad 9 values per region point, column-major
     *
     * @details Element coordinates are restricted from the grid function, the region's elements
     * copied out, and GradCalc applied with the reference Jacobians, so F is exact for the given
     * nodes. With the B-bar integrator, FBarElement then gives every point of an element the
     * element's volume ratio V / V_0.
     */
    void ComputeDefGrad(const mfem::ParGridFunction& nodes,
                        const mfem::Vector& loc_grad,
                        mfem::Vector& def_grad);

    /**
     * @brief Fill the UMAT input arrays for every region point (device kernel)
     *
     * @param jacobian Current Jacobians (rank elements), for CELENT
     * @param local2global Region element -> rank element map (device pointer), or nullptr
     */
    void PackUmatInputs(const mfem::Vector& jacobian, const int* local2global);

    /**
     * @brief Shared UMAT scalars of the current increment: TIME (both entries at the start of
     * the increment), DTIME, KSTEP (1) and KINC (1-based time step number)
     */
    void IncrementInfo(double time[2], double& dtime, int& kstep, int& kinc);

    /**
     * @brief Call the standard UMAT once per point on the host
     */
    void CallUmatPerPoint();

    /**
     * @brief Call the batched UMAT once with all points, in the run's memory space
     */
    void CallUmatBatched();

    /**
     * @brief Copy the UMAT results back into ExaConstit's conventions (device kernel), and
     * stop if a point asked for a smaller time step
     */
    void UnpackUmatOutputs();
};

#endif
