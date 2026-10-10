#ifndef MECHANICS_KERNELS
#define MECHANICS_KERNELS

#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

/**
 * @brief ExaConstit computational kernels for finite element operations.
 *
 * This namespace contains high-performance computational kernels used throughout
 * ExaConstit for finite element operations, particularly gradient calculations
 * and field transformations. The kernels are designed to work with both full
 * and partial element sets, supporting multi-material simulations with optimal
 * performance.
 *
 * @ingroup ExaConstit_utilities
 */
namespace exaconstit {
namespace kernel {

/**
 * @brief Main gradient calculation function with partial element mapping support.
 *
 * @param nqpts Number of quadrature points per element
 * @param nelems Number of local elements to process in the partial set
 * @param global_nelems Total number of elements in global arrays (for input data sizing)
 * @param nnodes Number of nodes per element (typically 8 for hexahedral elements)
 * @param jacobian_data Global jacobian data array (sized for global_nelems)
 * @param loc_grad_data Global local gradient data array (shape function derivatives)
 * @param field_data Global field data array (velocity or displacement field, sized for
 * global_nelems)
 * @param field_grad_array Local output array for computed gradients (sized for nelems)
 * @param local2global Optional mapping from local to global element indices
 *
 * This function computes field gradients (typically velocity gradients) at quadrature
 * points for a subset of mesh elements. It supports both full mesh processing and
 * partial element processing for multi-material simulations.
 *
 * The function performs the fundamental finite element operation:
 * ∇u = ∑(N_i,α * u_i) where N_i,α are shape function derivatives and u_i are nodal values.
 *
 * Key features:
 * - RAJA-based implementation for performance portability (CPU/GPU)
 * - Support for partial element processing via local2global mapping
 * - Efficient memory layout optimized for vectorization
 * - Automatic handling of Jacobian inverse computation
 * - Compatible with MFEM's FORALL construct for device execution
 *
 * The computation involves:
 * 1. Mapping local element indices to global indices (if partial processing)
 * 2. Computing Jacobian inverse at each quadrature point
 * 3. Transforming shape function derivatives from reference to physical space
 * 4. Computing field gradients using chain rule: ∇u = (∂N/∂ξ)(∂ξ/∂x)u
 *
 * @note The jacobian_data and loc_grad_data are sized for global elements,
 *       while field_grad_array is sized for local elements only.
 * @note When local2global is nullptr, assumes nelems == global_nelems (full processing).
 * @note All arrays must be properly sized and allocated before calling this function.
 */
void GradCalc(const int nqpts,
              const int nelems,
              const int global_nelems,
              const int nnodes,
              const double* jacobian_data,
              const double* loc_grad_data,
              const double* field_data,
              double* field_grad_array,
              const int* const local2global = nullptr);

/**
 * @brief Backward compatibility overload - assumes full element processing.
 *
 * @param nqpts Number of quadrature points per element
 * @param nelems Number of elements to process
 * @param nnodes Number of nodes per element
 * @param jacobian_data Jacobian data array
 * @param loc_grad_data Local gradient data array (shape function derivatives)
 * @param field_data Field data array (velocity or displacement field)
 * @param field_grad_array Output gradient array
 *
 * This overload provides backward compatibility for code that processes all
 * elements in the mesh without partial element mapping. It internally calls
 * the main GradCalc function with local2global = nullptr.
 *
 * This is equivalent to calling the main function with:
 * - global_nelems = nelems
 * - local2global = nullptr
 *
 * @deprecated Use the full signature with explicit global_nelems for clarity
 *             and better support of partial element processing.
 */
inline void GradCalc(const int nqpts,
                     const int nelems,
                     const int nnodes,
                     const double* jacobian_data,
                     const double* loc_grad_data,
                     const double* field_data,
                     double* field_grad_array) {
    // Call the full version with no partial mapping (backward compatibility)
    GradCalc(nqpts,
             nelems,
             nelems,
             nnodes,
             jacobian_data,
             loc_grad_data,
             field_data,
             field_grad_array,
             nullptr);
}

/**
 * @brief Compute volume-averaged tensor values from quadrature function data.
 *
 * @tparam vol_avg Boolean template parameter controlling averaging behavior
 * @param fes Parallel finite element space defining the mesh and element structure
 * @param qf Quadrature function containing the tensor data at quadrature points
 * @param tensor Output vector for the volume-averaged tensor components
 * @param size Number of tensor components per quadrature point
 * @param class_device Runtime model for device execution policy
 *
 * This template function computes volume-averaged values of tensor quantities
 * stored at quadrature points. It supports both simple averaging and proper
 * volume-weighted averaging depending on the template parameter.
 *
 * The volume averaging computation follows:
 * <T> = (∫ T(x) dV) / (∫ dV) = (∑ T_qp * |J_qp| * w_qp) / (∑ |J_qp| * w_qp)
 *
 * where:
 * - T(x) is the tensor field to be averaged
 * - T_qp are the tensor values at quadrature points
 * - |J_qp| are the Jacobian determinants at quadrature points
 * - w_qp are the quadrature weights
 *
 * Algorithm steps:
 * 1. Set up RAJA views for efficient memory access patterns
 * 2. Loop over all elements and quadrature points
 * 3. Accumulate weighted tensor values and total volume
 * 4. Normalize by total volume if vol_avg is true
 *
 * Template behavior:
 * - vol_avg = true: Performs proper volume averaging (tensor /= total_volume)
 * - vol_avg = false: Returns volume-weighted sum without normalization
 *
 * This function is essential for:
 * - Computing homogenized material properties
 * - Extracting representative volume element (RVE) responses
 * - Postprocessing stress and strain fields
 * - Volume averaging for multiscale analysis
 *
 * @note The tensor vector is resized automatically to match the size parameter.
 * @note RAJA views are used for performance portability across CPU/GPU.
 * @note MPI parallelization requires additional reduction across processes.
 *
 * @ingroup ExaConstit_utilities_kernels
 */
template <bool vol_avg>
void ComputeVolAvgTensor(const mfem::ParFiniteElementSpace* fes,
                         const mfem::QuadratureFunction* qf,
                         mfem::Vector& tensor,
                         int size,
                         RTModel& class_device) {
    mfem::Mesh* mesh = fes->GetMesh();
    const mfem::FiniteElement& el = *fes->GetFE(0);
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    ;

    const int nqpts = ir->GetNPoints();
    const int nelems = fes->GetNE();
    const int npts = nqpts * nelems;

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    double el_vol = 0.0;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{nqpts, nelems}}, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> wts_view(wts.ReadWrite(),
                                                                         layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> j_view(geom->detJ.Read(),
                                                                             layout_geom);

    RAJA::RangeSegment default_range(0, npts);

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = qf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int i_npts) {
                const double* val = &(qf_data[i_npts * size]);
                seq_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = seq_sum.get();
            el_vol = vol_sum.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = qf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i_npts) {
                const double* val = &(qf_data[i_npts * size]);
                omp_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = omp_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = qf->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> gpu_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i_npts) {
                const double* val = &(qf_data[i_npts * size]);
                gpu_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = gpu_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(
        data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (vol_avg) {
        double temp = el_vol;

        // Here we find what el_vol should be equal to
        MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = 1.0 / el_vol;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
}

/**
 * @brief Compute filtered volume-averaged tensor values from full QuadratureFunction.
 *
 * @tparam vol_avg Boolean template parameter controlling averaging behavior
 * @param fes Parallel finite element space defining the mesh and element structure
 * @param qf Quadrature function containing the tensor data at quadrature points
 * @param filter Boolean array indicating which quadrature points to include
 * @param tensor Output vector for the volume-averaged tensor components
 * @param size Number of tensor components per quadrature point
 * @param class_device Runtime model for device execution policy
 * @return Total volume of the filtered region
 *
 * This template function provides filtering capability for full mesh
 * QuadratureFunction data. It processes the entire mesh but includes only
 * those quadrature points where the filter condition is satisfied.
 *
 * This function bridges the gap between:
 * - Full mesh processing (ComputeVolAvgTensor)
 * - Partial mesh processing (ComputeVolAvgTensorFromPartial)
 * - Filtered partial processing (ComputeVolAvgTensorFilterFromPartial)
 *
 * Use cases include:
 * - Legacy code integration with filtering requirements
 * - Dynamic filtering where partial spaces are not pre-defined
 * - Multi-criteria filtering across the entire domain
 * - Exploratory data analysis on full simulation results
 *
 * Performance considerations:
 * - Processes entire mesh but conditionally accumulates
 * - More memory bandwidth than partial space alternatives
 * - Suitable when filter changes frequently or is complex
 * - RAJA views optimize memory access patterns
 *
 * The filter array organization follows standard QuadratureFunction layout:
 * - Element-major ordering: filter[elem][qp]
 * - Total size: nqpts * nelems
 * - Boolean values minimize memory footprint
 *
 * Integration with ExaConstit workflows:
 * - Supports all material models and integration rules
 * - Compatible with existing postprocessing infrastructure
 * - Enables gradual migration to partial space architectures
 * - Returns volume for consistency with other averaging functions
 *
 * @note Filter array must cover all quadrature points in the mesh.
 * @note Performance scales with total mesh size, not filtered size.
 * @note Return volume enables multi-region volume fraction calculations.
 *
 * @ingroup ExaConstit_utilities_kernels
 */
template <bool vol_avg>
double ComputeVolAvgTensorFilter(const mfem::ParFiniteElementSpace* fes,
                                 const mfem::QuadratureFunction* qf,
                                 const mfem::Array<bool>* filter,
                                 mfem::Vector& tensor,
                                 int size,
                                 const RTModel& class_device) {
    mfem::Mesh* mesh = fes->GetMesh();
    const mfem::FiniteElement& el = *fes->GetFE(0);
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    ;

    const int nqpts = ir->GetNPoints();
    const int nelems = fes->GetNE();
    const int npts = nqpts * nelems;

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    double el_vol = 0.0;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{nqpts, nelems}}, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> wts_view(wts.ReadWrite(),
                                                                         layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> j_view(geom->detJ.Read(),
                                                                             layout_geom);

    RAJA::RangeSegment default_range(0, npts);

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = qf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int i_npts) {
                if (!filter_data[i_npts])
                    return;
                const double* val = &(qf_data[i_npts * size]);
                seq_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = seq_sum.get();
            el_vol = vol_sum.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = qf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i_npts) {
                if (!filter_data[i_npts])
                    return;
                const double* val = &(qf_data[i_npts * size]);
                omp_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = omp_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = qf->Read();
        const bool* filter_data = filter->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> gpu_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i_npts) {
                if (!filter_data[i_npts])
                    return;
                const double* val = &(qf_data[i_npts * size]);
                gpu_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = gpu_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(
        data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double temp = el_vol;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(el_vol) > 1e-14) ? 1.0 / el_vol : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return el_vol;
}

/**
 * @brief Compute volume-averaged tensor values from PartialQuadratureFunction.
 *
 * @tparam vol_avg Boolean template parameter controlling averaging behavior
 * @param pqf Partial quadrature function containing region-specific tensor data
 * @param tensor Output vector for the volume-averaged tensor components
 * @param size Number of tensor components per quadrature point
 * @param class_device Runtime model for device execution policy
 * @param region_comm MPI communicator associated with a given region
 * @return Total volume of the region processed
 *
 * This template function computes volume-averaged values directly from a
 * PartialQuadratureFunction, which contains data only for a specific material
 * region or subdomain. This enables efficient region-specific postprocessing
 * without processing the entire mesh.
 *
 * The function handles the complexity of partial element mapping:
 * 1. Extracts local-to-global element mapping from PartialQuadratureSpace
 * 2. Maps local data offsets to global geometric factors
 * 3. Performs volume averaging over only the active elements
 * 4. Returns the total volume of the processed region
 *
 * Key advantages over full mesh processing:
 * - Reduced computational cost for region-specific calculations
 * - Automatic handling of multi-material simulations
 * - Efficient memory usage for sparse material distributions
 * - Direct integration with PartialQuadratureFunction workflow
 *
 * Data layout handling:
 * - Local data uses PartialQuadratureSpace offsets
 * - Global geometric factors indexed via local-to-global mapping
 * - Automatic optimization for full-space vs. partial-space cases
 *
 * The returned volume can be used for:
 * - Volume fraction calculations in composites
 * - Normalization of other regional quantities
 * - Quality assurance and verification
 * - Multi-scale homogenization procedures
 *
 * @note Debug builds include assertion checking for size/vdim consistency.
 * @note The function assumes uniform element types within the region.
 * @note Return value enables volume-based postprocessing workflows.
 *
 * @ingroup ExaConstit_utilities_kernels
 */
template <bool vol_avg>
double ComputeVolAvgTensorFilterFromPartial(const mfem::expt::PartialQuadratureFunction* pqf,
                                            const mfem::Array<bool>* filter,
                                            mfem::Vector& tensor,
                                            int size,
                                            const RTModel& class_device,
                                            MPI_Comm region_comm = MPI_COMM_WORLD) {
    auto pqs = pqf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();

    // Get finite element and integration rule info
    // Note: We need to get this from the global finite element space since
    // the PartialQuadratureSpace doesn't have direct FE access
    const int fe_order = pqs->GetOrder();
    mfem::Geometry::Type geom_type = mesh->GetElementBaseGeometry(0); // Assume uniform elements
    const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(geom_type, fe_order));

    const int nqpts = ir->GetNPoints();
    const int local_nelems = pqs->GetNE(); // Number of elements in this partial space
    const int nelems = mesh->GetNE();

    // Verify size matches vdim
#if defined(MFEM_USE_DEBUG)
    const int vdim = pqf->GetVDim();
    MFEM_ASSERT_0(size == vdim, "Size parameter must match quadrature function vector dimension");
#endif

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    // Get the local-to-global element mapping and data layout info
    auto l2g = pqs->GetLocal2Global().Read(); // Maps local element index to global element index
    auto loc_offsets =
        pqs->Offsets(mfem::QSpaceOffsetStorage::COMPRESSED).Read(); // Offsets for local data layout
    auto global_offsets = (pqs->GetGlobalOffset().Size() > 1)
                              ? pqs->GetGlobalOffset().Read()
                              : loc_offsets; // Offsets for global data layout

    double el_vol = 0.0;
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{nqpts, nelems}}, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> wts_view(wts.ReadWrite(),
                                                                         layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> j_view(geom->detJ.Read(),
                                                                             layout_geom);

    RAJA::RangeSegment default_range(0, local_nelems);

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = pqf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> data_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k])
                        continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = pqf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> data_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k])
                        continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = pqf->Read();
        const bool* filter_data = filter->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> data_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k])
                        continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, region_comm);

    double temp = el_vol;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, region_comm);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(el_vol) > 1e-14) ? 1.0 / el_vol : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return el_vol;
}

/**
 * @brief Compute filtered volume-averaged tensor values from PartialQuadratureFunction.
 *
 * @tparam vol_avg Boolean template parameter controlling averaging behavior
 * @param pqf Partial quadrature function containing region-specific tensor data
 * @param filter Boolean array indicating which quadrature points to include
 * @param tensor Output vector for the volume-averaged tensor components
 * @param size Number of tensor components per quadrature point
 * @param class_device Runtime model for device execution policy
 * @param region_comm MPI communicator associated with a given region
 * @return Total volume of the filtered region
 *
 * This template function extends ComputeVolAvgTensorFromPartial by adding
 * point-wise filtering capability. It computes volume averages over only
 * those quadrature points where the filter array is true, enabling selective
 * postprocessing based on material state, stress levels, or other criteria.
 *
 * Filtering applications:
 * - Stress-based filtering (e.g., only plastic regions)
 * - Phase-specific averaging in multiphase materials
 * - Damage-based selective averaging
 * - Grain-specific calculations in polycrystals
 * - Temperature or strain-rate dependent processing
 *
 * Algorithm with filtering:
 * 1. Loop over all local elements in the PartialQuadratureFunction
 * 2. For each quadrature point, check the filter condition
 * 3. Include only filtered points in volume and tensor accumulation
 * 4. Normalize by filtered volume if vol_avg is true
 *
 * The filter array indexing must match the quadrature point layout:
 * - One boolean value per quadrature point
 * - Organized element-by-element, then point-by-point within elements
 * - Size should equal nqpts * nelems for the partial space
 *
 * Memory efficiency considerations:
 * - Filter array can be generated on-the-fly or cached
 * - Boolean filter minimizes memory overhead
 * - Processing only active points reduces computational cost
 *
 * Return value enables cascaded filtering operations and
 * provides volume information for normalization in subsequent
 * calculations or multiscale homogenization procedures.
 *
 * @note Filter array size must match total quadrature points in partial space.
 * @note Filtering reduces computational cost but adds conditional overhead.
 * @note Zero filtered volume will result in division by zero if vol_avg is true.
 *
 * @ingroup ExaConstit_utilities_kernels
 */
template <bool vol_avg>
double ComputeVolAvgTensorFromPartial(const mfem::expt::PartialQuadratureFunction* pqf,
                                      mfem::Vector& tensor,
                                      int size,
                                      const RTModel& class_device,
                                      MPI_Comm region_comm = MPI_COMM_WORLD) {
    auto pqs = pqf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();

    // Get finite element and integration rule info
    // Note: We need to get this from the global finite element space since
    // the PartialQuadratureSpace doesn't have direct FE access
    const int fe_order = pqs->GetOrder();
    mfem::Geometry::Type geom_type = mesh->GetElementBaseGeometry(0); // Assume uniform elements
    const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(geom_type, fe_order));

    const int nqpts = ir->GetNPoints();
    const int local_nelems = pqs->GetNE(); // Number of elements in this partial space
    const int nelems = mesh->GetNE();

    // Verify size matches vdim
#if defined(MFEM_USE_DEBUG)
    const int vdim = pqf->GetVDim();
    MFEM_ASSERT_0(size == vdim, "Size parameter must match quadrature function vector dimension");
#endif

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    // Get the local-to-global element mapping and data layout info
    auto l2g = pqs->GetLocal2Global().Read(); // Maps local element index to global element index
    auto loc_offsets =
        pqs->Offsets(mfem::QSpaceOffsetStorage::COMPRESSED).Read(); // Offsets for local data layout
    auto global_offsets = (pqs->GetGlobalOffset().Size() > 1)
                              ? pqs->GetGlobalOffset().Read()
                              : loc_offsets; // Offsets for global data layout

    // Initialize output tensor and volume
    tensor.SetSize(size);
    tensor = 0.0;
    double total_volume = 0.0;
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{nqpts, nelems}}, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> wts_view(wts.ReadWrite(),
                                                                         layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> j_view(geom->detJ.Read(),
                                                                             layout_geom);

    RAJA::RangeSegment default_range(0, local_nelems);

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = pqf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    seq_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = seq_sum.get();
            total_volume = vol_sum.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = pqf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    omp_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = omp_sum.get();
            total_volume = vol_sum.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = pqf->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> gpu_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int ie) {
                const int global_elem = l2g[ie];          // Map local element to global element
                const int local_offset = loc_offsets[ie]; // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] -
                                      local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    gpu_sum += wts_data[global_offset + k] * val[j];
                    vol_sum += wts_data[global_offset + k];
                }
            });
            data[j] = gpu_sum.get();
            total_volume = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, region_comm);

    double temp = total_volume;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &total_volume, 1, MPI_DOUBLE, MPI_SUM, region_comm);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(total_volume) > 1e-14) ? 1.0 / total_volume : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return total_volume;
}

} // namespace kernel
} // namespace exaconstit
#endif
