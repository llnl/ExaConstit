#include "models/mechanics_umat.hpp"

#include "utilities/kinematics_kernels.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem/general/forall.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <math.h> // cbrt in device code (plain C math library)
#include <stdexcept>

namespace {

/// Allocate a vector in the run's memory space (device memory on GPU runs), set to value
void MakeDeviceVector(mfem::Vector& v, const int size, const double value = 0.0) {
    v.SetSize(size, mfem::Device::GetMemoryType());
    v.UseDevice(true);
    v = value;
}

/// Set every point's 3x3 tensor to the identity (9 values per point)
void SetIdentity(mfem::Vector& tensors) {
    const int npts = tensors.Size() / 9;
    double* data = tensors.Write();
    mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i) {
        for (int k = 0; k < 9; ++k) {
            data[9 * i + k] = (k % 4 == 0) ? 1.0 : 0.0;
        }
    });
}

} // namespace

AbaqusUmatModel::AbaqusUmatModel(const int region,
                                 int n_state_vars,
                                 std::shared_ptr<SimulationState> sim_state,
                                 const std::filesystem::path& umat_library_path_,
                                 const exaconstit::LoadStrategy& load_strategy_,
                                 const std::string umat_function_name_,
                                 const bool batched,
                                 const bool thread_safe,
                                 const double temperature,
                                 const std::string& material_name,
                                 const long long global_elem_offset,
                                 const bool log_strain,
                                 const bool rotate_stress)
    : ExaModel(region, n_state_vars, sim_state), umat_library_path(umat_library_path_),
      umat_function(nullptr), umat_batch_function(nullptr), load_strategy(load_strategy_),
      use_dynamic_loading(!umat_library_path_.empty()), umat_function_name(umat_function_name_),
      m_batched(batched), m_thread_safe(thread_safe), m_temperature(temperature),
      m_log_strain(log_strain), m_rotate_stress(rotate_stress), m_nqpts(0),
      m_nnodes(0), m_nelems(0), m_reference_ready(false) {
    MFEM_VERIFY_0(!m_batched || use_dynamic_loading,
                  "The batched UMAT interface needs a UMAT library (there is no built-in "
                  "batched UMAT)");

    // Sizes of the region on this rank
    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    m_nqpts = qspace->GetIntRule(0).GetNPoints();
    m_nelems = qspace->GetNE(); // the region's elements on this rank
    m_nnodes = m_sim_state->GetMeshParFiniteElementSpace()->GetTypicalFE()->GetDof();
    const int npts = m_nqpts * m_nelems;

    // Work arrays, allocated once in the run's memory space
    MakeDeviceVector(m_ref_jacobians, 9 * npts);
    MakeDeviceVector(m_ref_wdetj, npts);
    MakeDeviceVector(m_region_coords, 3 * m_nnodes * m_nelems);
    MakeDeviceVector(m_def_grad_beg, 9 * npts);
    MakeDeviceVector(m_def_grad_end, 9 * npts);
    SetIdentity(m_def_grad_beg);
    SetIdentity(m_def_grad_end);

    MakeDeviceVector(m_stress, 6 * npts);
    MakeDeviceVector(m_stran, 6 * npts);
    MakeDeviceVector(m_dstran, 6 * npts);
    MakeDeviceVector(m_drot, 9 * npts);
    MakeDeviceVector(m_coords, 3 * npts);
    MakeDeviceVector(m_celent, npts);
    MakeDeviceVector(m_pnewdt, npts);
    MakeDeviceVector(m_temp, npts);
    MakeDeviceVector(m_dtemp, npts);
    MakeDeviceVector(m_predef, npts);
    MakeDeviceVector(m_dpred, npts);
    MakeDeviceVector(m_rpl, npts);
    MakeDeviceVector(m_drpldt, npts);
    MakeDeviceVector(m_ddsdt, 6 * npts);
    MakeDeviceVector(m_drplde, 6 * npts);

    // Material properties: constant over the run
    {
        const auto& props = GetMaterialProperties();
        MakeDeviceVector(m_props, std::max(static_cast<int>(props.size()), 1));
        double* props_data = m_props.HostWrite();
        for (size_t i = 0; i < props.size(); ++i) {
            props_data[i] = props[i];
        }
    }

    // NOEL / NPT, 1-based as in Abaqus
    m_noel.SetSize(npts, mfem::Device::GetMemoryType());
    m_npt.SetSize(npts, mfem::Device::GetMemoryType());
    SetupElementNumbers(global_elem_offset);

    // CMNAME: the material name, blank-padded to 80 characters like a Fortran CHARACTER*80
    m_cmname.fill(' ');
    m_cmname[80] = '\0';
    material_name.copy(m_cmname.data(), std::min<size_t>(material_name.size(), 80));

    // Load the UMAT now unless it is loaded at each setup
    if (use_dynamic_loading && load_strategy == exaconstit::LoadStrategy::PERSISTENT) {
        if (!LoadUmatLibrary()) {
            throw std::runtime_error("Failed to load UMAT library: " + umat_library_path.string());
        }
    } else if (!use_dynamic_loading) {
        // Built-in UMAT: UnifiedUmatLoader searches the executable when the path is empty
        umat_function = exaconstit::UnifiedUmatLoader::LoadUmat(
            "", exaconstit::LoadStrategy::PERSISTENT, "");
        if (!umat_function) {
            throw std::runtime_error("No built-in UMAT available: " +
                                     exaconstit::UnifiedUmatLoader::GetLastError());
        }
    }
}

AbaqusUmatModel::~AbaqusUmatModel() {
    if (use_dynamic_loading && load_strategy != exaconstit::LoadStrategy::PERSISTENT) {
        UnloadUmatLibrary();
    }
}

std::shared_ptr<mfem::expt::PartialQuadratureFunction> AbaqusUmatModel::GetDefGrad0() {
    return m_sim_state->GetQuadratureFunction("def_grad_beg", m_region);
}

void AbaqusUmatModel::UpdateModelVars() {
    // Output / restart copy of the converged deformation gradient (same point ordering)
    mfem::Vector& def_grad_beg = *GetDefGrad0();
    def_grad_beg = m_def_grad_end;
}

void AbaqusUmatModel::SetupElementNumbers(const long long global_elem_offset) {
    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    const int* l2g = qspace->IsFullSpace() ? nullptr : qspace->GetLocal2Global().Read();
    const int nqpts = m_nqpts;
    int* noel = m_noel.Write();
    int* npt = m_npt.Write();
    mfem::forall(m_noel.Size(), [=] MFEM_HOST_DEVICE(int i) {
        const int e = i / nqpts;
        const int rank_elem = l2g ? l2g[e] : e;
        // NOEL is a default (32-bit) integer in the UMAT interface
        noel[i] = static_cast<int>(global_elem_offset + rank_elem) + 1;
        npt[i] = i % nqpts + 1;
    });
}

void AbaqusUmatModel::SetupReferenceGeometry(const mfem::Vector& loc_grad) {
    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    const mfem::FiniteElement& fe = *fes->GetTypicalFE();
    // The operator's quadrature rule (NonlinearMechOperator::Setup)
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + 1);
    MFEM_VERIFY_0(ir.GetNPoints() == m_nqpts,
                  "UMAT model: the integration rule differs from the region's quadrature space");
    MFEM_VERIFY_0(loc_grad.Size() == 3 * m_nnodes * m_nqpts,
                  "UMAT model: unexpected size of the reference shape derivatives");

    // Reference nodes -> element coordinates (NATIVE order, as the operator's loc_grad)
    auto ref_nodes = m_sim_state->GetRefCoords();
    const mfem::Operator* restriction = fes->GetElementRestriction(
        mfem::ElementDofOrdering::NATIVE);
    MFEM_VERIFY_0(ref_nodes->Size() == restriction->Width(),
                  "UMAT model: the mesh nodes are not on the mesh finite element space");
    MakeDeviceVector(m_elem_coords, restriction->Height());
    restriction->Mult(*ref_nodes, m_elem_coords);

    // d X / d xi at every point of every rank element; layout byVDIM = (3, 3, nqpts, elements)
    const int nelems_rank = fes->GetNE();
    mfem::Vector jac_rank;
    MakeDeviceVector(jac_rank, 9 * m_nqpts * nelems_rank);
    const mfem::QuadratureInterpolator* qi = fes->GetQuadratureInterpolator(ir);
    qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
    qi->DisableTensorProducts();
    qi->Derivatives(m_elem_coords, jac_rank);

    // Copy out the region's elements, and w_q det(J_ref) for F-bar
    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    const int* l2g = qspace->IsFullSpace() ? nullptr : qspace->GetLocal2Global().Read();
    const int nqpts = m_nqpts;
    const double* weights = ir.GetWeights().Read();
    const double* jac_all = jac_rank.Read();
    double* jac_region = m_ref_jacobians.Write();
    double* wdetj = m_ref_wdetj.Write();
    mfem::forall(m_nelems * nqpts, [=] MFEM_HOST_DEVICE(int i) {
        const int e = i / nqpts;
        const int q = i % nqpts;
        const int ge = l2g ? l2g[e] : e;
        const double* src = jac_all + (static_cast<std::ptrdiff_t>(ge) * nqpts + q) * 9;
        double* dst = jac_region + static_cast<std::ptrdiff_t>(i) * 9;
        for (int k = 0; k < 9; ++k) {
            dst[k] = src[k];
        }
        wdetj[i] = weights[q] * mfem::kernels::Det<3>(src);
    });
    m_reference_ready = true;
}

void AbaqusUmatModel::ComputeDefGrad(const mfem::ParGridFunction& nodes,
                                     const mfem::Vector& loc_grad,
                                     mfem::Vector& def_grad) {
    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    const mfem::Operator* restriction = fes->GetElementRestriction(
        mfem::ElementDofOrdering::NATIVE);
    MFEM_VERIFY_0(nodes.Size() == restriction->Width(),
                  "UMAT model: the mesh nodes are not on the mesh finite element space");
    restriction->Mult(nodes, m_elem_coords);

    // Region elements' coordinates, layout (nnodes, 3, region elements)
    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    const int* l2g = qspace->IsFullSpace() ? nullptr : qspace->GetLocal2Global().Read();
    const int elem_size = 3 * m_nnodes;
    const double* all_coords = m_elem_coords.Read();
    double* region_coords = m_region_coords.Write();
    mfem::forall(m_nelems * elem_size, [=] MFEM_HOST_DEVICE(int i) {
        const int e = i / elem_size;
        const int k = i % elem_size;
        const int ge = l2g ? l2g[e] : e;
        region_coords[i] = all_coords[static_cast<std::ptrdiff_t>(ge) * elem_size + k];
    });

    // F = sum_a x_a (x) grad_X N_a (GradCalc accumulates, so start from zero)
    def_grad = 0.0;
    exaconstit::kernel::GradCalc(m_nqpts,
                                 m_nelems,
                                 m_nelems,
                                 m_nnodes,
                                 m_ref_jacobians.Read(),
                                 loc_grad.Read(),
                                 m_region_coords.Read(),
                                 def_grad.ReadWrite(),
                                 nullptr);

    // B-bar integrator: the UMAT gets the deformation-gradient form of the same volume
    // averaging (F-bar): every point of an element gets the element's V / V_0, the isochoric
    // part of each point's F is unchanged
    if (m_sim_state->GetOptions().solvers.integ_model == IntegrationModel::BBAR) {
        const int nqpts = m_nqpts;
        const double* wdetj = m_ref_wdetj.Read();
        double* f = def_grad.ReadWrite();
        mfem::forall(m_nelems, [=] MFEM_HOST_DEVICE(int e) {
            exaconstit::kernel::FBarElement(nqpts,
                                            9,
                                            wdetj + static_cast<std::ptrdiff_t>(e) * nqpts,
                                            f + static_cast<std::ptrdiff_t>(e) * nqpts * 9);
        });
    }
}

void AbaqusUmatModel::PackUmatInputs(const mfem::Vector& jacobian, const int* local2global) {
    const int nqpts = m_nqpts;
    const int npts = m_nqpts * m_nelems;
    const int nelems_rank = m_sim_state->GetMeshParFiniteElementSpace()->GetNE();
    const double temperature = m_temperature;
    const bool log_strain = m_log_strain;
    const bool rotate = m_rotate_stress;
    // Volume of the reference element (1 for the unit cube, 1/6 for the unit tetrahedron), so
    // that det(J) times it is the element volume near the point
    const double ref_volume = mfem::Geometry::Volume[
        m_sim_state->GetMeshParFiniteElementSpace()->GetTypicalFE()->GetGeomType()];

    // Current point positions (the mesh holds the current nodes during the solve)
    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    auto geom = m_sim_state->GetMesh()->GetGeometricFactors(qspace->GetIntRule(0),
                                                            mfem::GeometricFactors::COORDINATES);
    const auto x_qpt = mfem::Reshape(geom->X.Read(), nqpts, 3, nelems_rank);

    const double* stress_beg =
        m_sim_state->GetQuadratureFunction("cauchy_stress_beg", m_region)->Read();
    const double* jac = jacobian.Read();
    const double* f_beg = m_def_grad_beg.Read();
    const double* f_end = m_def_grad_end.Read();

    double* stress = m_stress.Write();
    double* stran = m_stran.Write();
    double* dstran = m_dstran.Write();
    double* drot = m_drot.Write();
    double* coords = m_coords.Write();
    double* celent = m_celent.Write();
    double* pnewdt = m_pnewdt.Write();
    double* temp = m_temp.Write();
    double* dtemp = m_dtemp.Write();
    double* predef = m_predef.Write();
    double* dpred = m_dpred.Write();
    double* rpl = m_rpl.Write();
    double* drpldt = m_drpldt.Write();
    double* ddsdt = m_ddsdt.Write();
    double* drplde = m_drplde.Write();

    mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i) {
        const int e = i / nqpts;
        const int q = i % nqpts;
        const int ge = local2global ? local2global[e] : e;

        // Stress at the beginning of the step, Abaqus order
        exaconstit::kernel::StressToAbaqusVoigt(stress_beg + 6 * i, stress + 6 * i);

        // Increment F_incr = F_end F_beg^-1 and its exact rotation (polar decomposition)
        const double* fb = f_beg + 9 * i;
        double f_incr[9], e_tens[9];
        double* r_incr = drot + 9 * i;
        exaconstit::kernel::IncrementalDefGrad(fb, f_end + 9 * i, f_incr);
        exaconstit::kernel::PolarRotation(f_incr, r_incr);

        // Strains, in the chosen measure (logarithmic ln V or Eulerian 1/2 (I - B^-1)): STRAN
        // at the beginning of the increment from F_beg, DSTRAN over the increment from F_incr
        if (log_strain) {
            exaconstit::kernel::LogStrain(fb, e_tens);
        } else {
            exaconstit::kernel::EulerianStrain(fb, e_tens);
        }
        exaconstit::kernel::StrainToAbaqusVoigt(e_tens, stran + 6 * i);
        if (log_strain) {
            exaconstit::kernel::LogStrain(f_incr, e_tens);
        } else {
            exaconstit::kernel::EulerianStrain(f_incr, e_tens);
        }
        exaconstit::kernel::StrainToAbaqusVoigt(e_tens, dstran + 6 * i);

        // rotate_stress: carry the beginning-of-increment stress and strain along with the
        // rigid rotation of the increment (R T R^T with the exact DROT), so a UMAT written for
        // unrotated input (isotropic models) sees them in the current orientation
        if (rotate) {
            exaconstit::kernel::RotateAbaqusVoigt(r_incr, stress + 6 * i, 1.0);
            exaconstit::kernel::RotateAbaqusVoigt(r_incr, stran + 6 * i, 2.0);
        }

        // Point position, and characteristic length: cube root of the local element volume,
        // det(J) times the reference element's volume (the same as the original cbrt(det J) for
        // hexahedra)
        for (int c = 0; c < 3; ++c) {
            coords[3 * i + c] = x_qpt(q, c, ge);
        }
        const double det_j = mfem::kernels::Det<3>(
            jac + (static_cast<std::ptrdiff_t>(ge) * nqpts + q) * 9);
        celent[i] = cbrt(ref_volume * det_j);

        // No thermal coupling or predefined fields; outputs start from zero
        temp[i] = temperature;
        dtemp[i] = 0.0;
        predef[i] = 0.0;
        dpred[i] = 0.0;
        rpl[i] = 0.0;
        drpldt[i] = 0.0;
        for (int k = 0; k < 6; ++k) {
            ddsdt[6 * i + k] = 0.0;
            drplde[6 * i + k] = 0.0;
        }
        // A UMAT lowers PNEWDT below 1 to ask for a smaller time step
        pnewdt[i] = 10.0;
    });
}

void AbaqusUmatModel::IncrementInfo(double time[2], double& dtime, int& kstep, int& kinc) {
    dtime = m_sim_state->GetDeltaTime();
    // During the solve GetTime() is the time at the end of the increment; Abaqus passes the
    // times at its beginning. ExaConstit runs one analysis step, so the step time TIME(1) and
    // the total time TIME(2) are the same.
    const double time_beg = m_sim_state->GetTime() - dtime;
    time[0] = time_beg;
    time[1] = time_beg;
    kstep = 1;
    // Increment number, 1-based: the time steps completed so far plus the current one
    kinc = static_cast<int>(m_sim_state->GetSimulationCycle()) + 1;
}

void AbaqusUmatModel::CallUmatPerPoint() {
    MFEM_VERIFY_0(umat_function, "UMAT function not available");
    const int npts = m_nqpts * m_nelems;

    // Host copies of every argument (copied from the GPU on GPU runs)
    auto state_end = m_sim_state->GetQuadratureFunction("state_var_end", m_region);
    auto tangent = m_sim_state->GetQuadratureFunction("tangent_stiffness", m_region);
    double* stress = m_stress.HostReadWrite();
    double* statev = state_end->HostReadWrite();
    double* ddsdde = tangent->HostReadWrite();
    double* sse = m_sim_state->GetQuadratureFunction("umat_sse_end", m_region)->HostReadWrite();
    double* spd = m_sim_state->GetQuadratureFunction("umat_spd_end", m_region)->HostReadWrite();
    double* scd = m_sim_state->GetQuadratureFunction("umat_scd_end", m_region)->HostReadWrite();
    double* rpl = m_rpl.HostReadWrite();
    double* ddsdt = m_ddsdt.HostReadWrite();
    double* drplde = m_drplde.HostReadWrite();
    double* drpldt = m_drpldt.HostReadWrite();
    double* stran = m_stran.HostReadWrite();
    double* dstran = m_dstran.HostReadWrite();
    double* temp = m_temp.HostReadWrite();
    double* dtemp = m_dtemp.HostReadWrite();
    double* predef = m_predef.HostReadWrite();
    double* dpred = m_dpred.HostReadWrite();
    double* props = m_props.HostReadWrite();
    double* coords = m_coords.HostReadWrite();
    double* drot = m_drot.HostReadWrite();
    double* pnewdt = m_pnewdt.HostReadWrite();
    double* celent = m_celent.HostReadWrite();
    double* dfgrd0 = m_def_grad_beg.HostReadWrite();
    double* dfgrd1 = m_def_grad_end.HostReadWrite();
    int* noel = m_noel.HostReadWrite();
    int* npt = m_npt.HostReadWrite();

    // Scalars shared by all points
    double time_beg[2], dt;
    int kstep_c, kinc_c;
    IncrementInfo(time_beg, dt, kstep_c, kinc_c);
    const int nstatv_c = num_state_vars;
    const int nprops_c = static_cast<int>(GetMaterialProperties().size());
    const bool parallel = m_thread_safe;

    auto call_point = [&](const int i) {
        // Per-call copies: a UMAT receives every argument by reference and may overwrite them
        int ndi = 3, nshr = 3, ntens = 6, nstatv = nstatv_c, nprops = nprops_c;
        int layer = 0, kspt = 0, kstep = kstep_c, kinc = kinc_c;
        double time[2] = {time_beg[0], time_beg[1]};
        double dtime = dt;
        std::array<char, 81> cmname = m_cmname;
        umat_function(stress + 6 * i,
                      statev + static_cast<std::ptrdiff_t>(nstatv_c) * i,
                      ddsdde + 36 * i,
                      sse + i,
                      spd + i,
                      scd + i,
                      rpl + i,
                      ddsdt + 6 * i,
                      drplde + 6 * i,
                      drpldt + i,
                      stran + 6 * i,
                      dstran + 6 * i,
                      time,
                      &dtime,
                      temp + i,
                      dtemp + i,
                      predef + i,
                      dpred + i,
                      cmname.data(),
                      &ndi,
                      &nshr,
                      &ntens,
                      &nstatv,
                      props,
                      &nprops,
                      coords + 3 * i,
                      drot + 9 * i,
                      pnewdt + i,
                      celent + i,
                      dfgrd0 + 9 * i,
                      dfgrd1 + 9 * i,
                      noel + i,
                      npt + i,
                      &layer,
                      &kspt,
                      &kstep,
                      &kinc);
    };

#if defined(RAJA_ENABLE_OPENMP)
#pragma omp parallel for if (parallel)
#endif
    for (int i = 0; i < npts; ++i) {
        call_point(i);
    }
    (void)parallel;
}

void AbaqusUmatModel::CallUmatBatched() {
    MFEM_VERIFY_0(umat_batch_function, "Batched UMAT function not available");
    auto state_end = m_sim_state->GetQuadratureFunction("state_var_end", m_region);
    auto tangent = m_sim_state->GetQuadratureFunction("tangent_stiffness", m_region);

    // Sizes and other shared scalars: host memory (the user's code reads them to launch its
    // work); arrays: the run's memory space
    int npts = m_nqpts * m_nelems;
    int ndi = 3, nshr = 3, ntens = 6, nstatv = num_state_vars;
    int nprops = static_cast<int>(GetMaterialProperties().size());
    int layer = 0, kspt = 0, kstep = 0, kinc = 0;
    double dtime = 0.0;
    double time[2];
    IncrementInfo(time, dtime, kstep, kinc);
    std::array<char, 81> cmname = m_cmname;

    umat_batch_function(&npts,
                        m_stress.ReadWrite(),
                        state_end->ReadWrite(),
                        tangent->ReadWrite(),
                        m_sim_state->GetQuadratureFunction("umat_sse_end", m_region)->ReadWrite(),
                        m_sim_state->GetQuadratureFunction("umat_spd_end", m_region)->ReadWrite(),
                        m_sim_state->GetQuadratureFunction("umat_scd_end", m_region)->ReadWrite(),
                        m_rpl.ReadWrite(),
                        m_ddsdt.ReadWrite(),
                        m_drplde.ReadWrite(),
                        m_drpldt.ReadWrite(),
                        m_stran.ReadWrite(),
                        m_dstran.ReadWrite(),
                        time,
                        &dtime,
                        m_temp.ReadWrite(),
                        m_dtemp.ReadWrite(),
                        m_predef.ReadWrite(),
                        m_dpred.ReadWrite(),
                        cmname.data(),
                        &ndi,
                        &nshr,
                        &ntens,
                        &nstatv,
                        m_props.ReadWrite(),
                        &nprops,
                        m_coords.ReadWrite(),
                        m_drot.ReadWrite(),
                        m_pnewdt.ReadWrite(),
                        m_celent.ReadWrite(),
                        m_def_grad_beg.ReadWrite(),
                        m_def_grad_end.ReadWrite(),
                        m_noel.ReadWrite(),
                        m_npt.ReadWrite(),
                        &layer,
                        &kspt,
                        &kstep,
                        &kinc);
    // The user's GPU work may still be running when the call returns
    MFEM_DEVICE_SYNC;
}

void AbaqusUmatModel::UnpackUmatOutputs() {
    const int npts = m_nqpts * m_nelems;
    const double dt = m_sim_state->GetDeltaTime();

    // Smallest requested time step factor (one value per point)
    {
        const double* pnewdt = m_pnewdt.HostRead();
        double min_pnewdt = std::numeric_limits<double>::max();
        for (int i = 0; i < npts; ++i) {
            min_pnewdt = std::min(min_pnewdt, pnewdt[i]);
        }
        if (min_pnewdt < 1.0) {
            throw std::runtime_error(
                "UMAT time stepping needs to be reduced for at least 1 integration point");
        }
    }

    // Stress back to ExaConstit's order; tangent reordered and in rate form (DDSDDE dt).
    // STATEV was updated in place in state_var_end.
    const double* stress_abq = m_stress.Read();
    double* stress_end =
        m_sim_state->GetQuadratureFunction("cauchy_stress_end", m_region)->Write();
    double* ddsdde = m_sim_state->GetQuadratureFunction("tangent_stiffness", m_region)->ReadWrite();
    mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i) {
        exaconstit::kernel::StressFromAbaqusVoigt(stress_abq + 6 * i, stress_end + 6 * i);
        exaconstit::kernel::AbaqusTangentToExaConstit(dt, ddsdde + 36 * i);
    });
}

void AbaqusUmatModel::ModelSetup(const int nqpts,
                                 const int /*nelems*/,
                                 const int /*space_dim*/,
                                 const int nnodes,
                                 const mfem::Vector& jacobian,
                                 const mfem::Vector& loc_grad,
                                 const mfem::Vector& /*vel*/) {
    auto& logger = exaconstit::UnifiedLogger::get_instance();
    std::string material_log = logger.get_material_log_filename("umat", m_region);
    exaconstit::UnifiedLogger::ScopedCapture capture(material_log);

    MFEM_VERIFY_0(nqpts == m_nqpts && nnodes == m_nnodes,
                  "UMAT model: the operator's element sizes differ from the model's");

    // Load the UMAT if it isn't loaded yet: every setup with load_on_setup, the first setup
    // with lazy_load (persistent libraries are loaded by the constructor)
    if (use_dynamic_loading && !UmatLoaded()) {
        if (!LoadUmatLibrary()) {
            throw std::runtime_error("Failed to load UMAT library during ModelSetup: " +
                                     umat_library_path.string());
        }
    }
    // load_on_setup: unload when this setup ends, also when it ends with an exception (a point
    // asking for a smaller time step)
    struct UnloadOnExit {
        AbaqusUmatModel* model;
        ~UnloadOnExit() {
            if (model) {
                model->UnloadUmatLibrary();
            }
        }
    } unload_on_exit{(use_dynamic_loading &&
                      load_strategy == exaconstit::LoadStrategy::LOAD_ON_SETUP)
                         ? this
                         : nullptr};

    auto qspace = GetDefGrad0()->GetPartialSpaceShared();
    const int* local2global = qspace->IsFullSpace() ? nullptr
                                                    : qspace->GetLocal2Global().Read();

    // 1. Kinematics: exact F_beg and F_end from the stored nodes
    CALI_MARK_BEGIN("umat_kinematics");
    if (!m_reference_ready) {
        SetupReferenceGeometry(loc_grad);
    }
    ComputeDefGrad(*m_sim_state->GetTimeStartCoords(), loc_grad, m_def_grad_beg);
    ComputeDefGrad(*m_sim_state->GetCurrentCoords(), loc_grad, m_def_grad_end);
    CALI_MARK_END("umat_kinematics");

    // 2. UMAT inputs. DDSDDE starts from zero.
    CALI_MARK_BEGIN("umat_setup");
    PackUmatInputs(jacobian, local2global);
    {
        // STATEV and the accumulated energies SSE, SPD, SCD start from the converged values of
        // the previous step and are updated in place by the UMAT
        for (const char* name : {"state_var", "umat_sse", "umat_spd", "umat_scd"}) {
            mfem::Vector& end = *m_sim_state->GetQuadratureFunction(std::string(name) + "_end",
                                                                    m_region);
            end = *m_sim_state->GetQuadratureFunction(std::string(name) + "_beg", m_region);
        }
        mfem::Vector& tangent = *m_sim_state->GetQuadratureFunction("tangent_stiffness", m_region);
        tangent = 0.0;
    }
    CALI_MARK_END("umat_setup");

    // 3. The UMAT
    CALI_MARK_BEGIN("umat_kernel");
    if (m_nelems > 0) {
        if (m_batched) {
            CallUmatBatched();
        } else {
            CallUmatPerPoint();
        }
    }
    CALI_MARK_END("umat_kernel");

    // 4. Results back in ExaConstit's conventions, then into the global quadrature functions
    CALI_MARK_BEGIN("umat_postprocessing");
    UnpackUmatOutputs();
    auto global_stress = m_sim_state->GetQuadratureFunction("cauchy_stress_end");
    m_sim_state->GetQuadratureFunction("cauchy_stress_end", m_region)
        ->FillQuadratureFunction(*global_stress);
    auto global_tangent = m_sim_state->GetQuadratureFunction("tangent_stiffness");
    m_sim_state->GetQuadratureFunction("tangent_stiffness", m_region)
        ->FillQuadratureFunction(*global_tangent);
    CALI_MARK_END("umat_postprocessing");
}

bool AbaqusUmatModel::SetUmatLibrary(const std::filesystem::path& library_path,
                                     exaconstit::LoadStrategy strategy) {
    if (use_dynamic_loading) {
        UnloadUmatLibrary();
    }

    umat_library_path = library_path;
    load_strategy = strategy;
    use_dynamic_loading = !library_path.empty();
    umat_function = nullptr;
    umat_batch_function = nullptr;

    if (use_dynamic_loading && strategy == exaconstit::LoadStrategy::PERSISTENT) {
        return LoadUmatLibrary();
    }
    return true;
}

bool AbaqusUmatModel::ReloadUmatLibrary() {
    if (!use_dynamic_loading) {
        return true;
    }
    UnloadUmatLibrary();
    return LoadUmatLibrary();
}

bool AbaqusUmatModel::LoadUmatLibrary() {
    if (!use_dynamic_loading || UmatLoaded()) {
        return true; // Already loaded or not using dynamic loading
    }

    std::string error;
    if (m_batched) {
        umat_batch_function = exaconstit::UnifiedUmatLoader::LoadUmatBatch(
            umat_library_path.string(), load_strategy, umat_function_name);
        error = exaconstit::UnifiedUmatLoader::GetLastBatchError();
    } else {
        umat_function = exaconstit::UnifiedUmatLoader::LoadUmat(
            umat_library_path.string(), load_strategy, umat_function_name);
        error = exaconstit::UnifiedUmatLoader::GetLastError();
    }

    if (!UmatLoaded()) {
        MFEM_ABORT_0("Failed to load " << (m_batched ? "batched " : "") << "UMAT '"
                                       << umat_function_name << "' from library "
                                       << umat_library_path.string() << "\nError: " << error);
        return false;
    }
    return true;
}

void AbaqusUmatModel::UnloadUmatLibrary() {
    if (use_dynamic_loading && !umat_library_path.empty()) {
        if (m_batched) {
            exaconstit::UnifiedUmatLoader::UnloadUmatBatch(umat_library_path.string(),
                                                           umat_function_name);
        } else {
            exaconstit::UnifiedUmatLoader::UnloadUmat(umat_library_path.string());
        }
        umat_function = nullptr;
        umat_batch_function = nullptr;
    }
}
