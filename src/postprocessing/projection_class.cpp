#include "postprocessing/projection_class.hpp"

#include "utilities/rotations.hpp"
#include "utilities/unified_logger.hpp"

#include "ECMech_const.h"
#include "SNLS_linalg.h"

//=============================================================================
// GEOMETRY PROJECTIONS
//=============================================================================
void CentroidProjection::ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function) {
    auto* fes = grid_function->ParFESpace();
    auto* mesh = fes->GetMesh();
    const mfem::FiniteElement& el = *fes->GetFE(0);
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    const int nqpts = ir->GetNPoints();
    const int nelems = fes->GetNE();
    const int vdim = mesh->SpaceDimension();

    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);

    const double* W = ir->GetWeights().Read();
    const double* const detJ = geom->detJ.Read();
    const auto x_coords = mfem::Reshape(geom->X.Read(), nqpts, vdim, nelems);

    double* centroid_data = grid_function->ReadWrite();

    // Calculate element centroids
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int ie) {
        double vol = 0.0;
        for (int iv = 0; iv < vdim; ++iv) {
            centroid_data[ie * vdim + iv] = 0.0;
        }

        for (int iq = 0; iq < nqpts; ++iq) {
            const double wt = detJ[ie * nqpts + iq] * W[iq];
            vol += wt;
            for (int iv = 0; iv < vdim; ++iv) {
                const double coord = x_coords(iq, iv, ie);
                centroid_data[ie * vdim + iv] += coord * wt;
            }
        }

        const double inv_vol = 1.0 / vol;
        for (int iv = 0; iv < vdim; ++iv) {
            centroid_data[ie * vdim + iv] *= inv_vol;
        }
    });
}

void VolumeProjection::ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function) {
    auto* fes = grid_function->ParFESpace();
    auto* mesh = fes->GetMesh();
    const mfem::FiniteElement& el = *fes->GetFE(0);
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    const int nqpts = ir->GetNPoints();
    const int nelems = fes->GetNE();

    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    const double* const W = ir->GetWeights().Read();
    const double* const detJ = geom->detJ.Read();

    double* volume_data = grid_function->ReadWrite();

    // Calculate element volumes
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int ie) {
        double vol = 0.0;
        for (int iq = 0; iq < nqpts; ++iq) {
            vol += detJ[ie * nqpts + iq] * W[iq];
        }
        volume_data[ie] = vol;
    });
}

//=============================================================================
// STRESS-BASED PROJECTIONS
//=============================================================================

void CauchyStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> stress_gf,
    mfem::Array<int>& qpts2mesh) {
    // Get stress data and compute Von Mises
    const int nelems = stress_gf->ParFESpace()->GetNE();
    const auto part_quad_space = stress_qf->GetPartialSpaceShared();
    const int local_nelems = part_quad_space->GetNE();

    const auto l2g = qpts2mesh.Read();
    const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
    auto stress_gf_data = mfem::Reshape(stress_gf->Write(), 6, nelems);

    // Compute element-averaged Von Mises stress
    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_idx = l2g[ie];

        stress_gf_data(0, global_idx) = stress_data(0, ie);
        stress_gf_data(1, global_idx) = stress_data(1, ie);
        stress_gf_data(2, global_idx) = stress_data(2, ie);
        stress_gf_data(3, global_idx) = stress_data(3, ie);
        stress_gf_data(4, global_idx) = stress_data(4, ie);
        stress_gf_data(5, global_idx) = stress_data(5, ie);
    });
}

void VonMisesStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> von_mises,
    mfem::Array<int>& qpts2mesh) {
    // Get stress data and compute Von Mises
    const int nelems = von_mises->ParFESpace()->GetNE();
    const auto part_quad_space = stress_qf->GetPartialSpaceShared();
    const int local_nelems = part_quad_space->GetNE();

    const auto l2g = qpts2mesh.Read();
    const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
    auto von_mises_data = mfem::Reshape(von_mises->Write(), nelems);

    // Compute element-averaged Von Mises stress
    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_idx = l2g[ie];

        double term1 = stress_data(0, ie) - stress_data(1, ie);
        double term2 = stress_data(1, ie) - stress_data(2, ie);
        double term3 = stress_data(2, ie) - stress_data(0, ie);
        double term4 = stress_data(3, ie) * stress_data(3, ie) +
                       stress_data(4, ie) * stress_data(4, ie) +
                       stress_data(5, ie) * stress_data(5, ie);

        term1 *= term1;
        term2 *= term2;
        term3 *= term3;
        term4 *= 6.0;

        von_mises_data(global_idx) = sqrt(0.5 * (term1 + term2 + term3 + term4));
    });
}

void HydrostaticStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> hydro_static,
    mfem::Array<int>& qpts2mesh) {
    // Get stress data and compute Von Mises
    const int nelems = hydro_static->ParFESpace()->GetNE();
    const auto part_quad_space = stress_qf->GetPartialSpaceShared();
    const int local_nelems = part_quad_space->GetNE();

    const auto l2g = qpts2mesh.Read();
    const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
    auto hydro_static_data = mfem::Reshape(hydro_static->Write(), nelems);

    // Compute element-averaged Von Mises stress
    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_idx = l2g[ie];

        hydro_static_data(global_idx) = ecmech::onethird *
                                        (stress_data(0, ie) + stress_data(1, ie) +
                                         stress_data(2, ie));
    });
}

//=============================================================================
// STATE VARIABLE PROJECTIONS
//=============================================================================

void StateVariableProjection::Execute(std::shared_ptr<SimulationState> sim_state,
                                      std::shared_ptr<mfem::ParGridFunction> state_gf,
                                      mfem::Array<int>& qpts2mesh,
                                      int region) {
    // Get state variable quadrature function for this region
    auto state_qf = sim_state->GetQuadratureFunction("state_var_avg", region);
    if (!state_qf)
        return; // Region doesn't have state variables

    // Project the specific component(s)
    const int nelems = state_gf->ParFESpace()->GetNE();
    const auto part_quad_space = state_qf->GetPartialSpaceShared();
    const int local_nelems = part_quad_space->GetNE();
    const int vdim = state_qf->GetVDim();
    m_component_length = (m_component_length == -1) ? vdim : m_component_length;

    MFEM_VERIFY_0((m_component_length + m_component_index) <= vdim,
                  "StateVariableProjection provided a length and index that pushes us past the "
                  "state variable length");
    MFEM_VERIFY_0(m_component_length <= state_gf->VectorDim(),
                  "StateVariableProjection provided length is greater than the gridfunction "
                  "vector length");

    const auto l2g = qpts2mesh.Read();
    const auto state_qf_data = mfem::Reshape(state_qf->Read(), vdim, local_nelems);
    auto state_gf_data = mfem::Reshape(state_gf->Write(), state_gf->VectorDim(), nelems);

    // Compute element-averaged Von Mises stress
    const auto component_length = m_component_length;
    const auto component_index = m_component_index;
    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_idx = l2g[ie];
        for (int j = 0; j < component_length; j++) {
            state_gf_data(j, global_idx) = state_qf_data(j + component_index, ie);
        }
    });

    // Apply any post-processing
    PostProcessStateVariable(state_gf, part_quad_space, qpts2mesh);
}

void NNegStateProjection::PostProcessStateVariable(
    std::shared_ptr<mfem::ParGridFunction> grid_function,
    [[maybe_unused]] std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
    [[maybe_unused]] mfem::Array<int>& qpts2mesh) const {
    auto data = grid_function->Write();
    const int local_nelems = grid_function->Size();

    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int i) {
        data[i] = fmax(data[i], 0.0);
    });
}

void XtalOrientationProjection::PostProcessStateVariable(
    std::shared_ptr<mfem::ParGridFunction> grid_function,
    std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
    mfem::Array<int>& qpts2mesh) const {
    const int nelems = grid_function->ParFESpace()->GetNE();
    auto ori = mfem::Reshape(grid_function->Write(), grid_function->VectorDim(), nelems);
    const auto l2g = qpts2mesh.Read();
    const int local_nelems = qspace->GetNE();

    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int ie = l2g[i];
        const double inv_norm = 1.0 / (ori(0, ie) * ori(0, ie) + ori(1, ie) * ori(1, ie) +
                                       ori(2, ie) * ori(2, ie) + ori(3, ie) * ori(3, ie));

        ori(0, ie) = ori(0, ie) * inv_norm;
        ori(1, ie) = ori(1, ie) * inv_norm;
        ori(2, ie) = ori(2, ie) * inv_norm;
        ori(3, ie) = ori(3, ie) * inv_norm;
    });
}

void ElasticStrainProjection::Execute(std::shared_ptr<SimulationState> sim_state,
                                      std::shared_ptr<mfem::ParGridFunction> elastic_strain_gf,
                                      mfem::Array<int>& qpts2mesh,
                                      int region) {
    // Get state variable quadrature function for this region
    auto state_qf = sim_state->GetQuadratureFunction("state_var_avg", region);
    if (!state_qf)
        return; // Region doesn't have state variables

    const int nelems = elastic_strain_gf->ParFESpace()->GetNE();
    const auto part_quad_space = state_qf->GetPartialSpaceShared();

    const auto l2g = qpts2mesh.Read();
    const int local_nelems = part_quad_space->GetNE();
    const int vdim = state_qf->GetVDim();
    const int gf_vdim = elastic_strain_gf->VectorDim();
    m_component_length = (m_component_length == -1) ? vdim : m_component_length;

    MFEM_VERIFY_0((m_component_length + m_component_index) <= vdim,
                  "ElasticStrainProjection provided a length and index that pushes us past the "
                  "state variable length");
    MFEM_VERIFY_0(m_component_length <= gf_vdim,
                  "ElasticStrainProjection provided length is greater than the gridfunction "
                  "vector length");

    const int estrain_ind =
        sim_state->GetQuadratureFunctionStatePair("elastic_strain", region).first;
    const int quats_ind = sim_state->GetQuadratureFunctionStatePair("quats", region).first;
    const int rel_vol_ind =
        sim_state->GetQuadratureFunctionStatePair("relative_volume", region).first;

    auto state_vars = sim_state->GetQuadratureFunction("state_var_end", region)->Read();
    auto strain = mfem::Reshape(elastic_strain_gf->Write(), gf_vdim, nelems);

    mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE(int ie) {
        const int global_idx = l2g[ie];

        const auto strain_lat = &state_vars[ie * vdim + estrain_ind];
        const auto quats = &state_vars[ie * vdim + quats_ind];
        const auto rel_vol = state_vars[ie * vdim + rel_vol_ind];
        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain;                      // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain;                     // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4];                    // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3];                    // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2];                    // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};

            Quat2RMat(quats, rmat);
            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];

            strain(0, global_idx) = strain_m[0][0];
            strain(1, global_idx) = strain_m[1][1];
            strain(2, global_idx) = strain_m[2][2];
            strain(3, global_idx) = strain_m[1][2];
            strain(4, global_idx) = strain_m[0][2];
            strain(5, global_idx) = strain_m[0][1];
        }
    });
}