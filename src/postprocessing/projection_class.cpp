#include "postprocessing/projection_class.hpp"

#include "utilities/rotations.hpp"
#include "utilities/unified_logger.hpp"

#include "ECMech_const.h"
#include "SNLS_linalg.h"

//=============================================================================
// GEOMETRY PROJECTIONS
//=============================================================================
namespace {
/**
 * @brief Integration rule for element geometry quantities
 *
 * Chosen from the mesh geometry (its nodal order when the mesh is curved) rather than from the
 * output field, so zone (order 0) and quadrature (mesh order) output give the same centroids
 * and volumes.
 */
const mfem::IntegrationRule& GeometryIntRule(const mfem::ParFiniteElementSpace& fes) {
    const auto* mesh = fes.GetMesh();
    const auto* nodal_fes = mesh->GetNodalFESpace();
    const int geom_order = nodal_fes ? nodal_fes->GetMaxElementOrder() : 1;
    return mfem::IntRules.Get(fes.GetTypicalFE()->GetGeomType(), 2 * geom_order + 1);
}
} // namespace

void CentroidProjection::ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                         const VizPointMap& points) {
    auto* fes = grid_function->ParFESpace();
    auto* mesh = fes->GetMesh();
    const mfem::IntegrationRule* ir = &GeometryIntRule(*fes);

    const int nqpts = ir->GetNPoints();
    const int nelems_mesh = mesh->GetNE();
    const int nelems = points.GetNumElements();
    const int space_dim = mesh->SpaceDimension();
    const int gf_vdim = grid_function->VectorDim();

    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);

    const auto W = mfem::Reshape(ir->GetWeights().Read(), nqpts);
    const auto detJ = mfem::Reshape(geom->detJ.Read(), nqpts, nelems_mesh);
    const auto x_coords = mfem::Reshape(geom->X.Read(), nqpts, space_dim, nelems_mesh);
    const auto out_elem = points.ReadOutElement();
    const auto elem_offsets = points.ReadElementOffsets();
    const auto dest = points.ReadDest();
    auto centroid = mfem::Reshape(grid_function->ReadWrite(), gf_vdim, fes->GetNDofs());

    // Calculate element centroids and write them on every output node of the element
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int ie = out_elem(i);
        double vol = 0.0;
        double x_sum[3] = {0.0, 0.0, 0.0};

        for (int iq = 0; iq < nqpts; ++iq) {
            const double wt = detJ(iq, ie) * W(iq);
            vol += wt;
            for (int iv = 0; iv < space_dim; ++iv) {
                x_sum[iv] += x_coords(iq, iv, ie) * wt;
            }
        }

        const double inv_vol = 1.0 / vol;
        for (int k = elem_offsets(i); k < elem_offsets(i + 1); ++k) {
            for (int iv = 0; iv < space_dim; ++iv) {
                centroid(iv, dest(k)) = x_sum[iv] * inv_vol;
            }
        }
    });
}

void VolumeProjection::ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                       const VizPointMap& points) {
    auto* fes = grid_function->ParFESpace();
    auto* mesh = fes->GetMesh();
    const mfem::IntegrationRule* ir = &GeometryIntRule(*fes);

    const int nqpts = ir->GetNPoints();
    const int nelems_mesh = mesh->GetNE();
    const int nelems = points.GetNumElements();

    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::DETERMINANTS);

    const auto W = mfem::Reshape(ir->GetWeights().Read(), nqpts);
    const auto detJ = mfem::Reshape(geom->detJ.Read(), nqpts, nelems_mesh);
    const auto out_elem = points.ReadOutElement();
    const auto elem_offsets = points.ReadElementOffsets();
    const auto dest = points.ReadDest();
    auto volume = mfem::Reshape(
        grid_function->ReadWrite(), grid_function->VectorDim(), fes->GetNDofs());

    // Calculate element volumes and write them on every output node of the element
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int ie = out_elem(i);
        double vol = 0.0;
        for (int iq = 0; iq < nqpts; ++iq) {
            vol += detJ(iq, ie) * W(iq);
        }
        for (int k = elem_offsets(i); k < elem_offsets(i + 1); ++k) {
            volume(0, dest(k)) = vol;
        }
    });
}

//=============================================================================
// STRESS-BASED PROJECTIONS
//=============================================================================
//
// Every kernel below has the same shape: for each entry k of the region's VizPointMap, read the
// stress at source point src(k) and write the result on output node dst(k). Both sides are
// mfem::Reshape views sized (components, points), so a wrong index is caught by the view's bounds
// check in debug builds. This one loop serves both zone output (source = element average, one
// node per element) and quadrature output (source = nearest quadrature point, one node per H1
// node location).

namespace {
/// (components, points) view of a region's quadrature function data
mfem::DeviceTensor<2, const double> ReadPointView(const mfem::expt::PartialQuadratureFunction& qf) {
    const int vdim = qf.GetVDim();
    return mfem::Reshape(qf.Read(), vdim, qf.Size() / vdim);
}

/// (components, nodes) view of an output grid function (byVDIM ordering)
mfem::DeviceTensor<2, double> ReadWriteNodeView(mfem::ParGridFunction& gf) {
    return mfem::Reshape(gf.ReadWrite(), gf.VectorDim(), gf.ParFESpace()->GetNDofs());
}
} // namespace

void CauchyStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> stress_gf,
    const VizPointMap& points) {
    const auto src = points.ReadSource();
    const auto dst = points.ReadDest();
    const auto stress = ReadPointView(*stress_qf);
    auto stress_out = ReadWriteNodeView(*stress_gf);

    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        for (int i = 0; i < 6; i++) {
            stress_out(i, dst(k)) = stress(i, src(k));
        }
    });
}

void VonMisesStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> von_mises,
    const VizPointMap& points) {
    const auto src = points.ReadSource();
    const auto dst = points.ReadDest();
    const auto stress = ReadPointView(*stress_qf);
    auto von_mises_out = ReadWriteNodeView(*von_mises);

    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        const int s = src(k);

        double term1 = stress(0, s) - stress(1, s);
        double term2 = stress(1, s) - stress(2, s);
        double term3 = stress(2, s) - stress(0, s);
        double term4 = stress(3, s) * stress(3, s) + stress(4, s) * stress(4, s) +
                       stress(5, s) * stress(5, s);

        term1 *= term1;
        term2 *= term2;
        term3 *= term3;
        term4 *= 6.0;

        von_mises_out(0, dst(k)) = sqrt(0.5 * (term1 + term2 + term3 + term4));
    });
}

void HydrostaticStressProjection::ProjectStress(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
    std::shared_ptr<mfem::ParGridFunction> hydro_static,
    const VizPointMap& points) {
    const auto src = points.ReadSource();
    const auto dst = points.ReadDest();
    const auto stress = ReadPointView(*stress_qf);
    auto hydro_static_out = ReadWriteNodeView(*hydro_static);

    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        const int s = src(k);
        hydro_static_out(0, dst(k)) = ecmech::onethird *
                                      (stress(0, s) + stress(1, s) + stress(2, s));
    });
}

//=============================================================================
// STATE VARIABLE PROJECTIONS
//=============================================================================

void StateVariableProjection::Execute(std::shared_ptr<SimulationState> sim_state,
                                      std::shared_ptr<mfem::ParGridFunction> state_gf,
                                      const VizPointMap& points,
                                      int region) {
    // Element averages (zone output) or end-of-step quadrature values (quadrature output)
    auto state_qf = sim_state->GetQuadratureFunction(points.SourceName("state_var"), region);
    if (!state_qf)
        return; // Region doesn't have state variables

    const int vdim = state_qf->GetVDim();
    m_component_length = (m_component_length == -1) ? vdim : m_component_length;

    MFEM_VERIFY_0((m_component_length + m_component_index) <= vdim,
                  "StateVariableProjection provided a length and index that pushes us past the "
                  "state variable length");
    MFEM_VERIFY_0(m_component_length <= state_gf->VectorDim(),
                  "StateVariableProjection provided length is greater than the gridfunction "
                  "vector length");

    const auto src = points.ReadSource();
    const auto dst = points.ReadDest();
    const auto state = ReadPointView(*state_qf);
    auto state_out = ReadWriteNodeView(*state_gf);

    // Copy the requested slice of the state variables from each source point to its node
    const auto component_length = m_component_length;
    const auto component_index = m_component_index;
    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        for (int j = 0; j < component_length; j++) {
            state_out(j, dst(k)) = state(j + component_index, src(k));
        }
    });

    // Apply any post-processing
    PostProcessStateVariable(state_gf, points);
}

void NNegStateProjection::PostProcessStateVariable(
    std::shared_ptr<mfem::ParGridFunction> grid_function, const VizPointMap& points) const {
    const int vdim = grid_function->VectorDim();
    const auto dst = points.ReadDest();
    auto values = ReadWriteNodeView(*grid_function);

    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        for (int j = 0; j < vdim; j++) {
            values(j, dst(k)) = fmax(values(j, dst(k)), 0.0);
        }
    });
}

void XtalOrientationProjection::PostProcessStateVariable(
    std::shared_ptr<mfem::ParGridFunction> grid_function, const VizPointMap& points) const {
    const auto dst = points.ReadDest();
    auto quats = ReadWriteNodeView(*grid_function);

    // Scale each quaternion back to unit length. A zero length can only come from averaging
    // quaternions of opposite sign (q and -q are the same rotation) in zone output; leave such a
    // value at zero rather than writing NaN.
    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        const int d = dst(k);
        const double norm = sqrt(quats(0, d) * quats(0, d) + quats(1, d) * quats(1, d) +
                                 quats(2, d) * quats(2, d) + quats(3, d) * quats(3, d));
        const double inv_norm = (norm > 0.0) ? 1.0 / norm : 0.0;
        for (int i = 0; i < 4; i++) {
            quats(i, d) *= inv_norm;
        }
    });
}

void ElasticStrainProjection::Execute(std::shared_ptr<SimulationState> sim_state,
                                      std::shared_ptr<mfem::ParGridFunction> elastic_strain_gf,
                                      const VizPointMap& points,
                                      int region) {
    // Element averages (zone output) or end-of-step quadrature values (quadrature output)
    auto state_qf = sim_state->GetQuadratureFunction(points.SourceName("state_var"), region);
    if (!state_qf)
        return; // Region doesn't have state variables

    const int vdim = state_qf->GetVDim();

    // The three state variable slices read below: deviatoric elastic strain in the lattice frame
    // (5 values), the lattice orientation quaternion (4) and the relative volume (1). The output
    // is the full symmetric strain in the sample frame (6 values).
    const auto estrain_slice = sim_state->GetQuadratureFunctionStatePair("elastic_strain", region);
    const auto quats_slice = sim_state->GetQuadratureFunctionStatePair("quats", region);
    const auto rel_vol_slice = sim_state->GetQuadratureFunctionStatePair("relative_volume", region);
    const int estrain_ind = estrain_slice.first;
    const int quats_ind = quats_slice.first;
    const int rel_vol_ind = rel_vol_slice.first;

    MFEM_VERIFY_0(estrain_ind >= 0 && estrain_slice.second == ecmech::ntvec &&
                      estrain_ind + estrain_slice.second <= vdim,
                  "ElasticStrainProjection: elastic strain slice is missing or outside the state "
                  "variables");
    MFEM_VERIFY_0(quats_ind >= 0 && quats_slice.second == 4 &&
                      quats_ind + quats_slice.second <= vdim,
                  "ElasticStrainProjection: quaternion slice is missing or outside the state "
                  "variables");
    MFEM_VERIFY_0(rel_vol_ind >= 0 && rel_vol_slice.second == 1 &&
                      rel_vol_ind + rel_vol_slice.second <= vdim,
                  "ElasticStrainProjection: relative volume slice is missing or outside the state "
                  "variables");
    MFEM_VERIFY_0(elastic_strain_gf->VectorDim() >= 6,
                  "ElasticStrainProjection: output grid function needs 6 components");

    const auto src = points.ReadSource();
    const auto dst = points.ReadDest();
    const auto state = ReadPointView(*state_qf);
    auto strain = ReadWriteNodeView(*elastic_strain_gf);

    mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
        const int s = src(k);
        // Unit quaternion: element averaging (zone output) shortens it, so renormalize before
        // building the rotation.
        double quats[4] = {state(quats_ind + 0, s),
                           state(quats_ind + 1, s),
                           state(quats_ind + 2, s),
                           state(quats_ind + 3, s)};
        const double quat_norm = sqrt(quats[0] * quats[0] + quats[1] * quats[1] +
                                      quats[2] * quats[2] + quats[3] * quats[3]);
        const double inv_norm = (quat_norm > 0.0) ? 1.0 / quat_norm : 0.0;
        for (int i = 0; i < 4; i++) {
            quats[i] *= inv_norm;
        }

        // Full elastic strain in the lattice frame (symmetric 3x3, so storage order is irrelevant)
        double strain_lat_data[3 * 3] = {};
        auto strain_lat = mfem::Reshape(strain_lat_data, 3, 3);
        const double t1 = ecmech::sqr2i * state(estrain_ind + 0, s);
        const double t2 = ecmech::sqr6i * state(estrain_ind + 1, s);
        //
        // Volume strain is ln(V^e_mean) term aka ln(relative volume)
        // Our plastic deformation has a det(1) aka no change in volume change
        const double elas_vol_strain = log(state(rel_vol_ind, s));
        // We output elastic strain formulation such that the relationship
        // between V^e and \varepsilon is just V^e = I + \varepsilon
        strain_lat(0, 0) = (t1 - t2) + elas_vol_strain;                                  // 11
        strain_lat(1, 1) = (-t1 - t2) + elas_vol_strain;                                 // 22
        strain_lat(2, 2) = ecmech::sqr2b3 * state(estrain_ind + 1, s) + elas_vol_strain; // 33
        strain_lat(1, 2) = ecmech::sqr2i * state(estrain_ind + 4, s);                    // 23
        strain_lat(2, 0) = ecmech::sqr2i * state(estrain_ind + 3, s);                    // 31
        strain_lat(0, 1) = ecmech::sqr2i * state(estrain_ind + 2, s);                    // 12

        strain_lat(2, 1) = strain_lat(1, 2);
        strain_lat(0, 2) = strain_lat(2, 0);
        strain_lat(1, 0) = strain_lat(0, 1);

        // Rotate to the sample frame. rotMatrix works on row-major 3x3 data while the view is
        // column-major, so entry (row i, column j) of the result is strain_samp(j, i). The result
        // is symmetric only to round-off, so read the same entries as the row-major layout.
        double rmat[3 * 3] = {};
        double strain_samp_data[3 * 3] = {};
        Quat2RMat(quats, rmat);
        snls::linalg::rotMatrix<3, false>(strain_lat_data, rmat, strain_samp_data);
        const auto strain_samp = mfem::Reshape(strain_samp_data, 3, 3);

        const int d = dst(k);
        strain(0, d) = strain_samp(0, 0); // 11
        strain(1, d) = strain_samp(1, 1); // 22
        strain(2, d) = strain_samp(2, 2); // 33
        strain(3, d) = strain_samp(2, 1); // 23
        strain(4, d) = strain_samp(2, 0); // 13
        strain(5, d) = strain_samp(1, 0); // 12
    });
}
