#include "postprocessing/surface_quantities.hpp"

#include "postprocessing/postprocessing_file_manager.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem/general/forall.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <unordered_map>

namespace {

/// Number of per-face partial sums written by FaceSumsKernel
constexpr int NUM_FACE_SUMS = 10;
// Layout of the per-face partial sums
constexpr int SUM_DA = 0;   // da
constexpr int SUM_NDA = 1;  // n da (3)
constexpr int SUM_XDA = 4;  // x da (3)
constexpr int SUM_DISP = 7; // u dA0 (3)

/// Number of per-node values written by NodeSumsKernel: force (3), x cross force (3)
constexpr int NUM_NODE_SUMS = 6;

/**
 * @brief Per-face partial sums of the surface integrals
 */
void FaceSumsKernel(const int nfaces,
                    const int* face_qoff,
                    const double* x,
                    const double* nda,
                    const double* da,
                    const double* disp,
                    const double* dA0,
                    double* sums) {
    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int i) {
        double s[NUM_FACE_SUMS];
        for (int k = 0; k < NUM_FACE_SUMS; k++) {
            s[k] = 0.0;
        }
        for (int q = face_qoff[i]; q < face_qoff[i + 1]; q++) {
            s[SUM_DA] += da[q];
            for (int c = 0; c < 3; c++) {
                s[SUM_NDA + c] += nda[3 * q + c];
                s[SUM_XDA + c] += x[3 * q + c] * da[q];
                s[SUM_DISP + c] += disp[3 * q + c] * dA0[q];
            }
        }
        for (int k = 0; k < NUM_FACE_SUMS; k++) {
            sums[NUM_FACE_SUMS * i + k] = s[k];
        }
    });
}

/**
 * @brief Gather each surface node's weighted internal force and its moment about the origin
 */
void NodeSumsKernel(const int nnodes,
                    const int* node_tdofs,
                    const double* weights,
                    const double* force,
                    const double* coords,
                    double* sums) {
    mfem::forall(nnodes, [=] MFEM_HOST_DEVICE(int j) {
        const int* t = &node_tdofs[3 * j];
        const double* w = &weights[3 * j];
        const double f[3] = {w[0] * force[t[0]], w[1] * force[t[1]], w[2] * force[t[2]]};
        const double x[3] = {coords[t[0]], coords[t[1]], coords[t[2]]};
        double* s = &sums[NUM_NODE_SUMS * j];
        s[0] = f[0];
        s[1] = f[1];
        s[2] = f[2];
        s[3] = x[1] * f[2] - x[2] * f[1];
        s[4] = x[2] * f[0] - x[0] * f[2];
        s[5] = x[0] * f[1] - x[1] * f[0];
    });
}

/// Mesh boundary attributes present on any rank
std::set<int> GlobalBoundaryAttributes(const mfem::ParMesh& mesh) {
    // ParMesh::bdr_attributes already holds the attributes of all ranks
    std::set<int> attrs;
    for (int i = 0; i < mesh.bdr_attributes.Size(); i++) {
        attrs.insert(mesh.bdr_attributes[i]);
    }
    return attrs;
}

double Norm3(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} // namespace

BoundarySurface::BoundarySurface(std::shared_ptr<mfem::ParMesh> mesh,
                                 std::shared_ptr<mfem::ParFiniteElementSpace> fes,
                                 const std::vector<int>& boundary_ids,
                                 int order)
    : m_mesh(std::move(mesh)), m_fes(std::move(fes)), m_order(order),
      m_num_surfaces(static_cast<int>(boundary_ids.size())) {
    const int dim = m_mesh->Dimension();
    MFEM_VERIFY(dim == 3 && m_fes->GetVDim() == 3,
                "Surface quantities require a 3D mesh and a 3-component H1 space");

    std::map<int, int> attr_to_surface;
    for (int s = 0; s < m_num_surfaces; s++) {
        attr_to_surface[boundary_ids[static_cast<size_t>(s)]] = s;
    }

    std::vector<int> face_surface, face_table, face_qoff, face_vdof_off, vdofs;
    face_qoff.push_back(0);
    face_vdof_off.push_back(0);
    mfem::Array<int> elem_vdofs;
    for (int be = 0; be < m_mesh->GetNBE(); be++) {
        const auto it = attr_to_surface.find(m_mesh->GetBdrAttribute(be));
        if (it == attr_to_surface.end()) {
            continue;
        }
        const int face = m_mesh->GetBdrElementFaceIndex(be);
        const auto info = m_mesh->GetFaceInformation(face);
        // A boundary element always has its first adjacent element on this rank, and for an
        // interior face carrying a boundary attribute either side gives the same geometry
        const int elem = info.element[0].index;
        const int table = GetTable(m_mesh->GetElementBaseGeometry(elem),
                                   m_mesh->GetFaceGeometry(face),
                                   info.element[0].local_face_id);

        m_fes->GetElementVDofs(elem, elem_vdofs);
        MFEM_VERIFY(elem_vdofs.Size() == 3 * m_tab_ndof[table],
                    "Unexpected number of H1 DOFs for a surface element");
        for (int k = 0; k < elem_vdofs.Size(); k++) {
            // H1 DOFs carry no sign, but decode the convention anyway
            const int vd = elem_vdofs[k];
            vdofs.push_back(vd >= 0 ? vd : -1 - vd);
        }

        face_surface.push_back(it->second);
        face_table.push_back(table);
        face_qoff.push_back(face_qoff.back() + m_tab_nfq[table]);
        face_vdof_off.push_back(static_cast<int>(vdofs.size()));
    }
    m_num_face_points = face_qoff.back();

    auto to_array = [](const std::vector<int>& src, mfem::Array<int>& dst) {
        dst.SetSize(static_cast<int>(src.size()));
        std::copy(src.begin(), src.end(), dst.HostWrite());
    };
    to_array(face_surface, m_face_surface);
    to_array(face_table, m_face_table);
    to_array(face_qoff, m_face_qoff);
    to_array(face_vdof_off, m_face_vdof_off);
    to_array(vdofs, m_vdofs);
}

int BoundarySurface::GetTable(mfem::Geometry::Type elem_geom,
                              mfem::Geometry::Type face_geom,
                              int local_face) {
    const auto key = std::make_pair(static_cast<int>(elem_geom), local_face);
    const auto found = m_table_ids.find(key);
    if (found != m_table_ids.end()) {
        return found->second;
    }

    const mfem::IntegrationRule& face_ir = mfem::IntRules.Get(face_geom, 2 * m_order + 1);
    const int nfq = face_ir.GetNPoints();

    const mfem::FiniteElement* h1_fe = m_fes->FEColl()->FiniteElementForGeometry(elem_geom);
    MFEM_VERIFY(h1_fe,
                "Surface quantities do not support element geometry "
                    << mfem::Geometry::Name[elem_geom]);
    const int ndof = h1_fe->GetDof();

    // Map from the face reference element to this element's reference element. The local
    // face's own orientation is used (orientation 0), since faces are integrated in the
    // element's parametrization.
    mfem::IsoparametricTransformation loc;
    m_mesh->GetLocalFaceTransformation(mfem::Element::TypeFromGeometry(face_geom),
                                       mfem::Element::TypeFromGeometry(elem_geom),
                                       loc,
                                       64 * local_face);

    // Outward reference normal: the map is affine, so the face tangents are constant. The sign
    // is fixed by pointing away from the (convex) reference element's center.
    mfem::IntegrationPoint face_center = mfem::Geometries.GetCenter(face_geom);
    loc.SetIntPoint(&face_center);
    const mfem::DenseMatrix& jloc = loc.Jacobian();
    double nref[3] = {jloc(1, 0) * jloc(2, 1) - jloc(2, 0) * jloc(1, 1),
                      jloc(2, 0) * jloc(0, 1) - jloc(0, 0) * jloc(2, 1),
                      jloc(0, 0) * jloc(1, 1) - jloc(1, 0) * jloc(0, 1)};
    const double ref_jac = Norm3(nref);
    mfem::Vector xi(3);
    loc.Transform(face_center, xi);
    const mfem::IntegrationPoint& elem_center = mfem::Geometries.GetCenter(elem_geom);
    const double outward = (xi(0) - elem_center.x) * nref[0] + (xi(1) - elem_center.y) * nref[1] +
                           (xi(2) - elem_center.z) * nref[2];
    const double sign = (outward < 0.0) ? -1.0 : 1.0;
    for (double& n : nref) {
        n *= sign / ref_jac;
    }

    mfem::DenseMatrix shape_n(nfq, ndof);
    std::vector<double> dn(static_cast<size_t>(nfq * ndof * 3));
    std::vector<double> weights(static_cast<size_t>(nfq));
    mfem::Vector shape_h1(ndof);
    mfem::DenseMatrix dshape_h1(ndof, 3);
    for (int q = 0; q < nfq; q++) {
        const mfem::IntegrationPoint& ip_f = face_ir.IntPoint(q);
        loc.Transform(ip_f, xi);
        mfem::IntegrationPoint ip_e;
        ip_e.Set3(xi(0), xi(1), xi(2));

        h1_fe->CalcShape(ip_e, shape_h1);
        h1_fe->CalcDShape(ip_e, dshape_h1);
        for (int a = 0; a < ndof; a++) {
            shape_n(q, a) = shape_h1(a);
            for (int d = 0; d < 3; d++) {
                dn[static_cast<size_t>((q * ndof + a) * 3 + d)] = dshape_h1(a, d);
            }
        }
        weights[static_cast<size_t>(q)] = ip_f.weight * ref_jac;
    }

    // Append to the flat tables
    const int table = m_tab_nfq.Size();
    m_tab_nfq.Append(nfq);
    m_tab_ndof.Append(ndof);
    m_tab_off_n.Append(m_tab_n.Size());
    m_tab_off_w.Append(m_tab_w.Size());

    for (int q = 0; q < nfq; q++) {
        for (int a = 0; a < ndof; a++) {
            m_tab_n.Append(shape_n(q, a));
        }
    }
    m_tab_dn.Append(dn.data(), nfq * ndof * 3);
    m_tab_w.Append(weights.data(), nfq);
    m_tab_nref.Append(nref, 3);

    m_table_ids.emplace(key, table);
    return table;
}

void BoundarySurface::ComputeGeometry(const mfem::Vector& coords,
                                      mfem::Vector& x,
                                      mfem::Vector& nda,
                                      mfem::Vector& da) const {
    const int nfaces = GetNumFaces();
    x.SetSize(3 * m_num_face_points);
    nda.SetSize(3 * m_num_face_points);
    da.SetSize(m_num_face_points);
    if (nfaces == 0) {
        return;
    }

    const auto* face_table = m_face_table.Read();
    const auto* face_qoff = m_face_qoff.Read();
    const auto* face_vdof_off = m_face_vdof_off.Read();
    const auto* vdofs = m_vdofs.Read();
    const auto* tab_nfq = m_tab_nfq.Read();
    const auto* tab_ndof = m_tab_ndof.Read();
    const auto* tab_off_n = m_tab_off_n.Read();
    const auto* tab_off_w = m_tab_off_w.Read();
    const double* tab_n = m_tab_n.Read();
    const double* tab_dn = m_tab_dn.Read();
    const double* tab_w = m_tab_w.Read();
    const double* tab_nref = m_tab_nref.Read();
    const double* X = coords.Read();
    double* xq = x.Write();
    double* nq = nda.Write();
    double* daq = da.Write();

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int i) {
        const int t = face_table[i];
        const int nfq = tab_nfq[t];
        const int ndof = tab_ndof[t];
        const int* dofs = &vdofs[face_vdof_off[i]];
        const double* nref = &tab_nref[3 * t];
        for (int q = 0; q < nfq; q++) {
            const double* shape = &tab_n[tab_off_n[t] + q * ndof];
            const double* dshape = &tab_dn[3 * (tab_off_n[t] + q * ndof)];
            double pos[3] = {0.0, 0.0, 0.0};
            // J(c, d) = d x_c / d xi_d
            double J[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
            for (int a = 0; a < ndof; a++) {
                for (int c = 0; c < 3; c++) {
                    // Element vdofs are grouped by component
                    const double xa = X[dofs[c * ndof + a]];
                    pos[c] += shape[a] * xa;
                    for (int d = 0; d < 3; d++) {
                        J[c][d] += xa * dshape[3 * a + d];
                    }
                }
            }
            // Nanson's formula: n da = det(J) J^{-T} n_ref dA_ref = cof(J) n_ref dA_ref
            double cof[3][3];
            cof[0][0] = J[1][1] * J[2][2] - J[1][2] * J[2][1];
            cof[0][1] = J[1][2] * J[2][0] - J[1][0] * J[2][2];
            cof[0][2] = J[1][0] * J[2][1] - J[1][1] * J[2][0];
            cof[1][0] = J[0][2] * J[2][1] - J[0][1] * J[2][2];
            cof[1][1] = J[0][0] * J[2][2] - J[0][2] * J[2][0];
            cof[1][2] = J[0][1] * J[2][0] - J[0][0] * J[2][1];
            cof[2][0] = J[0][1] * J[1][2] - J[0][2] * J[1][1];
            cof[2][1] = J[0][2] * J[1][0] - J[0][0] * J[1][2];
            cof[2][2] = J[0][0] * J[1][1] - J[0][1] * J[1][0];
            const double w = tab_w[tab_off_w[t] + q];
            const int iq = face_qoff[i] + q;
            double area_vec[3];
            for (int c = 0; c < 3; c++) {
                xq[3 * iq + c] = pos[c];
                area_vec[c] = cof[c][0] * nref[0] + cof[c][1] * nref[1] + cof[c][2] * nref[2];
                nq[3 * iq + c] = w * area_vec[c];
            }
            // Signed like the weight, so not |nda|
            daq[iq] = w * sqrt(area_vec[0] * area_vec[0] + area_vec[1] * area_vec[1] +
                               area_vec[2] * area_vec[2]);
        }
    });
}

void BoundarySurface::InterpolateH1(const mfem::Vector& field, mfem::Vector& values) const {
    const int nfaces = GetNumFaces();
    values.SetSize(3 * m_num_face_points);
    if (nfaces == 0) {
        return;
    }

    const auto* face_table = m_face_table.Read();
    const auto* face_qoff = m_face_qoff.Read();
    const auto* face_vdof_off = m_face_vdof_off.Read();
    const auto* vdofs = m_vdofs.Read();
    const auto* tab_nfq = m_tab_nfq.Read();
    const auto* tab_ndof = m_tab_ndof.Read();
    const auto* tab_off_n = m_tab_off_n.Read();
    const double* tab_n = m_tab_n.Read();
    const double* U = field.Read();
    double* uq = values.Write();

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int i) {
        const int t = face_table[i];
        const int nfq = tab_nfq[t];
        const int ndof = tab_ndof[t];
        const int* dofs = &vdofs[face_vdof_off[i]];
        for (int q = 0; q < nfq; q++) {
            const double* shape = &tab_n[tab_off_n[t] + q * ndof];
            const int iq = face_qoff[i] + q;
            for (int c = 0; c < 3; c++) {
                double val = 0.0;
                for (int a = 0; a < ndof; a++) {
                    val += shape[a] * U[dofs[c * ndof + a]];
                }
                uq[3 * iq + c] = val;
            }
        }
    });
}

SurfaceQuantities::SurfaceQuantities(std::shared_ptr<SimulationState> sim_state,
                                     const ExaOptions& options)
    : m_sim_state(std::move(sim_state)),
      m_boundary_ids(options.post_processing.surface_quantities.boundary_ids),
      m_moments(options.post_processing.surface_quantities.moments),
      m_output_frequency(options.post_processing.surface_quantities.output_frequency) {
    auto mesh = m_sim_state->GetMesh();
    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    MFEM_VERIFY(fes->GetOrdering() == mfem::Ordering::byNODES,
                "Surface quantities expect the mesh space ordered byNODES");
    m_ntdofs_per_comp = fes->GetTrueVSize() / fes->GetVDim();

    const auto attrs = GlobalBoundaryAttributes(*mesh);
    for (const int id : m_boundary_ids) {
        if (attrs.count(id) == 0) {
            std::ostringstream err;
            err << "PostProcessing.surface_quantities: boundary ID " << id
                << " is not a boundary attribute of the mesh. Available boundary attributes:";
            for (const int a : attrs) {
                err << " " << a;
            }
            MFEM_ABORT_0(err.str());
        }
    }

    m_surface = std::make_unique<BoundarySurface>(mesh, fes, m_boundary_ids, options.mesh.order);

    for (auto* v : {&m_dA0,
                    &m_x,
                    &m_nda,
                    &m_da,
                    &m_disp,
                    &m_face_sums,
                    &m_coords_t,
                    &m_node_sums,
                    &m_node_weights}) {
        v->UseDevice(true);
    }

    m_all_attrs.assign(attrs.begin(), attrs.end());
    BuildSurfaceNodes();

    // Reference surface: face-point weights dA0, area and vector area
    m_surface->ComputeGeometry(*m_sim_state->GetRefCoords(), m_x, m_nda, m_dA0);

    const int nsurf = m_surface->GetNumSurfaces();
    std::vector<double> ref_sums(static_cast<size_t>(4 * nsurf), 0.0);
    {
        const auto& face_surface = m_surface->GetFaceSurface();
        const auto& face_qoff = m_surface->GetFacePointOffsets();
        const double* nda = m_nda.HostRead();
        const double* dA0 = m_dA0.HostRead();
        const int* fs = face_surface.HostRead();
        const int* fq = face_qoff.HostRead();
        for (int i = 0; i < m_surface->GetNumFaces(); i++) {
            double* s = &ref_sums[static_cast<size_t>(4 * fs[i])];
            for (int q = fq[i]; q < fq[i + 1]; q++) {
                s[0] += dA0[q];
                s[1] += nda[3 * q];
                s[2] += nda[3 * q + 1];
                s[3] += nda[3 * q + 2];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, ref_sums.data(), 4 * nsurf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    m_ref_area.resize(static_cast<size_t>(nsurf));
    m_zero_normal.resize(static_cast<size_t>(nsurf));
    m_preambles.resize(static_cast<size_t>(nsurf));
    for (size_t s = 0; s < static_cast<size_t>(nsurf); s++) {
        const double area = ref_sums[4 * s];
        const double* vec_area = &ref_sums[4 * s + 1];
        const double vec_area_norm = Norm3(vec_area);
        // Projected area over true area: 1 for a planar surface, 0 for a closed one
        const double planarity = (area > 0.0) ? vec_area_norm / area : 0.0;
        m_ref_area[s] = area;
        m_zero_normal[s] = planarity < 1.0e-3;
        if (m_zero_normal[s]) {
            MFEM_WARNING_0("PostProcessing.surface_quantities: surface "
                           << m_boundary_ids[s]
                           << " is (nearly) closed, so it has no meaningful average normal; "
                              "its normal is written as zero");
        }

        std::ostringstream pre;
        pre << std::scientific << std::setprecision(8);
        pre << "# Surface quantities for mesh boundary attribute " << m_boundary_ids[s] << "\n";
        pre << "# Area: current surface area\n";
        pre << "# Disp: displacement averaged over the reference surface\n";
        pre << "# Force: sum of the internal nodal forces (integral of sigma . grad N) over the "
               "surface's nodes;\n";
        pre << "#   at equilibrium this is the reaction force. At nodes shared with other "
               "boundary attributes each\n";
        pre << "#   component goes to the attributes constraining it (split equally), or is "
               "split among all of them\n";
        pre << "# Normal: normalized vector area (integral of n da) of the current surface, "
               "outward\n";
        if (m_moments) {
            pre << "# Moment: moment of the nodal forces about the current surface centroid\n";
        }
        pre << "# Units follow the mesh length and material stress units "
               "(force = stress x length^2)\n";
        pre << "# Reference area: " << area << "\n";
        pre << "# Reference normal:";
        for (int c = 0; c < 3; c++) {
            pre << " " << ((vec_area_norm > 0.0) ? vec_area[c] / vec_area_norm : 0.0);
        }
        pre << "\n";
        pre << "# Reference planarity (projected area / true area): " << planarity << "\n";
        m_preambles[s] = pre.str();
    }
}

void SurfaceQuantities::Update(double time, PostProcessingFileManager& file_manager) {
    CALI_CXX_MARK_SCOPE("surface_quantities_update");

    const int nfaces = m_surface->GetNumFaces();
    const int nsurf = m_surface->GetNumSurfaces();

    m_surface->ComputeGeometry(*m_sim_state->GetCurrentCoords(), m_x, m_nda, m_da);
    m_surface->InterpolateH1(*m_sim_state->GetDisplacement(), m_disp);

    // Internal nodal forces of the converged stresses, saved by SystemDriver, and the
    // matching nodal coordinates
    UpdateNodeWeights();
    const auto internal_force = m_sim_state->GetInternalForce();
    m_coords_t.SetSize(internal_force->Size());
    m_sim_state->GetCurrentCoords()->ParallelAverage(m_coords_t);
    const int nnodes = m_node_surface.Size();
    m_node_sums.SetSize(NUM_NODE_SUMS * nnodes);
    if (nnodes > 0) {
        NodeSumsKernel(nnodes,
                       m_node_tdofs.Read(),
                       m_node_weights.Read(),
                       internal_force->Read(),
                       m_coords_t.Read(),
                       m_node_sums.Write());
    }

    m_face_sums.SetSize(NUM_FACE_SUMS * nfaces);
    if (nfaces > 0) {
        FaceSumsKernel(nfaces,
                       m_surface->GetFacePointOffsets().Read(),
                       m_x.Read(),
                       m_nda.Read(),
                       m_da.Read(),
                       m_disp.Read(),
                       m_dA0.Read(),
                       m_face_sums.Write());
    }

    // Per surface: the face sums followed by the node sums
    constexpr int NUM_SUMS = NUM_FACE_SUMS + NUM_NODE_SUMS;
    std::vector<double> sums(static_cast<size_t>(NUM_SUMS * nsurf), 0.0);
    {
        const double* face_sums = m_face_sums.HostRead();
        const int* face_surface = m_surface->GetFaceSurface().HostRead();
        for (int i = 0; i < nfaces; i++) {
            double* s = &sums[static_cast<size_t>(NUM_SUMS * face_surface[i])];
            for (int k = 0; k < NUM_FACE_SUMS; k++) {
                s[k] += face_sums[NUM_FACE_SUMS * i + k];
            }
        }
        const double* node_sums = m_node_sums.HostRead();
        const int* node_surface = m_node_surface.HostRead();
        for (int j = 0; j < nnodes; j++) {
            double* s = &sums[static_cast<size_t>(NUM_SUMS * node_surface[j] + NUM_FACE_SUMS)];
            for (int k = 0; k < NUM_NODE_SUMS; k++) {
                s[k] += node_sums[NUM_NODE_SUMS * j + k];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), NUM_SUMS * nsurf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (size_t s = 0; s < static_cast<size_t>(nsurf); s++) {
        const double* sm = &sums[NUM_SUMS * s];
        const double area = sm[SUM_DA];
        const double* force = &sm[NUM_FACE_SUMS];
        const double vec_area_norm = Norm3(&sm[SUM_NDA]);

        std::vector<double> values;
        values.reserve(12);
        for (int c = 0; c < 3; c++) {
            values.push_back((m_ref_area[s] > 0.0) ? sm[SUM_DISP + c] / m_ref_area[s] : 0.0);
        }
        for (int c = 0; c < 3; c++) {
            values.push_back(force[c]);
        }
        for (int c = 0; c < 3; c++) {
            const bool valid = !m_zero_normal[s] && vec_area_norm > 0.0;
            values.push_back(valid ? sm[SUM_NDA + c] / vec_area_norm : 0.0);
        }
        if (m_moments) {
            // Shift the moment from the origin to the centroid: M_c = M_0 - x_c x F
            double xc[3];
            for (int c = 0; c < 3; c++) {
                xc[c] = (area > 0.0) ? sm[SUM_XDA + c] / area : 0.0;
            }
            const double* m0 = &sm[NUM_FACE_SUMS + 3];
            values.push_back(m0[0] - (xc[1] * force[2] - xc[2] * force[1]));
            values.push_back(m0[1] - (xc[2] * force[0] - xc[0] * force[2]));
            values.push_back(m0[2] - (xc[0] * force[1] - xc[1] * force[0]));
        }

        file_manager.WriteSurfaceQuantity(
            m_boundary_ids[s], m_preambles[s], m_moments, time, area, values);
    }
}

void SurfaceQuantities::BuildSurfaceNodes() {
    auto mesh = m_sim_state->GetMesh();
    auto fes = m_sim_state->GetMeshParFiniteElementSpace();
    mfem::Array<int> marker(mesh->bdr_attributes.Max());
    mfem::Array<int> tdofs;

    // Owned nodes of every boundary attribute, by their x-component true DOF
    m_attr_nodes.assign(m_all_attrs.size(), {});
    std::unordered_map<int, std::vector<int>> node_to_attrs;
    for (size_t k = 0; k < m_all_attrs.size(); k++) {
        marker = 0;
        marker[m_all_attrs[k] - 1] = 1;
        fes->GetEssentialTrueDofs(marker, tdofs, 0);
        m_attr_nodes[k].assign(tdofs.begin(), tdofs.end());
        for (const int t : m_attr_nodes[k]) {
            node_to_attrs[t].push_back(static_cast<int>(k));
        }
    }

    std::vector<int> node_tdofs, node_surface;
    m_node_own_attr.clear();
    m_node_attr_offsets.assign(1, 0);
    m_node_attrs.clear();
    for (size_t s = 0; s < m_boundary_ids.size(); s++) {
        const auto own = std::lower_bound(
            m_all_attrs.begin(), m_all_attrs.end(), m_boundary_ids[s]);
        const int own_attr = static_cast<int>(own - m_all_attrs.begin());
        for (const int t : m_attr_nodes[static_cast<size_t>(own_attr)]) {
            for (int c = 0; c < 3; c++) {
                node_tdofs.push_back(ComponentTrueDof(t, c));
            }
            node_surface.push_back(static_cast<int>(s));
            m_node_own_attr.push_back(own_attr);
            const auto& node_attrs = node_to_attrs.at(t);
            m_node_attrs.insert(m_node_attrs.end(), node_attrs.begin(), node_attrs.end());
            m_node_attr_offsets.push_back(static_cast<int>(m_node_attrs.size()));
        }
    }
    m_node_tdofs.SetSize(static_cast<int>(node_tdofs.size()));
    std::copy(node_tdofs.begin(), node_tdofs.end(), m_node_tdofs.HostWrite());
    m_node_surface.SetSize(static_cast<int>(node_surface.size()));
    std::copy(node_surface.begin(), node_surface.end(), m_node_surface.HostWrite());
}

void SurfaceQuantities::UpdateNodeWeights() {
    // Which components each attribute constrains under the current boundary conditions: all
    // of its nodes have that component as an essential true DOF, on every rank. Recomputed at
    // every output, so it follows boundary condition changes.
    const auto& ess_tdofs = m_sim_state->GetEssentialTrueDofs();
    const int ntdofs = m_sim_state->GetMeshParFiniteElementSpace()->GetTrueVSize();
    std::vector<char> is_ess(static_cast<size_t>(ntdofs), 0);
    const int* ess = ess_tdofs.HostRead();
    for (int i = 0; i < ess_tdofs.Size(); i++) {
        MFEM_VERIFY(ess[i] >= 0 && ess[i] < ntdofs, "Essential true DOF out of range");
        is_ess[static_cast<size_t>(ess[i])] = 1;
    }
    const size_t nattr = m_all_attrs.size();
    std::vector<int> constrains(3 * nattr, 1);
    for (size_t k = 0; k < nattr; k++) {
        for (int c = 0; c < 3; c++) {
            for (const int t : m_attr_nodes[k]) {
                if (!is_ess[static_cast<size_t>(ComponentTrueDof(t, c))]) {
                    constrains[3 * k + static_cast<size_t>(c)] = 0;
                    break;
                }
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE,
                  constrains.data(),
                  static_cast<int>(constrains.size()),
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    // A node's component goes to the attributes constraining it, split equally; if none does,
    // the force is only solver residual and is split among all of the node's attributes. Either
    // way the shares of all attributes at a node sum to one.
    const int nnodes = m_node_surface.Size();
    m_node_weights.SetSize(3 * nnodes);
    double* weights = m_node_weights.HostWrite();
    for (int j = 0; j < nnodes; j++) {
        const size_t own = static_cast<size_t>(m_node_own_attr[static_cast<size_t>(j)]);
        const auto begin = m_node_attrs.begin() + m_node_attr_offsets[static_cast<size_t>(j)];
        const auto end = m_node_attrs.begin() + m_node_attr_offsets[static_cast<size_t>(j) + 1];
        const double nattrs_node = static_cast<double>(end - begin);
        for (size_t c = 0; c < 3; c++) {
            int owners = 0;
            for (auto it = begin; it != end; ++it) {
                owners += constrains[3 * static_cast<size_t>(*it) + c];
            }
            double w = 1.0 / nattrs_node;
            if (owners > 0) {
                w = constrains[3 * own + c] ? 1.0 / owners : 0.0;
            }
            weights[3 * j + static_cast<int>(c)] = w;
        }
    }
}
