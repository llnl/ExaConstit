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

namespace {

/// Number of per-face partial sums written by FaceSumsKernel
constexpr int NUM_FACE_SUMS = 16;
// Layout of the per-face partial sums
constexpr int SUM_DA = 0;     // da
constexpr int SUM_NDA = 1;    // n da (3)
constexpr int SUM_FORCE = 4;  // sigma n da (3)
constexpr int SUM_MOMENT = 7; // x cross (sigma n da) about the origin (3)
constexpr int SUM_XDA = 10;   // x da (3)
constexpr int SUM_DISP = 13;  // u dA0 (3)

/**
 * @brief Per-face partial sums of the surface integrals
 *
 * The Cauchy stress is in ExaConstit's Voigt order (xx, yy, zz, yz, xz, xy).
 */
void FaceSumsKernel(const int nfaces,
                    const int* face_qoff,
                    const double* x,
                    const double* nda,
                    const double* da,
                    const double* disp,
                    const double* dA0,
                    const double* stress,
                    double* sums) {
    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int i) {
        double s[NUM_FACE_SUMS];
        for (int k = 0; k < NUM_FACE_SUMS; k++) {
            s[k] = 0.0;
        }
        for (int q = face_qoff[i]; q < face_qoff[i + 1]; q++) {
            const double* xq = &x[3 * q];
            const double* nq = &nda[3 * q];
            const double* sig = &stress[6 * q];
            const double t[3] = {sig[0] * nq[0] + sig[5] * nq[1] + sig[4] * nq[2],
                                 sig[5] * nq[0] + sig[1] * nq[1] + sig[3] * nq[2],
                                 sig[4] * nq[0] + sig[3] * nq[1] + sig[2] * nq[2]};
            s[SUM_DA] += da[q];
            for (int c = 0; c < 3; c++) {
                s[SUM_NDA + c] += nq[c];
                s[SUM_FORCE + c] += t[c];
                s[SUM_XDA + c] += xq[c] * da[q];
                s[SUM_DISP + c] += disp[3 * q + c] * dA0[q];
            }
            s[SUM_MOMENT + 0] += xq[1] * t[2] - xq[2] * t[1];
            s[SUM_MOMENT + 1] += xq[2] * t[0] - xq[0] * t[2];
            s[SUM_MOMENT + 2] += xq[0] * t[1] - xq[1] * t[0];
        }
        for (int k = 0; k < NUM_FACE_SUMS; k++) {
            sums[NUM_FACE_SUMS * i + k] = s[k];
        }
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
    m_l2_fec = std::make_unique<mfem::L2_FECollection>(
        m_order, dim, mfem::BasisType::GaussLegendre);

    std::map<int, int> attr_to_surface;
    for (int s = 0; s < m_num_surfaces; s++) {
        attr_to_surface[boundary_ids[static_cast<size_t>(s)]] = s;
    }

    std::vector<int> face_surface, face_elem, face_table, face_qoff, face_vdof_off, vdofs;
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
        // A boundary element always has its first adjacent element on this rank. For an
        // interior face carrying a boundary attribute, the traction is taken from that side.
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
        face_elem.push_back(elem);
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
    to_array(face_elem, m_face_elem);
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

    const int rule_order = 2 * m_order + 1;
    const mfem::IntegrationRule& face_ir = mfem::IntRules.Get(face_geom, rule_order);
    const mfem::IntegrationRule& vol_ir = mfem::IntRules.Get(elem_geom, rule_order);
    const int nfq = face_ir.GetNPoints();
    const int nvq = vol_ir.GetNPoints();

    const mfem::FiniteElement* h1_fe = m_fes->FEColl()->FiniteElementForGeometry(elem_geom);
    const mfem::FiniteElement* l2_fe = m_l2_fec->FiniteElementForGeometry(elem_geom);
    MFEM_VERIFY(h1_fe && l2_fe,
                "Surface quantities do not support element geometry "
                    << mfem::Geometry::Name[elem_geom]);
    const int ndof = h1_fe->GetDof();
    const int nbasis = l2_fe->GetDof();
    MFEM_VERIFY(nvq >= nbasis,
                "Too few quadrature points to extrapolate stresses for element geometry "
                    << mfem::Geometry::Name[elem_geom]);

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

    // Least-squares extrapolation E = Phi_f (Phi_v^T Phi_v)^{-1} Phi_v^T
    mfem::DenseMatrix phi_v(nvq, nbasis), phi_f(nfq, nbasis);
    mfem::Vector shape_l2(nbasis);
    for (int v = 0; v < nvq; v++) {
        l2_fe->CalcShape(vol_ir.IntPoint(v), shape_l2);
        phi_v.SetRow(v, shape_l2);
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

        l2_fe->CalcShape(ip_e, shape_l2);
        phi_f.SetRow(q, shape_l2);

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

    mfem::DenseMatrix normal_eq(nbasis, nbasis);
    mfem::MultAtB(phi_v, phi_v, normal_eq);
    mfem::DenseMatrixInverse normal_inv(normal_eq);
    mfem::DenseMatrix normal_eq_inv(nbasis, nbasis);
    normal_inv.GetInverseMatrix(normal_eq_inv);
    mfem::DenseMatrix pinv(nbasis, nvq), extrap(nfq, nvq);
    mfem::MultABt(normal_eq_inv, phi_v, pinv);
    mfem::Mult(phi_f, pinv, extrap);

    // Append to the flat tables
    const int table = m_tab_nfq.Size();
    m_tab_nfq.Append(nfq);
    m_tab_nvq.Append(nvq);
    m_tab_ndof.Append(ndof);
    m_tab_off_e.Append(m_tab_e.Size());
    m_tab_off_n.Append(m_tab_n.Size());
    m_tab_off_w.Append(m_tab_w.Size());

    for (int q = 0; q < nfq; q++) {
        for (int v = 0; v < nvq; v++) {
            m_tab_e.Append(extrap(q, v));
        }
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

void BoundarySurface::ExtrapolateQF(const mfem::expt::PartialQuadratureFunction& qf,
                                    mfem::Vector& values) const {
    const int nfaces = GetNumFaces();
    const int vdim = qf.GetVDim();
    values.SetSize(vdim * m_num_face_points);
    if (nfaces == 0) {
        return;
    }

    auto pqs = qf.GetPartialSpaceShared();
    MFEM_VERIFY(pqs->GetOrder() == 2 * m_order + 1,
                "Surface extrapolation expects quadrature data on the order " << 2 * m_order + 1
                                                                              << " rule");
    const bool full_space = pqs->IsFullSpace();
    const auto* offsets = pqs->Offsets(mfem::QSpaceOffsetStorage::FULL).Read();
    // global2local has a single placeholder entry for a full space; it is not read then
    const auto* g2l = pqs->GetGlobal2Local().Read();

    const auto* face_elem = m_face_elem.Read();
    const auto* face_table = m_face_table.Read();
    const auto* face_qoff = m_face_qoff.Read();
    const auto* tab_nfq = m_tab_nfq.Read();
    const auto* tab_nvq = m_tab_nvq.Read();
    const auto* tab_off_e = m_tab_off_e.Read();
    const double* tab_e = m_tab_e.Read();
    const double* data = qf.Read();
    double* vals = values.Write();

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int i) {
        const int t = face_table[i];
        const int nfq = tab_nfq[t];
        const int nvq = tab_nvq[t];
        const int local_elem = full_space ? face_elem[i] : g2l[face_elem[i]];
        for (int q = 0; q < nfq; q++) {
            const int iq = face_qoff[i] + q;
            for (int k = 0; k < vdim; k++) {
                vals[vdim * iq + k] = 0.0;
            }
            if (local_elem < 0) {
                continue;
            }
            const double* erow = &tab_e[tab_off_e[t] + q * nvq];
            const int qstart = offsets[local_elem];
            for (int v = 0; v < nvq; v++) {
                for (int k = 0; k < vdim; k++) {
                    vals[vdim * iq + k] += erow[v] * data[vdim * (qstart + v) + k];
                }
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

    m_surface = std::make_unique<BoundarySurface>(
        mesh, m_sim_state->GetMeshParFiniteElementSpace(), m_boundary_ids, options.mesh.order);

    for (auto* v : {&m_dA0, &m_x, &m_nda, &m_da, &m_disp, &m_stress, &m_face_sums}) {
        v->UseDevice(true);
    }

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
        pre << "# Force: integral of the Cauchy traction sigma.n over the current surface\n";
        pre << "# Normal: normalized vector area (integral of n da) of the current surface, "
               "outward\n";
        if (m_moments) {
            pre << "# Moment: moment of the traction about the current surface centroid\n";
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
    m_surface->ExtrapolateQF(*m_sim_state->GetQuadratureFunction("cauchy_stress_end"), m_stress);

    m_face_sums.SetSize(NUM_FACE_SUMS * nfaces);
    if (nfaces > 0) {
        FaceSumsKernel(nfaces,
                       m_surface->GetFacePointOffsets().Read(),
                       m_x.Read(),
                       m_nda.Read(),
                       m_da.Read(),
                       m_disp.Read(),
                       m_dA0.Read(),
                       m_stress.Read(),
                       m_face_sums.Write());
    }

    std::vector<double> sums(static_cast<size_t>(NUM_FACE_SUMS * nsurf), 0.0);
    {
        const double* face_sums = m_face_sums.HostRead();
        const int* face_surface = m_surface->GetFaceSurface().HostRead();
        for (int i = 0; i < nfaces; i++) {
            double* s = &sums[static_cast<size_t>(NUM_FACE_SUMS * face_surface[i])];
            for (int k = 0; k < NUM_FACE_SUMS; k++) {
                s[k] += face_sums[NUM_FACE_SUMS * i + k];
            }
        }
    }
    MPI_Allreduce(
        MPI_IN_PLACE, sums.data(), NUM_FACE_SUMS * nsurf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (size_t s = 0; s < static_cast<size_t>(nsurf); s++) {
        const double* sm = &sums[NUM_FACE_SUMS * s];
        const double area = sm[SUM_DA];
        const double* force = &sm[SUM_FORCE];
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
            const double* m0 = &sm[SUM_MOMENT];
            values.push_back(m0[0] - (xc[1] * force[2] - xc[2] * force[1]));
            values.push_back(m0[1] - (xc[2] * force[0] - xc[0] * force[2]));
            values.push_back(m0[2] - (xc[0] * force[1] - xc[1] * force[0]));
        }

        file_manager.WriteSurfaceQuantity(
            m_boundary_ids[s], m_preambles[s], m_moments, time, area, values);
    }
}
