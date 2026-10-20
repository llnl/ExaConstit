#include "postprocessing/surface_quantities.hpp"

#include "utilities/unified_logger.hpp"

#include "mfem/general/forall.hpp"

#include <algorithm>
#include <cmath>

namespace {

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
