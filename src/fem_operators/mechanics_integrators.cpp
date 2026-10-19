

#include "fem_operators/mechanics_integrators.hpp"

#include "utilities/assembly_ops.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <iostream> // cerr
#include <math.h>   // log

// Outside of the UMAT function calls this should be the function called
// to assemble our residual vectors.
void ExaNLFIntegrator::AssembleElementVector(const mfem::FiniteElement& el,
                                             mfem::ElementTransformation& Ttr,
                                             const mfem::Vector& elfun,
                                             mfem::Vector& elvect) {
    CALI_CXX_MARK_SCOPE("enlfi_assembleElemVec");
    int dof = el.GetDof(), dim = el.GetDim();

    mfem::DenseMatrix DSh, DS;
    mfem::DenseMatrix Jpt;
    mfem::DenseMatrix PMatI, PMatO;
    // This is our stress tensor
    mfem::DenseMatrix P(3);

    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    Jpt.SetSize(dim);

    // PMatI would be our velocity in this case
    PMatI.UseExternalData(elfun.GetData(), dof, dim);
    elvect.SetSize(dof * dim);

    // PMatO would be our residual vector
    elvect = 0.0;
    PMatO.UseExternalData(elvect.HostReadWrite(), dof, dim);

    const mfem::IntegrationRule* ir = IntRule;
    if (!ir) {
        ir = &(mfem::IntRules.Get(el.GetGeomType(),
                                  2 * el.GetOrder() + 1)); // must match quadrature space
    }

    for (int i = 0; i < ir->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Ttr.SetIntPoint(&ip);

        // compute Jacobian of the transformation
        Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

        el.CalcDShape(ip, DSh);
        Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX

        double stress[6];
        GetQFData(
            Ttr.ElementNo, i, stress, m_sim_state->GetQuadratureFunction("cauchy_stress_end"));
        // Could probably later have this only set once...
        // Would reduce the number mallocs that we're doing and
        // should potentially provide a small speed boost.
        /**
         * @brief Map Voigt notation stress components to full 3x3 symmetric stress tensor.
         *
         * Converts stress data from Voigt notation [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
         * to full symmetric 3x3 stress tensor for use in matrix operations.
         * The symmetry is enforced by setting P(i,j) = P(j,i) for off-diagonal terms.
         */
        P(0, 0) = stress[0];
        P(1, 1) = stress[1];
        P(2, 2) = stress[2];
        P(1, 2) = stress[3];
        P(0, 2) = stress[4];
        P(0, 1) = stress[5];

        P(2, 1) = P(1, 2);
        P(2, 0) = P(0, 2);
        P(1, 0) = P(0, 1);

        DS *= (Ttr.Weight() * ip.weight);
        AddMult(DS, P, PMatO);
    }

    return;
}

void ExaNLFIntegrator::AssembleElementGrad(const mfem::FiniteElement& el,
                                           mfem::ElementTransformation& Ttr,
                                           const mfem::Vector& /*elfun*/,
                                           mfem::DenseMatrix& elmat) {
    CALI_CXX_MARK_SCOPE("enlfi_assembleElemGrad");
    int dof = el.GetDof(), dim = el.GetDim();

    mfem::DenseMatrix DSh, DS, Jrt;

    // Now time to start assembling stuff
    mfem::DenseMatrix grad_trans, temp;
    mfem::DenseMatrix tan_stiff;

    constexpr int ngrad_dim2 = 36;
    double matGrad[ngrad_dim2];

    // temp1 is now going to become the transpose Bmatrix as seen in
    // [B^t][tan_stiff][B]
    grad_trans.SetSize(dof * dim, 6);
    // We need a temp matrix to store our first matrix results as seen in here
    temp.SetSize(6, dof * dim);

    tan_stiff.UseExternalData(&matGrad[0], 6, 6);

    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    Jrt.SetSize(dim);
    elmat.SetSize(dof * dim);

    const mfem::IntegrationRule* ir = IntRule;
    if (!ir) {
        ir = &(mfem::IntRules.Get(el.GetGeomType(),
                                  2 * el.GetOrder() + 1)); // <--- must match quadrature space
    }

    elmat = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Ttr.SetIntPoint(&ip);
        CalcInverse(Ttr.Jacobian(), Jrt);

        el.CalcDShape(ip, DSh);
        Mult(DSh, Jrt, DS);

        GetQFData(
            Ttr.ElementNo, i, matGrad, m_sim_state->GetQuadratureFunction("tangent_stiffness"));
        // temp1 is B^t
        GenerateGradMatrix(DS, grad_trans);
        // We multiple our quadrature wts here to our tan_stiff matrix
        tan_stiff *= ip.weight * Ttr.Weight();
        // We use kgeom as a temporary matrix
        // kgeom = [Cstiff][B]
        MultABt(tan_stiff, grad_trans, temp);
        // We now add our [B^t][kgeom] product to our tangent stiffness matrix that
        // we want to output to our material tangent stiffness matrix
        AddMult(grad_trans, temp, elmat);
    }

    return;
}

// This performs the assembly step of our RHS side of our system:
// f_ik =
void ExaNLFIntegrator::AssemblePA(const mfem::FiniteElementSpace& fes) {
    CALI_CXX_MARK_SCOPE("enlfi_assemblePA");
    mfem::Mesh* mesh = fes.GetMesh();
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();

    auto W = ir->GetWeights().Read();
    geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);

    // return a pointer to beginning step stress. This is used for output visualization
    auto stress_end = m_sim_state->GetQuadratureFunction("cauchy_stress_end");

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    if (grad.Size() != (nqpts * dim * nnodes)) {
        grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
        {
            mfem::DenseMatrix DSh;
            const int offset = nnodes * dim;
            double* qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
                el.CalcDShape(ip, DSh);
            }
        }
        grad.UseDevice(true);
    }

    // geom->J really isn't going to work for us as of right now. We could just reorder it
    // to the version that we want it to be in instead...
    if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
        jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        jacobian.UseDevice(true);
    }

    if (dmat.Size() != (dim * dim * nqpts * nelems)) {
        dmat.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        dmat.UseDevice(true);
    }

    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.ReadWrite(),
                                                                  layout_jacob);

    RAJA::Layout<DIM3> layout_stress = RAJA::make_permuted_layout({{2 * dim, nqpts, nelems}},
                                                                  perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> S(stress_end->ReadWrite(),
                                                                        layout_stress);

    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> D(dmat.ReadWrite(), layout_jacob);

    RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{nqpts, dim, dim, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> geom_j_view(geom->J.Read(),
                                                                                  layout_geom);
    const int nqpts_ = nqpts;
    const int dim_ = dim;
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        for (int j = 0; j < nqpts_; j++) {
            for (int k = 0; k < dim_; k++) {
                for (int l = 0; l < dim_; l++) {
                    J(l, k, j, i) = geom_j_view(j, l, k, i);
                }
            }
        }
    });

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points. adj is actually in row
        // major memory order but if we set this to col. major than this view will act as the
        // transpose of adj A which is what we want.
        RAJA::View<const double, RAJA::Layout<DIM2>> A(&adj[0], dim_, dim_);
        // RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0],
        // layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }

            D(0, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(0, 0) +
                                       S(5, j_qpts, i_elems) * A(0, 1) +
                                       S(4, j_qpts, i_elems) * A(0, 2);
            D(1, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(1, 0) +
                                       S(5, j_qpts, i_elems) * A(1, 1) +
                                       S(4, j_qpts, i_elems) * A(1, 2);
            D(2, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(2, 0) +
                                       S(5, j_qpts, i_elems) * A(2, 1) +
                                       S(4, j_qpts, i_elems) * A(2, 2);

            D(0, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(0, 0) +
                                       S(1, j_qpts, i_elems) * A(0, 1) +
                                       S(3, j_qpts, i_elems) * A(0, 2);
            D(1, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(1, 0) +
                                       S(1, j_qpts, i_elems) * A(1, 1) +
                                       S(3, j_qpts, i_elems) * A(1, 2);
            D(2, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(2, 0) +
                                       S(1, j_qpts, i_elems) * A(2, 1) +
                                       S(3, j_qpts, i_elems) * A(2, 2);

            D(0, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(0, 0) +
                                       S(3, j_qpts, i_elems) * A(0, 1) +
                                       S(2, j_qpts, i_elems) * A(0, 2);
            D(1, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(1, 0) +
                                       S(3, j_qpts, i_elems) * A(1, 1) +
                                       S(2, j_qpts, i_elems) * A(1, 2);
            D(2, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(2, 0) +
                                       S(3, j_qpts, i_elems) * A(2, 1) +
                                       S(2, j_qpts, i_elems) * A(2, 2);
        } // End of doing J_{ij}\sigma_{jk} / nqpts loop
    }); // End of elements
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            for (int i = 0; i < dim_; i++) {
                for (int j = 0; j < dim_; j++) {
                    D(j, i, j_qpts, i_elems) *= W[j_qpts];
                }
            }
        }
    });
}

// In the below function we'll be applying the below action on our material
// tangent matrix C^{tan} at each quadrature point as:
// D_{ijkm} = 1 / det(J) * w_{qpt} * adj(J)^T_{ij} C^{tan}_{ijkl} adj(J)_{lm}
// where D is our new 4th order tensor, J is our jacobian calculated from the
// mesh geometric factors, and adj(J) is the adjugate of J.
void ExaNLFIntegrator::AssembleGradPA(const mfem::Vector& /* x */,
                                      const mfem::FiniteElementSpace& fes) {
    this->AssembleGradPA(fes);
}

// In the below function we'll be applying the below action on our material
// tangent matrix C^{tan} at each quadrature point as:
// D_{ijkm} = 1 / det(J) * w_{qpt} * adj(J)^T_{ij} C^{tan}_{ijkl} adj(J)_{lm}
// where D is our new 4th order tensor, J is our jacobian calculated from the
// mesh geometric factors, and adj(J) is the adjugate of J.
void ExaNLFIntegrator::AssembleGradPA(const mfem::FiniteElementSpace& fes) {
    CALI_CXX_MARK_SCOPE("enlfi_assemblePAG");
    mfem::Mesh* mesh = fes.GetMesh();
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();
    auto W = ir->GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    if (grad.Size() != (nqpts * dim * nnodes)) {
        grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
        {
            mfem::DenseMatrix DSh;
            const int offset = nnodes * dim;
            double* qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
                el.CalcDShape(ip, DSh);
            }
        }
        grad.UseDevice(true);
    }

    // geom->J really isn't going to work for us as of right now. We could just reorder it
    // to the version that we want it to be in instead...
    if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
        jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        jacobian.UseDevice(true);

        geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);

        const int DIM4 = 4;
        std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};

        RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                     perm4);
        RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.ReadWrite(),
                                                                      layout_jacob);

        RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{nqpts, dim, dim, nelems}},
                                                                    perm4);
        RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> geom_j_view(
            geom->J.Read(), layout_geom);
        const int nqpts_ = nqpts;
        const int dim_ = dim;
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
            for (int j = 0; j < nqpts_; j++) {
                for (int k = 0; k < dim_; k++) {
                    for (int l = 0; l < dim_; l++) {
                        J(l, k, j, i) = geom_j_view(j, l, k, i);
                    }
                }
            }
        });
    }

    if (pa_dmat.Size() != (dim * dim * dim * dim * nqpts * nelems)) {
        pa_dmat.SetSize(dim * dim * dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        pa_dmat.UseDevice(true);
    }

    if (pa_mat.Size() != (dim * dim * dim * dim * nqpts * nelems)) {
        pa_mat.SetSize(dim * dim * dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        pa_mat.UseDevice(true);
    }

    TransformMatGradTo4D(m_sim_state->GetQuadratureFunction("tangent_stiffness"), pa_mat);

    pa_dmat = 0.0;

    const int DIM2 = 2;
    const int DIM4 = 4;
    const int DIM6 = 6;
    std::array<RAJA::idx_t, DIM6> perm6{{5, 4, 3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    RAJA::Layout<DIM6> layout_4Dtensor = RAJA::make_permuted_layout(
        {{dim, dim, dim, dim, nqpts, nelems}}, perm6);
    RAJA::View<const double, RAJA::Layout<DIM6, RAJA::Index_type, 0>> C(pa_mat.Read(),
                                                                        layout_4Dtensor);
    // Swapped over to row order since it makes sense in later applications...
    // Should make C row order as well for PA operations
    RAJA::View<double, RAJA::Layout<DIM6>> D(
        pa_dmat.ReadWrite(), nelems, nqpts, dim, dim, dim, dim);

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.ReadWrite(),
                                                                  layout_jacob);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                c_detJ = 1.0 / detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            // If we were to get really ambitious we could eliminate also the m indexed
            // loop...
            for (int n = 0; n < dim_; n++) {
                for (int m = 0; m < dim_; m++) {
                    for (int l = 0; l < dim_; l++) {
                        D(i_elems, j_qpts, 0, 0, l, n) +=
                            (A(0, 0) * C(0, 0, l, m, j_qpts, i_elems) +
                             A(1, 0) * C(1, 0, l, m, j_qpts, i_elems) +
                             A(2, 0) * C(2, 0, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 0, 1, l, n) +=
                            (A(0, 0) * C(0, 1, l, m, j_qpts, i_elems) +
                             A(1, 0) * C(1, 1, l, m, j_qpts, i_elems) +
                             A(2, 0) * C(2, 1, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 0, 2, l, n) +=
                            (A(0, 0) * C(0, 2, l, m, j_qpts, i_elems) +
                             A(1, 0) * C(1, 2, l, m, j_qpts, i_elems) +
                             A(2, 0) * C(2, 2, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 1, 0, l, n) +=
                            (A(0, 1) * C(0, 0, l, m, j_qpts, i_elems) +
                             A(1, 1) * C(1, 0, l, m, j_qpts, i_elems) +
                             A(2, 1) * C(2, 0, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 1, 1, l, n) +=
                            (A(0, 1) * C(0, 1, l, m, j_qpts, i_elems) +
                             A(1, 1) * C(1, 1, l, m, j_qpts, i_elems) +
                             A(2, 1) * C(2, 1, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 1, 2, l, n) +=
                            (A(0, 1) * C(0, 2, l, m, j_qpts, i_elems) +
                             A(1, 1) * C(1, 2, l, m, j_qpts, i_elems) +
                             A(2, 1) * C(2, 2, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 2, 0, l, n) +=
                            (A(0, 2) * C(0, 0, l, m, j_qpts, i_elems) +
                             A(1, 2) * C(1, 0, l, m, j_qpts, i_elems) +
                             A(2, 2) * C(2, 0, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 2, 1, l, n) +=
                            (A(0, 2) * C(0, 1, l, m, j_qpts, i_elems) +
                             A(1, 2) * C(1, 1, l, m, j_qpts, i_elems) +
                             A(2, 2) * C(2, 1, l, m, j_qpts, i_elems)) *
                            A(m, n);
                        D(i_elems, j_qpts, 2, 2, l, n) +=
                            (A(0, 2) * C(0, 2, l, m, j_qpts, i_elems) +
                             A(1, 2) * C(1, 2, l, m, j_qpts, i_elems) +
                             A(2, 2) * C(2, 2, l, m, j_qpts, i_elems)) *
                            A(m, n);
                    }
                }
            } // End of Dikln = adj(J)_{ji} C_{jklm} adj(J)_{mn} loop

            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            for (int n = 0; n < dim_; n++) {
                for (int l = 0; l < dim_; l++) {
                    D(i_elems, j_qpts, l, n, 0, 0) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 0, 1) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 0, 2) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 1, 0) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 1, 1) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 1, 2) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 2, 0) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 2, 1) *= c_detJ;
                    D(i_elems, j_qpts, l, n, 2, 2) *= c_detJ;
                }
            } // End of D_{ijkl} *= 1/det(J) * w_{qpt} loop
        } // End of quadrature loop
    }); // End of Elements loop
}

// Here we're applying the following action operation using the assembled "D" 2nd order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jk}
void ExaNLFIntegrator::AddMultPA(const mfem::Vector& /*x*/, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("enlfi_amPAV");
    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM3 = 3;
    const int DIM4 = 4;

    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    // Swapped over to row order since it makes sense in later applications...
    // Should make C row order as well for PA operations
    RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                  perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> D(dmat.Read(), layout_tensor);
    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);
    // Transpose of the local gradient variable
    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            for (int k = 0; k < dim_; k++) {
                for (int j = 0; j < dim_; j++) {
                    for (int i = 0; i < nnodes_; i++) {
                        Y(i, k, i_elems) += Gt(i, j, j_qpts) * D(j, k, j_qpts, i_elems);
                    }
                }
            } // End of the final action of Y_{ik} += Gt_{ij} T_{jk}
        } // End of nQpts
    }); // End of nelems
}

// Here we're applying the following action operation using the assembled "D" 4th order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jklm} \nabla_{mn}\phi_{\epsilon} x_{nl}
void ExaNLFIntegrator::AddMultGradPA(const mfem::Vector& x, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("enlfi_amPAG");
    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM6 = 6;

    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};
    // Swapped over to row order since it makes sense in later applications...
    // Should make C row order as well for PA operations
    RAJA::View<const double, RAJA::Layout<DIM6>> D(
        pa_dmat.Read(), nelems, nqpts, dim, dim, dim, dim);
    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> X(x.Read(), layout_field);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);
    // Transpose of the local gradient variable
    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    // View for our temporary 2d array
    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);
    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            double T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < dim_; i++) {
                for (int j = 0; j < dim_; j++) {
                    for (int k = 0; k < nnodes_; k++) {
                        T[0] += D(i_elems, j_qpts, 0, 0, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[1] += D(i_elems, j_qpts, 1, 0, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[2] += D(i_elems, j_qpts, 2, 0, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[3] += D(i_elems, j_qpts, 0, 1, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[4] += D(i_elems, j_qpts, 1, 1, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[5] += D(i_elems, j_qpts, 2, 1, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[6] += D(i_elems, j_qpts, 0, 2, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[7] += D(i_elems, j_qpts, 1, 2, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                        T[8] += D(i_elems, j_qpts, 2, 2, i, j) * Gt(k, j, j_qpts) *
                                X(k, i, i_elems);
                    }
                }
            } // End of doing tensor contraction of D_{jkmo}G_{op}X_{pm}

            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> Tview(&T[0],
                                                                                    layout_adj);
            for (int k = 0; k < dim_; k++) {
                for (int j = 0; j < dim_; j++) {
                    for (int i = 0; i < nnodes_; i++) {
                        Y(i, k, i_elems) += Gt(i, j, j_qpts) * Tview(j, k);
                    }
                }
            } // End of the final action of Y_{ik} += Gt_{ij} T_{jk}
        } // End of nQpts
    }); // End of nelems
}

// -----------------------------------------------------------------------------
// ExaNLFIntegrator::AddMultTransposeGradPA
//
// Native PA kernel computing y += K^T * x where K = B^T D B is the standard
// (non-BBar) tangent stiffness. Mirrors AddMultGradPA exactly except for the
// contraction order against the assembled 4th-order tensor D.
//
// Algorithm per element, per quadrature point:
//   1. Compute physical velocity gradient from input vector and shape function
//      derivatives:
//         Gx(i,k) = sum_a Gt(a,i,qpt) * X(a,k,elem)
//      This is the same operation as the forward kernel since B is independent
//      of the gradient transposition.
//
//   2. Apply the TRANSPOSED D tensor contraction:
//         T(l,n) = sum_{i,k} D(i,k,n,l,qpt,elem) * Gx(i,k)
//      whereas the forward kernel does
//         T(i,k) = sum_{l,n} D(i,k,n,l,qpt,elem) * Gx(l,n)
//      (D's last two indices are (component, gradient direction) of the trial
//      field, while Gx is indexed (gradient direction, component))
//      The difference is *which pair* of D's indices are summed against Gx.
//      For symmetric C, D has major symmetry D(i,k,l,n) = D(l,n,i,k) and the
//      two contractions agree; for non-symmetric C they disagree.
//
//   3. Apply test-function gradients (same operation as forward kernel):
//         Y(a,n) += sum_l Gt(a,l,qpt) * T(l,n)
//
// All quadrature weights and Jacobian determinants are baked into D from the
// AssembleGradPA step, so this kernel does not need to reapply them.
// -----------------------------------------------------------------------------
void ExaNLFIntegrator::AddMultTransposeGradPA(const mfem::Vector& x, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("enlfi_amTGPA");
    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM3 = 3;
    const int DIM6 = 6;

    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};

    // D tensor from AssembleGradPA: D(elem, qpt, i, k, l, n), row-major with elem slowest,
    // the same view the forward kernel uses.
    RAJA::View<const double, RAJA::Layout<DIM6>> D(
        pa_dmat.Read(), nelems, nqpts, dim, dim, dim, dim);

    // Field variables: input/output E-vectors
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> X(x.Read(), layout_field);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);

    // Reference shape function derivatives: Gt(node, dim, qpt)
    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // Step 1: Compute velocity gradient at this quadrature point
            //   Gx(i, k) = sum_a Gt(a, i, qpt) * X(a, k, elem)
            double Gx[3][3];
            for (int ii = 0; ii < dim_; ii++) {
                for (int kk = 0; kk < dim_; kk++) {
                    Gx[ii][kk] = 0.0;
                    for (int a = 0; a < nnodes_; a++) {
                        Gx[ii][kk] += Gt(a, ii, j_qpts) * X(a, kk, i_elems);
                    }
                }
            }

            // Step 2: Apply TRANSPOSED D contraction
            //   T(l, n) = sum_{i,k} D(i, k, n, l, qpt, elem) * Gx(i, k)
            // Compare to forward kernel:
            //   T(i, k) = sum_{l,n} D(i, k, n, l, qpt, elem) * Gx(l, n)
            double T[3][3];
            for (int ll = 0; ll < dim_; ll++) {
                for (int nn = 0; nn < dim_; nn++) {
                    T[ll][nn] = 0.0;
                    for (int ii = 0; ii < dim_; ii++) {
                        for (int kk = 0; kk < dim_; kk++) {
                            T[ll][nn] += D(i_elems, j_qpts, ii, kk, nn, ll) * Gx[ii][kk];
                        }
                    }
                }
            }

            // Step 3: Apply test-function gradients (same as forward kernel)
            //   Y(a, n) += sum_l Gt(a, l, qpt) * T(l, n)
            for (int nn = 0; nn < dim_; nn++) {
                for (int ll = 0; ll < dim_; ll++) {
                    for (int a = 0; a < nnodes_; a++) {
                        Y(a, nn, i_elems) += Gt(a, ll, j_qpts) * T[ll][nn];
                    }
                }
            }
        } // End of nqpts
    }); // End of nelems
}

// This assembles the diagonal of our LHS which can be used as a preconditioner
void ExaNLFIntegrator::AssembleGradDiagonalPA(mfem::Vector& diag) const {
    CALI_CXX_MARK_SCOPE("enlfi_AssembleGradDiagonalPA");

    const mfem::IntegrationRule& ir =
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->GetSpaceShared()->GetIntRule(0);
    auto W = ir.GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;

    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout(
        {{2 * dim, 2 * dim, nqpts, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> K(
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->Read(), layout_tensor);

    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(diag.ReadWrite(), layout_field);

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.Read(),
                                                                        layout_jacob);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                c_detJ = 1.0 / detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knodes = 0; knodes < nnodes_; knodes++) {
                const double bx = Gt(knodes, 0, j_qpts) * A(0, 0) +
                                  Gt(knodes, 1, j_qpts) * A(0, 1) + Gt(knodes, 2, j_qpts) * A(0, 2);

                const double by = Gt(knodes, 0, j_qpts) * A(1, 0) +
                                  Gt(knodes, 1, j_qpts) * A(1, 1) + Gt(knodes, 2, j_qpts) * A(1, 2);

                const double bz = Gt(knodes, 0, j_qpts) * A(2, 0) +
                                  Gt(knodes, 1, j_qpts) * A(2, 1) + Gt(knodes, 2, j_qpts) * A(2, 2);

                Y(knodes, 0, i_elems) +=
                    c_detJ * (bx * (bx * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                    bz * K(0, 4, j_qpts, i_elems)) +
                              by * (bx * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                    bz * K(5, 4, j_qpts, i_elems)) +
                              bz * (bx * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                    bz * K(4, 4, j_qpts, i_elems)));

                Y(knodes, 1, i_elems) +=
                    c_detJ * (bx * (bx * K(5, 5, j_qpts, i_elems) + by * K(5, 1, j_qpts, i_elems) +
                                    bz * K(5, 3, j_qpts, i_elems)) +
                              by * (bx * K(1, 5, j_qpts, i_elems) + by * K(1, 1, j_qpts, i_elems) +
                                    bz * K(1, 3, j_qpts, i_elems)) +
                              bz * (bx * K(3, 5, j_qpts, i_elems) + by * K(3, 1, j_qpts, i_elems) +
                                    bz * K(3, 3, j_qpts, i_elems)));

                Y(knodes, 2, i_elems) +=
                    c_detJ * (bx * (bx * K(4, 4, j_qpts, i_elems) + by * K(4, 3, j_qpts, i_elems) +
                                    bz * K(4, 2, j_qpts, i_elems)) +
                              by * (bx * K(3, 4, j_qpts, i_elems) + by * K(3, 3, j_qpts, i_elems) +
                                    bz * K(3, 2, j_qpts, i_elems)) +
                              bz * (bx * K(2, 4, j_qpts, i_elems) + by * K(2, 3, j_qpts, i_elems) +
                                    bz * K(2, 2, j_qpts, i_elems)));
            }
        }
    });
}

/// Method defining element assembly.
/** The result of the element assembly is added and stored in the @a emat
 Vector. */
void ExaNLFIntegrator::AssembleGradEA(const mfem::Vector& /*x*/,
                                      const mfem::FiniteElementSpace& fes,
                                      mfem::Vector& emat) {
    AssembleEA(fes, emat);
}
void ExaNLFIntegrator::AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat) {
    CALI_CXX_MARK_SCOPE("enlfi_assembleEA");
    mfem::Mesh* mesh = fes.GetMesh();
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();
    auto W = ir->GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    if (grad.Size() != (nqpts * dim * nnodes)) {
        grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
        {
            mfem::DenseMatrix DSh;
            const int offset = nnodes * dim;
            double* qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
                el.CalcDShape(ip, DSh);
            }
        }
        grad.UseDevice(true);
    }

    // geom->J really isn't going to work for us as of right now. We could just reorder it
    // to the version that we want it to be in instead...
    if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
        jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        jacobian.UseDevice(true);

        geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);

        const int DIM4 = 4;
        std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};

        RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                     perm4);
        RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.ReadWrite(),
                                                                      layout_jacob);

        RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{nqpts, dim, dim, nelems}},
                                                                    perm4);
        RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> geom_j_view(
            geom->J.Read(), layout_geom);
        const int nqpts_ = nqpts;
        const int dim_ = dim;
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
            for (int j = 0; j < nqpts_; j++) {
                for (int k = 0; k < dim_; k++) {
                    for (int l = 0; l < dim_; l++) {
                        J(l, k, j, i) = geom_j_view(j, l, k, i);
                    }
                }
            }
        });
    }

    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;

    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout(
        {{2 * dim, 2 * dim, nqpts, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> K(
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->Read(), layout_tensor);

    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout(
        {{nnodes * dim, nnodes * dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> E(emat.ReadWrite(), layout_field);

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.Read(),
                                                                        layout_jacob);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                c_detJ = 1.0 / detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes_; knds++) {
                const double bx = Gt(knds, 0, j_qpts) * A(0, 0) + Gt(knds, 1, j_qpts) * A(0, 1) +
                                  Gt(knds, 2, j_qpts) * A(0, 2);

                const double by = Gt(knds, 0, j_qpts) * A(1, 0) + Gt(knds, 1, j_qpts) * A(1, 1) +
                                  Gt(knds, 2, j_qpts) * A(1, 2);

                const double bz = Gt(knds, 0, j_qpts) * A(2, 0) + Gt(knds, 1, j_qpts) * A(2, 1) +
                                  Gt(knds, 2, j_qpts) * A(2, 2);

                const double k11x = c_detJ *
                                    (bx * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 4, j_qpts, i_elems));
                const double k11y = c_detJ *
                                    (bx * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 4, j_qpts, i_elems));
                const double k11z = c_detJ *
                                    (bx * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 4, j_qpts, i_elems));

                const double k12x = c_detJ *
                                    (bx * K(0, 5, j_qpts, i_elems) + by * K(0, 1, j_qpts, i_elems) +
                                     bz * K(0, 3, j_qpts, i_elems));
                const double k12y = c_detJ *
                                    (bx * K(5, 5, j_qpts, i_elems) + by * K(5, 1, j_qpts, i_elems) +
                                     bz * K(5, 3, j_qpts, i_elems));
                const double k12z = c_detJ *
                                    (bx * K(4, 5, j_qpts, i_elems) + by * K(4, 1, j_qpts, i_elems) +
                                     bz * K(4, 3, j_qpts, i_elems));

                const double k13x = c_detJ *
                                    (bx * K(0, 4, j_qpts, i_elems) + by * K(0, 3, j_qpts, i_elems) +
                                     bz * K(0, 2, j_qpts, i_elems));
                const double k13y = c_detJ *
                                    (bx * K(5, 4, j_qpts, i_elems) + by * K(5, 3, j_qpts, i_elems) +
                                     bz * K(5, 2, j_qpts, i_elems));
                const double k13z = c_detJ *
                                    (bx * K(4, 4, j_qpts, i_elems) + by * K(4, 3, j_qpts, i_elems) +
                                     bz * K(4, 2, j_qpts, i_elems));

                const double k21x = c_detJ *
                                    (bx * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 4, j_qpts, i_elems));
                const double k21y = c_detJ *
                                    (bx * K(1, 0, j_qpts, i_elems) + by * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 4, j_qpts, i_elems));
                const double k21z = c_detJ *
                                    (bx * K(3, 0, j_qpts, i_elems) + by * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 4, j_qpts, i_elems));

                const double k22x = c_detJ *
                                    (bx * K(5, 5, j_qpts, i_elems) + by * K(5, 1, j_qpts, i_elems) +
                                     bz * K(5, 3, j_qpts, i_elems));
                const double k22y = c_detJ *
                                    (bx * K(1, 5, j_qpts, i_elems) + by * K(1, 1, j_qpts, i_elems) +
                                     bz * K(1, 3, j_qpts, i_elems));
                const double k22z = c_detJ *
                                    (bx * K(3, 5, j_qpts, i_elems) + by * K(3, 1, j_qpts, i_elems) +
                                     bz * K(3, 3, j_qpts, i_elems));

                const double k23x = c_detJ *
                                    (bx * K(5, 4, j_qpts, i_elems) + by * K(5, 3, j_qpts, i_elems) +
                                     bz * K(5, 2, j_qpts, i_elems));
                const double k23y = c_detJ *
                                    (bx * K(1, 4, j_qpts, i_elems) + by * K(1, 3, j_qpts, i_elems) +
                                     bz * K(1, 2, j_qpts, i_elems));
                const double k23z = c_detJ *
                                    (bx * K(3, 4, j_qpts, i_elems) + by * K(3, 3, j_qpts, i_elems) +
                                     bz * K(3, 2, j_qpts, i_elems));

                const double k31x = c_detJ *
                                    (bx * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 4, j_qpts, i_elems));
                const double k31y = c_detJ *
                                    (bx * K(3, 0, j_qpts, i_elems) + by * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 4, j_qpts, i_elems));
                const double k31z = c_detJ *
                                    (bx * K(2, 0, j_qpts, i_elems) + by * K(2, 5, j_qpts, i_elems) +
                                     bz * K(2, 4, j_qpts, i_elems));

                const double k32x = c_detJ *
                                    (bx * K(4, 5, j_qpts, i_elems) + by * K(4, 1, j_qpts, i_elems) +
                                     bz * K(4, 3, j_qpts, i_elems));
                const double k32y = c_detJ *
                                    (bx * K(3, 5, j_qpts, i_elems) + by * K(3, 1, j_qpts, i_elems) +
                                     bz * K(3, 3, j_qpts, i_elems));
                const double k32z = c_detJ *
                                    (bx * K(2, 5, j_qpts, i_elems) + by * K(2, 1, j_qpts, i_elems) +
                                     bz * K(2, 3, j_qpts, i_elems));

                const double k33x = c_detJ *
                                    (bx * K(4, 4, j_qpts, i_elems) + by * K(4, 3, j_qpts, i_elems) +
                                     bz * K(4, 2, j_qpts, i_elems));
                const double k33y = c_detJ *
                                    (bx * K(3, 4, j_qpts, i_elems) + by * K(3, 3, j_qpts, i_elems) +
                                     bz * K(3, 2, j_qpts, i_elems));
                const double k33z = c_detJ *
                                    (bx * K(2, 4, j_qpts, i_elems) + by * K(2, 3, j_qpts, i_elems) +
                                     bz * K(2, 2, j_qpts, i_elems));

                for (int lnds = 0; lnds < nnodes_; lnds++) {
                    const double gx = Gt(lnds, 0, j_qpts) * A(0, 0) +
                                      Gt(lnds, 1, j_qpts) * A(0, 1) + Gt(lnds, 2, j_qpts) * A(0, 2);

                    const double gy = Gt(lnds, 0, j_qpts) * A(1, 0) +
                                      Gt(lnds, 1, j_qpts) * A(1, 1) + Gt(lnds, 2, j_qpts) * A(1, 2);

                    const double gz = Gt(lnds, 0, j_qpts) * A(2, 0) +
                                      Gt(lnds, 1, j_qpts) * A(2, 1) + Gt(lnds, 2, j_qpts) * A(2, 2);

                    E(lnds, knds, i_elems) += gx * k11x + gy * k11y + gz * k11z;
                    E(lnds, knds + nnodes_, i_elems) += gx * k12x + gy * k12y + gz * k12z;
                    E(lnds, knds + 2 * nnodes_, i_elems) += gx * k13x + gy * k13y + gz * k13z;

                    E(lnds + nnodes_, knds, i_elems) += gx * k21x + gy * k21y + gz * k21z;
                    E(lnds + nnodes_, knds + nnodes_, i_elems) += gx * k22x + gy * k22y + gz * k22z;
                    E(lnds + nnodes_, knds + 2 * nnodes_, i_elems) += gx * k23x + gy * k23y +
                                                                      gz * k23z;

                    E(lnds + 2 * nnodes_, knds, i_elems) += gx * k31x + gy * k31y + gz * k31z;
                    E(lnds + 2 * nnodes_, knds + nnodes_, i_elems) += gx * k32x + gy * k32y +
                                                                      gz * k32z;
                    E(lnds + 2 * nnodes_, knds + 2 * nnodes_, i_elems) += gx * k33x + gy * k33y +
                                                                          gz * k33z;
                }
            }
        }
    });
}

// Outside of the UMAT function calls this should be the function called
// to assemble our residual vectors.
void ICExaNLFIntegrator::AssembleElementVector(const mfem::FiniteElement& el,
                                               mfem::ElementTransformation& Ttr,
                                               const mfem::Vector& elfun,
                                               mfem::Vector& elvect) {
    CALI_CXX_MARK_SCOPE("icenlfi_assembleElemVec");
    int dof = el.GetDof(), dim = el.GetDim();

    mfem::DenseMatrix DSh, DS, elem_deriv_shapes_loc;
    mfem::DenseMatrix Jpt;
    mfem::DenseMatrix PMatI, PMatO;
    // This is our stress tensor
    mfem::DenseMatrix P;
    mfem::DenseMatrix grad_trans;
    // temp1 is now going to become the transpose Bmatrix as seen in
    // [B^t][tan_stiff][B]
    grad_trans.SetSize(dof * dim, 6);

    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    elem_deriv_shapes_loc.SetSize(dof, dim);
    elem_deriv_shapes_loc = 0.0;
    Jpt.SetSize(dim);

    // PMatI would be our velocity in this case
    PMatI.UseExternalData(elfun.GetData(), dof, dim);
    elvect.SetSize(dof * dim);

    // PMatO would be our residual vector
    elvect = 0.0;
    PMatO.UseExternalData(elvect.HostReadWrite(), dof * dim, 1);

    const mfem::IntegrationRule* ir = IntRule;
    if (!ir) {
        ir = &(mfem::IntRules.Get(el.GetGeomType(),
                                  2 * el.GetOrder() + 1)); // must match quadrature space
    }

    const mfem::IntegrationRule* irc = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    double eVol = 0.0;
    /**
     * @brief Compute element-averaged shape function derivatives for B-bar method.
     *
     * This loop integrates shape function derivatives over the entire element volume
     * to compute volume-averaged quantities needed for the B-bar method. The averaged
     * derivatives prevent volumetric locking in incompressible material problems.
     *
     * Process:
     * 1. Integrate ∂N/∂x derivatives weighted by Jacobian and quadrature weights
     * 2. Accumulate total element volume (eVol)
     * 3. Normalize by total volume to obtain element averages
     */
    for (int i = 0; i < irc->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = irc->IntPoint(i);
        Ttr.SetIntPoint(&ip);

        // Shared B-bar implementation (utilities/mechanics_kernels.hpp): the same element
        // averages the PA/EA setup and the material models use.
        el.CalcDShape(ip, DSh);
        exaconstit::kernel::BBarAccumulateQpt(dof,
                                              Ttr.Jacobian().GetData(),
                                              DSh.GetData(),
                                              ip.weight,
                                              elem_deriv_shapes_loc.GetData(),
                                              eVol);
    }

    exaconstit::kernel::BBarFinalize(dof, eVol, elem_deriv_shapes_loc.GetData());

    double stress[6];

    P.UseExternalData(&stress[0], 6, 1);

    for (int i = 0; i < ir->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Ttr.SetIntPoint(&ip);

        // compute Jacobian of the transformation
        Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

        el.CalcDShape(ip, DSh);
        Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX

        GetQFData(
            Ttr.ElementNo, i, stress, m_sim_state->GetQuadratureFunction("cauchy_stress_end"));
        GenerateGradBarMatrix(DS, elem_deriv_shapes_loc, grad_trans);

        grad_trans *= (ip.weight * Ttr.Weight());
        AddMult(grad_trans, P, PMatO);
    }

    return;
}

void ICExaNLFIntegrator::AssembleElementGrad(const mfem::FiniteElement& el,
                                             mfem::ElementTransformation& Ttr,
                                             const mfem::Vector& /*elfun*/,
                                             mfem::DenseMatrix& elmat) {
    CALI_CXX_MARK_SCOPE("icenlfi_assembleElemGrad");
    int dof = el.GetDof(), dim = el.GetDim();

    mfem::DenseMatrix DSh, DS, elem_deriv_shapes_loc, Jrt;

    // Now time to start assembling stuff
    mfem::DenseMatrix grad_trans, temp;
    mfem::DenseMatrix tan_stiff;

    constexpr int ngrad_dim2 = 36;
    double matGrad[ngrad_dim2];

    // temp1 is now going to become the transpose Bmatrix as seen in
    // [B^t][tan_stiff][B]
    grad_trans.SetSize(dof * dim, 6);
    // We need a temp matrix to store our first matrix results as seen in here
    temp.SetSize(6, dof * dim);

    tan_stiff.UseExternalData(&matGrad[0], 6, 6);

    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    elem_deriv_shapes_loc.SetSize(dof, dim);
    elem_deriv_shapes_loc = 0.0;
    Jrt.SetSize(dim);
    elmat.SetSize(dof * dim);

    const mfem::IntegrationRule* ir = IntRule;
    if (!ir) {
        ir = &(mfem::IntRules.Get(el.GetGeomType(),
                                  2 * el.GetOrder() + 1)); // <--- must match quadrature space
    }

    elmat = 0.0;

    const mfem::IntegrationRule* irc = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    double eVol = 0.0;

    for (int i = 0; i < irc->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = irc->IntPoint(i);
        Ttr.SetIntPoint(&ip);

        // Shared B-bar implementation (utilities/mechanics_kernels.hpp): the same element
        // averages the PA/EA setup and the material models use.
        el.CalcDShape(ip, DSh);
        exaconstit::kernel::BBarAccumulateQpt(dof,
                                              Ttr.Jacobian().GetData(),
                                              DSh.GetData(),
                                              ip.weight,
                                              elem_deriv_shapes_loc.GetData(),
                                              eVol);
    }

    exaconstit::kernel::BBarFinalize(dof, eVol, elem_deriv_shapes_loc.GetData());

    for (int i = 0; i < ir->GetNPoints(); i++) {
        const mfem::IntegrationPoint& ip = ir->IntPoint(i);
        Ttr.SetIntPoint(&ip);
        CalcInverse(Ttr.Jacobian(), Jrt);

        el.CalcDShape(ip, DSh);
        Mult(DSh, Jrt, DS);

        GetQFData(
            Ttr.ElementNo, i, matGrad, m_sim_state->GetQuadratureFunction("tangent_stiffness"));
        // temp1 is B^t
        GenerateGradBarMatrix(DS, elem_deriv_shapes_loc, grad_trans);
        // We multiple our quadrature wts here to our tan_stiff matrix
        tan_stiff *= ip.weight * Ttr.Weight();
        // We use kgeom as a temporary matrix
        // kgeom = [Cstiff][B]
        MultABt(tan_stiff, grad_trans, temp);
        // We now add our [B^t][kgeom] product to our tangent stiffness matrix that
        // we want to output to our material tangent stiffness matrix
        AddMult(grad_trans, temp, elmat);
    }

    return;
}

// -----------------------------------------------------------------------------
// ICExaNLFIntegrator::AssembleGradPA
//
// Sets up geometric data and ensures element-averaged derivatives are ready.
// The B-bar gradient PA does NOT pre-assemble a D tensor (unlike the base
// class) because the volumetric correction couples element-constant data
// (volume-averaged derivatives N̄) with per-quadrature-point data (C, adj(J))
// in a way that does not fold cleanly into a single pre-assembled tensor.
// Instead, AddMultGradPA / AddMultTransposeGradPA access C directly from the
// quadrature function and apply the B-bar action on the fly in physical space.
// -----------------------------------------------------------------------------
void ICExaNLFIntegrator::AssembleGradPA(const mfem::Vector& /* x */,
                                        const mfem::FiniteElementSpace& fes) {
    this->AssembleGradPA(fes);
}

void ICExaNLFIntegrator::AssembleGradPA(const mfem::FiniteElementSpace& fes) {
    CALI_CXX_MARK_SCOPE("icenlfi_assembleGradPA");

    mfem::Mesh* mesh = fes.GetMesh();
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    // Cache geometric factors (Jacobians at quadrature points)
    geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);

    // Cache reference shape function derivatives
    if (grad.Size() != (nqpts * space_dims * nnodes)) {
        grad.SetSize(nqpts * space_dims * nnodes, mfem::Device::GetMemoryType());
        {
            mfem::DenseMatrix DSh;
            const int offset = nnodes * space_dims;
            double* qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, space_dims);
                el.CalcDShape(ip, DSh);
            }
        }
        grad.UseDevice(true);
    }

    // Element-averaged derivatives N̄(a, k, elem) are computed by AssemblePA().
    // If they have not been computed yet, force a call now so the gradient PA
    // kernels can use them. The AssemblePA path is idempotent and safe to call
    // even if it has been called previously (it re-zeroes and recomputes).
    if (elem_deriv_shapes.Size() != (nnodes * space_dims * nelems)) {
        this->AssemblePA(fes);
    }

    // The gradient PA kernels read the material tangent as a full 3x3x3x3 tensor per
    // quadrature point, while the "tangent_stiffness" QF stores 6x6 Voigt matrices.
    // Expand it into pa_mat the same way ExaNLFIntegrator::AssembleGradPA does.
    const int ntensor = space_dims * space_dims * space_dims * space_dims * nqpts * nelems;
    if (pa_mat.Size() != ntensor) {
        pa_mat.SetSize(ntensor, mfem::Device::GetMemoryType());
        pa_mat.UseDevice(true);
    }
    TransformMatGradTo4D(m_sim_state->GetQuadratureFunction("tangent_stiffness"), pa_mat);
}

/// Method defining element assembly.
/** The result of the element assembly is added and stored in the @a emat
    Vector. */
void ICExaNLFIntegrator::AssembleGradEA(const mfem::Vector& /*x*/,
                                        const mfem::FiniteElementSpace& fes,
                                        mfem::Vector& emat) {
    AssembleEA(fes, emat);
}

void ICExaNLFIntegrator::AssembleEA(const mfem::FiniteElementSpace& fes, mfem::Vector& emat) {
    CALI_CXX_MARK_SCOPE("icenlfi_assembleEA");
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();
    auto W = ir->GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;

    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    // Our field variables that are inputs and outputs

    RAJA::Layout<DIM3> layout_egrads = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> elem_deriv_shapes_view(
        elem_deriv_shapes.Read(), layout_egrads);

    RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout(
        {{2 * dim, 2 * dim, nqpts, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> K(
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->Read(), layout_tensor);

    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout(
        {{nnodes * dim, nnodes * dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> E(emat.ReadWrite(), layout_field);

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.Read(),
                                                                        layout_jacob);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    const double i3 = 1.0 / 3.0;
    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        double idetJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                idetJ = 1.0 / detJ;
                c_detJ = detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes_; knds++) {
                const double bx = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(0, 0) + Gt(knds, 1, j_qpts) * A(0, 1) +
                                   Gt(knds, 2, j_qpts) * A(0, 2));

                const double by = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(1, 0) + Gt(knds, 1, j_qpts) * A(1, 1) +
                                   Gt(knds, 2, j_qpts) * A(1, 2));

                const double bz = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(2, 0) + Gt(knds, 1, j_qpts) * A(2, 1) +
                                   Gt(knds, 2, j_qpts) * A(2, 2));
                const double b4 = i3 * (elem_deriv_shapes_view(knds, 0, i_elems) - bx);
                const double b5 = b4 + bx;
                const double b6 = i3 * (elem_deriv_shapes_view(knds, 1, i_elems) - by);
                const double b7 = b6 + by;
                const double b8 = i3 * (elem_deriv_shapes_view(knds, 2, i_elems) - bz);
                const double b9 = b8 + bz;

                const double k11w = c_detJ *
                                    (b4 * K(1, 1, j_qpts, i_elems) + b4 * K(1, 2, j_qpts, i_elems) +
                                     b5 * K(1, 0, j_qpts, i_elems) + by * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 4, j_qpts, i_elems) + b4 * K(2, 1, j_qpts, i_elems) +
                                     b4 * K(2, 2, j_qpts, i_elems) + b5 * K(2, 0, j_qpts, i_elems) +
                                     by * K(2, 5, j_qpts, i_elems) + bz * K(2, 4, j_qpts, i_elems));

                const double k11x = c_detJ *
                                    (b4 * K(0, 1, j_qpts, i_elems) + b4 * K(0, 2, j_qpts, i_elems) +
                                     b5 * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 4, j_qpts, i_elems));

                const double k11y = c_detJ *
                                    (b4 * K(5, 1, j_qpts, i_elems) + b4 * K(5, 2, j_qpts, i_elems) +
                                     b5 * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 4, j_qpts, i_elems));

                const double k11z = c_detJ *
                                    (b4 * K(4, 1, j_qpts, i_elems) + b4 * K(4, 2, j_qpts, i_elems) +
                                     b5 * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 4, j_qpts, i_elems));

                const double k12w = c_detJ *
                                    (b6 * K(1, 0, j_qpts, i_elems) + b6 * K(1, 2, j_qpts, i_elems) +
                                     b7 * K(1, 1, j_qpts, i_elems) + bx * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 3, j_qpts, i_elems) + b6 * K(2, 0, j_qpts, i_elems) +
                                     b6 * K(2, 2, j_qpts, i_elems) + b7 * K(2, 1, j_qpts, i_elems) +
                                     bx * K(2, 5, j_qpts, i_elems) + bz * K(2, 3, j_qpts, i_elems));

                const double k12x = c_detJ *
                                    (b6 * K(0, 0, j_qpts, i_elems) + b6 * K(0, 2, j_qpts, i_elems) +
                                     b7 * K(0, 1, j_qpts, i_elems) + bx * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 3, j_qpts, i_elems));

                const double k12y = c_detJ *
                                    (b6 * K(5, 0, j_qpts, i_elems) + b6 * K(5, 2, j_qpts, i_elems) +
                                     b7 * K(5, 1, j_qpts, i_elems) + bx * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 3, j_qpts, i_elems));

                const double k12z = c_detJ *
                                    (b6 * K(4, 0, j_qpts, i_elems) + b6 * K(4, 2, j_qpts, i_elems) +
                                     b7 * K(4, 1, j_qpts, i_elems) + bx * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 3, j_qpts, i_elems));

                const double k13w = c_detJ *
                                    (b8 * K(1, 0, j_qpts, i_elems) + b8 * K(1, 1, j_qpts, i_elems) +
                                     b9 * K(1, 2, j_qpts, i_elems) + bx * K(1, 4, j_qpts, i_elems) +
                                     by * K(1, 3, j_qpts, i_elems) + b8 * K(2, 0, j_qpts, i_elems) +
                                     b8 * K(2, 1, j_qpts, i_elems) + b9 * K(2, 2, j_qpts, i_elems) +
                                     bx * K(2, 4, j_qpts, i_elems) + by * K(2, 3, j_qpts, i_elems));

                const double k13x = c_detJ *
                                    (b8 * K(0, 0, j_qpts, i_elems) + b8 * K(0, 1, j_qpts, i_elems) +
                                     b9 * K(0, 2, j_qpts, i_elems) + bx * K(0, 4, j_qpts, i_elems) +
                                     by * K(0, 3, j_qpts, i_elems));

                const double k13y = c_detJ *
                                    (b8 * K(5, 0, j_qpts, i_elems) + b8 * K(5, 1, j_qpts, i_elems) +
                                     b9 * K(5, 2, j_qpts, i_elems) + bx * K(5, 4, j_qpts, i_elems) +
                                     by * K(5, 3, j_qpts, i_elems));

                const double k13z = c_detJ *
                                    (b8 * K(4, 0, j_qpts, i_elems) + b8 * K(4, 1, j_qpts, i_elems) +
                                     b9 * K(4, 2, j_qpts, i_elems) + bx * K(4, 4, j_qpts, i_elems) +
                                     by * K(4, 3, j_qpts, i_elems));

                const double k21w = c_detJ *
                                    (b4 * K(0, 1, j_qpts, i_elems) + b4 * K(0, 2, j_qpts, i_elems) +
                                     b5 * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 4, j_qpts, i_elems) + b4 * K(2, 1, j_qpts, i_elems) +
                                     b4 * K(2, 2, j_qpts, i_elems) + b5 * K(2, 0, j_qpts, i_elems) +
                                     by * K(2, 5, j_qpts, i_elems) + bz * K(2, 4, j_qpts, i_elems));

                const double k21x = c_detJ *
                                    (b4 * K(1, 1, j_qpts, i_elems) + b4 * K(1, 2, j_qpts, i_elems) +
                                     b5 * K(1, 0, j_qpts, i_elems) + by * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 4, j_qpts, i_elems));

                const double k21y = c_detJ *
                                    (b4 * K(5, 1, j_qpts, i_elems) + b4 * K(5, 2, j_qpts, i_elems) +
                                     b5 * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 4, j_qpts, i_elems));

                const double k21z = c_detJ *
                                    (b4 * K(3, 1, j_qpts, i_elems) + b4 * K(3, 2, j_qpts, i_elems) +
                                     b5 * K(3, 0, j_qpts, i_elems) + by * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 4, j_qpts, i_elems));

                const double k22w = c_detJ *
                                    (b6 * K(0, 0, j_qpts, i_elems) + b6 * K(0, 2, j_qpts, i_elems) +
                                     b7 * K(0, 1, j_qpts, i_elems) + bx * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 3, j_qpts, i_elems) + b6 * K(2, 0, j_qpts, i_elems) +
                                     b6 * K(2, 2, j_qpts, i_elems) + b7 * K(2, 1, j_qpts, i_elems) +
                                     bx * K(2, 5, j_qpts, i_elems) + bz * K(2, 3, j_qpts, i_elems));

                const double k22x = c_detJ *
                                    (b6 * K(1, 0, j_qpts, i_elems) + b6 * K(1, 2, j_qpts, i_elems) +
                                     b7 * K(1, 1, j_qpts, i_elems) + bx * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 3, j_qpts, i_elems));

                const double k22y = c_detJ *
                                    (b6 * K(5, 0, j_qpts, i_elems) + b6 * K(5, 2, j_qpts, i_elems) +
                                     b7 * K(5, 1, j_qpts, i_elems) + bx * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 3, j_qpts, i_elems));

                const double k22z = c_detJ *
                                    (b6 * K(3, 0, j_qpts, i_elems) + b6 * K(3, 2, j_qpts, i_elems) +
                                     b7 * K(3, 1, j_qpts, i_elems) + bx * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 3, j_qpts, i_elems));

                const double k23w = c_detJ *
                                    (b8 * K(0, 0, j_qpts, i_elems) + b8 * K(0, 1, j_qpts, i_elems) +
                                     b9 * K(0, 2, j_qpts, i_elems) + bx * K(0, 4, j_qpts, i_elems) +
                                     by * K(0, 3, j_qpts, i_elems) + b8 * K(2, 0, j_qpts, i_elems) +
                                     b8 * K(2, 1, j_qpts, i_elems) + b9 * K(2, 2, j_qpts, i_elems) +
                                     bx * K(2, 4, j_qpts, i_elems) + by * K(2, 3, j_qpts, i_elems));

                const double k23x = c_detJ *
                                    (b8 * K(1, 0, j_qpts, i_elems) + b8 * K(1, 1, j_qpts, i_elems) +
                                     b9 * K(1, 2, j_qpts, i_elems) + bx * K(1, 4, j_qpts, i_elems) +
                                     by * K(1, 3, j_qpts, i_elems));

                const double k23y = c_detJ *
                                    (b8 * K(5, 0, j_qpts, i_elems) + b8 * K(5, 1, j_qpts, i_elems) +
                                     b9 * K(5, 2, j_qpts, i_elems) + bx * K(5, 4, j_qpts, i_elems) +
                                     by * K(5, 3, j_qpts, i_elems));

                const double k23z = c_detJ *
                                    (b8 * K(3, 0, j_qpts, i_elems) + b8 * K(3, 1, j_qpts, i_elems) +
                                     b9 * K(3, 2, j_qpts, i_elems) + bx * K(3, 4, j_qpts, i_elems) +
                                     by * K(3, 3, j_qpts, i_elems));

                const double k31w = c_detJ *
                                    (b4 * K(0, 1, j_qpts, i_elems) + b4 * K(0, 2, j_qpts, i_elems) +
                                     b5 * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 4, j_qpts, i_elems) + b4 * K(1, 1, j_qpts, i_elems) +
                                     b4 * K(1, 2, j_qpts, i_elems) + b5 * K(1, 0, j_qpts, i_elems) +
                                     by * K(1, 5, j_qpts, i_elems) + bz * K(1, 4, j_qpts, i_elems));

                const double k31x = c_detJ *
                                    (b4 * K(2, 1, j_qpts, i_elems) + b4 * K(2, 2, j_qpts, i_elems) +
                                     b5 * K(2, 0, j_qpts, i_elems) + by * K(2, 5, j_qpts, i_elems) +
                                     bz * K(2, 4, j_qpts, i_elems));

                const double k31y = c_detJ *
                                    (b4 * K(4, 1, j_qpts, i_elems) + b4 * K(4, 2, j_qpts, i_elems) +
                                     b5 * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 4, j_qpts, i_elems));

                const double k31z = c_detJ *
                                    (b4 * K(3, 1, j_qpts, i_elems) + b4 * K(3, 2, j_qpts, i_elems) +
                                     b5 * K(3, 0, j_qpts, i_elems) + by * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 4, j_qpts, i_elems));

                const double k32w = c_detJ *
                                    (b6 * K(0, 0, j_qpts, i_elems) + b6 * K(0, 2, j_qpts, i_elems) +
                                     b7 * K(0, 1, j_qpts, i_elems) + bx * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 3, j_qpts, i_elems) + b6 * K(1, 0, j_qpts, i_elems) +
                                     b6 * K(1, 2, j_qpts, i_elems) + b7 * K(1, 1, j_qpts, i_elems) +
                                     bx * K(1, 5, j_qpts, i_elems) + bz * K(1, 3, j_qpts, i_elems));

                const double k32x = c_detJ *
                                    (b6 * K(2, 0, j_qpts, i_elems) + b6 * K(2, 2, j_qpts, i_elems) +
                                     b7 * K(2, 1, j_qpts, i_elems) + bx * K(2, 5, j_qpts, i_elems) +
                                     bz * K(2, 3, j_qpts, i_elems));

                const double k32y = c_detJ *
                                    (b6 * K(4, 0, j_qpts, i_elems) + b6 * K(4, 2, j_qpts, i_elems) +
                                     b7 * K(4, 1, j_qpts, i_elems) + bx * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 3, j_qpts, i_elems));

                const double k32z = c_detJ *
                                    (b6 * K(3, 0, j_qpts, i_elems) + b6 * K(3, 2, j_qpts, i_elems) +
                                     b7 * K(3, 1, j_qpts, i_elems) + bx * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 3, j_qpts, i_elems));

                const double k33w = c_detJ *
                                    (b8 * K(0, 0, j_qpts, i_elems) + b8 * K(0, 1, j_qpts, i_elems) +
                                     b9 * K(0, 2, j_qpts, i_elems) + bx * K(0, 4, j_qpts, i_elems) +
                                     by * K(0, 3, j_qpts, i_elems) + b8 * K(1, 0, j_qpts, i_elems) +
                                     b8 * K(1, 1, j_qpts, i_elems) + b9 * K(1, 2, j_qpts, i_elems) +
                                     bx * K(1, 4, j_qpts, i_elems) + by * K(1, 3, j_qpts, i_elems));

                const double k33x = c_detJ *
                                    (b8 * K(2, 0, j_qpts, i_elems) + b8 * K(2, 1, j_qpts, i_elems) +
                                     b9 * K(2, 2, j_qpts, i_elems) + bx * K(2, 4, j_qpts, i_elems) +
                                     by * K(2, 3, j_qpts, i_elems));

                const double k33y = c_detJ *
                                    (b8 * K(4, 0, j_qpts, i_elems) + b8 * K(4, 1, j_qpts, i_elems) +
                                     b9 * K(4, 2, j_qpts, i_elems) + bx * K(4, 4, j_qpts, i_elems) +
                                     by * K(4, 3, j_qpts, i_elems));

                const double k33z = c_detJ *
                                    (b8 * K(3, 0, j_qpts, i_elems) + b8 * K(3, 1, j_qpts, i_elems) +
                                     b9 * K(3, 2, j_qpts, i_elems) + bx * K(3, 4, j_qpts, i_elems) +
                                     by * K(3, 3, j_qpts, i_elems));

                for (int lnds = 0; lnds < nnodes_; lnds++) {
                    const double gx = idetJ * (Gt(lnds, 0, j_qpts) * A(0, 0) +
                                               Gt(lnds, 1, j_qpts) * A(0, 1) +
                                               Gt(lnds, 2, j_qpts) * A(0, 2));

                    const double gy = idetJ * (Gt(lnds, 0, j_qpts) * A(1, 0) +
                                               Gt(lnds, 1, j_qpts) * A(1, 1) +
                                               Gt(lnds, 2, j_qpts) * A(1, 2));

                    const double gz = idetJ * (Gt(lnds, 0, j_qpts) * A(2, 0) +
                                               Gt(lnds, 1, j_qpts) * A(2, 1) +
                                               Gt(lnds, 2, j_qpts) * A(2, 2));

                    const double g4 = i3 * (elem_deriv_shapes_view(lnds, 0, i_elems) - gx);
                    const double g5 = g4 + gx;
                    const double g6 = i3 * (elem_deriv_shapes_view(lnds, 1, i_elems) - gy);
                    const double g7 = g6 + gy;
                    const double g8 = i3 * (elem_deriv_shapes_view(lnds, 2, i_elems) - gz);
                    const double g9 = g8 + gz;

                    E(lnds, knds, i_elems) += g4 * k11w + g5 * k11x + gy * k11y + gz * k11z;
                    E(lnds, knds + nnodes_, i_elems) += g4 * k12w + g5 * k12x + gy * k12y +
                                                        gz * k12z;
                    E(lnds, knds + 2 * nnodes_, i_elems) += g4 * k13w + g5 * k13x + gy * k13y +
                                                            gz * k13z;

                    E(lnds + nnodes_, knds, i_elems) += g6 * k21w + g7 * k21x + gx * k21y +
                                                        gz * k21z;
                    E(lnds + nnodes_, knds + nnodes_, i_elems) += g6 * k22w + g7 * k22x +
                                                                  gx * k22y + gz * k22z;
                    E(lnds + nnodes_, knds + 2 * nnodes_, i_elems) += g6 * k23w + g7 * k23x +
                                                                      gx * k23y + gz * k23z;

                    E(lnds + 2 * nnodes_, knds, i_elems) += g8 * k31w + g9 * k31x + gx * k31y +
                                                            gy * k31z;
                    E(lnds + 2 * nnodes_, knds + nnodes_, i_elems) += g8 * k32w + g9 * k32x +
                                                                      gx * k32y + gy * k32z;
                    E(lnds + 2 * nnodes_, knds + 2 * nnodes_, i_elems) += g8 * k33w + g9 * k33x +
                                                                          gx * k33y + gy * k33z;
                }
            }
        }
    });
}

// This assembles the diagonal of our LHS which can be used as a preconditioner
void ICExaNLFIntegrator::AssembleGradDiagonalPA(mfem::Vector& diag) const {
    CALI_CXX_MARK_SCOPE("icenlfi_AssembleGradDiagonalPA");

    const mfem::IntegrationRule& ir =
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->GetSpaceShared()->GetIntRule(0);
    auto W = ir.GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

    RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout(
        {{2 * dim, 2 * dim, nqpts, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> K(
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->Read(), layout_tensor);

    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(diag.ReadWrite(), layout_field);

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.Read(),
                                                                        layout_jacob);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    RAJA::Layout<DIM3> layout_egrads = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> elem_deriv_shapes_view(
        elem_deriv_shapes.Read(), layout_egrads);

    const double i3 = 1.0 / 3.0;
    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;
    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        double idetJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                idetJ = 1.0 / detJ;
                c_detJ = detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes_; knds++) {
                const double bx = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(0, 0) + Gt(knds, 1, j_qpts) * A(0, 1) +
                                   Gt(knds, 2, j_qpts) * A(0, 2));

                const double by = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(1, 0) + Gt(knds, 1, j_qpts) * A(1, 1) +
                                   Gt(knds, 2, j_qpts) * A(1, 2));

                const double bz = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(2, 0) + Gt(knds, 1, j_qpts) * A(2, 1) +
                                   Gt(knds, 2, j_qpts) * A(2, 2));
                const double b4 = i3 * (elem_deriv_shapes_view(knds, 0, i_elems) - bx);
                const double b5 = b4 + bx;
                const double b6 = i3 * (elem_deriv_shapes_view(knds, 1, i_elems) - by);
                const double b7 = b6 + by;
                const double b8 = i3 * (elem_deriv_shapes_view(knds, 2, i_elems) - bz);
                const double b9 = b8 + bz;

                const double k11w = c_detJ *
                                    (b4 * K(1, 1, j_qpts, i_elems) + b4 * K(1, 2, j_qpts, i_elems) +
                                     b5 * K(1, 0, j_qpts, i_elems) + by * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 4, j_qpts, i_elems) + b4 * K(2, 1, j_qpts, i_elems) +
                                     b4 * K(2, 2, j_qpts, i_elems) + b5 * K(2, 0, j_qpts, i_elems) +
                                     by * K(2, 5, j_qpts, i_elems) + bz * K(2, 4, j_qpts, i_elems));

                const double k11x = c_detJ *
                                    (b4 * K(0, 1, j_qpts, i_elems) + b4 * K(0, 2, j_qpts, i_elems) +
                                     b5 * K(0, 0, j_qpts, i_elems) + by * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 4, j_qpts, i_elems));

                const double k11y = c_detJ *
                                    (b4 * K(5, 1, j_qpts, i_elems) + b4 * K(5, 2, j_qpts, i_elems) +
                                     b5 * K(5, 0, j_qpts, i_elems) + by * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 4, j_qpts, i_elems));

                const double k11z = c_detJ *
                                    (b4 * K(4, 1, j_qpts, i_elems) + b4 * K(4, 2, j_qpts, i_elems) +
                                     b5 * K(4, 0, j_qpts, i_elems) + by * K(4, 5, j_qpts, i_elems) +
                                     bz * K(4, 4, j_qpts, i_elems));

                const double k22w = c_detJ *
                                    (b6 * K(0, 0, j_qpts, i_elems) + b6 * K(0, 2, j_qpts, i_elems) +
                                     b7 * K(0, 1, j_qpts, i_elems) + bx * K(0, 5, j_qpts, i_elems) +
                                     bz * K(0, 3, j_qpts, i_elems) + b6 * K(2, 0, j_qpts, i_elems) +
                                     b6 * K(2, 2, j_qpts, i_elems) + b7 * K(2, 1, j_qpts, i_elems) +
                                     bx * K(2, 5, j_qpts, i_elems) + bz * K(2, 3, j_qpts, i_elems));

                const double k22x = c_detJ *
                                    (b6 * K(1, 0, j_qpts, i_elems) + b6 * K(1, 2, j_qpts, i_elems) +
                                     b7 * K(1, 1, j_qpts, i_elems) + bx * K(1, 5, j_qpts, i_elems) +
                                     bz * K(1, 3, j_qpts, i_elems));

                const double k22y = c_detJ *
                                    (b6 * K(5, 0, j_qpts, i_elems) + b6 * K(5, 2, j_qpts, i_elems) +
                                     b7 * K(5, 1, j_qpts, i_elems) + bx * K(5, 5, j_qpts, i_elems) +
                                     bz * K(5, 3, j_qpts, i_elems));

                const double k22z = c_detJ *
                                    (b6 * K(3, 0, j_qpts, i_elems) + b6 * K(3, 2, j_qpts, i_elems) +
                                     b7 * K(3, 1, j_qpts, i_elems) + bx * K(3, 5, j_qpts, i_elems) +
                                     bz * K(3, 3, j_qpts, i_elems));

                const double k33w = c_detJ *
                                    (b8 * K(0, 0, j_qpts, i_elems) + b8 * K(0, 1, j_qpts, i_elems) +
                                     b9 * K(0, 2, j_qpts, i_elems) + bx * K(0, 4, j_qpts, i_elems) +
                                     by * K(0, 3, j_qpts, i_elems) + b8 * K(1, 0, j_qpts, i_elems) +
                                     b8 * K(1, 1, j_qpts, i_elems) + b9 * K(1, 2, j_qpts, i_elems) +
                                     bx * K(1, 4, j_qpts, i_elems) + by * K(1, 3, j_qpts, i_elems));

                const double k33x = c_detJ *
                                    (b8 * K(2, 0, j_qpts, i_elems) + b8 * K(2, 1, j_qpts, i_elems) +
                                     b9 * K(2, 2, j_qpts, i_elems) + bx * K(2, 4, j_qpts, i_elems) +
                                     by * K(2, 3, j_qpts, i_elems));

                const double k33y = c_detJ *
                                    (b8 * K(4, 0, j_qpts, i_elems) + b8 * K(4, 1, j_qpts, i_elems) +
                                     b9 * K(4, 2, j_qpts, i_elems) + bx * K(4, 4, j_qpts, i_elems) +
                                     by * K(4, 3, j_qpts, i_elems));

                const double k33z = c_detJ *
                                    (b8 * K(3, 0, j_qpts, i_elems) + b8 * K(3, 1, j_qpts, i_elems) +
                                     b9 * K(3, 2, j_qpts, i_elems) + bx * K(3, 4, j_qpts, i_elems) +
                                     by * K(3, 3, j_qpts, i_elems));

                Y(knds, 0, i_elems) += b4 * k11w + b5 * k11x + by * k11y + bz * k11z;
                Y(knds, 1, i_elems) += b6 * k22w + b7 * k22x + bx * k22y + bz * k22z;
                Y(knds, 2, i_elems) += b8 * k33w + b9 * k33x + bx * k33y + by * k33z;
            }
        }
    });
}

// This performs the assembly step of our RHS side of our system:
// f_ik =
void ICExaNLFIntegrator::AssemblePA(const mfem::FiniteElementSpace& fes) {
    CALI_CXX_MARK_SCOPE("icenlfi_assemblePA");
    mfem::Mesh* mesh = fes.GetMesh();
    const mfem::FiniteElement& el = *fes.GetTypicalFE();
    space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

    nqpts = ir->GetNPoints();
    nnodes = el.GetDof();
    nelems = fes.GetNE();

    auto W = ir->GetWeights().Read();
    geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;

    if (grad.Size() != (nqpts * dim * nnodes)) {
        grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
        {
            mfem::DenseMatrix DSh;
            const int offset = nnodes * dim;
            double* qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
                el.CalcDShape(ip, DSh);
            }
        }
        grad.UseDevice(true);
    }

    if (elem_deriv_shapes.Size() != (nnodes * dim * nelems)) {
        elem_deriv_shapes.SetSize(nnodes * space_dims * nelems, mfem::Device::GetMemoryType());
        elem_deriv_shapes.UseDevice();
    }

    // geom->J really isn't going to work for us as of right now. We could just reorder it
    // to the version that we want it to be in instead...
    if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
        jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
        jacobian.UseDevice(true);
    }

    const int DIM4 = 4;
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.ReadWrite(),
                                                                  layout_jacob);

    RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{nqpts, dim, dim, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> geom_j_view(geom->J.Read(),
                                                                                  layout_geom);

    const int nqpts_ = nqpts;
    const int dim_ = dim;

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        for (int j = 0; j < nqpts_; j++) {
            for (int k = 0; k < dim_; k++) {
                for (int l = 0; l < dim_; l++) {
                    J(l, k, j, i) = geom_j_view(j, l, k, i);
                }
            }
        }
    });

    // Element-averaged shape gradients with the shared B-bar implementation
    // (utilities/mechanics_kernels.hpp), the same code the material models use.
    exaconstit::kernel::ElementAveragedShapeGradients(
        nqpts, nelems, nnodes, jacobian.Read(), grad.Read(), W, elem_deriv_shapes.Write());
}

// -----------------------------------------------------------------------------
// ICExaNLFIntegrator::AddMultGradPA
//
// Native B-bar tangent stiffness PA action: y += K̄ * x where
//   K̄ = ∫ B̄^T C B̄ dΩ
// and B̄ is the B-bar strain-displacement matrix from Hughes (1980).
//
// Because B̄ couples element-constant volume-averaged data with per-qpt data,
// we work in physical space and access C directly from the simulation state's
// tangent stiffness quadrature function.
//
// Algorithm per element, per quadrature point (q):
//   1. Hoist tr_bar (element-constant) outside the qpt loop:
//        tr_bar = sum_{a,k} N̄(a,k) * V(a,k)
//      This is the volume-averaged trace of the velocity gradient that B̄
//      uses in place of the per-qpt trace.
//
//   2. Compute the adjugate matrix and Jacobian determinant from the cached
//      Jacobian. Adjugate is used to transform reference derivatives Gt to
//      physical derivatives:
//        dN(a,j) = (1/detJ) * sum_k Gt(a,k,q) * adj(j,k)
//      (Adjugate uses inverse-transpose convention; same as in the standard
//      ExaNLFIntegrator AssembleGradPA kernel.)
//
//   3. Compute physical velocity gradient:
//        L(i,j) = sum_a dN(a,j) * V(a,i)
//
//   4. Compute B-bar trace correction:
//        Δtr = (tr_bar - tr(L)) / 3
//      and modified velocity gradient:
//        L̄(i,j) = L(i,j) + δ_ij * Δtr
//      which replaces the volumetric trace of L with tr_bar (Hughes' B-bar).
//
//   5. Apply material tangent (forward direction):
//        σ'(j,k) = sum_{l,m} C(j,k,l,m) * L̄(l,m)
//      C is fetched on the fly from the tangent_stiffness quadrature function.
//
//   6. Compute pressure (volumetric) part of σ':
//        p' = (1/3) * tr(σ')
//
//   7. Accumulate into Y with B-bar test side. The test side replaces the
//      pressure contribution to nodal forces using the volume-averaged
//      derivatives N̄ in place of the per-qpt dN:
//        Y(a,k) += [sum_j dN(a,j) σ'(j,k) + (N̄(a,k) - dN(a,k)) p'] * w * detJ
//      The first term is the standard B^T σ' force, the second redirects the
//      pressure piece through N̄.
//
// Verification properties:
//   - For symmetric C, the result must equal the forward action of any
//     symmetric formulation (B̄^T C B̄ is symmetric).
//   - For a uniform-Jacobian mesh where tr_bar agrees with the per-qpt
//     trace, Δtr → 0 at every qpt and the result must match the standard
//     (non-B-bar) result.
// -----------------------------------------------------------------------------
void ICExaNLFIntegrator::AddMultGradPA(const mfem::Vector& x, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("icenlfi_amGPA");
    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM3 = 3;
    const int DIM4 = 4;
    const int DIM6 = 6;

    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM6> perm6{{5, 4, 3, 2, 1, 0}};

    // Input / output E-vectors
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> X(x.Read(), layout_field);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);

    // Reference shape function derivatives Gt(node, dim, qpt)
    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    // Element-averaged derivatives N̄(node, dim, elem)
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Nbar(elem_deriv_shapes.Read(),
                                                                           layout_field);

    // Mesh Jacobians J(dim, dim, qpt, elem)
    RAJA::Layout<DIM4> layout_jac = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}}, perm4);
    // Use the reordered `jacobian` (filled in AssemblePA): geom->J itself is stored
    // quadrature-point-fastest, (nqpts, dim, dim, nelems), not in this layout.
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J_data(jacobian.Read(),
                                                                             layout_jac);

    // Quadrature function holding the 6x6 material tangent (used for its integration rule)
    auto tangent_qf = m_sim_state->GetQuadratureFunction("tangent_stiffness");
    RAJA::Layout<DIM6> layout_C = RAJA::make_permuted_layout({{dim, dim, dim, dim, nqpts, nelems}},
                                                             perm6);
    // 3x3x3x3 tangent built from the 6x6 "tangent_stiffness" QF in AssembleGradPA
    RAJA::View<const double, RAJA::Layout<DIM6, RAJA::Index_type, 0>> C(pa_mat.Read(), layout_C);

    // Integration weights from the tangent stiffness QF integration rule
    const mfem::IntegrationRule& ir = tangent_qf->GetSpaceShared()->GetIntRule(0);
    auto W = ir.GetWeights().Read();

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int e) {
        // Step 1: Hoist tr_bar outside the qpt loop (element-constant)
        double tr_bar = 0.0;
        for (int a = 0; a < nnodes_; a++) {
            for (int k = 0; k < dim_; k++) {
                tr_bar += Nbar(a, k, e) * X(a, k, e);
            }
        }

        for (int q = 0; q < nqpts_; q++) {
            // Step 2: Compute adjugate and Jacobian determinant
            // J_data(r, c) = J_rc = dx_r / dxi_c
            const double J11 = J_data(0, 0, q, e), J12 = J_data(0, 1, q, e),
                         J13 = J_data(0, 2, q, e);
            const double J21 = J_data(1, 0, q, e), J22 = J_data(1, 1, q, e),
                         J23 = J_data(1, 2, q, e);
            const double J31 = J_data(2, 0, q, e), J32 = J_data(2, 1, q, e),
                         J33 = J_data(2, 2, q, e);

            double adj[9];
            adj[0] = (J22 * J33) - (J23 * J32); // 0,0
            adj[1] = (J23 * J31) - (J21 * J33); // 0,1
            adj[2] = (J21 * J32) - (J22 * J31); // 0,2
            adj[3] = (J13 * J32) - (J12 * J33); // 1,0
            adj[4] = (J11 * J33) - (J13 * J31); // 1,1
            adj[5] = (J12 * J31) - (J11 * J32); // 1,2
            adj[6] = (J12 * J23) - (J13 * J22); // 2,0
            adj[7] = (J13 * J21) - (J11 * J23); // 2,1
            adj[8] = (J11 * J22) - (J12 * J21); // 2,2

            const double detJ = J11 * adj[0] + J21 * adj[3] + J31 * adj[6];
            const double idetJ = 1.0 / detJ;
            const double w_detJ = W[q] * detJ;

            // Step 3: Physical velocity gradient L(i,j) = sum_a dN(a,j) * V(a,i)
            // We compute dN(a, :) on-the-fly from Gt and adj.
            double L[3][3] = {{0.0}};
            for (int a = 0; a < nnodes_; a++) {
                double dNa[3];
                for (int j = 0; j < dim_; j++) {
                    dNa[j] = idetJ * (Gt(a, 0, q) * adj[j * 3 + 0] + Gt(a, 1, q) * adj[j * 3 + 1] +
                                      Gt(a, 2, q) * adj[j * 3 + 2]);
                }
                for (int i = 0; i < dim_; i++) {
                    for (int j = 0; j < dim_; j++) {
                        L[i][j] += dNa[j] * X(a, i, e);
                    }
                }
            }

            // Step 4: B-bar trace correction
            const double tr_std = L[0][0] + L[1][1] + L[2][2];
            const double dtr = (tr_bar - tr_std) / 3.0;

            double Lbar[3][3];
            for (int i = 0; i < dim_; i++) {
                for (int j = 0; j < dim_; j++) {
                    Lbar[i][j] = L[i][j];
                }
            }
            Lbar[0][0] += dtr;
            Lbar[1][1] += dtr;
            Lbar[2][2] += dtr;

            // Step 5: Apply material tangent — forward contraction
            //   σ'(j, k) = sum_{l,m} C(j, k, l, m) * L̄(l, m)
            double sigma[3][3] = {{0.0}};
            for (int j = 0; j < dim_; j++) {
                for (int k = 0; k < dim_; k++) {
                    for (int l = 0; l < dim_; l++) {
                        for (int m = 0; m < dim_; m++) {
                            sigma[j][k] += C(j, k, l, m, q, e) * Lbar[l][m];
                        }
                    }
                }
            }

            // Step 6: Pressure (volumetric) part of σ'
            const double p = (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;

            // Step 7: Accumulate forces with B-bar test side
            //   Y(a, k) += [sum_j dN(a,j) σ'(j,k) + (N̄(a,k) - dN(a,k)) p] * w * detJ
            for (int a = 0; a < nnodes_; a++) {
                double dNa[3];
                for (int j = 0; j < dim_; j++) {
                    dNa[j] = idetJ * (Gt(a, 0, q) * adj[j * 3 + 0] + Gt(a, 1, q) * adj[j * 3 + 1] +
                                      Gt(a, 2, q) * adj[j * 3 + 2]);
                }
                for (int k = 0; k < dim_; k++) {
                    double f_std = 0.0;
                    for (int j = 0; j < dim_; j++) {
                        f_std += dNa[j] * sigma[j][k];
                    }
                    double f_bbar = (Nbar(a, k, e) - dNa[k]) * p;
                    Y(a, k, e) += (f_std + f_bbar) * w_detJ;
                }
            }
        } // End of qpts
    }); // End of nelems
}

// -----------------------------------------------------------------------------
// ICExaNLFIntegrator::AddMultTransposeGradPA
//
// Native transposed B-bar tangent stiffness PA action: y += K̄^T * x.
//
// This is structurally IDENTICAL to AddMultGradPA except for one line: the
// material tangent contraction uses C(l,m,j,k) instead of C(j,k,l,m). The
// B-bar geometry (N̄, dN, trace correction, pressure redirection) is the
// same on both sides of K̄ = B̄^T C B̄ because:
//   (B̄^T C B̄)^T = B̄^T C^T B̄
// — only the middle factor C transposes; the outer B̄^T and B̄ remain in
// place.
//
// For symmetric C, this kernel produces results identical to AddMultGradPA
// (a useful verification check). For non-symmetric C (crystal plasticity
// with non-associated flow or non-symmetric Schmid coupling) it produces
// genuinely different results, as required for correct trust-region
// Cauchy point computation.
// -----------------------------------------------------------------------------
void ICExaNLFIntegrator::AddMultTransposeGradPA(const mfem::Vector& x, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("icenlfi_amTGPA");
    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM3 = 3;
    const int DIM4 = 4;
    const int DIM6 = 6;

    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM6> perm6{{5, 4, 3, 2, 1, 0}};

    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> X(x.Read(), layout_field);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);

    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Nbar(elem_deriv_shapes.Read(),
                                                                           layout_field);

    RAJA::Layout<DIM4> layout_jac = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}}, perm4);
    // Use the reordered `jacobian` (filled in AssemblePA): geom->J itself is stored
    // quadrature-point-fastest, (nqpts, dim, dim, nelems), not in this layout.
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J_data(jacobian.Read(),
                                                                             layout_jac);

    auto tangent_qf = m_sim_state->GetQuadratureFunction("tangent_stiffness");
    RAJA::Layout<DIM6> layout_C = RAJA::make_permuted_layout({{dim, dim, dim, dim, nqpts, nelems}},
                                                             perm6);
    // 3x3x3x3 tangent built from the 6x6 "tangent_stiffness" QF in AssembleGradPA
    RAJA::View<const double, RAJA::Layout<DIM6, RAJA::Index_type, 0>> C(pa_mat.Read(), layout_C);

    const mfem::IntegrationRule& ir = tangent_qf->GetSpaceShared()->GetIntRule(0);
    auto W = ir.GetWeights().Read();

    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;

    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int e) {
        // Step 1: Hoist tr_bar (element-constant)
        double tr_bar = 0.0;
        for (int a = 0; a < nnodes_; a++) {
            for (int k = 0; k < dim_; k++) {
                tr_bar += Nbar(a, k, e) * X(a, k, e);
            }
        }

        for (int q = 0; q < nqpts_; q++) {
            // Step 2: Adjugate and Jacobian determinant
            // J_data(r, c) = J_rc = dx_r / dxi_c
            const double J11 = J_data(0, 0, q, e), J12 = J_data(0, 1, q, e),
                         J13 = J_data(0, 2, q, e);
            const double J21 = J_data(1, 0, q, e), J22 = J_data(1, 1, q, e),
                         J23 = J_data(1, 2, q, e);
            const double J31 = J_data(2, 0, q, e), J32 = J_data(2, 1, q, e),
                         J33 = J_data(2, 2, q, e);

            double adj[9];
            adj[0] = (J22 * J33) - (J23 * J32);
            adj[1] = (J23 * J31) - (J21 * J33);
            adj[2] = (J21 * J32) - (J22 * J31);
            adj[3] = (J13 * J32) - (J12 * J33);
            adj[4] = (J11 * J33) - (J13 * J31);
            adj[5] = (J12 * J31) - (J11 * J32);
            adj[6] = (J12 * J23) - (J13 * J22);
            adj[7] = (J13 * J21) - (J11 * J23);
            adj[8] = (J11 * J22) - (J12 * J21);

            const double detJ = J11 * adj[0] + J21 * adj[3] + J31 * adj[6];
            const double idetJ = 1.0 / detJ;
            const double w_detJ = W[q] * detJ;

            // Step 3: Physical velocity gradient
            double L[3][3] = {{0.0}};
            for (int a = 0; a < nnodes_; a++) {
                double dNa[3];
                for (int j = 0; j < dim_; j++) {
                    dNa[j] = idetJ * (Gt(a, 0, q) * adj[j * 3 + 0] + Gt(a, 1, q) * adj[j * 3 + 1] +
                                      Gt(a, 2, q) * adj[j * 3 + 2]);
                }
                for (int i = 0; i < dim_; i++) {
                    for (int j = 0; j < dim_; j++) {
                        L[i][j] += dNa[j] * X(a, i, e);
                    }
                }
            }

            // Step 4: B-bar trace correction
            const double tr_std = L[0][0] + L[1][1] + L[2][2];
            const double dtr = (tr_bar - tr_std) / 3.0;

            double Lbar[3][3];
            for (int i = 0; i < dim_; i++) {
                for (int j = 0; j < dim_; j++) {
                    Lbar[i][j] = L[i][j];
                }
            }
            Lbar[0][0] += dtr;
            Lbar[1][1] += dtr;
            Lbar[2][2] += dtr;

            // Step 5: TRANSPOSED material tangent contraction
            //   σ'(j, k) = sum_{l,m} C(l, m, j, k) * L̄(l, m)
            // (Compare to forward: C(j, k, l, m) * L̄(l, m))
            double sigma[3][3] = {{0.0}};
            for (int j = 0; j < dim_; j++) {
                for (int k = 0; k < dim_; k++) {
                    for (int l = 0; l < dim_; l++) {
                        for (int m = 0; m < dim_; m++) {
                            sigma[j][k] += C(l, m, j, k, q, e) * Lbar[l][m];
                        }
                    }
                }
            }

            // Step 6: Pressure
            const double p = (sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0;

            // Step 7: Accumulate with B-bar test side (same as forward kernel)
            for (int a = 0; a < nnodes_; a++) {
                double dNa[3];
                for (int j = 0; j < dim_; j++) {
                    dNa[j] = idetJ * (Gt(a, 0, q) * adj[j * 3 + 0] + Gt(a, 1, q) * adj[j * 3 + 1] +
                                      Gt(a, 2, q) * adj[j * 3 + 2]);
                }
                for (int k = 0; k < dim_; k++) {
                    double f_std = 0.0;
                    for (int j = 0; j < dim_; j++) {
                        f_std += dNa[j] * sigma[j][k];
                    }
                    double f_bbar = (Nbar(a, k, e) - dNa[k]) * p;
                    Y(a, k, e) += (f_std + f_bbar) * w_detJ;
                }
            }
        } // End of qpts
    }); // End of nelems
}

// Here we're applying the following action operation using the assembled "D" 2nd order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jk}
void ICExaNLFIntegrator::AddMultPA(const mfem::Vector& /*x*/, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("icenlfi_amPAV");

    // return a pointer to beginning step stress. This is used for output visualization
    auto stress_end = m_sim_state->GetQuadratureFunction("cauchy_stress_end");

    const mfem::IntegrationRule& ir =
        m_sim_state->GetQuadratureFunction("tangent_stiffness")->GetSpaceShared()->GetIntRule(0);
    auto W = ir.GetWeights().Read();

    MFEM_VERIFY_0(space_dims == 3, "Dimensions of 1 or 2 not supported.");

    const int dim = 3;
    const int DIM2 = 2;
    const int DIM3 = 3;
    const int DIM4 = 4;

    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                 perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J(jacobian.Read(),
                                                                        layout_jacob);

    RAJA::Layout<DIM3> layout_stress = RAJA::make_permuted_layout({{2 * dim, nqpts, nelems}},
                                                                  perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> S(stress_end->ReadWrite(),
                                                                        layout_stress);

    // Our field variables that are inputs and outputs
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Y(y.ReadWrite(), layout_field);
    // Transpose of the local gradient variable
    RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> Gt(grad.Read(), layout_grads);

    RAJA::Layout<DIM3> layout_egrads = RAJA::make_permuted_layout({{nnodes, dim, nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> elem_deriv_shapes_view(
        elem_deriv_shapes.Read(), layout_egrads);

    RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    const double i3 = 1.0 / 3.0;
    const int nqpts_ = nqpts;
    const int dim_ = dim;
    const int nnodes_ = nnodes;

    // This loop we'll want to parallelize the rest are all serial for now.
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_elems) {
        double adj[dim_ * dim_];
        double c_detJ;
        double idetJ;
        // So, we're going to say this view is constant however we're going to mutate the values
        // only in that one scoped section for the quadrature points.
        RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> A(&adj[0], layout_adj);
        for (int j_qpts = 0; j_qpts < nqpts_; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around
            // with us for the adjugate term.
            {
                const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
                const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
                const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
                const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
                const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
                const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
                const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
                const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
                const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
                const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                    /* */ J21 * (J12 * J33 - J32 * J13) +
                                    /* */ J31 * (J12 * J23 - J22 * J13);
                idetJ = 1.0 / detJ;
                c_detJ = detJ * W[j_qpts];
                // adj(J)
                adj[0] = (J22 * J33) - (J23 * J32); // 0,0
                adj[1] = (J32 * J13) - (J12 * J33); // 0,1
                adj[2] = (J12 * J23) - (J22 * J13); // 0,2
                adj[3] = (J31 * J23) - (J21 * J33); // 1,0
                adj[4] = (J11 * J33) - (J13 * J31); // 1,1
                adj[5] = (J21 * J13) - (J11 * J23); // 1,2
                adj[6] = (J21 * J32) - (J31 * J22); // 2,0
                adj[7] = (J31 * J12) - (J11 * J32); // 2,1
                adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes_; knds++) {
                const double bx = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(0, 0) + Gt(knds, 1, j_qpts) * A(0, 1) +
                                   Gt(knds, 2, j_qpts) * A(0, 2));

                const double by = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(1, 0) + Gt(knds, 1, j_qpts) * A(1, 1) +
                                   Gt(knds, 2, j_qpts) * A(1, 2));

                const double bz = idetJ *
                                  (Gt(knds, 0, j_qpts) * A(2, 0) + Gt(knds, 1, j_qpts) * A(2, 1) +
                                   Gt(knds, 2, j_qpts) * A(2, 2));

                const double b4 = i3 * (elem_deriv_shapes_view(knds, 0, i_elems) - bx);
                const double b5 = b4 + bx;
                const double b6 = i3 * (elem_deriv_shapes_view(knds, 1, i_elems) - by);
                const double b7 = b6 + by;
                const double b8 = i3 * (elem_deriv_shapes_view(knds, 2, i_elems) - bz);
                const double b9 = b8 + bz;

                Y(knds, 0, i_elems) += c_detJ *
                                       (b4 * S(1, j_qpts, i_elems) + b4 * S(2, j_qpts, i_elems) +
                                        b5 * S(0, j_qpts, i_elems) + by * S(5, j_qpts, i_elems) +
                                        bz * S(4, j_qpts, i_elems));

                Y(knds, 1, i_elems) += c_detJ *
                                       (b6 * S(0, j_qpts, i_elems) + b6 * S(2, j_qpts, i_elems) +
                                        b7 * S(1, j_qpts, i_elems) + bx * S(5, j_qpts, i_elems) +
                                        bz * S(3, j_qpts, i_elems));

                Y(knds, 2, i_elems) += c_detJ *
                                       (b8 * S(0, j_qpts, i_elems) + b8 * S(1, j_qpts, i_elems) +
                                        b9 * S(2, j_qpts, i_elems) + bx * S(4, j_qpts, i_elems) +
                                        by * S(3, j_qpts, i_elems));
            } // End of nnodes
        } // End of nQpts
    }); // End of nelems
}
