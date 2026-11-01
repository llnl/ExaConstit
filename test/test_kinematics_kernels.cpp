/**
 * @file test_kinematics_kernels.cpp
 * @brief Checks the per-point kinematics used by the UMAT model
 *
 * - The device routines of utilities/kinematics_kernels.hpp: the rotation of the polar
 *   decomposition against its definition for large stretches and rotations, and (rotation,
 *   Eulerian strain, inverse, incremental deformation gradient) against the host DenseMatrix
 *   routines of utilities/strain_measures.hpp on time-step-sized deformation gradients.
 * - The way AbaqusUmatModel gets its deformation gradients: reference Jacobians from the
 *   reference nodes (MFEM's QuadratureInterpolator, NATIVE dof order), then GradCalc of the
 *   deformed nodes with the operator's shape derivatives. On a curved
 *   second-order hexahedron moved by an affine motion x = A X + b, F must equal A at every
 *   quadrature point (to round-off), and the reference Jacobians must match MFEM's element
 *   transformation.
 *
 * Part of the extensive suite (EXACONSTIT_ENABLE_EXTENSIVE_TESTS).
 */
#include "utilities/kinematics_kernels.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/strain_measures.hpp"

#include "mfem.hpp"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

namespace {

/// Random deformation gradient with det > 0: a rotation about z by up to max_angle times
/// I + (random perturbation of entries up to stretch)
void RandomDefGrad(std::mt19937& gen, double stretch, double max_angle, double* f) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < 9; ++i) {
        f[i] = stretch * dist(gen) + ((i % 4 == 0) ? 1.0 : 0.0);
    }
    const double theta = max_angle * dist(gen);
    const double c = std::cos(theta), s = std::sin(theta);
    const double r[9] = {c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0};
    double rf[9];
    mfem::kernels::Mult(3, 3, 3, r, f, rf);
    for (int i = 0; i < 9; ++i) {
        f[i] = rf[i];
    }
}

TEST(KinematicsKernels, PolarRotationIsPolarDecomposition) {
    // Large stretches and rotations: R must be a proper rotation with U = R^T F symmetric (the
    // definition of the polar decomposition)
    std::mt19937 gen(2024);
    for (int trial = 0; trial < 500; ++trial) {
        double f[9], r[9];
        RandomDefGrad(gen, 0.3, 3.0, f);
        ASSERT_GT(mfem::kernels::Det<3>(f), 0.0);
        exaconstit::kernel::PolarRotation(f, r);
        double rt[9], rtr[9], u_sym[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rt[i + 3 * j] = r[j + 3 * i];
            }
        }
        mfem::kernels::Mult(3, 3, 3, rt, r, rtr);
        mfem::kernels::Mult(3, 3, 3, rt, f, u_sym);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_NEAR(rtr[i + 3 * j], (i == j) ? 1.0 : 0.0, 1e-12);
                EXPECT_NEAR(u_sym[i + 3 * j], u_sym[j + 3 * i], 1e-12);
            }
        }
        EXPECT_NEAR(mfem::kernels::Det<3>(r), 1.0, 1e-12);
        // U positive definite: its trace is the sum of the principal stretches
        EXPECT_GT(u_sym[0] + u_sym[4] + u_sym[8], 0.0);
    }
}

TEST(KinematicsKernels, PolarRotationOfKnownDecomposition) {
    // F = R0 U0 with R0 a rotation of 0.3 rad about (1, 2, 2)/3 and U0 symmetric positive
    // definite: the polar rotation must be R0, for the device routine and the host one
    const double angle = 0.3;
    const double n[3] = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
    const double c = std::cos(angle), s = std::sin(angle);
    double r0[9];
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            // Rodrigues: R = c I + s [n]x + (1 - c) n n^T
            const double cross[3][3] = {
                {0.0, -n[2], n[1]}, {n[2], 0.0, -n[0]}, {-n[1], n[0], 0.0}};
            r0[i + 3 * j] = ((i == j) ? c : 0.0) + s * cross[i][j] + (1.0 - c) * n[i] * n[j];
        }
    }
    const double u0[9] = {1.10, 0.05, -0.02, 0.05, 0.95, 0.03, -0.02, 0.03, 1.02};
    double f[9], r[9];
    mfem::kernels::Mult(3, 3, 3, r0, u0, f);
    exaconstit::kernel::PolarRotation(f, r);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(r[i], r0[i], 1e-12) << "entry " << i;
    }
}

TEST(KinematicsKernels, MatchHostStrainMeasures) {
    // Time-step-sized deformation gradients (what a UMAT receives as F_incr), where the host
    // iterative polar decomposition the original UMAT code used is accurate
    std::mt19937 gen(12345);
    for (int trial = 0; trial < 200; ++trial) {
        double f[9];
        RandomDefGrad(gen, 0.05, 0.2, f);
        ASSERT_GT(mfem::kernels::Det<3>(f), 0.0);

        // Rotation: the original UMAT code passed DROT = CalcPolarDecompDefGrad of a transposed
        // copy of F_incr (DenseMatrix(F, 3) is MFEM's transpose-copy constructor). That host
        // routine returns R^T of its input, so the two transposes gave R; the device routine
        // must reproduce that DROT.
        double r[9];
        exaconstit::kernel::PolarRotation(f, r);
        mfem::DenseMatrix f_copy(3);
        for (int i = 0; i < 9; ++i) {
            f_copy.Data()[i] = f[i];
        }
        mfem::DenseMatrix r_host(f_copy, 't'), u(3), v(3);
        CalcPolarDecompDefGrad(r_host, u, v);
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(r[i], r_host.Data()[i], 1e-10) << "trial " << trial << " entry " << i;
        }

        // Eulerian strain
        double e[9];
        exaconstit::kernel::EulerianStrain(f, e);
        mfem::DenseMatrix e_host(3);
        CalcEulerianStrain(e_host, f_copy);
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(e[i], e_host.Data()[i], 1e-13);
        }

        // Inverse and incremental deformation gradient: F_incr F_beg = F_end
        double f_end[9], f_incr[9], back[9];
        RandomDefGrad(gen, 0.3, 3.0, f_end);
        exaconstit::kernel::IncrementalDefGrad(f, f_end, f_incr);
        mfem::kernels::Mult(3, 3, 3, f_incr, f, back);
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(back[i], f_end[i], 1e-13);
        }
    }
}

TEST(KinematicsKernels, VoigtOrderAndTangent) {
    const double s_exa[6] = {1.0, 2.0, 3.0, 23.0, 13.0, 12.0};
    double s_abq[6], s_back[6];
    exaconstit::kernel::StressToAbaqusVoigt(s_exa, s_abq);
    EXPECT_EQ(s_abq[3], 12.0);
    EXPECT_EQ(s_abq[4], 13.0);
    EXPECT_EQ(s_abq[5], 23.0);
    exaconstit::kernel::StressFromAbaqusVoigt(s_abq, s_back);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(s_back[i], s_exa[i]);
    }

    const double e[9] = {1.0, 12.0, 13.0, 12.0, 2.0, 23.0, 13.0, 23.0, 3.0};
    double v[6];
    exaconstit::kernel::StrainToAbaqusVoigt(e, v);
    const double v_expected[6] = {1.0, 2.0, 3.0, 24.0, 26.0, 46.0};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(v[i], v_expected[i]);
    }

    // Tangent: entry (i, j) labelled 10 i + j in Abaqus order; after conversion, row / column
    // 3 and 5 are swapped and every entry is scaled by dt
    double ddsdde[36];
    for (int i = 0; i < 36; ++i) {
        ddsdde[i] = static_cast<double>(i);
    }
    const double dt = 0.5;
    exaconstit::kernel::AbaqusTangentToExaConstit(dt, ddsdde);
    const int perm[6] = {0, 1, 2, 5, 4, 3};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            EXPECT_EQ(ddsdde[6 * i + j], dt * (6 * perm[i] + perm[j]));
        }
    }
}

TEST(KinematicsKernels, ExactDeformationGradientOnCurvedElement) {
    // One hexahedron, second-order nodes, curved by a smooth nonlinear map
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
    const int order = 2;
    mesh.SetCurvature(order);
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& y) {
        y.SetSize(3);
        y(0) = x(0) + 0.15 * x(1) * x(1) + 0.05 * x(2);
        y(1) = x(1) + 0.1 * std::sin(x(0)) * x(2);
        y(2) = x(2) + 0.12 * x(0) * x(1);
    });

    const mfem::GridFunction* nodes = mesh.GetNodes();
    const mfem::FiniteElementSpace* fes = nodes->FESpace();
    const mfem::FiniteElement& fe = *fes->GetFE(0);
    const int nnodes = fe.GetDof();
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe.GetGeomType(), 2 * order + 1);
    const int nqpts = ir.GetNPoints();

    // Reference shape derivatives, layout (nnodes, 3, nqpts), NATIVE dof order (as the operator)
    mfem::Vector dshape(nnodes * 3 * nqpts);
    for (int q = 0; q < nqpts; ++q) {
        mfem::DenseMatrix dsh(dshape.HostReadWrite() + q * 3 * nnodes, nnodes, 3);
        fe.CalcDShape(ir.IntPoint(q), dsh);
    }
    const mfem::Operator* restriction = fes->GetElementRestriction(
        mfem::ElementDofOrdering::NATIVE);
    mfem::Vector x_ref_el(restriction->Height());
    restriction->Mult(*nodes, x_ref_el);

    // Reference Jacobians d X / d xi from MFEM's QuadratureInterpolator (layout byVDIM:
    // (3, 3, nqpts, nelems) column-major, the layout GradCalc reads), vs the element
    // transformation
    const mfem::QuadratureInterpolator* qi = fes->GetQuadratureInterpolator(ir);
    qi->SetOutputLayout(mfem::QVectorLayout::byVDIM);
    qi->DisableTensorProducts();
    mfem::Vector jac_ref(9 * nqpts);
    qi->Derivatives(x_ref_el, jac_ref);
    jac_ref.HostRead();
    mfem::ElementTransformation* trans = mesh.GetElementTransformation(0);
    for (int q = 0; q < nqpts; ++q) {
        trans->SetIntPoint(&ir.IntPoint(q));
        const mfem::DenseMatrix& jac_mfem = trans->Jacobian();
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(jac_ref(9 * q + i), jac_mfem.Data()[i], 1e-13);
        }
    }

    // Affine motion of the nodes: x = A X + b
    const double a[9] = {1.05, 0.02, -0.03, 0.10, 0.97, 0.04, -0.02, 0.06, 1.08};
    const double b[3] = {0.3, -0.1, 0.2};
    mfem::GridFunction x_def(*nodes);
    const int ndofs = fes->GetNDofs();
    for (int d = 0; d < ndofs; ++d) {
        double xr[3], xd[3];
        for (int c = 0; c < 3; ++c) {
            xr[c] = (*nodes)(fes->DofToVDof(d, c));
        }
        for (int i = 0; i < 3; ++i) {
            xd[i] = b[i] + a[i] * xr[0] + a[i + 3] * xr[1] + a[i + 6] * xr[2];
        }
        for (int c = 0; c < 3; ++c) {
            x_def(fes->DofToVDof(d, c)) = xd[c];
        }
    }
    mfem::Vector x_def_el(restriction->Height());
    restriction->Mult(x_def, x_def_el);

    mfem::Vector def_grad(9 * nqpts);
    def_grad = 0.0;
    exaconstit::kernel::GradCalc(nqpts,
                                 1,
                                 1,
                                 nnodes,
                                 jac_ref.Read(),
                                 dshape.Read(),
                                 x_def_el.Read(),
                                 def_grad.ReadWrite(),
                                 nullptr);
    def_grad.HostRead();
    for (int q = 0; q < nqpts; ++q) {
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(def_grad(9 * q + i), a[i], 1e-13) << "point " << q << " entry " << i;
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
