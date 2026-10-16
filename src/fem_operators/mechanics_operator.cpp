
#include "fem_operators/mechanics_operator.hpp"

#include "models/mechanics_multi_model.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
void GetTrueDofsParallel(const mfem::ParGridFunction& gf, mfem::Vector& true_dofs) {
    // used to do something like:
    // gf.GetTrueDofs(true_dofs);
    // but looks like there are issues with that on the GPUs with newer versions of MFEM
    gf.ParallelAverage(true_dofs);
}
} // namespace

NonlinearMechOperator::NonlinearMechOperator(mfem::Array<int>& ess_bdr,
                                             mfem::Array2D<bool>& ess_bdr_comp,
                                             std::shared_ptr<SimulationState> sim_state)
    : mfem::NonlinearForm(sim_state->GetMeshParFiniteElementSpace().get()),
      ess_bdr_comps(ess_bdr_comp), m_sim_state(sim_state) {
    CALI_CXX_MARK_SCOPE("mechop_class_setup");
    mfem::Vector* rhs;
    rhs = nullptr;

    const auto& options = m_sim_state->GetOptions();
    auto loc_fe_space = m_sim_state->GetMeshParFiniteElementSpace();

    // Define the parallel nonlinear form
    h_form = std::make_unique<mfem::ParNonlinearForm>(
        m_sim_state->GetMeshParFiniteElementSpace().get());

    // Set the essential boundary conditions
    h_form->SetEssentialBC(ess_bdr, ess_bdr_comps, rhs);

    // Set the essential boundary conditions that we can store on our class
    SetEssentialBC(ess_bdr, ess_bdr_comps, rhs);

    assembly = options.solvers.assembly;

    model = std::make_shared<MultiExaModel>(m_sim_state, options);
    // Add the user defined integrator
    if (options.solvers.integ_model == IntegrationModel::DEFAULT) {
        h_form->AddDomainIntegrator(new ExaNLFIntegrator(m_sim_state));
    } else if (options.solvers.integ_model == IntegrationModel::BBAR) {
        h_form->AddDomainIntegrator(new ICExaNLFIntegrator(m_sim_state));
    }

    if (assembly == AssemblyType::PA) {
        h_form->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL, mfem::ElementDofOrdering::NATIVE);
        diag.SetSize(loc_fe_space->GetTrueVSize(), mfem::Device::GetMemoryType());
        diag.UseDevice(true);
        diag = 1.0;
        prec_oper = std::make_shared<MechOperatorJacobiSmoother>(diag,
                                                                 this->GetEssentialTrueDofs());
    } else if (assembly == AssemblyType::EA) {
        h_form->SetAssemblyLevel(mfem::AssemblyLevel::ELEMENT, mfem::ElementDofOrdering::NATIVE);
        diag.SetSize(loc_fe_space->GetTrueVSize(), mfem::Device::GetMemoryType());
        diag.UseDevice(true);
        diag = 1.0;
        prec_oper = std::make_shared<MechOperatorJacobiSmoother>(diag,
                                                                 this->GetEssentialTrueDofs());
    }

    // So, we're going to originally support non tensor-product type elements originally.
    const mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE;
    // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
    elem_restrict_lex = loc_fe_space->GetElementRestriction(ordering);

    el_x.SetSize(elem_restrict_lex->Height(), mfem::Device::GetMemoryType());
    el_x.UseDevice(true);
    px.SetSize(P->Height(), mfem::Device::GetMemoryType());
    px.UseDevice(true);

    {
        const mfem::FiniteElement& el = *loc_fe_space->GetTypicalFE();
        const int space_dims = el.GetDim();
        const mfem::IntegrationRule* ir = &(
            mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        ;

        const int nqpts = ir->GetNPoints();
        const int ndofs = el.GetDof();
        const int nelems = loc_fe_space->GetNE();

        el_jac.SetSize(space_dims * space_dims * nqpts * nelems, mfem::Device::GetMemoryType());
        el_jac.UseDevice(true);

        qpts_dshape.SetSize(nqpts * space_dims * ndofs, mfem::Device::GetMemoryType());
        qpts_dshape.UseDevice(true);
        {
            mfem::DenseMatrix DSh;
            const int offset = ndofs * space_dims;
            double* qpts_dshape_data = qpts_dshape.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
                const mfem::IntegrationPoint& ip = ir->IntPoint(i);
                DSh.UseExternalData(&qpts_dshape_data[offset * i], ndofs, space_dims);
                el.CalcDShape(ip, DSh);
            }
        }
    }
}

const mfem::Array<int>& NonlinearMechOperator::GetEssTDofList() {
    return h_form->GetEssentialTrueDofs();
}

void NonlinearMechOperator::UpdateEssTDofs(const mfem::Array<int>& ess_bdr, bool mono_def_flag) {
    if (mono_def_flag) {
        h_form->SetEssentialTrueDofs(ess_bdr);
        ess_tdof_list = ess_bdr;
    } else {
        // Set the essential boundary conditions
        h_form->SetEssentialBC(ess_bdr, ess_bdr_comps, nullptr);
        auto tmp = h_form->GetEssentialTrueDofs();
        // Set the essential boundary conditions that we can store on our class
        SetEssentialBC(ess_bdr, ess_bdr_comps, nullptr);
    }
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const mfem::Vector& k, mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("mechop_Mult");
    // We first run a setup step before actually doing anything.
    // We'll want to move this outside of Mult() at some given point in time
    // and have it live in the NR solver itself or whatever solver
    // we're going to be using.
    Setup<true>(k);
    // We now perform our element vector operation.
    CALI_MARK_BEGIN("mechop_mult_setup");
    // Assemble our operator
    h_form->Setup();
    CALI_MARK_END("mechop_mult_setup");
    CALI_MARK_BEGIN("mechop_mult_Mult");
    h_form->Mult(k, y);
    CALI_MARK_END("mechop_mult_Mult");
}

template <bool upd_crds>
void NonlinearMechOperator::Setup(const mfem::Vector& k) const {
    CALI_CXX_MARK_SCOPE("mechop_setup");
    // Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
    // Solver class to use the NonlinearMechOperator instead of Operator class.
    // We now update our end coordinates based on the solved for velocity.
    if (upd_crds) {
        UpdateEndCoords(k);
    }

    // This performs the computation of the velocity gradient if needed,
    // det(J), material tangent stiffness matrix, state variable update,
    // stress update, and other stuff that might be needed in the integrators.
    auto loc_fe_space = m_sim_state->GetMeshParFiniteElementSpace();

    const mfem::FiniteElement& el = *loc_fe_space->GetTypicalFE();
    const int space_dims = el.GetDim();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    ;

    const int nqpts = ir->GetNPoints();
    const int ndofs = el.GetDof();
    const int nelems = loc_fe_space->GetNE();

    SetupJacobianTerms();

    // We can now make the call to our material model set-up stage...
    // Everything else that we need should live on the class.
    // Within this function the model just needs to produce the Cauchy stress
    // and the material tangent matrix (d \sigma / d Vgrad_{sym})
    // bool succeed_t = false;
    bool succeed = false;
    try {
        // Takes in k vector and transforms into into our E-vector array
        P->Mult(k, px);
        elem_restrict_lex->Mult(px, el_x);
        model->ModelSetup(nqpts, nelems, space_dims, ndofs, el_jac, qpts_dshape, el_x);
        succeed = true;
    } catch (const std::exception& exc) {
        // catch anything thrown within try block that derives from std::exception
        MFEM_WARNING_0(exc.what());
        succeed = false;
    } catch (...) {
        succeed = false;
    }
    // MPI_Allreduce(&succeed_t, &succeed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (!succeed) {
        throw std::runtime_error(std::string(
            "Material model setup portion of code failed for at least one integration point."));
    }
} // End of model setup

void NonlinearMechOperator::SetupJacobianTerms() const {
    auto mesh = m_sim_state->GetMesh();
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    const mfem::FiniteElement& el = *fe_space->GetTypicalFE();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    ;

    const int space_dims = el.GetDim();
    const int nqpts = ir->GetNPoints();
    const int nelems = fe_space->GetNE();

    // We need to make sure these are deleted at the start of each iteration
    // since we have meshes that are constantly changing.
    mesh->DeleteGeometricFactors();
    const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
        *ir, mfem::GeometricFactors::JACOBIANS);
    // geom->J really isn't going to work for us as of right now. We could just reorder it
    // to the version that we want it to be in instead...

    const int DIM4 = 4;
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout(
        {{space_dims, space_dims, nqpts, nelems}}, perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> jac_view(el_jac.ReadWrite(),
                                                                         layout_jacob);

    RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout(
        {{nqpts, space_dims, space_dims, nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> geom_j_view(geom->J.Read(),
                                                                                  layout_geom);

    const int nqpts1 = nqpts;
    const int space_dims1 = space_dims;
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i) {
        const int nqpts_ = nqpts1;
        const int space_dims_ = space_dims1;
        for (int j = 0; j < nqpts_; j++) {
            for (int k = 0; k < space_dims_; k++) {
                for (int l = 0; l < space_dims_; l++) {
                    jac_view(l, k, j, i) = geom_j_view(j, l, k, i);
                }
            }
        }
    });
}

void NonlinearMechOperator::CalculateDeformationGradient(mfem::QuadratureFunction& def_grad) const {
    auto mesh = m_sim_state->GetMesh();
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    const mfem::FiniteElement& el = *fe_space->GetTypicalFE();
    const mfem::IntegrationRule* ir = &(
        mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
    ;

    const int nqpts = ir->GetNPoints();
    const int nelems = fe_space->GetNE();
    const int ndofs = fe_space->GetTypicalFE()->GetDof();

    auto x_ref = m_sim_state->GetRefCoords();
    auto x_cur = m_sim_state->GetCurrentCoords();
    // Since we never modify our mesh nodes during this operations this is okay.
    mfem::GridFunction* nodes =
        x_ref.get(); // set a nodes grid function to global current configuration
    int owns_nodes = 0;
    mesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
    SetupJacobianTerms();

    mfem::Vector x_true(fe_space->TrueVSize(), mfem::Device::GetMemoryType());

    GetTrueDofsParallel(*x_cur, x_true);
    // Takes in k vector and transforms into into our E-vector array
    P->Mult(x_true, px);
    elem_restrict_lex->Mult(px, el_x);

    def_grad = 0.0;
    exaconstit::kernel::GradCalc(
        nqpts, nelems, ndofs, el_jac.Read(), qpts_dshape.Read(), el_x.Read(), def_grad.ReadWrite());

    // We're returning our mesh nodes to the original object they were pointing to.
    // So, we need to cast away the const here.
    // We just don't want other functions outside this changing things.
    nodes = x_cur.get();
    mesh->SwapNodes(nodes, owns_nodes);
    // Delete the old geometric factors since they dealt with the original reference frame.
    mesh->DeleteGeometricFactors();
}

// Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const mfem::Vector& vel) const {
    m_sim_state->GetPrimalField()->operator=(vel);
    m_sim_state->UpdateNodalEndCoords();
}

// Compute the Jacobian from the nonlinear form
mfem::Operator& NonlinearMechOperator::GetGradient(const mfem::Vector& x) const {
    CALI_CXX_MARK_SCOPE("mechop_getgrad");
    jacobian = &h_form->GetGradient(x);
    // Reset our preconditioner operator aka recompute the diagonal for our jacobi.
    jacobian->AssembleDiagonal(diag);
    return *jacobian;
}

// Compute the Jacobian from the nonlinear form
mfem::Operator& NonlinearMechOperator::GetUpdateBCsAction(const mfem::Vector& k,
                                                          const mfem::Vector& x,
                                                          mfem::Vector& y) const {
    CALI_CXX_MARK_SCOPE("mechop_GetUpdateBCsAction");
    // We first run a setup step before actually doing anything.
    // We'll want to move this outside of Mult() at some given point in time
    // and have it live in the NR solver itself or whatever solver
    // we're going to be using.
    Setup<false>(k);
    // We now perform our element vector operation.
    mfem::Vector resid(y);
    resid.UseDevice(true);
    mfem::Array<int> zero_tdofs;
    CALI_MARK_BEGIN("mechop_h_form_LocalGrad");
    h_form->Setup();
    h_form->SetEssentialTrueDofs(zero_tdofs);
    auto& loc_jacobian = h_form->GetGradient(x);
    loc_jacobian.Mult(x, y);
    h_form->SetEssentialTrueDofs(ess_tdof_list);
    h_form->Mult(k, resid);
    jacobian = &h_form->GetGradient(x);
    CALI_MARK_END("mechop_h_form_LocalGrad");

    {
        auto I = ess_tdof_list.Read();
        auto size = ess_tdof_list.Size();
        auto Y = y.Write();
        // Need to get rid of all the constrained values here
        mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) {
            Y[I[i]] = 0.0;
        });
    }

    y += resid;
    return *jacobian;
}