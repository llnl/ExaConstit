
#include "system_driver.hpp"

#include "boundary_conditions/BCData.hpp"
#include "boundary_conditions/BCManager.hpp"
#include "solvers/trust_region_solver.hpp"
#include "utilities/mechanics_kernels.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "ECMech_const.h"
#include "RAJA/RAJA.hpp"

#include <iostream>
#include <limits>

/**
 * @brief Dirichlet boundary condition function for MFEM integration
 *
 * @param attr_id Boundary attribute identifier from the mesh
 * @param y Output vector where boundary condition values will be set
 *
 * @details This function serves as the interface between MFEM's boundary condition
 * system and ExaConstit's boundary condition management. It is used as a callback
 * function during finite element assembly to apply Dirichlet boundary conditions.
 *
 * The function:
 * 1. Gets the singleton BCManager instance
 * 2. Retrieves the appropriate BCData instance for the given boundary attribute
 * 3. Applies the boundary condition values to the output vector
 *
 * This function is typically passed to MFEM's VectorFunctionRestrictedCoefficient
 * or similar boundary condition mechanisms during system setup.
 *
 * @note The attr_id corresponds to mesh boundary attributes and must match the
 * boundary IDs used during BCManager initialization.
 */
void DirBdrFunc(int attr_id, mfem::Vector& y) {
    BCManager& bcManager = BCManager::GetInstance();
    BCData& bc = bcManager.GetBCInstance(attr_id);

    bc.SetDirBCs(y);
}

namespace {

void GetTrueDofsParallel(const mfem::ParGridFunction& gf, mfem::Vector& true_dofs) {
    // used to do something like:
    // gf.GetTrueDofs(true_dofs);
    // but looks like there are issues with that on the GPUs with newer versions of MFEM
    gf.ParallelAverage(true_dofs);
}

/**
 * @brief Helper function to find mesh bounding box for velocity gradient calculations
 *
 * @tparam T Device execution policy type (CPU/GPU)
 * @param space_dim Spatial dimension of the problem (2D or 3D)
 * @param nnodes Number of nodes in the mesh
 * @param class_device Device execution policy instance
 * @param nodes Pointer to mesh node coordinates vector
 * @param origin Output vector containing min and max coordinates [min_x, min_y, min_z, max_x,
 * max_y, max_z]
 *
 * @details Calculates the minimum and maximum coordinates of the mesh nodes across all
 * spatial dimensions. This information is needed for velocity gradient boundary conditions
 * that require knowledge of the mesh extent.
 *
 * The function:
 * 1. Handles the MFEM node ordering (xxx..., yyy..., zzz... rather than xyz, xyz...)
 * 2. Uses device-compatible reduction operations for GPU execution
 * 3. Performs MPI reductions to find global min/max across all processes
 * 4. Stores results in the origin vector with min values first, then max values
 *
 * @note This is a template function to support different device execution policies.
 * The "NVCC is the bane of my existence" comment refers to CUDA compiler limitations
 * that necessitated this template approach.
 */
template <class T>
void min_max_helper(const int space_dim,
                    const size_t nnodes,
                    const T& class_device,
                    mfem::Vector* const nodes,
                    mfem::Vector& origin) {
    // Our nodes are by default saved in xxx..., yyy..., zzz... ordering rather
    // than xyz, xyz, ...
    // So, the below should get us a device reference that can be used.
    const auto X = mfem::Reshape(nodes->Read(), nnodes, space_dim);
    mfem::Vector min_origin(space_dim);
    min_origin = std::numeric_limits<double>::max();
    mfem::Vector max_origin(space_dim);
    max_origin = -std::numeric_limits<double>::max();

    min_origin.HostReadWrite();
    max_origin.HostReadWrite();
    // We need to calculate the minimum point in the mesh to get the correct velocity gradient
    // across the part.
    RAJA::RangeSegment default_range(0, static_cast<long>(nnodes));
    if (class_device == RTModel::CPU) {
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<RAJA::seq_reduce, double> seq_min(std::numeric_limits<double>::max());
            RAJA::ReduceMax<RAJA::seq_reduce, double> seq_max(-std::numeric_limits<double>::max());
            RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
                seq_min.min(X(i, j));
                seq_max.max(X(i, j));
            });
            min_origin(j) = seq_min.get();
            max_origin(j) = seq_max.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<RAJA::omp_reduce_ordered, double> omp_min(
                std::numeric_limits<double>::max());
            RAJA::ReduceMax<RAJA::omp_reduce_ordered, double> omp_max(
                -std::numeric_limits<double>::max());
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i) {
                omp_min.min(X(i, j));
                omp_max.max(X(i, j));
            });
            min_origin(j) = omp_min.get();
            max_origin(j) = omp_max.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < space_dim; j++) {
            RAJA::ReduceMin<gpu_reduce, double> gpu_min(std::numeric_limits<double>::max());
            RAJA::ReduceMax<gpu_reduce, double> gpu_max(-std::numeric_limits<double>::max());
            RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i) {
                gpu_min.min(X(i, j));
                gpu_max.max(X(i, j));
            });
            min_origin(j) = gpu_min.get();
            max_origin(j) = gpu_max.get();
        }
    }
#endif
    MPI_Allreduce(min_origin.HostRead(),
                  origin.HostReadWrite(),
                  space_dim,
                  MPI_DOUBLE,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(max_origin.HostRead(),
                  &origin.HostReadWrite()[space_dim],
                  space_dim,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
} // End of finding max and min locations
} // namespace

bool is_vgrad_option_flag(const std::shared_ptr<SimulationState> sim_state) {
    const auto& bo = sim_state->GetOptions().boundary_conditions;
    if (bo.vgrad_bcs.size() > 0) {
        if (bo.vgrad_bcs[0].origin) {
            return true;
        }
    }
    return false;
}

bool is_expt_mono_flag(const std::shared_ptr<SimulationState> sim_state) {
    return sim_state->GetOptions().boundary_conditions.mono_def_bcs;
}

SystemDriver::SystemDriver(std::shared_ptr<SimulationState> sim_state)
    : class_device(sim_state->GetOptions().solvers.rtmodel),
      auto_time(sim_state->GetOptions().time.time_type == TimeStepType::AUTO),
      vgrad_origin_flag(is_vgrad_option_flag(sim_state)),
      mono_def_flag(is_expt_mono_flag(sim_state)), m_sim_state(sim_state) {
    CALI_CXX_MARK_SCOPE("system_driver_init");

    const auto& options = sim_state->GetOptions();

    auto mesh = m_sim_state->GetMesh();
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    const int space_dim = mesh->SpaceDimension();
    // set the size of the essential boundary conditions attribute array
    ess_bdr["total"] = mfem::Array<int>();
    ess_bdr["total"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["total"] = 0;
    ess_bdr["ess_vel"] = mfem::Array<int>();
    ess_bdr["ess_vel"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["ess_vel"] = 0;
    ess_bdr["ess_vgrad"] = mfem::Array<int>();
    ess_bdr["ess_vgrad"].SetSize(mesh->bdr_attributes.Max());
    ess_bdr["ess_vgrad"] = 0;

    ess_bdr_component["total"] = mfem::Array2D<bool>();
    ess_bdr_component["total"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["total"] = false;
    ess_bdr_component["ess_vel"] = mfem::Array2D<bool>();
    ess_bdr_component["ess_vel"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["ess_vel"] = false;
    ess_bdr_component["ess_vgrad"] = mfem::Array2D<bool>();
    ess_bdr_component["ess_vgrad"].SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_component["ess_vgrad"] = false;

    ess_bdr_scale.SetSize(mesh->bdr_attributes.Max(), space_dim);
    ess_bdr_scale = 0.0;
    ess_velocity_gradient.SetSize(space_dim * space_dim, mfem::Device::GetMemoryType());
    ess_velocity_gradient.UseDevice(true);

    vgrad_origin.SetSize(space_dim, mfem::Device::GetMemoryType());
    vgrad_origin.UseDevice(true);
    if (vgrad_origin_flag) {
        vgrad_origin.HostReadWrite();
        vgrad_origin = 0.0;
        // already checked if this exists
        auto origin = sim_state->GetOptions().boundary_conditions.vgrad_bcs[0].origin;
        vgrad_origin(0) = (*origin)[0];
        vgrad_origin(1) = (*origin)[1];
        vgrad_origin(2) = (*origin)[2];
    }

    // Set things to the initial step
    BCManager::GetInstance().GetUpdateStep(1);
    BCManager::GetInstance().UpdateBCData(
        ess_bdr, ess_bdr_scale, ess_velocity_gradient, ess_bdr_component);
    mech_operator = std::make_shared<NonlinearMechOperator>(
        ess_bdr["total"], ess_bdr_component["total"], m_sim_state);
    model = mech_operator->GetModel();

    if (mono_def_flag) {
        const auto nodes = mesh->GetNodes();
        const int nnodes = nodes->Size() / space_dim;
        mfem::Vector origin(space_dim * 2, mfem::Device::GetMemoryType());
        origin.UseDevice(true);
        origin = 0.0;
        // Just scoping variable usage so we can reuse variables if we'd want to
        // CUDA once again is limiting us from writing normal C++
        // code so had to move to a helper function for this part...
        min_max_helper(space_dim, static_cast<size_t>(nnodes), class_device, nodes, origin);

        mfem::Array<int> ess_vdofs, ess_tdofs, ess_true_dofs;
        ess_vdofs.SetSize(fe_space->GetVSize());
        ess_vdofs = 0;
        // We need to set the ess_vdofs doing something like ess_vdofs[i] = -1;
        // However, the compiler thinks ess_vdofs is const when trying to do this in
        // the later loop, so we turn to lambda fcns to do this so the compiler picks
        // the right mfem::Array::operator[](int i) fcn.
        auto f = [&ess_vdofs](int i) {
            ess_vdofs[i] = -1;
        };
        const auto X = mfem::Reshape(nodes->HostRead(), nnodes, space_dim);
        // For this we would need to set up the true dofs at start of simulation
        // before anything actually moves
        // X's dofs would be at global min(x, z)
        // Y's dofs would be at global min(x, y, z)
        // Z's dofs would be at global min(z) | global max(z)
        RAJA::RangeSegment default_range(0, nnodes);
        RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
            const double x_diff_min = std::abs(X(i, 0) - origin(0));
            const double y_diff_min = std::abs(X(i, 1) - origin(1));
            const double z_diff_min = std::abs(X(i, 2) - origin(2));
            const double z_diff_max = std::abs(X(i, 2) - origin(5));
            if (x_diff_min < 1e-12 && z_diff_min < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 0);
                f(dof);
            }
            if (x_diff_min < 1e-12 && y_diff_min < 1e-12 && z_diff_min < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 1);
                f(dof);
            }
            if (z_diff_min < 1e-12 || z_diff_max < 1e-12) {
                auto dof = fe_space->DofToVDof(i, 2);
                f(dof);
            }
        }); // end loop over nodes
        // Taken from mfem::FiniteElementSpace::GetEssentialTrueDofs(...)
        fe_space->Synchronize(ess_vdofs);
        fe_space->GetRestrictionMatrix()->BooleanMult(ess_vdofs, ess_tdofs);
        fe_space->MarkerToList(ess_tdofs, ess_true_dofs);
        mech_operator->UpdateEssTDofs(ess_true_dofs, mono_def_flag);
    }

    ess_bdr_func = std::make_unique<mfem::VectorFunctionRestrictedCoefficient>(
        space_dim, DirBdrFunc, ess_bdr["ess_vel"], ess_bdr_scale);

    // Partial assembly we need to use a matrix free option instead for our preconditioner
    // Everything else remains the same.
    auto& linear_solvers = options.solvers.linear_solver;
    if (options.solvers.assembly != AssemblyType::FULL) {
        J_prec = mech_operator->GetPAPreconditioner();
    } else {
        if (linear_solvers.preconditioner == PreconditionerType::AMG) {
            auto prec_amg = std::make_shared<mfem::HypreBoomerAMG>();
            const int problem_dim = m_sim_state->GetMesh()->SpaceDimension();
            const bool order_bynodes = (fe_space->GetOrdering() == mfem::Ordering::byNODES);
            // Use MFEM's supported systems-AMG configuration so Hypre sees
            // the correct vector-valued DOF ordering on newer MFEM/Hypre builds.
            prec_amg->SetSystemsOptions(problem_dim, order_bynodes);
            prec_amg->SetPrintLevel(linear_solvers.print_level);
            J_prec = prec_amg;
        } else if (linear_solvers.preconditioner == PreconditionerType::ILU) {
            auto J_hypreEuclid = std::make_shared<mfem::HypreEuclid>(fe_space->GetComm());
            J_prec = J_hypreEuclid;
        } else if (linear_solvers.preconditioner == PreconditionerType::L1GS) {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::l1GS);
            J_hypreSmoother->SetPositiveDiagonal(true);
            J_prec = J_hypreSmoother;
        } else if (linear_solvers.preconditioner == PreconditionerType::CHEBYSHEV) {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::Chebyshev);
            J_prec = J_hypreSmoother;
        } else {
            auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
            J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
            J_hypreSmoother->SetPositiveDiagonal(true);
            J_prec = J_hypreSmoother;
        }
    }

    if (linear_solvers.solver_type == LinearSolverType::GMRES) {
        J_solver = std::make_shared<mfem::GMRESSolver>(fe_space->GetComm());
    } else if (linear_solvers.solver_type == LinearSolverType::CG) {
        J_solver = std::make_shared<mfem::CGSolver>(fe_space->GetComm());
    } else if (linear_solvers.solver_type == LinearSolverType::BICGSTAB) {
        J_solver = std::make_shared<mfem::BiCGSTABSolver>(fe_space->GetComm());
    } else {
        J_solver = std::make_shared<mfem::MINRESSolver>(fe_space->GetComm());
    }

    // The relative tolerance should be at this point or smaller
    J_solver->SetRelTol(linear_solvers.rel_tol);
    // The absolute tolerance could probably get even smaller then this
    J_solver->SetAbsTol(linear_solvers.abs_tol);
    J_solver->SetMaxIter(linear_solvers.max_iter);
    J_solver->SetPrintLevel(linear_solvers.print_level);
    J_solver->SetPreconditioner(*J_prec);

    auto nonlinear_solver = options.solvers.nonlinear_solver;
    newton_iter = nonlinear_solver.iter;
    if (nonlinear_solver.nl_solver == NonlinearSolverType::NR) {
        newton_solver = std::make_unique<ExaNewtonSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());
    } else if (nonlinear_solver.nl_solver == NonlinearSolverType::NRLS) {
        newton_solver = std::make_unique<ExaNewtonLSSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());
    } else if (nonlinear_solver.nl_solver == NonlinearSolverType::TRDOG) {
        // Build the trust-region dogleg solver and configure delta-control
        // parameters from the parsed TOML options. If the user did not supply
        // a [trust_region] sub-table, the solver's internal defaults (matching
        // SNLS's TrDeltaControl defaults) are used.
        auto tr_solver = std::make_unique<ExaTrustRegionSolver>(
            m_sim_state->GetMeshParFiniteElementSpace()->GetComm());

        if (nonlinear_solver.trust_region.has_value()) {
            const auto& tr_opts = nonlinear_solver.trust_region.value();
            TrDeltaControl ctrl;
            ctrl.deltaInit = tr_opts.delta_init;
            ctrl.deltaMin = tr_opts.delta_min;
            ctrl.deltaMax = tr_opts.delta_max;
            ctrl.xiLG = tr_opts.xi_lg;
            ctrl.xiUG = tr_opts.xi_ug;
            ctrl.xiLO = tr_opts.xi_lo;
            ctrl.xiUO = tr_opts.xi_uo;
            ctrl.xiIncDelta = tr_opts.xi_inc;
            ctrl.xiDecDelta = tr_opts.xi_dec;
            ctrl.xiForcedIncDelta = tr_opts.xi_forced_inc;
            ctrl.rejectResIncrease = tr_opts.reject_increase;
            tr_solver->SetTrustRegionControl(ctrl);
        }

        newton_solver = std::move(tr_solver);

        // Sanity check: TRDOG requires gradient transpose support (J^T*r). For
        // PA mode, this requires the native PA transpose kernels in the
        // integrator. EA and FULL always support transpose. We warn rather than
        // hard-fail here because PA support exists once the kernels are wired.
        if (options.solvers.assembly == AssemblyType::PA) {
            mfem::out << "Note: TRDOG with PA assembly requires native PA transpose "
                      << "kernels in the gradient operator.\n";
        }
    }

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver(J_solver);
    newton_solver->SetOperator(mech_operator);
    newton_solver->SetPrintLevel(1);
    newton_solver->SetRelTol(nonlinear_solver.rel_tol);
    newton_solver->SetAbsTol(nonlinear_solver.abs_tol);
    newton_solver->SetMaxIter(nonlinear_solver.iter);
}

const mfem::Array<int>& SystemDriver::GetEssTDofList() {
    return mech_operator->GetEssTDofList();
}

// Solve the Newton system
void SystemDriver::Solve() {
    mfem::Vector zero;
    auto x = m_sim_state->GetPrimalField();
    // Global convergence status across all ranks
    bool converged = false;
    if (auto_time) {
        // This would only happen on the last time step
        const auto x_prev = m_sim_state->GetPrimalFieldPrev();
        // Vector xprev(x); xprev.UseDevice(true);
        // We provide an initial guess for what our current coordinates will look like
        // based on what our last time steps solution was for our velocity field.
        // The end nodes are updated before the 1st step of the solution here so we're good.
        bool succeed_t = false;
        bool succeed = false;
        try {
            newton_solver->Mult(zero, *x);
            succeed_t = newton_solver->GetConverged();
        } catch (const std::exception& exc) {
            // catch anything thrown within try block that derives from std::exception
            MFEM_WARNING_0(exc.what());
            succeed_t = false;
        } catch (...) {
            MFEM_WARNING_0("An unknown exception was thrown in Krylov solver step");
            succeed_t = false;
        }
        MPI_Allreduce(&succeed_t, &succeed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        TimeStep state = m_sim_state->UpdateDeltaTime(newton_solver->GetNumIterations(), succeed);
        if (!succeed) {
            while (state == TimeStep::RETRIAL) {
                MFEM_WARNING_0("Solution did not converge decreasing dt by input scale factor");
                if (m_sim_state->GetMPIID() == 0) {
                    m_sim_state->PrintRetrialTimeStats();
                }
                m_sim_state->RestartCycle();
                try {
                    newton_solver->Mult(zero, *x);
                    succeed_t = newton_solver->GetConverged();
                } catch (...) {
                    succeed_t = false;
                }
                MPI_Allreduce(&succeed_t, &succeed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
                state = m_sim_state->UpdateDeltaTime(newton_solver->GetNumIterations(), succeed);
            } // Do final converge check outside of this while loop
            if (state == TimeStep::FAILED) {
                MFEM_WARNING_0("Solution did not converge and the minimum dt or maximum number "
                               "of retries has been reached");
            }
        }
        converged = succeed;
    } else {
        // We provide an initial guess for what our current coordinates will look like
        // based on what our last time steps solution was for our velocity field.
        // The end nodes are updated before the 1st step of the solution here so we're good.
        newton_solver->Mult(zero, *x);
        m_sim_state->UpdateDeltaTime(newton_solver->GetNumIterations(), true);
        converged = newton_solver->GetConverged();
    }

    // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
    // back to the current configuration...
    // Once the system has finished solving, our current coordinates configuration are based on what
    // our converged velocity field ended up being equal to.
    if (m_sim_state->GetMPIID() == 0 && converged) {
        ess_bdr_func->SetTime(m_sim_state->GetTime());
    }
    MFEM_VERIFY_0(converged, "Newton Solver did not converge.");
}

// Solve the Newton system for the 1st time step
// It was found that for large meshes a ramp up to our desired applied BC might
// be needed.
void SystemDriver::SolveInit() const {
    const auto x = m_sim_state->GetPrimalField();
    const auto x_prev = m_sim_state->GetPrimalFieldPrev();
    mfem::Vector b(*x);
    b.UseDevice(true);

    mfem::Vector deltaF(*x);
    deltaF.UseDevice(true);
    b = 0.0;
    // Want our vector for everything not on the Ess BCs to be 0
    // This means when we do K * diffF = b we're actually do the following:
    // K_uc * (x - x_prev)_c = deltaF_u
    {
        deltaF = 0.0;
        auto I = mech_operator->GetEssentialTrueDofs().Read();
        auto size = mech_operator->GetEssentialTrueDofs().Size();
        auto Y = deltaF.Write();
        auto XPREV = x_prev->Read();
        auto X = x->Read();
        mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) {
            Y[I[i]] = X[I[i]] - XPREV[I[i]];
        });
    }
    mfem::Operator& oper = mech_operator->GetUpdateBCsAction(*x_prev, deltaF, b);
    x->operator=(0.0);
    // This will give us our -change in velocity
    // So, we want to add the previous velocity terms to it
    newton_solver->CGSolver(oper, b, *x);
    auto X = x->ReadWrite();
    auto XPREV = x_prev->Read();
    mfem::forall(x->Size(), [=] MFEM_HOST_DEVICE(int i) {
        X[i] = -X[i] + XPREV[i];
    });
    m_sim_state->GetVelocity()->Distribute(*x);
}

void SystemDriver::UpdateEssBdr() {
    if (!mono_def_flag) {
        BCManager::GetInstance().UpdateBCData(
            ess_bdr, ess_bdr_scale, ess_velocity_gradient, ess_bdr_component);
        mech_operator->UpdateEssTDofs(ess_bdr["total"], mono_def_flag);
    }
}

// In the current form, we could honestly probably make use of velocity as our working array
void SystemDriver::UpdateVelocity() {
    auto fe_space = m_sim_state->GetMeshParFiniteElementSpace();
    auto mesh = m_sim_state->GetMesh();
    auto velocity = m_sim_state->GetVelocity();
    auto vel_tdofs = m_sim_state->GetPrimalField();

    if (ess_bdr["ess_vel"].Sum() > 0) {
        // Now that we're doing velocity based we can just overwrite our data with the ess_bdr_func
        velocity->ProjectBdrCoefficient(*ess_bdr_func); // don't need attr list as input
                                                        // pulled off the
                                                        // VectorFunctionRestrictedCoefficient
        // populate the solution vector, v_sol, with the true dofs entries in v_cur.
        GetTrueDofsParallel(*velocity, *vel_tdofs);
    }

    if (ess_bdr["ess_vgrad"].Sum() > 0) {
        // Just scoping variable usage so we can reuse variables if we'd want to
        {
            const auto nodes = mesh->GetNodes();
            const int space_dim = mesh->SpaceDimension();
            const int nnodes = nodes->Size() / space_dim;

            // Our nodes are by default saved in xxx..., yyy..., zzz... ordering rather
            // than xyz, xyz, ...
            // So, the below should get us a device reference that can be used.
            const auto X = mfem::Reshape(nodes->Read(), nnodes, space_dim);
            const auto VGRAD = mfem::Reshape(ess_velocity_gradient.Read(), space_dim, space_dim);
            velocity->operator=(0.0);
            auto VT = mfem::Reshape(velocity->ReadWrite(), nnodes, space_dim);

            if (!vgrad_origin_flag) {
                vgrad_origin.HostReadWrite();
                // We need to calculate the minimum point in the mesh to get the correct velocity
                // gradient across the part.
                RAJA::RangeSegment default_range(0, nnodes);
                if (class_device == RTModel::CPU) {
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<RAJA::seq_reduce, double> seq_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<RAJA::seq_exec>(default_range, [=](int i) {
                            seq_min.min(X(i, j));
                        });
                        vgrad_origin(j) = seq_min.get();
                    }
                }
#if defined(RAJA_ENABLE_OPENMP)
                if (class_device == RTModel::OPENMP) {
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<RAJA::omp_reduce_ordered, double> omp_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [=](int i) {
                            omp_min.min(X(i, j));
                        });
                        vgrad_origin(j) = omp_min.get();
                    }
                }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                if (class_device == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA)
                    using gpu_reduce = RAJA::cuda_reduce;
                    using gpu_policy = RAJA::cuda_exec<1024>;
#else
                    using gpu_reduce = RAJA::hip_reduce;
                    using gpu_policy = RAJA::hip_exec<1024>;
#endif
                    for (int j = 0; j < space_dim; j++) {
                        RAJA::ReduceMin<gpu_reduce, double> gpu_min(
                            std::numeric_limits<double>::max());
                        RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i) {
                            gpu_min.min(X(i, j));
                        });
                        vgrad_origin(j) = gpu_min.get();
                    }
                }
#endif
            } // End if vgrad_origin_flag
            mfem::Vector origin(space_dim, mfem::Device::GetMemoryType());
            origin.UseDevice(true);
            MPI_Allreduce(vgrad_origin.HostRead(),
                          origin.HostReadWrite(),
                          space_dim,
                          MPI_DOUBLE,
                          MPI_MIN,
                          MPI_COMM_WORLD);
            const double* dmin_x = origin.Read();
            // We've now found our minimum points so we can now go and calculate everything.
            mfem::forall(nnodes, [=] MFEM_HOST_DEVICE(int i) {
                for (int ii = 0; ii < space_dim; ii++) {
                    for (int jj = 0; jj < space_dim; jj++) {
                        // mfem::Reshape assumes Fortran memory layout
                        // which is why everything is the transpose down below...
                        VT(i, ii) += VGRAD(jj, ii) * (X(i, jj) - dmin_x[jj]);
                    }
                }
            });
        }
        {
            mfem::Vector vel_tdof_tmp(*vel_tdofs);
            vel_tdof_tmp.UseDevice(true);
            vel_tdof_tmp = 0.0;
            GetTrueDofsParallel(*velocity, vel_tdof_tmp);

            mfem::Array<int> ess_tdofs(mech_operator->GetEssentialTrueDofs());
            if (!mono_def_flag) {
                fe_space->GetEssentialTrueDofs(
                    ess_bdr["ess_vgrad"], ess_tdofs, ess_bdr_component["ess_vgrad"]);
            }
            auto I = ess_tdofs.Read();
            auto size = ess_tdofs.Size();
            auto Y = vel_tdofs->ReadWrite();
            const auto X = vel_tdof_tmp.Read();
            // vel_tdofs should already have the current solution
            mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) {
                Y[I[i]] = X[I[i]];
            });
        }
    } // end of if constant strain rate
}

void SystemDriver::UpdateModel() {
    model->UpdateModelVars();
    m_sim_state->UpdateModel();
    m_sim_state->SetupModelVariables();

    auto def_grad = m_sim_state->GetQuadratureFunction("kinetic_grads");
    mech_operator->CalculateDeformationGradient(*def_grad.get());
}