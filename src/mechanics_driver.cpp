/**
 * @file mechanics_driver.cpp
 * @brief Main application driver for ExaConstit velocity-based finite element simulations.
 *
 * @details ExaConstit is a high-performance, parallel finite element application for nonlinear
 * solid mechanics simulations with emphasis on crystal plasticity and micromechanics modeling.
 * This driver implements a velocity-based, updated Lagrangian finite element framework designed
 * for large-scale materials science applications on leadership-class computing systems.
 *
 * **Key Capabilities:**
 * - **Velocity-Based Formulation**: Updated Lagrangian kinematics with velocity primary variables
 * - **Crystal Plasticity**: Advanced polycrystalline material modeling with grain-level resolution
 * - **Large Deformation**: Geometrically nonlinear analysis for finite strain applications
 * - **Multi-Material Support**: Heterogeneous material regions with different constitutive models
 * - **Adaptive Time Stepping**: Automatic time step control based on Newton-Raphson convergence
 * - **High-Performance Computing**: MPI parallelization with GPU acceleration support
 * - **Advanced Solvers**: Newton-Raphson with line search and Krylov iterative linear solvers
 *
 * **Supported Material Models:**
 * - **ExaCMech**: LLNL's crystal plasticity library with advanced hardening models
 * - **UMAT Interface**: Abaqus-compatible user material subroutines
 * - **Multi-Model Regions**: Different material models in different mesh regions
 * - **History Variables**: Full support for internal state variable evolution
 *
 * **Computational Architecture:**
 * - **MFEM Framework**: Built on LLNL's MFEM finite element library
 * - **RAJA Performance Portability**: CPU/OpenMP/GPU execution with unified code
 * - **Device-Aware Memory**: Automatic host/device memory management
 * - **Partial Assembly**: Memory-efficient matrix-free operator evaluation
 * - **Scalable I/O**: Parallel visualization and data output capabilities
 *
 * **Simulation Workflow:**
 * 1. **Initialization**: MPI setup, option parsing, device configuration
 * 2. **Mesh Setup**: Parallel mesh loading and finite element space creation
 * 3. **Material Initialization**: State variables, grain orientations, and material properties
 * 4. **Solver Configuration**: Newton-Raphson and linear solver setup with preconditioning
 * 5. **Time Stepping**: Main simulation loop with boundary condition updates
 * 6. **Post-Processing**: Field projection, volume averaging, and visualization output
 * 7. **Performance Analysis**: Timing data and scalability metrics collection
 *
 * **Input Requirements:**
 * - **options.toml**: Primary configuration file with all simulation parameters
 * - **Mesh File**: Parallel-compatible mesh (typically .mesh format)
 * - **Material Properties**: Material parameter files (props.txt for crystal plasticity)
 * - **State Variables**: Initial internal state variable values (state.txt)
 * - **Grain Data**: Crystal orientation data (grain.txt for crystal plasticity applications)
 *
 * **Key Dependencies:**
 * - **MFEM**: Finite element framework with parallel/GPU support
 * - **HYPRE**: Algebraic multigrid preconditioning and linear solvers
 * - **ExaCMech**: Crystal plasticity constitutive model library
 * - **RAJA**: Performance portability and GPU execution framework
 * - **Conduit**: Data management and I/O for visualization
 * - **Caliper**: Performance profiling and analysis toolkit
 *
 * **Usage:**
 * ```bash
 * mpirun -np <nprocs> ./mechanics [-opt options_file.toml]
 * ```
 *
 * **Performance Considerations:**
 * - Designed for leadership-class HPC systems (CPU clusters and GPU systems)
 * - Scales to thousands of MPI processes with efficient domain decomposition
 * - GPU acceleration available for material model evaluation and linear algebra
 * - Memory-efficient algorithms suitable for large-scale polycrystalline simulations
 *
 * @note This application is designed for materials science research and industrial
 * applications requiring high-fidelity simulation of polycrystalline materials
 * under complex loading conditions.
 *
 * @author LLNL ExaConstit Development Team (Lead Author: Robert Carson (carson16@llnl.gov))
 * @ingroup ExaConstit_applications
 */
#include "boundary_conditions/BCData.hpp"
#include "boundary_conditions/BCManager.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "mfem_expt/partial_qspace.hpp"
#include "options/option_parser_v2.hpp"
#include "postprocessing/postprocessing_driver.hpp"
#include "postprocessing/postprocessing_file_manager.hpp"
#include "sim_state/simulation_state.hpp"
#include "system_driver.hpp"
#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "SNLS_config.h"
#if defined(SNLS_RAJA_PORT_SUITE)
#include <umpire/util/io.hpp>
#endif

#include <memory>
#include <sstream>
#include <string>

/**
 * @brief Main application entry point for ExaConstit finite element simulations.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit status (0 for success, 1 for failure)
 *
 * Orchestrates the complete ExaConstit simulation workflow from initialization through
 * final results output. The function implements a time-stepping algorithm for solving
 * nonlinear solid mechanics problems with advanced material models and boundary conditions.
 *
 * **PHASE 1: PARALLEL INITIALIZATION AND SETUP**
 */
int main(int argc, char* argv[]) {
    // Initialize Caliper performance profiling system for detailed performance analysis
    CALI_INIT
    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN("main_driver_init");
    /*
     * MPI Environment Setup:
     * - Initialize MPI for parallel execution across distributed memory systems
     * - Query total process count and local rank for parallel coordination
     * - Initialize HYPRE if version supports it (parallel linear algebra)
     */
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#if (MFEM_HYPRE_VERSION >= 21900)
    mfem::Hypre::Init();
#endif
    // Scope block to ensure proper MPI cleanup and resource deallocation
    {
        /*
         * Performance Timing Setup:
         * - Initialize wall-clock timer for total simulation time measurement
         * - Set up per-timestep timing vector for performance analysis
         * - Enable detailed timing data for strong/weak scaling studies
         */
        double start = MPI_Wtime();
        /**
         * **PHASE 2: COMMAND LINE PROCESSING AND CONFIGURATION**
         */

        /*
         * Options File Processing:
         * - Default to "options.toml" configuration file
         * - Support command line override with -opt/--option flag
         * - Enable multiple configuration scenarios without recompilation
         */
        const char* toml_file = "options.toml";
        mfem::OptionsParser args(argc, argv);
        args.AddOption(&toml_file, "-opt", "--option", "Option file to use.");
        args.Parse();
        // Error handling for invalid command line arguments
        if (!args.Good()) {
            if (myid == 0) {
                args.PrintUsage(std::cout);
            }
            CALI_MARK_END("main_driver_init");
            MPI_Finalize();
            return 1;
        }

        // Print MFEM version information for reproducibility and debugging
        if (myid == 0) {
            printf("MFEM Version: %d \n", mfem::GetVersion());
        }

        /*
         * Configuration File Parsing:
         * - Load complete simulation configuration from TOML file
         * - Validate all required parameters and consistency checks
         * - Print configuration summary for verification
         */
        ExaOptions toml_opt;
        toml_opt.parse_options(toml_file, myid);

        exaconstit::UnifiedLogger& logger = exaconstit::UnifiedLogger::get_instance();
        logger.initialize(toml_opt);
#if defined(SNLS_RAJA_PORT_SUITE)
        umpire::util::initialize_io(false);
#endif
        toml_opt.print_options();

        /**
         * **PHASE 3: DEVICE CONFIGURATION AND MEMORY MANAGEMENT**
         */

        /*
         * Device Execution Model Setup:
         * - Configure CPU, OpenMP, or GPU execution based on options
         * - Set up RAJA performance portability layer for device-agnostic kernels
         * - Priority order: GPU (CUDA/HIP) > OpenMP > CPU for optimal performance
         */
        std::string device_config = "cpu";

        if (toml_opt.solvers.rtmodel == RTModel::CPU) {
            device_config = "cpu";
        } else if (toml_opt.solvers.rtmodel == RTModel::OPENMP) {
            device_config = "raja-omp";
        } else if (toml_opt.solvers.rtmodel == RTModel::GPU) {
#if defined(RAJA_ENABLE_CUDA)
            device_config = "raja-cuda";
#elif defined(RAJA_ENABLE_HIP)
            device_config = "raja-hip";
#endif
        }
        /*
         * MFEM Device Configuration:
         * - Configure device memory management for host/device data movement
         * - Set up automatic memory synchronization for CPU/GPU execution
         * - Enable high-performance device kernels for linear algebra operations
         */
        mfem::Device device;
        if (toml_opt.solvers.rtmodel == RTModel::GPU) {
#if defined(MFEM_USE_UMPIRE)
            device.SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);
#else
            device.SetMemoryTypes(mfem::MemoryType::HOST_64, mfem::MemoryType::DEVICE);
#endif
        } else {
#if defined(MFEM_USE_UMPIRE)
            device.SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE, mfem::MemoryType::DEVICE_UMPIRE);
#else
            device.SetMemoryTypes(mfem::MemoryType::HOST_64, mfem::MemoryType::DEVICE);
#endif
        }
        device.Configure(device_config.c_str());

        // Print device configuration for verification and debugging
        if (myid == 0) {
            printf("\n");
            device.Print();
            printf("\n");
        }

        /**
         * **PHASE 4: SIMULATION STATE AND MESH INITIALIZATION**
         */

        /*
         * SimulationState Creation:
         * - Initialize complete simulation state from parsed options
         * - Set up parallel mesh with domain decomposition
         * - Create finite element spaces and degree-of-freedom mappings
         * - Initialize all quadrature functions for material state variables
         * - Set up boundary condition management systems
         */
        auto sim_state = std::make_shared<SimulationState>(toml_opt);

        auto pmesh = sim_state->GetMesh();

        CALI_MARK_END("main_driver_init");
        /*
         * Mesh and DOF Information:
         * - Query mesh dimension and finite element space size
         * - Print parallel mesh statistics for load balancing verification
         * - Display total degrees of freedom for memory estimation
         */
        HYPRE_Int glob_size = sim_state->GetMeshParFiniteElementSpace()->GlobalTrueVSize();
        pmesh->PrintInfo();

        if (myid == 0) {
            std::cout << "***********************************************************\n";
            std::cout << "dim(u) = " << glob_size << "\n";
            std::cout << "***********************************************************\n";
        }

        /**
         * **PHASE 5: FIELD INITIALIZATION AND GRID FUNCTIONS**
         */

        /*
         * Grid Function Setup:
         * - Get displacement and velocity field references from simulation state
         * - Initialize vector coefficient function for zero initial conditions
         * - Project initial conditions onto finite element spaces
         * - Prepare fields for time-stepping algorithm
         */

        auto x_diff = sim_state->GetDisplacement();
        auto v_cur = sim_state->GetVelocity();

        x_diff->operator=(0.0);
        v_cur->operator=(0.0);

        /**
         * **PHASE 6: SOLVER SYSTEM CONSTRUCTION**
         */

        /*
         * SystemDriver Initialization:
         * - Create main simulation driver with complete solver configuration
         * - Initialize Newton-Raphson nonlinear solver with line search options
         * - Set up Krylov iterative linear solvers with algebraic multigrid preconditioning
         * - Configure essential boundary condition enforcement
         * - Prepare material model interfaces and state variable management
         */
        SystemDriver oper(sim_state);

        // Get essential true DOF list for boundary condition enforcement
        const mfem::Array<int> ess_tdof_list = oper.GetEssTDofList();
        /*
         * PostProcessing Setup:
         * - Initialize post-processing driver for field projection and output
         * - Set up volume averaging calculations for homogenization
         * - Configure visualization data collection (VisIt, ParaView, ADIOS2)
         * - Prepare performance and convergence monitoring
         */
        PostProcessingDriver post_process(sim_state, toml_opt);
        /**
         * **PHASE 7: MAIN TIME-STEPPING LOOP**
         */

        /*
         * Time-Stepping Algorithm:
         * - Implements implicit time integration with Newton-Raphson iteration
         * - Supports adaptive time stepping based on convergence behavior
         * - Handles time-dependent boundary conditions with smooth transitions
         * - Performs material state updates and post-processing at each step
         */
        int ti = 0;
        auto v_sol = sim_state->GetPrimalField();
        while (!sim_state->IsFinished()) {
            ti++;
            // Print timestep information and timing statistics
            if (myid == 0) {
                std::cout << "Simulation cycle: " << ti << std::endl;
                sim_state->PrintTimeStats();
            }
            /*
             * Current Time Step Processing:
             * - Retrieve current simulation time and time step size
             * - Update time-dependent material properties and boundary conditions
             * - Prepare solver state for current time increment
             */

            /*
             * Boundary Condition Change Detection:
             * - Check if boundary conditions change for current time step
             * - Apply corrector step (SolveInit) for smooth BC transitions
             * - This prevents convergence issues with sudden load changes
             */
            if (BCManager::GetInstance().GetUpdateStep(ti)) {
                if (myid == 0) {
                    std::cout << "Changing boundary conditions this step: " << ti << std::endl;
                }

                // Update boundary condition data and apply corrector step
                oper.UpdateEssBdr();
                oper.UpdateVelocity();
                oper.SolveInit();
            }
            /*
             * Main Solution Process:
             * 1. Update velocity field with current boundary conditions
             * 2. Solve nonlinear system using Newton-Raphson iteration
             * 3. Check convergence and handle potential failures
             */
            oper.UpdateVelocity();
            oper.Solve();

            /*
             * Time Step Completion:
             * - Advance simulation time and check for final step
             * - Update material state variables with converged solution
             * - Perform post-processing calculations and output generation
             */
            sim_state->FinishCycle();
            oper.UpdateModel();
            post_process.Update(ti, sim_state->GetTrueCycleTime());
        } // end loop over time steps

        /**
         * **PHASE 8: SIMULATION COMPLETION AND PERFORMANCE ANALYSIS**
         */

        /*
         * Performance Timing Collection:
         * - Measure total simulation wall-clock time
         * - Compute average timing across all MPI processes
         * - Report performance metrics for scalability analysis
         */
        double end = MPI_Wtime();

        double sim_time = end - start;
        double avg_sim_time;

        MPI_Allreduce(&sim_time, &avg_sim_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (myid == 0) {
            printf("The process took %lf seconds to run\n", (avg_sim_time / world_size));
        }
#if defined(SNLS_RAJA_PORT_SUITE)
        umpire::util::finalize_io();
#endif
        logger.shutdown();
    } // End of main simulation scope for proper resource cleanup

    /*
     * MPI Cleanup and Termination:
     * - Synchronize all processes before exit
     * - Finalize MPI environment and clean up resources
     * - Return success status to operating system
     */

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}