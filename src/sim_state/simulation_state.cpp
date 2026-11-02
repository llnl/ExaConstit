#include "sim_state/simulation_state.hpp"

#include "utilities/unified_logger.hpp"
#if defined(EXACONSTIT_USE_AXOM)
#include "sim_state/restart_checkpoint.hpp"
#endif

namespace {

void setupBoundaryConditions(ExaOptions& options) {
    BCManager& bcm = BCManager::GetInstance();
    auto& bcs_opts = options.boundary_conditions;
    // update_steps (not time_info.cycles) also covers BCs.time_info being omitted
    bcm.Init(bcs_opts.update_steps,
             bcs_opts.map_ess_vel,
             bcs_opts.map_ess_vgrad,
             bcs_opts.map_ess_comp,
             bcs_opts.map_ess_id);
}

void setBdrConditions(mfem::Mesh& mesh) {
    // modify MFEM auto cuboidal hex mesh generation boundary
    // attributes to correspond to correct ExaConstit boundary conditions.
    // Look at ../../mesh/mesh.cpp Make3D() to see how boundary attributes
    // are set and modify according to ExaConstit convention

    // loop over boundary elements
    for (int i = 0; i < mesh.GetNBE(); ++i) {
        int bdrAttr = mesh.GetBdrAttribute(i);

        switch (bdrAttr) {
        // note, srw wrote SetBdrAttribute() in ../../mesh/mesh.hpp
        case 1:
            mesh.SetBdrAttribute(i, 1); // bottom
            break;
        case 2:
            mesh.SetBdrAttribute(i, 3); // front
            break;
        case 3:
            mesh.SetBdrAttribute(i, 5); // right
            break;
        case 4:
            mesh.SetBdrAttribute(i, 6); // back
            break;
        case 5:
            mesh.SetBdrAttribute(i, 2); // left
            break;
        case 6:
            mesh.SetBdrAttribute(i, 4); // top
            break;
        }
    }

    return;
}

void setElementGrainIDs(mfem::Mesh& mesh, const mfem::Vector& grainMap, int ncols, int offset) {
    // after a call to reorderMeshElements, the elements in the serial
    // MFEM mesh should be ordered the same as the input grainMap
    // vector. Set the element attribute to the grain id. This vector
    // has stride of 4 with the id in the 3rd position indexing from 0

    const double* data = grainMap.HostRead();

    // loop over elements
    for (int i = 0; i < mesh.GetNE(); ++i) {
        const int grainID = static_cast<int>(data[ncols * i + offset]);
        mesh.SetAttribute(i, grainID);
    }

    return;
}

// Projects the element attribute to GridFunction nodes
// This also assumes this the GridFunction is an L2 FE space
// void projectElemAttr2GridFunc(std::shared_ptr<mfem::Mesh> mesh,
// std::shared_ptr<mfem::ParGridFunction> elem_attr) {
//    // loop over elementsQ
//    elem_attr->HostRead();
//    mfem::ParFiniteElementSpace *pfes = elem_attr->ParFESpace();
//    mfem::Array<int> vdofs;
//    for (int i = 0; i < mesh->GetNE(); ++i) {
//       pfes->GetElementVDofs(i, vdofs);
//       const double ea = static_cast<double>(mesh->GetAttribute(i));
//       elem_attr->SetSubVector(vdofs, ea);
//    }
// }

std::shared_ptr<mfem::ParMesh> makeMesh(ExaOptions& options, const int my_id) {
    mfem::Mesh mesh;
    if (options.mesh.mesh_type == MeshType::FILE) {
        if (my_id == 0) {
            std::cout << "Opening mesh file: " << options.mesh.mesh_file << std::endl;
        }

        mesh = mfem::Mesh(options.mesh.mesh_file.string().c_str(), 1, 1, true);
    }
    // We're using the auto mesh generator
    else {
        MFEM_VERIFY_0(options.mesh.nxyz[0] > 0 && options.mesh.mxyz[0] > 0,
                      "Must input mesh geometry/discretization for hex_mesh_gen");

        if (my_id == 0) {
            std::cout << "Using mfem's hex mesh generator" << std::endl;
        }

        // use constructor to generate a 3D cuboidal mesh with 8 node hexes
        // The false at the end is to tell the inline mesh generator to use the lexicographic
        // ordering of the mesh The newer space-filling ordering option that was added in the
        // pre-okina tag of MFEM resulted in a noticeable divergence of the material response for a
        // monotonic tension test using symmetric boundary conditions out to 1% strain.
        mesh = mfem::Mesh::MakeCartesian3D(options.mesh.nxyz[0],
                                           options.mesh.nxyz[1],
                                           options.mesh.nxyz[2],
                                           mfem::Element::HEXAHEDRON,
                                           options.mesh.mxyz[0],
                                           options.mesh.mxyz[1],
                                           options.mesh.mxyz[2],
                                           false);
        // read in the grain map if using a MFEM auto generated cuboidal mesh
        if (options.grain_file) {
            std::ifstream gfile(*options.grain_file);
            MFEM_VERIFY_0(gfile, "Cannot open grain map file: " << *options.grain_file);

            const int gmap_size = mesh.GetNE();
            mfem::Vector gmap(gmap_size);
            gmap.Load(gfile, gmap_size);
            // A short file leaves grain ids of 0 or uninitialized values, which are
            // later used as (id - 1) indices
            MFEM_VERIFY_0(!gfile.fail(),
                          "Grain map file " << *options.grain_file << " has fewer than the "
                                            << gmap_size << " entries needed for the "
                                            << "auto-generated mesh (one per element)");
            gfile.close();

            // set grain ids as element attributes on the mesh
            // The offset of where the grain index is located is
            // location - 1.
            ::setElementGrainIDs(mesh, gmap, 1, 0);
        }
        //// reorder elements to conform to ordering convention in grain map file
        // No longer needed for the CA stuff. It's now ordered as X->Y->Z
        // reorderMeshElements(mesh, &toml_opt.nxyz[0]);

        // reset boundary conditions from
        ::setBdrConditions(mesh);
    }

    // We need to check to see if our provided mesh has a different order than
    // the order provided. If we see a difference we either increase our order seen
    // in the options file or we increase the mesh ordering. I'm pretty sure this
    // was causing a problem earlier with our auto-generated mesh and if we wanted
    // to use a higher order FE space.
    // So we can't really do the GetNodalFESpace it appears if we're given
    // an initial mesh. It looks like NodalFESpace is initially set to
    // NULL and only if we swap the mesh nodes does this actually
    // get set...
    // So, we're just going to set the mesh order to at least be 1. Although,
    // I would like to see this change sometime in the future.
    int mesh_order = 1;
    if (mesh_order > options.mesh.order) {
        options.mesh.order = mesh_order;
    }
    if (mesh_order <= options.mesh.order) {
        if (my_id == 0) {
            std::cout << "Increasing mesh order of the mesh to " << options.mesh.order << std::endl;
        }
        mesh_order = options.mesh.order;
        mesh.SetCurvature(mesh_order);
    }

    // mesh refinement if specified in input
    for (int lev = 0; lev < options.mesh.ref_ser; lev++) {
        mesh.UniformRefinement();
    }

    std::shared_ptr<mfem::ParMesh> pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, mesh);

    for (int lev = 0; lev < options.mesh.ref_par; lev++) {
        pmesh->UniformRefinement();
    }
    pmesh->SetAttributes();

    return pmesh;
}

std::map<int, int> create_grains_to_map(const ExaOptions& options, const mfem::Array<int>& grains) {
    std::map<int, int> grain2regions;

    if (!options.region_mapping_file) {
        for (const auto& item : grains) {
            const int key = item;
            grain2regions.emplace(key, 1);
        }
    } else {
        std::ifstream file(*options.region_mapping_file);

        MFEM_VERIFY_0(file.is_open(),
                      "Failed to open region mapping file: " << *options.region_mapping_file);

        std::string line;
        int key, value;
        size_t lineNumber = 0;

        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.empty()) {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            if (!(iss >> key >> value)) {
                MFEM_WARNING_0("Skipping unreadable line "
                               << lineNumber << " of region mapping file "
                               << *options.region_mapping_file << ": " << line);
                continue;
            }
            // Insert into the map
            // Since keys are assumed to be unique, this won't overwrite any existing entry.
            grain2regions.emplace(key, value);
        }
        file.close();
    }

    return grain2regions;
}

/**
 * @brief Helper function to initialize deformation gradient QuadratureFunction to identity
 */
void initializeDeformationGradientToIdentity(mfem::expt::PartialQuadratureFunction& defGrad) {
    // This function would need to be implemented to properly initialize
    // a 9-component QuadratureFunction representing 3x3 identity matrices
    // at each quadrature point

    double* data = defGrad.HostReadWrite();
    const int npts = defGrad.Size() / defGrad.GetVDim();

    // Initialize each 3x3 matrix to identity
    for (int i = 0; i < npts; i++) {
        double* mat = &data[i * 9];
        // Set to identity: [1,0,0,0,1,0,0,0,1]
        mat[0] = 1.0;
        mat[1] = 0.0;
        mat[2] = 0.0; // first row
        mat[3] = 0.0;
        mat[4] = 1.0;
        mat[5] = 0.0; // second row
        mat[6] = 0.0;
        mat[7] = 0.0;
        mat[8] = 1.0; // third row
    }
}

} // end namespace

TimeManagement::TimeManagement(ExaOptions& options) : time_type(options.time.time_type) {
    if (time_type == TimeStepType::FIXED) {
        dt = options.time.fixed_time->dt;
        dt_fixed = dt;
        dt_min = std::pow(dt_scale, max_failures) * dt;
        time_final = options.time.fixed_time->t_final;
    } else if (time_type == TimeStepType::AUTO) {
        dt = options.time.auto_time->dt_start;
        dt_min = options.time.auto_time->dt_min;
        dt_max = options.time.auto_time->dt_max;
        dt_scale = options.time.auto_time->dt_scale;
        time_final = options.time.auto_time->t_final;
        max_nr_steps = static_cast<size_t>(options.solvers.nonlinear_solver.iter);
        // insert logic to write out the first time step maybe?
    } else if (time_type == TimeStepType::CUSTOM) {
        // const auto dt_beg = options.time.custom_time->dt_values.begin();
        // const auto dt_end = options.time.custom_time->dt_values.end();
        custom_dt = options.time.custom_time->dt_values;
        dt = custom_dt[0];
        dt_min = std::pow(dt_scale, max_failures) *
                 static_cast<double>(*std::min_element(custom_dt.begin(), custom_dt.end()));
        time_final = std::accumulate(custom_dt.begin(), custom_dt.end(), 0.0);
    }

    prev_dt = dt;
    // Set our first cycle to the initial dt value;
    time = dt;

    const double tf_dt = std::abs(time_final - dt);
    if (tf_dt <= std::abs(1e-3 * dt)) {
        internal_tracker = TimeStep::FINAL;
    }
}

TimeStep TimeManagement::UpdateDeltaTime(const int nr_steps, const bool success) {
    // If simulation failed we want to scale down our dt by some factor
    if (!success) {
        // If we were already sub-stepping through a simulation and encouter this just fail out
        if (internal_tracker == TimeStep::SUBSTEP) {
            return TimeStep::FAILED;
        }
        // If we're already at the smallest allowable dt then there's nothing left to try
        if (dt <= dt_min) {
            return TimeStep::FAILED;
        }
        // For the very first failure we want to save off the initial guessed time step
        if (num_failures == 0) {
            dt_orig = dt;
        }
        // reset the time, update dt, and then update the time to correct time
        ResetTime();
        dt *= dt_scale;
        if (dt < dt_min) {
            dt = dt_min;
        }
        UpdateTime();
        num_failures++;
        num_sub_steps = 1;
        if (internal_tracker == TimeStep::FINAL) {
            const double tf_dt = std::abs(time - time_final);
            if (tf_dt > std::abs(1e-3 * dt)) {
                internal_tracker = TimeStep::RETRIAL;
            }
        }
        // If we've failed too many times just give up at this point
        if (num_failures > max_failures) {
            return TimeStep::FAILED;
        }
        // else we need to let the simulation now it's retrying it's time step again
        else {
            return TimeStep::RETRIAL;
        }
    }

    old_time = time;

    if (internal_tracker == TimeStep::FINAL) {
        internal_tracker = TimeStep::FINISHED;
        return TimeStep::FINISHED;
    }
    // This means we had a successful time step but previously we failed
    // Since we were using a fixed / custom dt here that means we need to substep
    // to get our desired dt that the user was asking for
    if (num_failures > 0) {
        required_num_sub_steps = (time_type != TimeStepType::AUTO)
                                     ? (static_cast<size_t>(1.0 / std::pow(dt_scale, num_failures)))
                                     : 0;
        num_failures = 0;
    }
    // If sub-stepping through our original dt then need to update the time while we go along
    if ((num_sub_steps < required_num_sub_steps) and (time_type != TimeStepType::AUTO)) {
        num_sub_steps += 1;
        UpdateTime();
        internal_tracker = TimeStep::SUBSTEP;
        return TimeStep::SUBSTEP;
    }

    prev_dt = dt;
    simulation_cycle++;
    // update our time based on the following logic
    if (time_type == TimeStepType::AUTO) {
        // update the dt
        const double niter_scale = static_cast<double>(max_nr_steps) * dt_scale;
        const int nr_temp = (nr_steps == 0) ? 1 : nr_steps;
        const double nr_iter = static_cast<double>(nr_temp);
        // Will approach dt_scale as nr_iter -> newton_iter
        // dt increases as long as nr_iter > niter_scale
        const double factor = niter_scale / nr_iter;
        dt *= factor;
        if (dt < dt_min) {
            dt = dt_min;
        }
        if (dt > dt_max) {
            dt = dt_max;
        }
    } else if (time_type == TimeStepType::CUSTOM) {
        dt = custom_dt[simulation_cycle];
    } else {
        dt = dt_fixed;
    }
    const double tnew = time + dt;
    const double tf_dt = std::abs(tnew - time_final);
    if (tf_dt <= std::abs(1e-3 * dt)) {
        internal_tracker = TimeStep::FINAL;
        time = tnew;
        return TimeStep::FINAL;
    } else if ((tnew - time_final) > 0) {
        internal_tracker = TimeStep::FINAL;
        dt = time_final - time;
        time = time_final;
        return TimeStep::FINAL;
    }
    time = tnew;
    // We're back on a normal time stepping procedure
    internal_tracker = TimeStep::NORMAL;
    return TimeStep::NORMAL;
}

bool TimeManagement::BCTime(const double desired_bc_time) {
    // if time is already past the desired_bc_time before updating this then we're not going to
    // update things to nail it
    if (time > desired_bc_time) {
        return false;
    }
    const double tnew = time + dt;
    const double tf_dt = desired_bc_time - tnew;
    // First check if we're when the radius when the next time step would be don't care about sign
    // yet
    if (std::abs(tf_dt) < std::abs(dt)) {
        // Now only update the dt value if we're past the original value
        if (tf_dt < 0.0) {
            ResetTime();
            dt += tf_dt;
            UpdateTime();
            return true;
        }
    }
    return false;
}

/**
 * @details Implementation notes. After a successful step, UpdateDeltaTime() leaves the time
 * manager describing the *next* step: `old_time` is the end of the completed step, `dt` and
 * `time` are the next step's size and end time, `simulation_cycle` counts completed steps and
 * the tracker says whether the next step is a NORMAL or the FINAL one. This function rebuilds
 * exactly that state from the checkpoint, using this run's time-step settings so a
 * continuation can change them:
 *
 * 1. Completed time and cycle count come from the checkpoint. The retrial and sub-step
 *    counters are reset: checkpoints are only written when IsCheckpointSafe(), i.e. when no
 *    retrial or sub-step sequence is in flight, so they were zero (or irrelevant) then too.
 * 2. prev_dt and dt_orig come from the checkpoint. Only log messages use them, and taking them
 *    from the checkpoint makes the restarted run print the same messages as the uninterrupted
 *    run.
 *    - prev_dt is the size of the last completed step. The driver prints it as "dt old was" at
 *      the start of the next cycle. If this run's settings change dt, the message shows the
 *      last step taken before the restart followed by the new dt.
 *    - dt_orig is reset at the first failure of a retrial sequence, before it is ever printed.
 *      It is restored only so both runs' time managers hold the same values.
 * 3. The next dt is chosen the way the success branch of UpdateDeltaTime() would choose it:
 *    CUSTOM takes custom_dt[completed cycles] (or the stored dt if this run's list is shorter),
 *    FIXED takes dt_fixed, and AUTO takes the stored next dt, which the writing run had
 *    already computed from its last Newton iteration count, clamped to this run's
 *    [dt_min, dt_max]. With unchanged settings this reproduces the writing run's dt exactly.
 * 4. The final-time logic of UpdateDeltaTime() is applied against this run's time_final (same
 *    1e-3 * dt tolerance): the next step becomes FINAL if it reaches t_final, and its dt is
 *    shortened if it would overshoot. One case UpdateDeltaTime() never meets is added: a
 *    checkpoint written at t_final (by write_final) resumes as FINISHED, so the driver loop
 *    does not run at all unless t_final has been increased.
 */
void TimeManagement::ResumeFromCheckpoint(const CheckpointState& state) {
    // Step 1: completed time and cycles; no retrial or sub-step sequence is in flight.
    old_time = state.completed_time;
    simulation_cycle = state.completed_cycles;
    num_failures = 0;
    num_sub_steps = 0;
    required_num_sub_steps = 0;

    // Step 2: the time steps used only in log messages.
    prev_dt = state.previous_dt;
    dt_orig = state.original_dt;

    // Step 3: next dt from this run's scheme.
    if (time_type == TimeStepType::CUSTOM) {
        dt = (simulation_cycle < custom_dt.size()) ? custom_dt[simulation_cycle] : state.next_dt;
    } else if (time_type == TimeStepType::FIXED) {
        dt = dt_fixed;
    } else {
        dt = std::clamp(state.next_dt, dt_min, dt_max);
    }

    // Step 4: final-time logic against this run's time_final.
    const double tolerance = std::abs(1e-3 * dt);
    if (time_final - old_time <= tolerance) {
        internal_tracker = TimeStep::FINISHED;
        time = old_time;
        return;
    }
    const double tnew = old_time + dt;
    if (std::abs(tnew - time_final) <= tolerance) {
        internal_tracker = TimeStep::FINAL;
        time = tnew;
    } else if (tnew > time_final) {
        internal_tracker = TimeStep::FINAL;
        dt = time_final - old_time;
        time = time_final;
    } else {
        internal_tracker = TimeStep::NORMAL;
        time = tnew;
    }
}

SimulationState::SimulationState(ExaOptions& options)
    : m_time_manager(options), m_options(options), class_device(options.solvers.rtmodel) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    m_time_manager = TimeManagement(options);

    // Restart runs ([Restart] read = true) build this object exactly as a fresh run does,
    // except that the few things a fresh run reads from input files are taken from the
    // checkpoint instead, since those files may no longer exist ("phase 1" of a restart; see
    // the background notes in restart_checkpoint.cpp). Each restart-specific step below is
    // marked. The evolving state itself is restored later, from the driver, by
    // RestoreFromCheckpoint() ("phase 2"), once every component has registered its
    // quadrature functions.
    //
    // Restart step: open the checkpoint and rebuild the mesh from it instead of reading the
    // mesh file. Both are collective, so they run on every rank, including ranks that end up
    // owning no elements. m_restart is also created for runs that only write checkpoints.
    const bool restart_run = options.restart.read;
#if defined(EXACONSTIT_USE_AXOM)
    if (options.restart.read || options.restart.write) {
        m_restart = std::make_unique<exaconstit::RestartCheckpoint>(options.restart,
                                                                    MPI_COMM_WORLD);
    }
    if (restart_run) {
        m_restart->OpenForRestart();
        m_mesh = m_restart->ReconstructMesh();
    } else {
        m_mesh = ::makeMesh(options, my_id);
    }
#else
    // RestartOptions::validate() already rejects this; kept as a safeguard for callers that
    // construct SimulationState without validating their options.
    MFEM_VERIFY_0(!restart_run, "Restarting from a checkpoint requires ExaConstit built with Axom");
    m_mesh = ::makeMesh(options, my_id);
#endif
    ::setupBoundaryConditions(options);
    // m_bc_manager = BCManager::GetInstance();

    // Set-up the mesh FEC and PFES
    {
        const int space_dim = m_mesh->SpaceDimension();
        std::string mesh_fec_str = "H1_" + std::to_string(space_dim) + "D_P" +
                                   std::to_string(options.mesh.order);
        m_map_fec[mesh_fec_str] = std::make_shared<mfem::H1_FECollection>(options.mesh.order,
                                                                          space_dim);
        m_mesh_fes = std::make_shared<mfem::ParFiniteElementSpace>(
            m_mesh.get(), m_map_fec[mesh_fec_str].get(), space_dim);
    }

    // Set-up our various mesh nodes / mesh QoI and
    // primal variables
    {
        // Create our mesh nodes
        m_mesh_nodes["mesh_current"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());
        // Create our mesh nodes
        m_mesh_nodes["mesh_t_beg"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());
        // Create our mesh nodes
        m_mesh_nodes["mesh_ref"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());

        // Set them to the current default vaules
        m_mesh->GetNodes(*m_mesh_nodes["mesh_current"]);
        (*m_mesh_nodes["mesh_t_beg"]) = *m_mesh_nodes["mesh_current"];
        (*m_mesh_nodes["mesh_ref"]) = *m_mesh_nodes["mesh_current"];

        {
            mfem::GridFunction* nodes =
                m_mesh_nodes["mesh_current"]
                    .get(); // set a nodes grid function to global current configuration
            int owns_nodes = 0;
            m_mesh->SwapNodes(nodes, owns_nodes); // m_mesh has current configuration nodes
            delete nodes;
        }

        m_mesh_qoi_nodes["displacement"] = std::make_shared<mfem::ParGridFunction>(
            m_mesh_fes.get());

        m_mesh_qoi_nodes["velocity"] = std::make_shared<mfem::ParGridFunction>(m_mesh_fes.get());

        (*m_mesh_qoi_nodes["displacement"]) = 0.0;
        (*m_mesh_qoi_nodes["velocity"]) = 0.0;
        // This is our velocity field
        m_primal_field = std::make_shared<mfem::Vector>(m_mesh_fes->TrueVSize());
        m_primal_field->UseDevice(true);
        m_primal_field_prev = std::make_shared<mfem::Vector>(m_mesh_fes->TrueVSize());
        m_primal_field_prev->UseDevice(true);
        (*m_primal_field) = 0.0;
        (*m_primal_field_prev) = 0.0;

#if defined(EXACONSTIT_USE_AXOM)
        // Restart step: overwrite the coordinate fields just created with the checkpointed
        // reference coordinates, and verify the DOF layout, before anything reads them. The
        // rebuilt mesh has no Nodes, so GetNodes() above projected from the vertex positions,
        // which differs from the original run's coordinates in the last bits. Components built
        // before RestoreFromCheckpoint() keep values derived from the starting geometry
        // (SystemDriver's mono_def_bcs DOF search, the UMAT reference shape gradients), so they
        // must see exactly the original coordinates for the restart to be bitwise exact. See
        // RestartCheckpoint::RestoreReferenceConfiguration(). Collective.
        if (restart_run) {
            m_restart->RestoreReferenceConfiguration(*this);
        }
#endif
    }

    {
        const int space_dim = m_mesh->SpaceDimension();
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
        m_map_fec[l2_fec_str] = std::make_shared<mfem::L2_FECollection>(0, space_dim);
    }

    // Global QuadratureSpace Setup and QFs
    const int int_order = 2 * options.mesh.order + 1;
    {
        mfem::Array<bool> global_index;
        m_map_qs["global"] = std::make_shared<mfem::expt::PartialQuadratureSpace>(
            m_mesh, int_order, global_index);

        m_map_qs["global_ord_0"] = std::make_shared<mfem::expt::PartialQuadratureSpace>(
            m_mesh, 1, global_index);

        m_map_qfs["cauchy_stress_beg"] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            m_map_qs["global"], 6, 0.0);
        m_map_qfs["cauchy_stress_end"] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            m_map_qs["global"], 6, 0.0);
        m_map_qfs["cauchy_stress_avg"] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            m_map_qs["global_ord_0"], 6, 0.0);

        m_map_qfs["cauchy_stress_beg"]->operator=(0.0);
        m_map_qfs["cauchy_stress_end"]->operator=(0.0);
        m_map_qfs["cauchy_stress_avg"]->operator=(0.0);

        m_model_update_qf_pairs.push_back(std::make_pair("cauchy_stress_beg", "cauchy_stress_end"));

        auto kinetic_grads_name = GetQuadratureFunctionMapName("kinetic_grads", -1);
        m_map_qfs[kinetic_grads_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            m_map_qs["global"], 9, 0.0);
        ::initializeDeformationGradientToIdentity(*m_map_qfs[kinetic_grads_name]);

        auto tangent_stiffness_name = GetQuadratureFunctionMapName("tangent_stiffness", -1);
        m_map_qfs[tangent_stiffness_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
            m_map_qs["global"], 36, 0.0);
        m_map_qfs[tangent_stiffness_name]->operator=(0.0);
    }

    // Material state variable and qspace setup
    {
        // Update our options file with the correct number of state variables now
        UpdateExaOptionsWithOrientationCounts();
        // create our region map now
        const int loc_nelems = m_mesh->GetNE();
        mfem::Array2D<bool> region_map(static_cast<int>(options.materials.size()), loc_nelems);
        region_map = false;
        mfem::Array<int> grains(loc_nelems);

        // Grain IDs. A fresh run reads them from the mesh's element attributes. In a
        // checkpointed mesh the element attributes already hold region IDs (they are
        // overwritten below on a fresh run), so the restart step reads the grain IDs the
        // checkpoint stored separately. Collective on restart runs.
        if (restart_run) {
#if defined(EXACONSTIT_USE_AXOM)
            m_restart->ReadElementGrainIds(grains);
#endif
        } else {
            for (int i = 0; i < loc_nelems; i++) {
                grains.operator[](i) = m_mesh->GetAttribute(i);
            }
        }

        {
            auto pfes = GetParFiniteElementSpace(1);
            m_grains = std::make_shared<mfem::ParGridFunction>(pfes.get());
            m_grains->HostWrite();
            for (int i = 0; i < loc_nelems; i++) {
                m_grains->operator[](i) = grains.operator[](i);
            }
        }

        std::map<int, int> grains2region;
        if (!restart_run) {
            grains2region = ::create_grains_to_map(options, grains);

            for (int i = 0; i < loc_nelems; i++) {
                const int grain_id = grains.operator[](i);
                const int region_id = grains2region.at(grain_id);
                m_mesh->SetAttribute(i, region_id);
                region_map(region_id - 1, i) = true;
            }
        } else {
            // Restart step: a fresh run maps grains to regions with the region-mapping file and
            // then stores the region IDs in the element attributes. The checkpointed mesh
            // already carries those 1-based region IDs, so the region map is built from them
            // directly and the region-mapping file is never read.
            //
            // A region ID outside [1, number of materials] means the material list shrank
            // since the checkpoint was written. That check depends on each rank's own
            // elements, so the ranks agree on it with an MPI_Allreduce and all abort together;
            // a rank with no elements contributes "no problem".
            const int num_regions = static_cast<int>(options.materials.size());
            int local_out_of_range = 0;
            for (int i = 0; i < loc_nelems; i++) {
                const int region_id = m_mesh->GetAttribute(i);
                if (region_id < 1 || region_id > num_regions) {
                    local_out_of_range = 1;
                    continue;
                }
                region_map(region_id - 1, i) = true;
            }
            int any_out_of_range = 0;
            MPI_Allreduce(
                &local_out_of_range, &any_out_of_range, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            MFEM_VERIFY_0(any_out_of_range == 0,
                          "The checkpoint has elements in regions beyond the "
                              << num_regions
                              << " material(s) configured for this run; the material list "
                                 "cannot change on a restart");
        }

        // update all of our attributes
        m_mesh->SetAttributes();

        for (auto matl : options.materials) {
            const int region_id = matl.region_id - 1;
            m_region_material_type.push_back(matl.mech_type);
            m_material_name_region.push_back(std::make_pair(matl.material_name, region_id));
            std::string qspace_name = GetRegionName(region_id);
            std::string qspace_name_0 = qspace_name + "_ord_0";

            m_material_properties.emplace(qspace_name, matl.properties.properties);
            mfem::Array<bool> loc_index(region_map.GetRow(region_id), loc_nelems, false);
            // Check if this region is active on this rank
            const int loc_num_elems = std::accumulate(loc_index.begin(), loc_index.end(), 0);
            m_is_region_active[region_id] = (loc_num_elems > 0);
            if (loc_num_elems == 0) {
                continue;
            }

            m_map_qs[qspace_name] = std::make_shared<mfem::expt::PartialQuadratureSpace>(
                m_mesh, int_order, loc_index);

            m_map_qs[qspace_name_0] = std::make_shared<mfem::expt::PartialQuadratureSpace>(
                m_mesh, 1, loc_index);

            auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
            auto state_var_end_name = GetQuadratureFunctionMapName("state_var_end", region_id);
            auto state_var_avg_name = GetQuadratureFunctionMapName("state_var_avg", region_id);
            auto cauchy_stress_beg_name = GetQuadratureFunctionMapName("cauchy_stress_beg",
                                                                       region_id);
            auto cauchy_stress_end_name = GetQuadratureFunctionMapName("cauchy_stress_end",
                                                                       region_id);
            auto cauchy_stress_avg_name = GetQuadratureFunctionMapName("cauchy_stress_avg",
                                                                       region_id);
            auto tangent_stiffness_name = GetQuadratureFunctionMapName("tangent_stiffness",
                                                                       region_id);

            if (m_options.post_processing.volume_averages.def_grad ||
                m_options.post_processing.volume_averages.euler_strain ||
                m_options.post_processing.volume_averages.elastic_strain) {
                auto def_grad = GetQuadratureFunctionMapName("kinetic_grads", region_id);
                m_map_qfs[def_grad] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 9, 0.0);
                ::initializeDeformationGradientToIdentity(*m_map_qfs[def_grad]);
            }

            if (m_options.post_processing.volume_averages.plastic_work ||
                m_options.post_processing.volume_averages.eq_pl_strain) {
                auto scalar = GetQuadratureFunctionMapName("scalar", region_id);
                m_map_qfs[scalar] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 1, 0.0);
            }

            m_map_qfs[state_var_beg_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);
            m_map_qfs[state_var_end_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                m_map_qs[qspace_name], matl.state_vars.num_vars, 0.0);
            m_map_qfs[cauchy_stress_beg_name] =
                std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 6, 0.0);
            m_map_qfs[cauchy_stress_end_name] =
                std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 6, 0.0);
            m_map_qfs[tangent_stiffness_name] =
                std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 36, 0.0);

            m_map_qfs[cauchy_stress_avg_name] =
                std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name_0], 6, 0.0);
            m_map_qfs[state_var_avg_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                m_map_qs[qspace_name_0], matl.state_vars.num_vars, 0.0);

            m_map_qfs[state_var_beg_name]->operator=(0.0);
            m_map_qfs[state_var_end_name]->operator=(0.0);
            m_map_qfs[cauchy_stress_beg_name]->operator=(0.0);
            m_map_qfs[cauchy_stress_end_name]->operator=(0.0);
            m_map_qfs[tangent_stiffness_name]->operator=(0.0);
            m_map_qfs[state_var_avg_name]->operator=(0.0);
            m_map_qfs[cauchy_stress_avg_name]->operator=(0.0);

            if (matl.mech_type == MechType::UMAT) {
                auto def_grad_name = GetQuadratureFunctionMapName("def_grad_beg", region_id);
                m_map_qfs[def_grad_name] = std::make_shared<mfem::expt::PartialQuadratureFunction>(
                    m_map_qs[qspace_name], 9, 0.0);
                ::initializeDeformationGradientToIdentity(*m_map_qfs[def_grad_name]);
            }

            m_model_update_qf_pairs.push_back(
                std::make_pair(state_var_beg_name, state_var_end_name));
            m_model_update_qf_pairs.push_back(
                std::make_pair(cauchy_stress_beg_name, cauchy_stress_end_name));
        }

        CreateRegionCommunicators();
        // Restart step: skip the state-variable and orientation initialization. Its input
        // files may no longer exist, and RestoreFromCheckpoint() overwrites all state
        // variables (orientations included) with the checkpointed values anyway.
        if (!restart_run) {
            InitializeStateVariables(grains2region);
        }
    }
#if defined(EXACONSTIT_USE_AXOM)
    // Restart step: the mesh and grain IDs have been copied out of the checkpoint, so free
    // those arrays now; the rest of the checkpoint stays loaded for RestoreFromCheckpoint().
    if (restart_run) {
        m_restart->ReleaseMeshArrays();
    }
#endif
}

SimulationState::~SimulationState() {
    for (auto& [region_id, comm] : m_region_communicators) {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }
}

void SimulationState::RestoreFromCheckpoint() {
#if defined(EXACONSTIT_USE_AXOM)
    MFEM_VERIFY_0(m_restart && m_options.restart.read,
                  "RestoreFromCheckpoint() was called on a run that is not restarting");
    m_restart->RestoreState(*this);
#else
    MFEM_ABORT_0("Restarting from a checkpoint requires ExaConstit built with Axom");
#endif
}

void SimulationState::WriteCheckpointIfDue(const size_t driver_step) {
#if defined(EXACONSTIT_USE_AXOM)
    if (m_restart) {
        m_restart->WriteCheckpointIfDue(*this, driver_step);
    }
#else
    static_cast<void>(driver_step);
#endif
}

size_t SimulationState::GetRestartDriverStep() const {
#if defined(EXACONSTIT_USE_AXOM)
    return m_restart ? m_restart->GetRestartDriverStep() : 0;
#else
    return 0;
#endif
}

bool SimulationState::AddQuadratureFunction(const std::string_view& qf_name,
                                            const int vdim,
                                            const int region) {
    std::string qf_name_mat = GetQuadratureFunctionMapName(qf_name, region);
    if (m_map_qfs.find(qf_name_mat) != m_map_qfs.end()) {
        return false;
    }
    // find(), not operator[]: a region with no elements on this rank has no quadrature space
    // here, and operator[] would insert a null entry into m_map_qs, which every loop over the
    // spaces (including the checkpoint writer) would then dereference.
    const auto space = m_map_qs.find(GetRegionName(region));
    if (space == m_map_qs.end() || !space->second) {
        return false;
    }
    m_map_qfs.emplace(
        qf_name_mat,
        std::make_shared<mfem::expt::PartialQuadratureFunction>(space->second, vdim, 0.0));
    return true;
}

std::pair<int, int>
SimulationState::GetQuadratureFunctionStatePair(const std::string_view& state_name,
                                                const int region) const {
    std::string mat_name = GetQuadratureFunctionMapName(state_name, region);
    if (m_map_qf_mappings.find(mat_name) == m_map_qf_mappings.end()) {
        return {-1, -1};
    }
    const std::pair<int, int> output = m_map_qf_mappings.at(mat_name);
    return output;
}

bool SimulationState::AddQuadratureFunctionStatePair(const std::string_view state_name,
                                                     std::pair<int, int> state_pair,
                                                     const int region) {
    std::string mat_name = GetQuadratureFunctionMapName(state_name, region);
    if (m_map_qf_mappings.find(mat_name) == m_map_qf_mappings.end()) {
        m_map_qf_mappings.emplace(mat_name, state_pair);
        return true;
    }
    return false;
}

void SimulationState::FinishCycle() {
    (*m_primal_field_prev) = *m_primal_field;
    (*m_mesh_qoi_nodes["displacement"]) = *m_mesh_nodes["mesh_current"];
    (*m_mesh_qoi_nodes["displacement"]) -= *m_mesh_nodes["mesh_ref"];
    m_mesh_qoi_nodes["velocity"]->Distribute(*m_primal_field);
    // Code previously had beg time coords updated after the update model aspect of things
    // UpdateModel();
    (*m_mesh_nodes["mesh_t_beg"]) = *m_mesh_nodes["mesh_current"];
}

std::shared_ptr<mfem::ParFiniteElementSpace>
SimulationState::GetParFiniteElementSpace(const int vdim) {
    if (m_map_pfes.find(vdim) == m_map_pfes.end()) {
        const int space_dim = m_mesh->SpaceDimension();
        std::string l2_fec_str = "L2_" + std::to_string(space_dim) + "D_P" + std::to_string(0);
        auto l2_fec = m_map_fec[l2_fec_str];
        auto value = std::make_shared<mfem::ParFiniteElementSpace>(
            m_mesh.get(), l2_fec.get(), vdim, mfem::Ordering::byVDIM);
        m_map_pfes.emplace(vdim, std::move(value));
    }
    return m_map_pfes[vdim];
}

std::string SimulationState::GetRegionDisplayName(const int region) const {
    std::string raw_name = GetRegionName(region);
    if (raw_name.empty())
        return raw_name;

    std::string display_name = raw_name;

    // Replace underscores with spaces
    std::replace(display_name.begin(), display_name.end(), '_', ' ');

    // Capitalize first letter and letters after spaces
    bool capitalize_next = true;
    std::transform(display_name.begin(),
                   display_name.end(),
                   display_name.begin(),
                   [&capitalize_next](unsigned char c) -> char { // Explicitly specify return type
                       if (std::isspace(c)) {
                           capitalize_next = true;
                           return static_cast<char>(c);
                       } else if (capitalize_next) {
                           capitalize_next = false;
                           return static_cast<char>(std::toupper(c)); // No cast needed now
                       }
                       return static_cast<char>(c);
                   });

    return display_name;
}

void SimulationState::CreateRegionCommunicators() {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Get all unique region IDs across all materials
    std::set<int> all_region_ids;
    for (const auto& mat : m_material_name_region) {
        all_region_ids.insert(mat.second);
    }

    // For each region, create a communicator containing only ranks with that region
    for (int region_id : all_region_ids) {
        // Each rank contributes whether it has this region
        int has_region = m_is_region_active[region_id] ? 1 : 0;
        std::vector<int> all_has_region(static_cast<size_t>(mpi_size));

        MPI_Allgather(&has_region, 1, MPI_INT, all_has_region.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Build list of ranks that have this region
        std::vector<int> ranks_with_region;
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (all_has_region[static_cast<size_t>(rank)]) {
                ranks_with_region.push_back(rank);
            }
        }

        // Create MPI group and communicator for this region
        if (!ranks_with_region.empty()) {
            m_region_root_rank[region_id] = ranks_with_region[0]; // First is lowest
            if (!has_region) {
                continue;
            }
            MPI_Group world_group, region_group;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);
            MPI_Group_incl(world_group,
                           static_cast<int>(ranks_with_region.size()),
                           ranks_with_region.data(),
                           &region_group);

            MPI_Comm region_comm;
            MPI_Comm_create_group(MPI_COMM_WORLD, region_group, 0, &region_comm);

            // Only store the communicator if this rank is part of it
            m_region_communicators[region_id] = region_comm;

            MPI_Group_free(&region_group);
            MPI_Group_free(&world_group);
        }
    }
}

// Modified InitializeStateVariables to load shared orientation data first
void SimulationState::InitializeStateVariables(const std::map<int, int>& grains2region) {
    // First, load shared orientation data if any material needs it
    for (const auto& material : m_options.materials) {
        if (material.grain_info.has_value() && material.grain_info->orientation_file.has_value()) {
            LoadSharedOrientationData(material.grain_info->orientation_file.value(),
                                      material.grain_info->num_grains);
            break; // Only need to load once since it's shared
        }
    }

    // Initialize state variables for each material region
    for (size_t i = 0; i < m_options.materials.size(); ++i) {
        if (!IsRegionActive(static_cast<int>(i))) {
            continue;
        }
        const auto& material = m_options.materials[i];
        const int region_id = material.region_id - 1;
        InitializeRegionStateVariables(region_id, material, grains2region);

        auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
        auto state_var_qf_beg = m_map_qfs[state_var_beg_name];

        auto state_var_end_name = GetQuadratureFunctionMapName("state_var_end", region_id);
        auto state_var_qf_end = m_map_qfs[state_var_end_name];
        state_var_qf_end->operator=(*state_var_qf_beg.get());
    }

    // Clean up shared orientation data after all regions are initialized
    CleanupSharedOrientationData();
}

// Refactored InitializeRegionStateVariables function
void SimulationState::InitializeRegionStateVariables(int region_id,
                                                     const MaterialOptions& material,
                                                     const std::map<int, int>& grains2region) {
    // Get the state variable QuadratureFunction for this region
    auto state_var_beg_name = GetQuadratureFunctionMapName("state_var_beg", region_id);
    auto state_var_qf = m_map_qfs[state_var_beg_name];

    // Get the QuadratureSpace for this region
    std::string qspace_name = GetRegionName(region_id);
    auto qspace = m_map_qs[qspace_name];
    // Prepare orientation data for this region
    OrientationConfig orientation_config = PrepareOrientationForRegion(material);
    MFEM_VERIFY_0(!material.grain_info.has_value() || orientation_config.is_valid,
                  "Unable to set up the orientation data for material " << material.material_name);

    // Calculate effective state variable count (includes orientations)
    const int effective_state_var_size = material.state_vars.num_vars;
    const int base_state_var_size = material.state_vars.num_vars - orientation_config.stride;

    // Load base state variable initial values
    std::vector<double> state_var_data;
    if (!material.state_vars.initial_values.empty()) {
        state_var_data = material.state_vars.initial_values;
    } else if (!material.state_vars.state_file.empty()) {
        // Load from file if not already loaded
        std::ifstream file(material.state_vars.state_file);
        MFEM_VERIFY_0(file.is_open(),
                      "Cannot open state variables file: " << material.state_vars.state_file);

        double value;
        while (file >> value) {
            state_var_data.push_back(value);
        }
        file.close();
    }

    // Validate state variable data size
    if (state_var_data.size() != static_cast<size_t>(base_state_var_size)) {
        MFEM_WARNING_0("State variable data size ("
                       << state_var_data.size() << ") doesn't match expected size ("
                       << base_state_var_size << ") for material " << material.material_name
                       << ". Any missing state variables are set to 0.");
    }

    // Get the data pointer for the QuadratureFunction
    double* qf_data = state_var_qf->HostReadWrite();
    const int qf_vdim = state_var_qf->GetVDim();

    // Validate that our total size matches
    MFEM_VERIFY_0(qf_vdim == effective_state_var_size,
                  "QuadratureFunction vdim (" << qf_vdim << ") doesn't match effective total size ("
                                              << effective_state_var_size << ") for material "
                                              << material.material_name);

    // Get the local to global element mapping for this region
    const auto& local2global = qspace->GetLocal2Global();
    const int num_local_elements = qspace->GetNumLocalElements();

    // Loop over local elements in this region
    for (int local_elem = 0; local_elem < num_local_elements; ++local_elem) {
        const int global_elem = local2global[local_elem];

        // Get the grain ID for this element (before region mapping)
        const int grain_id = static_cast<int>(m_grains->operator[](global_elem));

        // Verify this element belongs to the current region
        const int elem_region = grains2region.at(grain_id);
        if (elem_region != (region_id + 1)) { // grains2region uses 1-based indexing
            continue;                         // Skip elements that don't belong to this region
        }

        // Get the integration rule for this element
        const mfem::IntegrationRule* ir = &(state_var_qf->GetSpaceShared()->GetIntRule(local_elem));
        const int num_qpts = ir->GetNPoints();

        // Loop over quadrature points in this element
        for (int qpt = 0; qpt < num_qpts; ++qpt) {
            // Calculate the base index for this quadrature point's data
            const int qpt_base_index = (local_elem * num_qpts + qpt) * qf_vdim;

            // Fill state variables
            size_t state_var_idx = 0;
            for (int k = 0; k < qf_vdim; ++k) {
                // Check if this component is NOT orientation data
                if (orientation_config.is_valid && k > orientation_config.offset_start &&
                    k < orientation_config.offset_end) {
                    // Skip orientation components - they'll be filled separately
                    continue;
                } else {
                    // This is state variable data
                    double var_data = 0.0;
                    // Missing values were already reported by the size check above
                    if (state_var_idx < state_var_data.size()) {
                        var_data = state_var_data[state_var_idx];
                    }
                    qf_data[qpt_base_index + k] = var_data;
                    state_var_idx++;
                }
            }

            // Fill orientation data (converted to format required by this material)
            FillOrientationData(qf_data, qpt_base_index, qf_vdim, grain_id, orientation_config);
        }
    }

    if (my_id == 0) {
        std::cout << "Initialized state variables for material " << material.material_name
                  << " (region " << region_id << ")" << std::endl;
        if (orientation_config.is_valid) {
            const std::string format_name = (material.mech_type == MechType::EXACMECH)
                                                ? "unit quaternions"
                                                : "values from the orientation file";
            std::cout << "  - Orientation data: " << format_name << " (stride "
                      << orientation_config.stride << ")" << std::endl;
        }
    }
}

// Additional utility function to update ExaOptions with correct state variable counts
void SimulationState::UpdateExaOptionsWithOrientationCounts() {
    for (auto& material : m_options.materials) {
        int effective_count = CalculateEffectiveStateVarCount(material);
        if (effective_count != material.state_vars.num_vars) {
            if (my_id == 0) {
                std::cout << "Updated state variable count for material " << material.material_name
                          << " from " << material.state_vars.num_vars << " to " << effective_count
                          << " (includes orientations)" << std::endl;
            }
            material.state_vars.num_vars = effective_count;
        }
    }
}

int SimulationState::CalculateEffectiveStateVarCount(const MaterialOptions& material) {
    int base_count = material.state_vars.num_vars;

    if (material.grain_info.has_value()) {
        const auto& grain_info = material.grain_info.value();

        // Add orientation variables based on what format this material needs
        int orientation_vars = 0;
        if (grain_info.ori_type == OriType::QUAT) {
            orientation_vars = 4; // Quaternions
        } else if (grain_info.ori_type == OriType::EULER) {
            orientation_vars = 3; // Euler angles
        } else if (grain_info.ori_type == OriType::CUSTOM) {
            orientation_vars = grain_info.ori_stride; // Custom stride
        }

        base_count += orientation_vars;
    }

    return base_count;
}

void SimulationState::LoadSharedOrientationData(const std::string& orientation_file,
                                                int num_grains) {
    if (m_shared_orientation_data.is_loaded) {
        // Already loaded, just verify grain count matches
        MFEM_VERIFY_0(m_shared_orientation_data.num_grains == num_grains,
                      "Grain count mismatch. Expected " << num_grains << " but shared data has "
                                                        << m_shared_orientation_data.num_grains);
        return;
    }

    std::ifstream orient_file(orientation_file);
    MFEM_VERIFY_0(orient_file.is_open(), "Cannot open orientation file: " << orientation_file);

    // Read every value in the file. How these values are interpreted (quaternions for
    // ExaCMech models, or whatever format a UMAT expects) depends on the material, so
    // that is handled per region in PrepareOrientationForRegion.
    double value;
    while (orient_file >> value) {
        m_shared_orientation_data.values.push_back(value);
    }
    orient_file.close();

    MFEM_VERIFY_0(!m_shared_orientation_data.values.empty(),
                  "Orientation file " << orientation_file << " has no values");

    m_shared_orientation_data.num_grains = num_grains;
    m_shared_orientation_data.is_loaded = true;

    if (my_id == 0) {
        std::cout << "Loaded shared orientation data: " << m_shared_orientation_data.values.size()
                  << " values for " << num_grains << " grains" << std::endl;
    }
}

void SimulationState::CleanupSharedOrientationData() {
    m_shared_orientation_data.values.clear();
    m_shared_orientation_data.values.shrink_to_fit();
    m_shared_orientation_data.num_grains = 0;
    m_shared_orientation_data.is_loaded = false;

    if (my_id == 0) {
        std::cout << "Cleaned up shared orientation data to free memory" << std::endl;
    }
}

std::vector<double>
SimulationState::ConvertQuaternionsToEuler(const std::vector<double>& quaternions, int num_grains) {
    std::vector<double> euler_angles;
    euler_angles.reserve(static_cast<size_t>(num_grains) * 3);

    auto bunge_func = [](const double* const quat, double* bunge_ang) {
        // below is equivalent to std::sqrt(std::numeric_limits<T>::epsilon);
        constexpr double tol = 1.4901161193847656e-08;
        const auto q03 = quat[0] * quat[0] + quat[3] * quat[3];
        const auto q12 = quat[1] * quat[1] + quat[2] * quat[2];
        const auto xi = std::sqrt(q03 * q12);
        // We get to now go through all of the different cases that this might break down into
        if (std::abs(xi) < tol && std::abs(q12) < tol) {
            bunge_ang[0] = std::atan2(-2.0 * quat[0] * quat[3],
                                      quat[0] * quat[0] - quat[3] * quat[3]);
            // All of the other values are zero
        } else if (std::abs(xi) < tol && std::abs(q03) < tol) {
            bunge_ang[0] = std::atan2(2.0 * quat[1] * quat[2],
                                      quat[1] * quat[1] - quat[2] * quat[2]);
            bunge_ang[1] = 3.141592653589793;
            // The other value is zero
        } else {
            const double inv_xi = 1.0 / xi;
            // The atan2 terms are pretty long so we're breaking it down into a couple of temp terms
            const double t1 = inv_xi * (quat[1] * quat[3] - quat[0] * quat[2]);
            const double t2 = inv_xi * (-quat[0] * quat[1] - quat[2] * quat[3]);
            // We can now assign the first two bunge angles
            bunge_ang[0] = std::atan2(t1, t2);
            bunge_ang[1] = std::atan2(2.0 * xi, q03 - q12);
            // Once again these terms going into the atan2 term are pretty long
            {
                const double t1_ = inv_xi * (quat[0] * quat[2] + quat[1] * quat[3]);
                const double t2_ = inv_xi * (quat[2] * quat[3] - quat[0] * quat[1]);
                // We can finally find the final bunge angle
                bunge_ang[2] = std::atan2(t1_, t2_);
            }
        }
    };

    for (int i = 0; i < num_grains; ++i) {
        const int base_idx = i * 4;
        const double* const quat = &(quaternions.data()[base_idx]);
        double bunge[3] = {};

        bunge_func(quat, bunge);

        euler_angles.push_back(bunge[0]);
        euler_angles.push_back(bunge[1]);
        euler_angles.push_back(bunge[2]);
    }

    return euler_angles;
}

std::vector<double>
SimulationState::ConvertQuaternionsToMatrix(const std::vector<double>& quaternions,
                                            int num_grains) {
    std::vector<double> matrices;
    matrices.reserve(static_cast<size_t>(num_grains) * 9);

    for (int i = 0; i < num_grains; ++i) {
        const int base_idx = i * 4;
        const double* const quat = &(quaternions.data()[base_idx]);

        const double qbar = quat[0] * quat[0] -
                            (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

        // Row-major order: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
        matrices.push_back(qbar + 2.0 * quat[1] * quat[1]);
        matrices.push_back(2.0 * (quat[1] * quat[2] - quat[0] * quat[3]));
        matrices.push_back(2.0 * (quat[1] * quat[3] + quat[0] * quat[2]));

        matrices.push_back(2.0 * (quat[1] * quat[2] + quat[0] * quat[3]));
        matrices.push_back(qbar + 2.0 * quat[2] * quat[2]);
        matrices.push_back(2.0 * (quat[2] * quat[3] - quat[0] * quat[1]));

        matrices.push_back(2.0 * (quat[1] * quat[3] - quat[0] * quat[2]));
        matrices.push_back(2.0 * (quat[2] * quat[3] + quat[0] * quat[1]));
        matrices.push_back(qbar + 2.0 * quat[3] * quat[3]);
    }

    return matrices;
}

SimulationState::OrientationConfig
SimulationState::PrepareOrientationForRegion(const MaterialOptions& material) {
    OrientationConfig config;

    if (!material.grain_info.has_value() || !m_shared_orientation_data.is_loaded) {
        return config; // Return invalid config
    }

    const auto& grain_info = material.grain_info.value();

    // Verify grain count consistency
    MFEM_VERIFY_0(m_shared_orientation_data.num_grains == grain_info.num_grains,
                  "Grain count mismatch for material "
                      << material.material_name << ". Expected " << grain_info.num_grains
                      << " but shared data has " << m_shared_orientation_data.num_grains);

    // ExaCMech models always take unit quaternions (w, x, y, z) representing passive rotations
    // from the crystal to the sample frame. Any other model (e.g. a UMAT) is given the values
    // from the orientation file as is, ori_stride values per grain, in whatever format it expects.
    const bool is_exacmech = (material.mech_type == MechType::EXACMECH);
    int stride = 0;
    if (is_exacmech || grain_info.ori_type == OriType::QUAT) {
        stride = 4;
    } else if (grain_info.ori_type == OriType::EULER) {
        stride = 3;
    } else if (grain_info.ori_type == OriType::CUSTOM) {
        stride = grain_info.ori_stride;
    }

    MFEM_VERIFY_0(stride > 0,
                  "Invalid orientation stride (" << stride << ") for material "
                                                 << material.material_name);

    const auto& values = m_shared_orientation_data.values;
    const size_t expected_size = static_cast<size_t>(stride) *
                                 static_cast<size_t>(grain_info.num_grains);
    MFEM_VERIFY_0(values.size() >= expected_size,
                  "Orientation file has " << values.size() << " values but material "
                                          << material.material_name << " needs " << expected_size
                                          << " (" << grain_info.num_grains << " grains x " << stride
                                          << " values per orientation)");
    if (values.size() > expected_size) {
        MFEM_WARNING_0("Orientation file has "
                       << values.size() << " values but material " << material.material_name
                       << " only uses the first " << expected_size << " (" << grain_info.num_grains
                       << " grains x " << stride << " values per orientation)");
    }

    config.data.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(expected_size));
    config.stride = stride;

    if (is_exacmech) {
        // Make sure the quaternions are normalized
        int num_normalized = 0;
        for (size_t i = 0; i < static_cast<size_t>(grain_info.num_grains); ++i) {
            double* quat = &config.data[i * 4];
            const double norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                          quat[2] * quat[2] + quat[3] * quat[3]);
            MFEM_VERIFY_0(norm != 0.0,
                          "Quaternion " << i << " for material " << material.material_name
                                        << " has zero norm");
            if (std::fabs(norm - 1.0) > 1e-6) {
                num_normalized++;
                for (int j = 0; j < 4; ++j) {
                    quat[j] /= norm;
                }
            }
        }
        if (num_normalized > 0) {
            MFEM_WARNING_0(num_normalized << " quaternions for material " << material.material_name
                                          << " were not normalized and have been normalized");
        }
    }

    // Calculate placement offsets
    auto offsets = CalculateOrientationOffsets(material, config.stride);
    config.offset_start = offsets.first;
    config.offset_end = offsets.second;
    config.is_valid = true;

    return config;
}

std::pair<int, int> SimulationState::CalculateOrientationOffsets(const MaterialOptions& material,
                                                                 int orientation_stride) {
    if (!material.grain_info.has_value() || orientation_stride == 0) {
        return {-1, 0};
    }

    const auto& grain_info = material.grain_info.value();
    // UpdateExaOptionsWithOrientationCounts has already added the orientation
    // components to num_vars, so this is the total state variable size
    const int state_var_size = material.state_vars.num_vars;

    int offset_start, offset_end;

    if (grain_info.ori_state_var_loc < 0) {
        // Put orientation data at the end, in the last orientation_stride components
        if (my_id == 0) {
            std::cout << "Note: Orientation data placed at end of state variable array "
                      << "for material " << material.material_name << std::endl;
        }
        offset_start = state_var_size - orientation_stride - 1;
        offset_end = state_var_size;
    } else if (grain_info.ori_state_var_loc == 0) {
        // Put orientation data at the beginning
        offset_start = -1;
        offset_end = orientation_stride;
    } else {
        // Put orientation data at specified location
        offset_start = grain_info.ori_state_var_loc - 1;
        offset_end = grain_info.ori_state_var_loc + orientation_stride;
    }

    return {offset_start, offset_end};
}

void SimulationState::FillOrientationData(double* qf_data,
                                          int qpt_base_index,
                                          int qf_vdim,
                                          int grain_id,
                                          const OrientationConfig& orientation_config) {
    if (!orientation_config.is_valid || orientation_config.stride == 0) {
        return; // No orientation data to fill
    }

    for (int k = 0; k < qf_vdim; ++k) {
        if (k > orientation_config.offset_start && k < orientation_config.offset_end) {
            // This is orientation data
            const int grain_idx = k - orientation_config.offset_start - 1;
            const size_t orient_idx = static_cast<size_t>(
                orientation_config.stride * (grain_id - 1) + grain_idx);

            MFEM_VERIFY_0(orient_idx < orientation_config.data.size(),
                          "Missing orientation data for grain " << grain_id << ", component "
                                                                << grain_idx);
            qf_data[qpt_base_index + k] = orientation_config.data[orient_idx];
        }
    }
}