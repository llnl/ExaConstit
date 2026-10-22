#pragma once

/**
 * @file option_microstructure_hook.hpp
 * @brief The two [Mesh.Microstructure] functions the options code calls.
 *
 * Declared here, apart from sim_state/microstructure_hook.hpp, so that the options library does
 * not include any microstructure generator header: these declarations need nothing but
 * ExaOptions. They are implemented in sim_state/microstructure_hook.cpp, which also compiles
 * (and rejects the table) in builds without the generator.
 */

class ExaOptions;

namespace exaconstit {

/**
 * @brief True when this ExaConstit was built with the microstructure generator
 * (ENABLE_MICROSTRUCTURE=ON), i.e. when [Mesh.Microstructure] can be used
 */
bool MicrostructureHookAvailable();

/**
 * @brief Check the [Mesh.Microstructure] table before any mesh work
 *
 * @param options Parsed options with options.mesh.microstructure set. options.restart.read
 *        selects the restart behavior described below; options.solvers.rtmodel gives the
 *        generator backend.
 * @return true if the table can be used; false after printing every problem on rank 0
 *
 * @details Called from ExaOptions::validate(), which separately refuses the grain, orientation
 * and region-mapping files that the hook replaces and checks the [Grain] tables. This function
 * checks the generator side:
 *  - exactly one form: `file = "..."` alone, or inline generator tables without `file`;
 *  - no `exec` key, and a generator [Mesh] table with only active_attributes /
 *    passthrough_attributes (the mesh-source keys type, floc, Auto are refused);
 *  - [Output]: mesh_format may only be "parallel" (the grain mesh is written as the mesh's own
 *    decomposition, readable by [Mesh] type = "parallel"), num_parts and grain_file are refused;
 *  - the options parse (MicrostructureOptions::from_toml) and, on fresh runs, pass
 *    MicrostructureOptions::validate(ValidationScope::GENERATOR_ONLY) (everything but the mesh
 *    source, which is ExaConstit's; it also checks that seed tables and voxel maps exist).
 * On a restart the hook does not run: the form checks and refusals still apply, but validate()
 * and its file checks are skipped, and a missing options file (form 1) only gives a note.
 *
 * Not collective (every rank checks its own identical copy, as ExaOptions::validate does). Host
 * only. Without the generator it always returns false with the "built without the
 * microstructure generator" message.
 */
bool ValidateMicrostructureHook(const ExaOptions& options);

} // namespace exaconstit
