#pragma once

#include "options/option_microstructure_hook.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <string>
#include <vector>

#if defined(EXACONSTIT_USE_MICROSTRUCTURE)
#include "microstructure/microstructure_options.hpp"
#endif

/**
 * @file microstructure_hook.hpp
 * @brief The [Mesh.Microstructure] hook: mechanics generates its own grains in memory.
 *
 * Normally a crystal-plasticity run reads three inputs prepared beforehand, for example with the
 * standalone microstructure_generator tool: the grain ID of every element (element attributes
 * of the mesh, or the grain_file of an auto mesh), one orientation per grain (the orientation
 * file) and, with more than one material, the grain-to-region map (region_mapping_file). With a
 * [Mesh.Microstructure] table in the input deck, mechanics instead runs the same generator
 * (exaconstit::microstructure::GenerateMicrostructure, the function the tool calls) on its own
 * final mesh, right after the mesh is built and refined, and keeps the result in memory:
 *  - each element's attribute becomes its grain ID (1..N);
 *  - every [Grain] table gets num_grains = N and ori_type = "quat";
 *  - the N unit quaternions (w x y z, crystal to sample, passive) are handed to the state
 *    variable initialization without a file;
 *  - the grain-to-region map is taken from the generator's region assignment.
 * The same generator options therefore give the same microstructure through the tool and
 * through the hook (on the same mesh). For the record, rank 0 still writes the orientation file,
 * the region map (more than one region) and the grain table, and optionally the grain mesh as a
 * parallel decomposition; nothing reads them back.
 *
 * Input deck forms (exactly one):
 * @code
 *   [Mesh.Microstructure]
 *       file = "micro.toml"      # a generator options file
 *   # or the generator's tables inline:
 *   [Mesh.Microstructure.Seeds]
 *       num_seeds = 200
 *   [Mesh.Microstructure.Output]
 *       basename = "grains"
 * @endcode
 * The mesh always comes from ExaConstit's own [Mesh] table, so the generator's mesh-source keys
 * (`type`, `floc`, [Mesh.Auto]) are refused; its [Mesh] table may only hold active_attributes and
 * passthrough_attributes. `exec` is refused too: the generator runs on the backend of
 * [Solvers] rtmodel, which the mechanics driver has already configured.
 *
 * Mesh resolution and element order (straight-sided meshes, e.g. every auto mesh):
 *  - the seeds (and so the underlying tessellation and the orientations, which are keyed by the
 *    seeds) do not depend on the refinement levels ref_ser / ref_par or on the element order
 *    p_refinement: the generator draws them in the material's bounding box, and Nodes that
 *    only re-describe straight elements are recognized as such;
 *  - at the same resolution, a straight mesh with order-p Nodes gives the labels of the same
 *    mesh without Nodes up to round-off (element centroids evaluated through the Nodes may
 *    break an exact tie differently); the generator works in place, without a mesh copy;
 *  - the element labels follow the mesh: each element takes the grain of its centroid (or of the
 *    quadrature-point vote), so a finer mesh resolves grain boundaries more sharply and can
 *    reveal small grains (or pieces kept by the fragment cleanup) that a coarse mesh missed.
 *    Grain counts and exact grain shapes therefore change with refinement, the polycrystal does
 *    not;
 *  - exception: element-weighted seeding (Seeds.placement = "element_weighted", or the automatic
 *    fallback the generator takes for very sparse domains) keys its random numbers on the
 *    elements, so its seeds change when the mesh is refined.
 * Genuinely curved meshes keep their curved geometry (the Nodes are never dropped).
 *
 * Restarts: a restart rebuilds the mesh, grain IDs and state variables (orientations included)
 * from the checkpoint, so the hook is skipped and only the deck is checked.
 *
 * Builds without the generator (ENABLE_MICROSTRUCTURE=OFF): the functions exist, but
 * ValidateMicrostructureHook rejects the table with "built without the microstructure generator"
 * and RunMicrostructureHook aborts.
 */

namespace exaconstit {

/**
 * @brief The grains a hook run hands to SimulationState (host data, identical on every rank)
 */
struct HookGrains {
    /** @brief Number of grains N; grain IDs are 1..N */
    int num_grains = 0;
    /** @brief Number of material regions R (equals the number of [[Materials]] blocks) */
    int num_regions = 0;
    /**
     * @brief 4 * N values: the unit quaternion of grain g at [4(g-1) .. 4(g-1)+3], order
     * w x y z (crystal to sample, passive), exactly as the orientation file would hold them
     */
    std::vector<double> grain_quat;
    /** @brief N values: region ID (1-based, selects the [[Materials]] block) of grain g at g-1 */
    std::vector<int> grain_region;
};


/**
 * @brief Generate the grains on the final mesh and inject them into the run (fresh runs only)
 *
 * @param options Simulation options. Every material's grain_info (when present) is updated:
 *        num_grains = N, ori_type = QUAT, ori_stride = 4.
 *        post_processing.volume_averages.output_directory / basename is the default directory of
 *        the record files.
 * @param pmesh The run's final mesh, after refinement and before the mesh nodes are created.
 *        On return every local element's attribute is its grain ID (1..N) and SetAttributes()
 *        has been called.
 * @return The grain count, quaternions and grain-to-region map, identical on every rank
 *
 * @details Collective over MPI_COMM_WORLD (the generator, the grain statistics and the optional
 * mesh and ParaView writers communicate). Runs on the host; the generator's kernels run on the
 * already configured mfem::Device. Steps:
 *  1. parse the options again (ExaOptions::validate already checked them) and fill in the
 *     backend and the record-file defaults;
 *  2. GenerateMicrostructure on pmesh in place, Nodes included and no mesh copy
 *     (straight-sided Nodes of any order are recognized by the generator and give the seeds of
 *     the same mesh without Nodes, with labels equal up to round-off; curved Nodes are used as
 *     they are);
 *  3. require as many regions as [[Materials]] blocks;
 *  4. grain statistics (element counts, volumes, centroids, neighbors) for the grain table;
 *  5. element attributes = grain IDs; [Grain] tables updated;
 *  6. record files and a rank-0 summary.
 * Aborts every rank (MFEM_VERIFY_0 / MFEM_ABORT) with the generator's message if generation
 * fails, if the region count does not match the materials, or if a record file cannot be
 * written.
 */
HookGrains RunMicrostructureHook(ExaOptions& options, mfem::ParMesh& pmesh);

#if defined(EXACONSTIT_USE_MICROSTRUCTURE)
/**
 * @brief The generator options of a [Mesh.Microstructure] table, plus which [Output] keys the
 * user wrote (the hook's defaults differ from the tool's for those)
 */
struct HookOptions {
    /** @brief Generator options, ready for GenerateMicrostructure (exec from rtmodel; the mesh
     * source keeps its defaults and is never used, see ParseMicrostructureHook) */
    exaconstit::microstructure::MicrostructureOptions generator;
    /** @brief Output.directory was given (otherwise ExaConstit's output directory is used) */
    bool output_directory_given = false;
    /** @brief Output.mesh_format was given (then it is "parallel"; otherwise no mesh file) */
    bool mesh_format_given = false;
    /** @brief Output.paraview was given (otherwise no ParaView files: mechanics has its own) */
    bool paraview_given = false;
    /** @brief Where the options came from, for the log: the options file or "inline" */
    std::string origin;
    /**
     * @brief The form-1 options file does not exist and ParseMicrostructureHook was told to
     * accept that (restarts); `generator` then holds defaults
     */
    bool options_file_missing = false;
};

/**
 * @brief Parse a [Mesh.Microstructure] table into generator options
 *
 * @param options ExaConstit options with options.mesh.microstructure set
 * @param problems Every problem found is appended (one message each); the returned options are
 *        only meaningful when no problem was added
 * @param allow_missing_file Accept a form-1 options file that does not exist (restarts, where
 *        the hook does not run): HookOptions::options_file_missing is set instead of a problem,
 *        after the form checks have run
 * @return The generator options with `exec` set from [Solvers] rtmodel and ParaView output off
 *         unless Output.paraview was given. The generator's mesh source (type, floc,
 *         [Mesh.Auto]) keeps its defaults: GenerateMicrostructure never reads it, and the hook
 *         validates with ValidationScope::GENERATOR_ONLY, which does not check it.
 *
 * @details Does not call MicrostructureOptions::validate() (ValidateMicrostructureHook does,
 * on fresh runs). Not collective, host only, never throws (exceptions from file reading and
 * parsing become problems). Exposed for the tests; ValidateMicrostructureHook and
 * RunMicrostructureHook are the normal entry points.
 */
HookOptions ParseMicrostructureHook(const ExaOptions& options,
                                    std::vector<std::string>& problems,
                                    bool allow_missing_file = false);
#endif

} // namespace exaconstit
