#include "sim_state/microstructure_hook.hpp"

#include "options/option_util.hpp"
#include "utilities/unified_logger.hpp"

#if defined(EXACONSTIT_USE_MICROSTRUCTURE)
#include "microstructure/io/output_writer.hpp"
#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/microstructure_report.hpp"
#include "microstructure/microstructure_types.hpp"

#include "ExaConstit_Version.h"
#endif

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
 * Implementation of the [Mesh.Microstructure] hook (see microstructure_hook.hpp for what it does
 * and where SimulationState calls it).
 *
 * Words used below:
 *  - "collective": an MPI operation every rank has to enter before any rank can leave it. The
 *    generator, the grain statistics and the parallel mesh writer are collective, so every rank
 *    must reach them, and an error found on identical data must stop all ranks together.
 *  - "record files": the orientation file, region map, grain table (and optional mesh) the hook
 *    writes so a run can be reproduced or inspected later. Nothing in the run reads them back.
 */

namespace exaconstit {

#if defined(EXACONSTIT_USE_MICROSTRUCTURE)

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

/** Prefix of every message about the hook, so users see which table a message refers to. */
constexpr const char* HOOK_PREFIX = "[Mesh.Microstructure]: ";

/*
 * ExecFromRtModel: the generator's `exec` spelling for ExaConstit's [Solvers] rtmodel. The
 * mechanics driver configures the mfem::Device from rtmodel (mechanics_driver.cpp, "cpu",
 * "raja-omp", "raja-cuda"/"raja-hip") before SimulationState is built, and the generator's
 * kernels follow whatever mfem::Device is configured (exec_dispatch.hpp). The generator's
 * validate() only checks that its `exec` names a backend this build has, so `exec` is set to
 * the backend actually in use.
 */
std::string ExecFromRtModel(const RTModel model) {
    if (model == RTModel::OPENMP) {
        return "openmp";
    }
    if (model == RTModel::GPU) {
        return "gpu";
    }
    return "cpu";
}

/*
 * CheckGeneratorTables: the refusals that apply to the generator options document itself
 * (either the options file of form 1 or the inline table of form 2). Each problem is appended
 * to `problems`; the [Output] keys the user wrote are recorded in `hook`.
 *  - `exec` is refused: the backend is ExaConstit's (see ExecFromRtModel).
 *  - [Mesh] may only hold active_attributes / passthrough_attributes: the mesh itself is
 *    ExaConstit's final mesh, so a mesh source (type, floc, [Mesh.Auto]) would be ignored and is
 *    refused to avoid silent surprises.
 *  - [Output]: only a parallel decomposition of the mesh can be written (each rank writes its
 *    own piece; a serial file would need the whole mesh on one rank), the number of parts is
 *    therefore the number of ranks (num_parts refused), and the per-element grain file of the
 *    tool is not written (grain_file refused; the grain IDs are the element attributes of the
 *    optional mesh output).
 */
void CheckGeneratorTables(const toml::value& doc,
                          HookOptions& hook,
                          std::vector<std::string>& problems) {
    if (doc.contains("exec")) {
        problems.push_back("`exec` is not allowed: the generator runs on the backend of "
                           "[Solvers] rtmodel");
    }

    if (doc.contains("Mesh")) {
        const toml::value& mesh = doc.at("Mesh");
        if (!mesh.is_table()) {
            problems.push_back("the generator's Mesh entry must be a table");
        } else {
            for (const auto& entry : mesh.as_table()) {
                const std::string& key = entry.first;
                if (key == "type" || key == "floc" || key == "Auto") {
                    problems.push_back("generator key Mesh." + key +
                                       " is not allowed: the mesh comes from ExaConstit's own "
                                       "[Mesh] table");
                } else if (key != "active_attributes" && key != "passthrough_attributes") {
                    problems.push_back("generator key Mesh." + key +
                                       " is not allowed: the generator's [Mesh] table may only "
                                       "hold active_attributes and passthrough_attributes");
                }
            }
        }
    }

    if (doc.contains("Output")) {
        const toml::value& output = doc.at("Output");
        if (output.is_table()) {
            hook.output_directory_given = output.contains("directory");
            hook.paraview_given = output.contains("paraview");
            if (output.contains("mesh_format")) {
                hook.mesh_format_given = true;
                const toml::value& format = output.at("mesh_format");
                if (!format.is_string() || format.as_string() != "parallel") {
                    problems.push_back("Output.mesh_format may only be \"parallel\" (each rank "
                                       "writes its own piece of the grain mesh); leave it out to "
                                       "write no mesh");
                }
            }
            if (output.contains("num_parts")) {
                problems.push_back("Output.num_parts is not allowed: the grain mesh is written "
                                   "with one part per MPI rank");
            }
            if (output.contains("grain_file")) {
                problems.push_back("Output.grain_file is not allowed: the hook keeps the grain "
                                   "IDs in memory (write them with Output.mesh_format = "
                                   "\"parallel\" if needed)");
            }
        }
        // A non-table Output entry is reported by MicrostructureOptions::from_toml
    }
}

/** "major.minor.patch" of ExaConstit, as the tool prints it. */
std::string VersionString() {
    std::ostringstream os;
    os << EXACONSTIT_VERSION_MAJOR << "." << EXACONSTIT_VERSION_MINOR << "."
       << EXACONSTIT_VERSION_PATCH;
    return os.str();
}

} // namespace

/*
 * ParseMicrostructureHook: turns the stored [Mesh.Microstructure] table into generator options.
 * Steps:
 *  1. Pick the form. Form 1 is a table whose only key is `file`, a generator options file (path
 *     relative to the working directory, like every other ExaConstit input file); the file is
 *     parsed with toml::parse. Form 2 is the table itself, holding the generator's tables
 *     directly. `file` next to any other key is both forms at once and refused.
 *  2. The refusals of CheckGeneratorTables.
 *  3. MicrostructureOptions::from_toml (it warns about unknown keys on rank 0 and throws for
 *     values it cannot store; the exception text becomes a problem).
 *  4. Fill in what the hook decides: `exec` from rtmodel, and no ParaView files unless the user
 *     asked for them (mechanics has its own visualization output). The generator's mesh source
 *     (type, floc, [Mesh.Auto]) stays at its defaults: it is never used, and the hook validates
 *     with ValidationScope::GENERATOR_ONLY, which does not check it.
 * Never throws: every failure is a problem message.
 */
HookOptions ParseMicrostructureHook(const ExaOptions& options,
                                    std::vector<std::string>& problems,
                                    bool allow_missing_file) {
    HookOptions hook;
    if (!options.mesh.microstructure) {
        problems.push_back("no [Mesh.Microstructure] table");
        return hook;
    }
    const toml::value& table = *options.mesh.microstructure;
    if (!table.is_table()) {
        problems.push_back("Mesh.Microstructure must be a table");
        return hook;
    }

    // Step 1: form 1 (file) or form 2 (inline tables)
    toml::value doc;
    try {
        if (table.contains("file")) {
            if (table.as_table().size() > 1) {
                problems.push_back("give either `file` (a generator options file) or inline "
                                   "generator tables, not both");
                return hook;
            }
            const fs::path file = toml::find<std::string>(table, "file");
            hook.origin = file.string();
            if (!fs::exists(file)) {
                if (allow_missing_file) {
                    hook.options_file_missing = true;
                } else {
                    problems.push_back("the generator options file '" + file.string() +
                                       "' does not exist");
                }
                return hook;
            }
            doc = toml::parse(file.string());
        } else {
            hook.origin = "inline";
            doc = table;
        }
    } catch (const std::exception& e) {
        problems.push_back(std::string("cannot read the generator options: ") + e.what());
        return hook;
    }

    // Step 2: keys the hook does not allow
    CheckGeneratorTables(doc, hook, problems);

    // Step 3: the generator's own parser
    try {
        hook.generator = ms::MicrostructureOptions::from_toml(doc);
    } catch (const std::exception& e) {
        problems.push_back(std::string("invalid generator options: ") + e.what());
        return hook;
    }

    // Step 4: what the hook fills in (the generator's mesh source is left alone, see above)
    ms::MicrostructureOptions& generator = hook.generator;
    generator.exec = ExecFromRtModel(options.solvers.rtmodel);
    if (!hook.paraview_given) {
        generator.output.paraview = false;
    }
    return hook;
}

bool MicrostructureHookAvailable() {
    return true;
}

/*
 * ValidateMicrostructureHook: the generator-side checks of ExaOptions::validate.
 *  - Both: ParseMicrostructureHook with its refusals (both forms together, refused keys), so a
 *    restart reports the same deck errors as a fresh run.
 *  - Restarts: the hook will not run, so a deleted options file (form 1) is only noted (after
 *    the form checks), and the generator's validate() (which checks that seed tables and voxel
 *    maps exist) is skipped.
 *  - Fresh runs: then MicrostructureOptions::validate(ValidationScope::GENERATOR_ONLY), which
 *    prints its own problems on rank 0 and does not check the mesh source (the mesh is
 *    ExaConstit's).
 */
bool ValidateMicrostructureHook(const ExaOptions& options) {
    const bool restart_run = options.restart.read;
    if (!options.mesh.microstructure) {
        return true;
    }

    std::vector<std::string> problems;
    HookOptions hook = ParseMicrostructureHook(options, problems, restart_run);
    for (const std::string& problem : problems) {
        WARNING_0_OPT("Error: " + std::string(HOOK_PREFIX) + problem);
    }
    if (!problems.empty()) {
        return false;
    }
    if (hook.options_file_missing) {
        INFO_0_OPT(std::string(HOOK_PREFIX) + "restart run: the generator options file '" +
                   hook.origin + "' is not read (the grains come from the checkpoint)");
        return true;
    }
    if (restart_run) {
        return true;
    }
    if (!hook.generator.validate(nullptr, ms::ValidationScope::GENERATOR_ONLY)) {
        WARNING_0_OPT("Error: " + std::string(HOOK_PREFIX) +
                      "the generator options are invalid (see the messages above)");
        return false;
    }
    return true;
}

/*
 * RunMicrostructureHook: see the header for the contract. The steps below follow the numbering
 * there. Error handling: the generator throws on every rank together (its checks are made on
 * identical data or globally reduced), and so does the region-count check, so those stop all
 * ranks with MFEM_VERIFY_0 (every rank aborts, rank 0 prints). The record files are written by
 * rank 0 alone, so a write failure aborts from rank 0 with MFEM_ABORT, which ends the whole MPI
 * job (mfem_error -> MPI_Abort).
 */
HookGrains RunMicrostructureHook(ExaOptions& options, mfem::ParMesh& pmesh) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool root = (rank == 0);

    // ---- 1. Options (already checked by ExaOptions::validate) -------------------------------
    std::vector<std::string> problems;
    HookOptions hook = ParseMicrostructureHook(options, problems);
    {
        std::ostringstream joined;
        for (const std::string& problem : problems) {
            joined << "\n  " << problem;
        }
        MFEM_VERIFY_0(problems.empty(), HOOK_PREFIX << "invalid options:" << joined.str());
    }
    ms::MicrostructureOptions& generator = hook.generator;
    // Record files default to the run's own output directory, <output_directory>/<basename>,
    // where the post-processing results go (postprocessing_file_manager.hpp)
    if (!hook.output_directory_given) {
        fs::path output_dir = options.post_processing.volume_averages.output_directory;
        if (output_dir.empty()) {
            output_dir = ".";
        }
        generator.output.directory = output_dir / options.basename;
    }

    // ---- 2. Generate on the final mesh ------------------------------------------------------
    // The mesh is passed as it is, Nodes included, and is not copied. makeMesh always gives it
    // order-p Nodes; when they describe straight-sided elements (any order) the generator
    // recognizes that and draws the same seeds as on the same mesh without Nodes (element
    // volumes and centroids may differ in the last bits, so labels agree up to round-off);
    // genuinely curved meshes are labeled through their Nodes (GenerateMicrostructure step 2a,
    // microstructure/mesh/straight_sided.hpp).
    ms::GenerationReport report;
    ms::MicrostructureResult result;
    {
        bool ok = true;
        std::string error;
        try {
            result = ms::GenerateMicrostructure(generator, pmesh, MPI_COMM_WORLD, &report);
        } catch (const std::exception& e) {
            ok = false;
            error = e.what();
        }
        MFEM_VERIFY_0(ok, HOOK_PREFIX << "the microstructure generator failed: " << error);
    }
    const int num_grains = result.num_grains;
    const int num_regions = result.num_regions;

    // ---- 3. One region per [[Materials]] block -----------------------------------------------
    // Region r selects [[Materials]] block r (input order), so the counts must agree. The
    // result is identical on every rank, so every rank takes the same branch.
    const int num_materials = static_cast<int>(options.materials.size());
    if (num_regions != num_materials) {
        std::ostringstream err;
        err << HOOK_PREFIX << "the generator produced " << num_regions << " material region(s), "
            << "but the input deck has " << num_materials << " [[Materials]] block(s); they "
            << "must match (region r uses [[Materials]] block r). Regions:";
        for (int r = 1; r <= num_regions; ++r) {
            err << "\n  region " << r << ": " << ms::RegionContents(report, result, r);
        }
        MFEM_VERIFY_0(false, err.str());
    }

    // ---- 4. Grain statistics for the grain table (collective) --------------------------------
    // Over the face adjacency the generator already built on pmesh (GenerationReport::
    // adjacency), as the tool does. The sums over elements are fixed-point sums: each value is
    // rounded to an integer multiple of a common small unit and the integers are added, so the
    // total is exact and does not depend on the order of addition, i.e. on the rank count.
    // Computed before the attributes change, while the adjacency matches the mesh as labeled.
    ms::GrainStatistics stats;
    {
        bool ok = (report.adjacency != nullptr);
        std::string error = "the generator report carries no mesh adjacency";
        if (ok) {
            try {
                stats = ms::ComputeGrainStatistics(*report.adjacency, result.element_grain,
                                                   num_grains, 0);
            } catch (const std::exception& e) {
                ok = false;
                error = e.what();
            }
        }
        MFEM_VERIFY_0(ok, HOOK_PREFIX << "grain statistics failed: " << error);
    }

    // ---- 5. Element attributes = grain IDs; [Grain] tables updated ---------------------------
    // element_grain is indexed by local element (0 .. GetNE()-1) and may live on the device, so
    // read its host copy. SetAttributes() then rebuilds the global attribute list (collective).
    {
        const int* grain = result.element_grain.HostRead();
        for (int i = 0; i < pmesh.GetNE(); ++i) {
            pmesh.SetAttribute(i, grain[i]);
        }
        pmesh.SetAttributes();
    }
    // Every crystal-plasticity material shares the one grain set: N grains, unit quaternions.
    // These replace the num_grains = 0 (or absent) and ori_type = "quat" (or absent) of the deck
    // before SimulationState counts the orientation state variables.
    for (auto& material : options.materials) {
        if (material.grain_info) {
            material.grain_info->num_grains = num_grains;
            material.grain_info->ori_type = OriType::QUAT;
            material.grain_info->ori_stride = 4;
        }
    }

    // ---- 6. Record files and summary ---------------------------------------------------------
    const ms::OutputOptions& out = generator.output;
    const fs::path base = out.directory / out.basename;
    const std::string orientation_path = base.string() + ".ori";
    const std::string region_path = base.string() + ".regions";
    const std::string table_path = base.string() + "_grains.csv";
    const std::string adjacency_path = base.string() + "_adjacency.csv";
    if (root) {
        try {
            fs::create_directories(out.directory);
            ms::WriteOrientationFile(orientation_path, result.grain_quat);
            if (num_regions > 1) {
                ms::WriteRegionMap(region_path, result.grain_region);
            }
            ms::WriteGrainTable(table_path, ms::BuildGrainTable(result, report, stats));
            if (out.grain_adjacency) {
                ms::WriteGrainAdjacency(adjacency_path, ms::BuildAdjacencyTable(stats));
            }
        } catch (const std::exception& e) {
            MFEM_ABORT(HOOK_PREFIX << "cannot write the record files in " << out.directory
                                   << ": " << e.what());
        }
    }
    // The optional collective writers below need the directory rank 0 just created
    MPI_Barrier(MPI_COMM_WORLD);

    // Optional grain mesh: each rank writes its piece with ParMesh::ParPrint (a decomposition
    // with one part per rank, readable by [Mesh] type = "parallel" on the same rank count)
    std::string mesh_prefix;
    if (hook.mesh_format_given) {
        mesh_prefix = base.string() + ".mesh.";
        std::ostringstream comments;
        comments << "Generated in memory by ExaConstit mechanics ([Mesh.Microstructure]), "
                 << "ExaConstit version " << VersionString() << "\n"
                 << "generator options: " << hook.origin << "\n"
                 << "rng_seed: " << generator.seeds.rng_seed << "\n"
                 << "grains: " << ms::RequestedGrains(report) << "\n"
                 << "final grains: " << num_grains << "\n"
                 << "regions: " << num_regions << "\n";
        bool ok = true;
        std::string error;
        try {
            ms::WriteParallelMesh(pmesh, mesh_prefix, out.compression, comments.str());
        } catch (const std::exception& e) {
            ok = false;
            error = e.what();
        }
        MFEM_VERIFY_0(ok, HOOK_PREFIX << "cannot write the grain mesh: " << error);
    }
    // Optional ParaView view of the grains (and regions when there are several)
    std::string paraview_path;
    if (out.paraview) {
        paraview_path = base.string() + "_paraview";
        std::vector<int> regions_for_view;
        if (num_regions > 1) {
            regions_for_view = result.grain_region;
        }
        bool ok = true;
        std::string error;
        try {
            ms::WriteParaView(pmesh, result.element_grain, regions_for_view, paraview_path);
        } catch (const std::exception& e) {
            ok = false;
            error = e.what();
        }
        MFEM_VERIFY_0(ok, HOOK_PREFIX << "cannot write the ParaView files: " << error);
    }

    if (root) {
        std::cout << "\n" << HOOK_PREFIX << "generated the grains in memory (generator options: "
                  << hook.origin << ")\n"
                  << "  grains: " << ms::RequestedGrains(report) << ", " << num_grains
                  << " final grains on " << report.num_elements << " elements\n"
                  << "  rng_seed: " << generator.seeds.rng_seed << "\n"
                  << "  regions: " << num_regions << "\n";
        for (int r = 1; r <= num_regions; ++r) {
            std::cout << "    region " << r << " ([[Materials]] block '"
                      << options.materials[static_cast<std::size_t>(r - 1)].material_name
                      << "'): " << ms::RegionContents(report, result, r) << "\n";
        }
        for (const std::string& warning : report.warnings) {
            std::cout << "  warning: " << warning << "\n";
        }
        // Region r uses [[Materials]] block r; flag the two combinations that are allowed but
        // probably not intended
        for (int r = 1; r <= num_regions; ++r) {
            const auto ri = static_cast<std::size_t>(r - 1);
            const bool crystalline =
                ri < result.region_crystalline.size() && result.region_crystalline[ri] != 0;
            const MaterialOptions& material = options.materials[ri];
            if (crystalline && !material.grain_info) {
                std::cout << "  warning: region " << r << " holds crystalline grains, but "
                          << "[[Materials]] block '" << material.material_name
                          << "' has no [Grain] table, so their orientations are not used\n";
            } else if (!crystalline && material.grain_info) {
                std::cout << "  warning: region " << r << " is not crystalline, but "
                          << "[[Materials]] block '" << material.material_name
                          << "' has a [Grain] table: its grains get the identity orientation\n";
            }
        }
        std::cout << "  record files (not read by this run):\n"
                  << "    orientations: " << orientation_path << "\n";
        if (num_regions > 1) {
            std::cout << "    region map: " << region_path << "\n";
        }
        std::cout << "    grain table: " << table_path << "\n";
        if (out.grain_adjacency) {
            std::cout << "    grain adjacency: " << adjacency_path << "\n";
        }
        if (!mesh_prefix.empty()) {
            std::cout << "    grain mesh (one part per rank): " << mesh_prefix << "*\n";
        }
        if (!paraview_path.empty()) {
            std::cout << "    ParaView: " << paraview_path << "\n";
        }
        std::cout << std::endl;
    }

    HookGrains grains;
    grains.num_grains = num_grains;
    grains.num_regions = num_regions;
    grains.grain_quat = result.grain_quat;
    grains.grain_region = result.grain_region;
    return grains;
}

#else // !EXACONSTIT_USE_MICROSTRUCTURE

/*
 * Builds without the generator: the hook cannot run. ExaOptions::validate calls
 * ValidateMicrostructureHook only when the deck has a [Mesh.Microstructure] table, so the user
 * gets this message before any mesh work, and RunMicrostructureHook is never reached through the
 * normal driver (it aborts if called anyway).
 */
bool MicrostructureHookAvailable() {
    return false;
}

bool ValidateMicrostructureHook(const ExaOptions& options) {
    if (!options.mesh.microstructure) {
        return true;
    }
    WARNING_0_OPT("Error: the input deck has a [Mesh.Microstructure] table, but this ExaConstit "
                  "was built without the microstructure generator (configure with "
                  "-DENABLE_MICROSTRUCTURE=ON, which needs Axom), so it cannot generate grains. "
                  "Run the standalone microstructure_generator tool and read its files instead.");
    return false;
}

HookGrains RunMicrostructureHook(ExaOptions& /*options*/, mfem::ParMesh& /*pmesh*/) {
    MFEM_ABORT_0("[Mesh.Microstructure]: this ExaConstit was built without the microstructure "
                 "generator");
    return HookGrains{};
}

#endif // EXACONSTIT_USE_MICROSTRUCTURE

} // namespace exaconstit
