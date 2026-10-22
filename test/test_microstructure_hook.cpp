/*
 * test_microstructure_hook: the [Mesh.Microstructure] hook of the mechanics driver
 * (src/sim_state/microstructure_hook.*, the hook call in the SimulationState constructor and the
 * hook checks in ExaOptions::validate).
 *
 * What is checked:
 *  1. Parsing: the two forms of the table (a generator options file, or the generator's tables
 *     inline) give the same generator options, and the hook fills in the backend from
 *     [Solvers] rtmodel.
 *  2. Refusals: every input the hook replaces or cannot honor makes ExaOptions::validate()
 *     return false (both forms together, generator mesh-source keys, exec, unsupported [Output]
 *     keys, grain/orientation/region-mapping files, a [Grain] num_grains or non-quat ori_type,
 *     invalid generator options, a missing options file). A restart accepts a deleted options
 *     file, because the hook does not run then. A build without the generator rejects the table
 *     with "built without the microstructure generator"; that build cannot run this test (it is
 *     only registered with ENABLE_MICROSTRUCTURE), so the message is checked by hand: configure
 *     with -DENABLE_MICROSTRUCTURE=OFF, build mechanics and run it on any deck with the table.
 *  3. SimulationState: a fresh run on an 8x8x8 auto mesh with 8 grains. The element grain IDs and
 *     the orientations placed in the state variables must equal, bit for bit, what a direct
 *     GenerateMicrostructure call returns on the same mesh with the same options, and every
 *     [Grain] table must now hold the generated grain count.
 *
 * Run with 1 or 2 MPI ranks (registered for np 1 and 2 behind EXACONSTIT_ENABLE_EXTENSIVE_TESTS).
 * Scratch files go to hook_unit_np<ranks>/ in the working directory.
 */
#include "sim_state/microstructure_hook.hpp"
#include "sim_state/simulation_state.hpp"
#include "utilities/auto_mesh.hpp"

#include "microstructure/microstructure_driver.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

/** @brief Directory with the test input files (test/data in the source tree) */
const fs::path kDataDir = EXACONSTIT_TEST_DATA_DIR;

/** @brief Number of MPI ranks of this run */
int NumRanks() {
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

/** @brief Rank of this process */
int Rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

/** @brief Scratch directory of this rank count, created on first use (all ranks) */
fs::path ScratchDir() {
    const fs::path dir = fs::current_path() / ("hook_unit_np" + std::to_string(NumRanks()));
    if (Rank() == 0) {
        fs::create_directories(dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return dir;
}

/**
 * @brief The generator tables used throughout: 8 random Voronoi seeds (rng_seed 0) on the auto
 * box, the same as test/data/microstructure/voronoi_auto.toml without its [Mesh] table
 */
const char* kGeneratorTables = R"(
[Seeds]
    num_seeds = 8
    rng_seed = 0
[Tessellation]
    metric = "voronoi"
[Output]
    basename = "hook_unit"
)";

/**
 * @brief An ExaConstit deck: one ExaCMech crystal-plasticity material on an 8x8x8 auto box
 *
 * @param mesh_extra Text appended inside [Mesh] (the hook table goes here)
 * @param grain_extra Text appended inside [Materials.Grain]
 * @param top_extra Text placed before the first table (top-level keys)
 * @param restart_read Whether the deck restarts from a checkpoint ([Restart] read = true)
 * @return The deck as TOML text. Properties and state variables come from test/data.
 */
std::string MakeDeck(const std::string& mesh_extra,
                     const std::string& grain_extra = "",
                     const std::string& top_extra = "",
                     const bool restart_read = false) {
    std::ostringstream deck;
    deck << "Version = \"0.10.0\"\n"
         << "basename = \"hook_unit\"\n"
         << top_extra << "\n"
         << "[[Materials]]\n"
         << "    name = \"polycrystal\"\n"
         << "    mech_type = \"exacmech\"\n"
         << "    temperature = 298.0\n"
         << "    [Materials.Properties]\n"
         << "        floc = \"" << (kDataDir / "props_cp_voce.txt").string() << "\"\n"
         << "        num_props = 17\n"
         << "    [Materials.State_Vars]\n"
         << "        floc = \"" << (kDataDir / "state_cp_voce.txt").string() << "\"\n"
         << "        num_vars = 24\n"
         << "    [Materials.Grain]\n"
         << "        ori_state_var_loc = 9\n"
         << "        ori_stride = 4\n"
         << grain_extra << "\n"
         << "    [Materials.Model]\n"
         << "        crystal_plasticity = true\n"
         << "        [Materials.Model.ExaCMech]\n"
         << "            shortcut = \"evptn_FCC_A\"\n"
         << "[BCs]\n"
         << "    [[BCs.velocity_bcs]]\n"
         << "        essential_ids = [1, 2, 3]\n"
         << "        essential_comps = [3, 1, 2]\n"
         << "        essential_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
         << "    [[BCs.velocity_gradient_bcs]]\n"
         << "        essential_ids = [4]\n"
         << "        essential_comps = [3]\n"
         << "        velocity_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.001]]\n"
         << "        origin = [0.0, 0.0, 0.0]\n"
         << "    [BCs.time_info]\n"
         << "        cycle_dependent = true\n"
         << "        cycles = [1]\n"
         << "[Time]\n"
         << "    [Time.Fixed]\n"
         << "        dt = 0.1\n"
         << "        t_final = 0.5\n"
         << "[PostProcessing]\n"
         << "    [PostProcessing.volume_averages]\n"
         << "        enabled = true\n"
         << "        output_directory = \"" << (ScratchDir() / "results").string() << "\"\n"
         << "[Solvers]\n"
         << "    assembly = \"FULL\"\n"
         << "    rtmodel = \"CPU\"\n";
    if (restart_read) {
        // RestartOptions::validate() requires the checkpoint directory to exist; its contents
        // are only looked at when a SimulationState is built, which these decks never are
        const fs::path restart_dir = ScratchDir() / "restart";
        if (Rank() == 0) {
            fs::create_directories(restart_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        deck << "[Restart]\n"
             << "    read = true\n"
             << "    directory = \"" << restart_dir.string() << "\"\n";
    }
    deck << "[Mesh]\n"
         << "    type = \"auto\"\n"
         << "    p_refinement = 1\n"
         << "    [Mesh.Auto]\n"
         << "        length = [1.0, 1.0, 1.0]\n"
         << "        ncuts = [8, 8, 8]\n"
         << mesh_extra << "\n";
    return deck.str();
}

/** @brief The inline form: kGeneratorTables with every table moved under [Mesh.Microstructure] */
std::string InlineHook(const std::string& tables = kGeneratorTables) {
    std::string text = tables;
    // "[Seeds]" -> "[Mesh.Microstructure.Seeds]", and so on for every table header
    std::string out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '[') {
            line = "[Mesh.Microstructure." + line.substr(1);
        }
        out += line + "\n";
    }
    return out;
}

/** @brief Writes a generator options file (form 1) into the scratch directory (rank 0) */
fs::path WriteOptionsFile(const std::string& name, const std::string& text) {
    const fs::path path = ScratchDir() / name;
    if (Rank() == 0) {
        std::ofstream(path) << text;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return path;
}

/** @brief The form-1 hook table pointing at `file` */
std::string FileHook(const fs::path& file) {
    return "    [Mesh.Microstructure]\n        file = \"" + file.string() + "\"\n";
}

/** @brief Parses a deck into ExaOptions (no validation, no output-directory setup) */
ExaOptions ParseDeck(const std::string& deck) {
    std::istringstream stream(deck);
    const toml::value toml_input = toml::parse(stream, "hook_unit_deck");
    ExaOptions options;
    options.parse_from_toml(toml_input);
    return options;
}

/** @brief Parses and validates a deck; true when ExaOptions::validate() accepts it */
bool DeckIsValid(const std::string& deck) {
    ExaOptions options = ParseDeck(deck);
    return options.validate();
}

} // namespace

// Both forms must describe the same microstructure: the inline tables and an options file with
// the same text give identical generator options, and the backend is taken from rtmodel ("CPU"
// -> "cpu"). The generator's mesh source is not touched (the mesh is ExaConstit's), so the
// options pass validate() only with ValidationScope::GENERATOR_ONLY.
TEST(microstructure_hook, both_forms_parse_to_the_same_options) {
    const fs::path file = WriteOptionsFile("form1.toml", kGeneratorTables);

    std::vector<std::string> problems_file;
    const ExaOptions options_file = ParseDeck(MakeDeck(FileHook(file)));
    ASSERT_TRUE(options_file.mesh.microstructure.has_value());
    const exaconstit::HookOptions from_file =
        exaconstit::ParseMicrostructureHook(options_file, problems_file);
    EXPECT_TRUE(problems_file.empty());
    EXPECT_EQ(from_file.origin, file.string());

    std::vector<std::string> problems_inline;
    const ExaOptions options_inline = ParseDeck(MakeDeck(InlineHook()));
    const exaconstit::HookOptions from_inline =
        exaconstit::ParseMicrostructureHook(options_inline, problems_inline);
    EXPECT_TRUE(problems_inline.empty());
    EXPECT_EQ(from_inline.origin, "inline");

    for (const exaconstit::HookOptions* hook : {&from_file, &from_inline}) {
        EXPECT_EQ(hook->generator.seeds.num_seeds, 8);
        EXPECT_EQ(hook->generator.seeds.rng_seed, 0u);
        EXPECT_EQ(hook->generator.tessellation.metric, ms::MetricType::VORONOI);
        EXPECT_EQ(hook->generator.output.basename, "hook_unit");
        EXPECT_EQ(hook->generator.exec, "cpu");
        // ParaView is off unless asked for (mechanics writes its own visualization)
        EXPECT_FALSE(hook->generator.output.paraview);
        EXPECT_FALSE(hook->output_directory_given);
        EXPECT_FALSE(hook->mesh_format_given);
        EXPECT_TRUE(hook->generator.validate(nullptr, ms::ValidationScope::GENERATOR_ONLY));
    }

    // Both complete decks pass ExaConstit's validation: no num_grains, no orientation file
    EXPECT_TRUE(DeckIsValid(MakeDeck(FileHook(file))));
    EXPECT_TRUE(DeckIsValid(MakeDeck(InlineHook())));
    // An explicit num_grains = 0 and ori_type = "quat" are what the hook sets anyway
    EXPECT_TRUE(DeckIsValid(MakeDeck(InlineHook(), "num_grains = 0\nori_type = \"quat\"")));
}

// Every input the hook replaces, or cannot honor, is refused before any mesh work. Each deck
// differs from a valid one in exactly one place, so a false result can only come from that place.
TEST(microstructure_hook, refusals) {
    const fs::path file = WriteOptionsFile("form1_refusals.toml", kGeneratorTables);
    const std::string seeds = "[Seeds]\n    num_seeds = 8\n";

    // Form 1 and form 2 together
    EXPECT_FALSE(DeckIsValid(MakeDeck(FileHook(file) + "        exec = \"cpu\"\n")));
    EXPECT_FALSE(DeckIsValid(MakeDeck(FileHook(file) + InlineHook(seeds))));
    // The mesh comes from ExaConstit: no generator mesh source, no other [Mesh] keys
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh]\n    type = \"auto\"\n"))));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh]\n    floc = \"a.mesh\"\n"))));
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh.Auto]\n    ncuts = [8, 8, 8]\n"))));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh]\n    order = 2\n"))));
    // ... while the attribute keys are allowed
    EXPECT_TRUE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh]\n    active_attributes = [1]\n"))));
    // The same refusals apply inside an options file
    const fs::path bad_file =
        WriteOptionsFile("form1_bad.toml", seeds + "[Mesh]\n    type = \"auto\"\n");
    EXPECT_FALSE(DeckIsValid(MakeDeck(FileHook(bad_file))));
    // The backend is rtmodel's
    EXPECT_FALSE(DeckIsValid(
        MakeDeck("    [Mesh.Microstructure]\n        exec = \"cpu\"\n" + InlineHook(seeds))));
    // Output: only a parallel grain mesh, one part per rank, no per-element grain file
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Output]\n    mesh_format = \"serial\"\n"))));
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Output]\n    mesh_format = \"both\"\n"))));
    EXPECT_TRUE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Output]\n    mesh_format = \"parallel\"\n"))));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Output]\n    num_parts = 2\n"))));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Output]\n    grain_file = true\n"))));
    // The files the hook replaces, in the [Grain] table or at the top level. The files need not
    // exist: the refusal comes first.
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "orientation_file = \"a.ori\"")));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "grain_file = \"a.grains\"")));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "", "orientation_file = \"a.ori\"")));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "", "grain_file = \"a.grains\"")));
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(InlineHook(seeds), "", "region_mapping_file = \"a.regions\"")));
    // The hook sets num_grains and ori_type
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "num_grains = 8")));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds), "ori_type = \"euler\"")));
    // The generator's own validation runs on fresh runs (num_seeds must be positive)
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook("[Seeds]\n    num_seeds = 0\n"))));
    // A generator key that cannot be stored (unknown metric) is a parse error
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(InlineHook(seeds + "[Tessellation]\n    metric = \"nope\"\n"))));
    // A missing options file: refused on a fresh run, only noted on a restart (the hook does not
    // run, and a restart reads none of the files a fresh run reads)
    const fs::path missing = ScratchDir() / "does_not_exist.toml";
    EXPECT_FALSE(DeckIsValid(MakeDeck(FileHook(missing))));
    EXPECT_TRUE(DeckIsValid(MakeDeck(FileHook(missing), "", "", true)));
    // The form checks run on restarts too, even when the options file is gone
    EXPECT_FALSE(
        DeckIsValid(MakeDeck(FileHook(missing) + "        exec = \"cpu\"\n", "", "", true)));
    // A restart skips the generator's validate() (its file checks), but still parses the table
    EXPECT_TRUE(DeckIsValid(MakeDeck(InlineHook("[Seeds]\n    num_seeds = 0\n"), "", "", true)));
    EXPECT_FALSE(DeckIsValid(MakeDeck(InlineHook(seeds + "[Mesh]\n    type = \"auto\"\n"), "", "",
                                      true)));
    // The build has the generator
    EXPECT_TRUE(exaconstit::MicrostructureHookAvailable());
}

// A fresh SimulationState with the hook must hold exactly the microstructure a direct
// GenerateMicrostructure call produces on the same mesh: same grain ID on every local element
// and, in every quadrature point's state variables (components 9..12, ori_state_var_loc = 9),
// the bit-identical unit quaternion of that element's grain. The comparison mesh is built the way
// makeMesh builds an auto mesh (MakeAutoHexMesh, ExaConstit boundary attributes, METIS partition
// through ParMesh(comm, mesh), order-1 Nodes from SetCurvature(1)), so both see the same mesh,
// the same Nodes and the same partition, and the generator, which works in place on the mesh it
// is given, must give bit-identical results.
TEST(microstructure_hook, simulation_state_matches_direct_generation) {
    ExaOptions options = ParseDeck(MakeDeck(InlineHook()));
    ASSERT_TRUE(options.validate());
    SimulationState sim(options);

    // Direct generation with the same parsed options on an identically built mesh
    std::vector<std::string> problems;
    exaconstit::HookOptions hook = exaconstit::ParseMicrostructureHook(options, problems);
    ASSERT_TRUE(problems.empty());
    mfem::Mesh serial = exaconstit::MakeAutoHexMesh({8, 8, 8}, {1.0, 1.0, 1.0});
    exaconstit::ApplyAutoMeshBoundaryAttributes(serial);
    serial.SetCurvature(1);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, serial);
    pmesh.SetAttributes();
    const ms::MicrostructureResult direct =
        ms::GenerateMicrostructure(hook.generator, pmesh, MPI_COMM_WORLD);
    EXPECT_EQ(direct.num_grains, 8);
    EXPECT_EQ(direct.num_regions, 1);

    // The [Grain] table now carries the generated grain count
    ASSERT_TRUE(options.materials[0].grain_info.has_value());
    EXPECT_EQ(options.materials[0].grain_info->num_grains, direct.num_grains);

    // Grain IDs element by element (local element order is the same on both meshes)
    ASSERT_EQ(sim.GetMesh()->GetNE(), pmesh.GetNE());
    const double* grains = sim.GetGrains()->HostRead();
    const int* direct_grain = direct.element_grain.HostRead();
    int wrong_grains = 0;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        wrong_grains += (static_cast<int>(grains[e]) == direct_grain[e]) ? 0 : 1;
        // After construction the element attributes are region IDs (one region here)
        EXPECT_EQ(sim.GetMesh()->GetAttribute(e), 1);
    }
    EXPECT_EQ(wrong_grains, 0);

    // Orientations in the state variables of region 0 (24 model variables + 4 quaternion
    // components = 28 per point; the quaternion occupies components 9..12)
    auto state = sim.GetQuadratureFunction("state_var_beg", 0);
    ASSERT_TRUE(state != nullptr);
    const int vdim = state->GetVDim();
    ASSERT_EQ(vdim, 28);
    auto space = state->GetPartialSpaceShared();
    const auto& local_to_global = space->GetLocal2Global();
    const int num_elements = space->GetNumLocalElements();
    int wrong_quats = 0;
    if (num_elements > 0) {
        const int points_per_element = state->Size() / vdim / num_elements;
        const double* data = state->HostRead();
        for (int element = 0; element < num_elements; ++element) {
            const int grain = direct_grain[local_to_global[element]];
            const double* q = &direct.grain_quat[static_cast<std::size_t>(4 * (grain - 1))];
            for (int point = 0; point < points_per_element; ++point) {
                const double* s = data +
                                  static_cast<std::size_t>(element * points_per_element + point) *
                                      vdim;
                const bool same = s[9] == q[0] && s[10] == q[1] && s[11] == q[2] && s[12] == q[3];
                wrong_quats += same ? 0 : 1;
            }
        }
    }
    EXPECT_EQ(wrong_quats, 0);

    // Record files written by rank 0 into <output_directory>/<basename>
    const fs::path record = ScratchDir() / "results" / "hook_unit";
    if (Rank() == 0) {
        EXPECT_TRUE(fs::exists(record / "hook_unit.ori"));
        EXPECT_TRUE(fs::exists(record / "hook_unit_grains.csv"));
        // One region: no region map
        EXPECT_FALSE(fs::exists(record / "hook_unit.regions"));
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    mfem::Hypre::Init();
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
