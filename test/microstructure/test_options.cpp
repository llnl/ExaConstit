/**
 * @file test_options.cpp
 * @brief Tests of the microstructure generator's options: TOML parsing, defaults, validation and
 * command-line overrides (src/microstructure/microstructure_options.cpp).
 *
 * The options pipeline is: parse the TOML file -> apply command-line flags -> validate(). These
 * tests check each stage on its own:
 *   - the two example files in test/data/microstructure parse to the values they spell out;
 *   - omitted keys keep the documented defaults;
 *   - every accepted spelling and alternative form is understood;
 *   - aliases and duplicate spellings resolve as documented;
 *   - unknown keys only warn, while values that cannot be stored (wrong length, integer out of
 *     range) throw;
 *   - validate() rejects each kind of inconsistent or impossible input, including NaN/inf and
 *     an exec backend this build does not have;
 *   - command-line flags override the file as documented.
 *
 * Runs as a normal MPI program (registered at np 1 in test_options.cmake, also run by hand at
 * np 2). Nothing here communicates; MPI is needed only because warnings are printed on rank 0.
 */
#include "microstructure/exec_dispatch.hpp"
#include "microstructure/microstructure_options.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

// test/data, set by test_options.cmake so the tests do not depend on the working directory.
const fs::path data_dir = fs::path(MS_TEST_DATA_DIR);

/** Parses TOML text held in a string (no file needed), exactly as from_toml sees a file. */
ms::MicrostructureOptions FromString(const std::string& text) {
    std::istringstream stream(text);
    return ms::MicrostructureOptions::from_toml(toml::parse(stream, "test_options.toml"));
}

/**
 * Minimal options that validate: a 2x2x2 auto hex mesh (no file needed on disk) with 4 random
 * seeds. ValidationFailures changes one field at a time from this baseline, so each failure can
 * only come from that field.
 */
ms::MicrostructureOptions ValidAutoOptions() {
    ms::MicrostructureOptions options;
    options.mesh.type = "auto";
    options.mesh.nxyz = {2, 2, 2};
    options.seeds.num_seeds = 4;
    return options;
}

} // namespace

// The commented example voronoi_auto.toml must parse to exactly the values written in it, and
// must validate as is (an auto mesh needs no input file). Also checks the ExaConstit-style
// [Mesh.Auto] spellings `length`/`ncuts` land in mxyz/nxyz.
TEST(MicrostructureOptions, ParseVoronoiAuto) {
    const auto options = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/voronoi_auto.toml");
    EXPECT_EQ(options.exec, "cpu");
    EXPECT_TRUE(options.deterministic);
    EXPECT_EQ(options.mesh.type, "auto");
    EXPECT_EQ(options.mesh.nxyz, (std::array<int, 3>{8, 8, 8}));
    EXPECT_EQ(options.mesh.mxyz, (std::array<double, 3>{1.0, 1.0, 1.0}));
    EXPECT_EQ(options.seeds.source, "random");
    EXPECT_EQ(options.seeds.num_seeds, 8);
    EXPECT_EQ(options.seeds.rng_seed, 0u);
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::REJECTION);
    EXPECT_EQ(options.tessellation.metric, ms::MetricType::VORONOI);
    EXPECT_EQ(options.tessellation.sampling, ms::SamplingMode::CENTROID);
    EXPECT_EQ(options.orientations.source, "random");
    EXPECT_EQ(options.output.basename, "voronoi_auto");
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::SERIAL);
    EXPECT_EQ(options.output.grain_file, "auto");
    EXPECT_TRUE(options.validate());
}

// The commented example voronoi_cube_tet.toml (file mesh, serial + 4-part parallel output) must
// parse to the values written in it, and validate once its relative mesh path is resolved.
TEST(MicrostructureOptions, ParseVoronoiCubeTet) {
    auto options = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/voronoi_cube_tet.toml");
    EXPECT_EQ(options.mesh.type, "file");
    EXPECT_EQ(options.mesh.floc, fs::path("cube_tet.mesh"));
    EXPECT_EQ(options.seeds.num_seeds, 20);
    EXPECT_EQ(options.seeds.rng_seed, 42u);
    EXPECT_EQ(options.seeds.sampling_domain, "material");
    EXPECT_EQ(options.tessellation.periodic, (std::array<bool, 3>{false, false, false}));
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::BOTH);
    EXPECT_EQ(options.output.num_parts, 4);

    // floc is relative to the working directory, which is not test/data here
    options.mesh.floc = data_dir / options.mesh.floc;
    EXPECT_TRUE(options.validate());
}

// Every key that is not written must keep the default documented in microstructure_options.hpp
// (the "-n N" quick path: Voronoi, rejection-sampled random seeds, gzip level 6 serial output).
// The literal values here are the documented defaults, so a changed default fails this test.
TEST(MicrostructureOptions, DefaultsWhenKeysOmitted) {
    const auto options = FromString("[Seeds]\nnum_seeds = 5\n");
    const ms::MicrostructureOptions defaults;
    EXPECT_EQ(options.seeds.num_seeds, 5);
    EXPECT_EQ(options.exec, defaults.exec);
    EXPECT_EQ(options.deterministic, defaults.deterministic);
    EXPECT_EQ(options.mesh.type, "file");
    EXPECT_EQ(options.seeds.source, "random");
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::REJECTION);
    EXPECT_EQ(options.seeds.min_distance_factor, 0.0);
    EXPECT_EQ(options.seeds.units_scale, 1.0);
    EXPECT_EQ(options.tessellation.metric, ms::MetricType::VORONOI);
    EXPECT_EQ(options.tessellation.global_tensor, defaults.tessellation.global_tensor);
    EXPECT_EQ(options.tessellation.num_bins, 8);
    EXPECT_EQ(options.tessellation.distance, ms::DistanceMode::EUCLIDEAN);
    EXPECT_EQ(options.cleanup.fragments, "auto");
    EXPECT_EQ(options.cleanup.min_grain_elements, 1);
    EXPECT_EQ(options.regions.mode, "single");
    EXPECT_EQ(options.output.directory, fs::path("."));
    EXPECT_EQ(options.output.basename, "microstructure");
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::SERIAL);
    EXPECT_EQ(options.output.num_parts, 0);
    EXPECT_EQ(options.output.compression, 6);
    EXPECT_EQ(options.output.grain_file, "auto");
    EXPECT_TRUE(options.output.paraview);

    // An empty document is all defaults
    const auto empty = FromString("");
    EXPECT_EQ(empty.seeds.num_seeds, 0);
    EXPECT_EQ(empty.output.basename, "microstructure");
}

// Every non-default enum spelling and every alternative input form is understood: integers for
// real-valued keys (units_scale = 2), the inline-table form of the anisotropy tensor, a boolean
// grain_file, and file placement implied by source = "file". The tensor order is
// [Axx, Ayy, Azz, Ayz, Axz, Axy], so 0.5 is the xy component.
TEST(MicrostructureOptions, ParsesAllEnumsAndForms) {
    const auto options = FromString(R"(
exec = "openmp"
deterministic = false
[Mesh]
type = "parallel"
floc = "parts/mesh."
[Source]
type = "tessellation"
[Seeds]
source = "file"
file = "seeds.csv"
units_scale = 2
translate = [1, 0.5, 0]
[Tessellation]
metric = "quadric"
anisotropy = {global_tensor = [2, 1, 1, 0, 0, 0.5]}
sampling = "quadrature_vote"
distance = "geodesic"
periodic = [true, false, true]
num_bins = 4
[Output]
mesh_format = "parallel"
grain_file = true
paraview = false
)");
    EXPECT_EQ(options.exec, "openmp");
    EXPECT_FALSE(options.deterministic);
    EXPECT_EQ(options.mesh.type, "parallel");
    EXPECT_EQ(options.seeds.source, "file");
    // A seed table implies file placement
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::FILE);
    EXPECT_EQ(options.seeds.units_scale, 2.0);
    EXPECT_EQ(options.seeds.translate, (std::array<double, 3>{1.0, 0.5, 0.0}));
    EXPECT_EQ(options.tessellation.metric, ms::MetricType::QUADRIC);
    EXPECT_EQ(options.tessellation.global_tensor,
              (std::array<double, 6>{2.0, 1.0, 1.0, 0.0, 0.0, 0.5}));
    EXPECT_EQ(options.tessellation.sampling, ms::SamplingMode::QUADRATURE_VOTE);
    EXPECT_EQ(options.tessellation.distance, ms::DistanceMode::GEODESIC);
    EXPECT_EQ(options.tessellation.periodic, (std::array<bool, 3>{true, false, true}));
    EXPECT_EQ(options.tessellation.num_bins, 4);
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::PARALLEL);
    EXPECT_EQ(options.output.grain_file, "true");
    EXPECT_FALSE(options.output.paraview);

    const auto laguerre = FromString(
        "[Seeds]\nplacement = \"johnson_mehl\"\n[Tessellation]\nmetric = \"laguerre\"\n");
    EXPECT_EQ(laguerre.seeds.placement, ms::SeedPlacement::JOHNSON_MEHL);
    EXPECT_EQ(laguerre.tessellation.metric, ms::MetricType::LAGUERRE);
}

// A typo (num_sedes) and a table of a later stage ([Conformal]) must only warn on rank 0 and
// must not change the values that were spelled correctly.
TEST(MicrostructureOptions, UnknownKeysAreIgnored) {
    const auto options = FromString(R"(
[Seeds]
num_seeds = 3
num_sedes = 7
[Conformal]
enabled = true
)");
    EXPECT_EQ(options.seeds.num_seeds, 3);
}

// Values that cannot be stored at all must throw while parsing, rather than silently falling back
// to a default: unknown enum names ("power" metric, "poisson" placement, "vtk" format), a
// negative RNG key, a 2-component translation, an unsupported [Source] type (voxel maps would be
// tessellated by mistake), a string where an integer is expected, and a missing options file.
TEST(MicrostructureOptions, ParseErrors) {
    EXPECT_THROW(FromString("[Tessellation]\nmetric = \"power\"\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Seeds]\nplacement = \"poisson\"\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Output]\nmesh_format = \"vtk\"\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Seeds]\nrng_seed = -1\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Seeds]\ntranslate = [1.0, 2.0]\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Source]\ntype = \"voxel_map\"\n"), std::invalid_argument);
    // Wrong TOML types surface as toml11 errors (std::exception)
    EXPECT_THROW(FromString("[Seeds]\nnum_seeds = \"many\"\n"), std::exception);
    EXPECT_THROW(ms::MicrostructureOptions::parse_options_file(data_dir / "does_not_exist.toml"),
                 std::runtime_error);
}

// validate() must reject each kind of impossible or inconsistent input. Each case changes one
// field of the valid baseline, so a false result can only come from that field: unknown string
// values, zero seeds or elements, missing mesh/seed files, a source/placement mismatch, an
// out-of-range output setting, and anisotropy tensors that are not positive definite
// ({1, 1, -1, ...} has a negative diagonal; {1, 1, 1, 0, 0, 2} has xx*yy - xy^2 = 1 - 4 < 0).
TEST(MicrostructureOptions, ValidationFailures) {
    EXPECT_TRUE(ValidAutoOptions().validate());

    // Applies `mutate` to a fresh copy of the baseline and returns validate()'s result.
    auto validate_with = [](auto&& mutate) {
        auto options = ValidAutoOptions();
        mutate(options);
        return options.validate();
    };
    using O = ms::MicrostructureOptions;

    // Unknown spellings of string-valued options (each is a typo of a real choice).
    EXPECT_FALSE(validate_with([](O& o) { o.exec = "seq"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.mesh.type = "cubit"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.source = "dream3d"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.sampling_domain = "box"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.lattice_type = "diamond"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.cleanup.fragments = "drop"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.orientations.source = "file"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.regions.mode = "multi"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.output.grain_file = "yes"; }));

    // Nothing to label: zero random seeds, or an auto box with zero elements along y.
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.num_seeds = 0; }));
    EXPECT_FALSE(validate_with([](O& o) { o.mesh.nxyz[1] = 0; }));

    // Mesh inputs that cannot be opened: a missing file, a file mesh without floc, and a parallel
    // prefix whose part 0 ("no_such_prefix.000000") does not exist.
    EXPECT_FALSE(validate_with([](O& o) {
        o.mesh.type = "file";
        o.mesh.floc = "no_such_mesh.mesh";
    }));
    EXPECT_FALSE(validate_with([](O& o) { o.mesh.type = "file"; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.mesh.type = "parallel";
        o.mesh.floc = "no_such_prefix.";
    }));

    // Output settings out of range: negative part count, gzip levels outside 0-9, no basename.
    EXPECT_FALSE(validate_with([](O& o) { o.output.num_parts = -1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.output.compression = 10; }));
    EXPECT_FALSE(validate_with([](O& o) { o.output.compression = -1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.output.basename.clear(); }));

    // Seed source and placement must agree: file placement with random seeds, and a seed table
    // that does not exist.
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.placement = ms::SeedPlacement::FILE; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.source = "file";
        o.seeds.placement = ms::SeedPlacement::FILE;
        o.seeds.file = "no_such_seeds.csv";
    }));

    // Tessellation: no seed group, and the two non-positive-definite tensors described above.
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.num_bins = 0; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.global_tensor = {1, 1, -1, 0, 0, 0}; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.global_tensor = {1, 1, 1, 0, 0, 2}; }));

    // A seed table that exists is fine without num_seeds (the grain count is the table's row
    // count). Only existence is checked here, so any file in test/data serves as a stand-in.
    EXPECT_TRUE(validate_with([](O& o) {
        o.seeds.num_seeds = 0;
        o.seeds.source = "file";
        o.seeds.placement = ms::SeedPlacement::FILE;
        o.seeds.file = data_dir / "grains.txt";
    }));
}

// TOML 1.0 accepts the floats nan and inf. They must not slip through validate(): NaN makes every
// "x <= 0" test false, so a plain sign check would accept it, and inf passes "x > 0". Each case
// puts one non-finite value into an otherwise valid baseline.
TEST(MicrostructureOptions, NonFiniteValuesFailValidation) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    auto validate_with = [](auto&& mutate) {
        auto options = ValidAutoOptions();
        mutate(options);
        return options.validate();
    };
    using O = ms::MicrostructureOptions;

    // Box size of the auto mesh.
    EXPECT_FALSE(validate_with([nan](O& o) { o.mesh.mxyz[0] = nan; }));
    EXPECT_FALSE(validate_with([inf](O& o) { o.mesh.mxyz[2] = inf; }));
    // Seed spacing, lattice jitter, and the seed-table transform.
    EXPECT_FALSE(validate_with([nan](O& o) { o.seeds.min_distance_factor = nan; }));
    EXPECT_FALSE(validate_with([inf](O& o) { o.seeds.boundary_clearance = inf; }));
    EXPECT_FALSE(validate_with([nan](O& o) { o.seeds.lattice_jitter = nan; }));
    EXPECT_FALSE(validate_with([nan](O& o) { o.seeds.units_scale = nan; }));
    EXPECT_FALSE(validate_with([inf](O& o) { o.seeds.units_scale = inf; }));
    EXPECT_FALSE(validate_with([inf](O& o) { o.seeds.translate[1] = -inf; }));
    // Anisotropy tensor: inf on the diagonal would otherwise pass all three minor tests
    // (inf > 0), and NaN off the diagonal makes the minors NaN.
    EXPECT_FALSE(validate_with([inf](O& o) { o.tessellation.global_tensor[0] = inf; }));
    EXPECT_FALSE(validate_with([nan](O& o) { o.tessellation.global_tensor[5] = nan; }));

    // The same path from an actual TOML file: `nan` parses as a float and is then rejected.
    auto from_file = FromString(R"(
[Mesh]
type = "auto"
[Mesh.Auto]
length = [nan, 1, 1]
ncuts = [2, 2, 2]
[Seeds]
num_seeds = 4
)");
    EXPECT_TRUE(std::isnan(from_file.mesh.mxyz[0]));
    EXPECT_FALSE(from_file.validate());
}

// exec must name a backend this build has. "cpu" always exists. "openmp" and "gpu" exist only in
// builds with that backend (EXEC_OPENMP_AVAILABLE / EXEC_GPU_AVAILABLE, exec_dispatch.hpp), so
// the expected result follows those flags. On the local CPU-only build both must fail.
TEST(MicrostructureOptions, ExecBackendAvailability) {
    auto options = ValidAutoOptions();
    options.exec = "cpu";
    EXPECT_TRUE(options.validate());

    options.exec = "openmp";
    EXPECT_EQ(options.validate(), ms::EXEC_OPENMP_AVAILABLE);

    options.exec = "gpu";
    EXPECT_EQ(options.validate(), ms::EXEC_GPU_AVAILABLE);
}

// Alternative spellings and precedence: [Mesh.Auto] accepts ExaConstit's mxyz/nxyz as aliases of
// length/ncuts. When both are written the canonical name wins (with a rank-0 warning), and the
// bare [Tessellation] global_tensor wins over the inline anisotropy form (also with a warning).
TEST(MicrostructureOptions, AliasesAndPrecedence) {
    // Aliases alone are read into the same fields.
    const auto aliases = FromString(R"(
[Mesh.Auto]
mxyz = [2, 3, 4]
nxyz = [5, 6, 7]
)");
    EXPECT_EQ(aliases.mesh.mxyz, (std::array<double, 3>{2.0, 3.0, 4.0}));
    EXPECT_EQ(aliases.mesh.nxyz, (std::array<int, 3>{5, 6, 7}));

    // Both written: length/ncuts are used, mxyz/nxyz ignored.
    const auto both = FromString(R"(
[Mesh.Auto]
length = [1, 1, 1]
mxyz = [9, 9, 9]
ncuts = [2, 2, 2]
nxyz = [9, 9, 9]
)");
    EXPECT_EQ(both.mesh.mxyz, (std::array<double, 3>{1.0, 1.0, 1.0}));
    EXPECT_EQ(both.mesh.nxyz, (std::array<int, 3>{2, 2, 2}));

    // Bare global_tensor is read after the inline form, so it is the one kept.
    const auto tensors = FromString(R"(
[Tessellation]
anisotropy = {global_tensor = [2, 2, 2, 0, 0, 0]}
global_tensor = [3, 3, 3, 0, 0, 0]
)");
    EXPECT_EQ(tensors.tessellation.global_tensor,
              (std::array<double, 6>{3.0, 3.0, 3.0, 0.0, 0.0, 0.0}));
}

// Shape and range errors that must throw while parsing: a 2-entry ncuts or periodic (one
// direction would be missing), and integers that do not fit in an int. toml11 would otherwise
// cast 4294967302 = 2^32 + 6 to 6 without complaint, so a typo would become a valid-looking value.
TEST(MicrostructureOptions, ShapeAndRangeErrors) {
    EXPECT_THROW(FromString("[Mesh.Auto]\nncuts = [2, 2]\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Tessellation]\nperiodic = [true, false]\n"),
                 std::invalid_argument);
    EXPECT_THROW(FromString("[Output]\ncompression = 4294967302\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Output]\nnum_parts = 3000000000\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Tessellation]\nnum_bins = -3000000000\n"), std::invalid_argument);
    EXPECT_THROW(FromString("[Cleanup]\nmin_grain_elements = 2147483648\n"),
                 std::invalid_argument);
    EXPECT_THROW(FromString("[Mesh.Auto]\nncuts = [2, 4294967298, 2]\n"), std::invalid_argument);
    // The largest int is still accepted.
    EXPECT_EQ(FromString("[Output]\nnum_parts = 2147483647\n").output.num_parts, 2147483647);
}

// Command-line flags (-m -n -s -o --parts) must replace the file values: -m switches to a file
// mesh, -n switches a seed table back to random rejection seeds, --parts turns serial output into
// serial + parallel. Unset flags must change nothing, and an explicit parallel-only format must
// survive --parts.
TEST(MicrostructureOptions, CliOverrides) {
    auto options = FromString(R"(
[Mesh]
type = "auto"
[Seeds]
source = "file"
file = "seeds.csv"
[Output]
basename = "from_file"
)");
    ms::CliOverrides overrides;
    overrides.mesh = data_dir / "cube_tet.mesh";
    overrides.num_seeds = 50;
    overrides.rng_seed = 7;
    overrides.basename = "cli";
    overrides.num_parts = 4;
    options.apply_cli_overrides(overrides);

    EXPECT_EQ(options.mesh.type, "file");
    EXPECT_EQ(options.mesh.floc, data_dir / "cube_tet.mesh");
    EXPECT_EQ(options.seeds.num_seeds, 50);
    // -n is the random quick path
    EXPECT_EQ(options.seeds.source, "random");
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::REJECTION);
    EXPECT_EQ(options.seeds.rng_seed, 7u);
    EXPECT_EQ(options.output.basename, "cli");
    EXPECT_EQ(options.output.num_parts, 4);
    // --parts adds the decomposition and keeps the serial file
    EXPECT_EQ(options.output.mesh_format, ms::MeshFormat::BOTH);
    EXPECT_TRUE(options.validate());

    // Unset overrides leave everything alone
    const auto before = options;
    options.apply_cli_overrides(ms::CliOverrides{});
    EXPECT_EQ(options.mesh.floc, before.mesh.floc);
    EXPECT_EQ(options.seeds.num_seeds, before.seeds.num_seeds);
    EXPECT_EQ(options.output.basename, before.output.basename);
    EXPECT_EQ(options.output.mesh_format, before.output.mesh_format);

    // An explicit parallel-only format is kept
    ms::MicrostructureOptions parallel_only;
    parallel_only.output.mesh_format = ms::MeshFormat::PARALLEL;
    ms::CliOverrides parts;
    parts.num_parts = 2;
    parallel_only.apply_cli_overrides(parts);
    EXPECT_EQ(parallel_only.output.mesh_format, ms::MeshFormat::PARALLEL);

    // -n on a file that already asks for random seeds only changes the count: an explicit
    // placement such as "lattice" is the user's choice and is kept.
    auto lattice = FromString("[Seeds]\nsource = \"random\"\nplacement = \"lattice\"\n");
    ms::CliOverrides count_only;
    count_only.num_seeds = 12;
    lattice.apply_cli_overrides(count_only);
    EXPECT_EQ(lattice.seeds.num_seeds, 12);
    EXPECT_EQ(lattice.seeds.placement, ms::SeedPlacement::LATTICE);
}

// MPI must be initialized because the options code prints warnings through WARNING_0_OPT, which
// asks MPI for the rank.
int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
