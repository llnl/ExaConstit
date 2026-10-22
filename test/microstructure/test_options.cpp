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
#include <vector>

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
    EXPECT_EQ(options.seeds.min_distance, 0.0);
    EXPECT_FALSE(options.seeds.use_sampling_box);
    EXPECT_FALSE(options.seeds.allow_fewer_seeds);
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

// The seed-spacing keys added for the seed generator: an absolute minimum distance (an integer
// is accepted for a real value), a sampling box given as one flat [lo corner, hi corner] array
// (so [0, 0.25, 0.5, 1, 0.75, 2] means x in [0, 1], y in [0.25, 0.75], z in [0.5, 2]), and the
// allow_fewer_seeds switch. A box with the wrong number of entries cannot be stored and throws.
TEST(MicrostructureOptions, SeedSpacingAndSamplingBox) {
    const auto options = FromString(R"(
[Seeds]
num_seeds = 10
min_distance = 2
sampling_box = [0, 0.25, 0.5, 1, 0.75, 2]
allow_fewer_seeds = true
)");
    EXPECT_EQ(options.seeds.min_distance, 2.0);
    EXPECT_TRUE(options.seeds.use_sampling_box);
    EXPECT_EQ(options.seeds.sampling_box_lo, (std::array<double, 3>{0.0, 0.25, 0.5}));
    EXPECT_EQ(options.seeds.sampling_box_hi, (std::array<double, 3>{1.0, 0.75, 2.0}));
    EXPECT_TRUE(options.seeds.allow_fewer_seeds);
    EXPECT_THROW(FromString("[Seeds]\nsampling_box = [0, 0, 0, 1, 1]\n"), std::invalid_argument);
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

    // Seed spacing and sampling box: a negative absolute distance, and boxes with no extent
    // (hi == lo along x) or inverted (hi < lo along z), which could never receive a seed. A
    // proper box validates.
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.min_distance = -1.0; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_lo = {0.5, 0.0, 0.0};
        o.seeds.sampling_box_hi = {0.5, 1.0, 1.0};
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_lo = {0.0, 0.0, 1.0};
        o.seeds.sampling_box_hi = {1.0, 1.0, 0.0};
    }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_lo = {0.0, 0.0, 0.0};
        o.seeds.sampling_box_hi = {0.5, 0.5, 0.5};
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

// validate(&problems) also hands every error message to the caller (on every rank, while the
// printed warnings appear on rank 0 only), so a non-root rank whose checks fail alone can report
// them. Two independent problems (zero seeds, gzip level 10) give two messages; valid options
// give none, and the list is not touched by a later validate() without a list.
TEST(MicrostructureOptions, ValidationCollectsProblems) {
    std::vector<std::string> problems;
    EXPECT_TRUE(ValidAutoOptions().validate(&problems));
    EXPECT_TRUE(problems.empty());

    auto options = ValidAutoOptions();
    options.seeds.num_seeds = 0;
    options.output.compression = 10;
    EXPECT_FALSE(options.validate(&problems));
    EXPECT_EQ(problems.size(), 2u);
    EXPECT_FALSE(options.validate());
    EXPECT_EQ(problems.size(), 2u);
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
    EXPECT_FALSE(validate_with([nan](O& o) { o.seeds.min_distance = nan; }));
    EXPECT_FALSE(validate_with([inf](O& o) {
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_hi = {inf, 1.0, 1.0};
    }));
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

// ---------------------------------------------------------------------------------------------
// Stage 1b/1c options (fragment cleanup, quadrature voting, periodicity, anisotropy kinds, volume
// fitting, Lloyd relaxation, size distributions, Johnson-Mehl nucleation)
// ---------------------------------------------------------------------------------------------

// The new keys must default to the documented "off" values, so an options file written before
// they existed means exactly what it meant then (apart from fragments = "auto", whose split
// behavior is now active).
TEST(MicrostructureOptions, Stage1bcDefaults) {
    const auto options = FromString("[Seeds]\nnum_seeds = 5\n");
    EXPECT_EQ(options.cleanup.keep_component, "largest");
    EXPECT_EQ(options.cleanup.min_fragment_elements, 8);
    EXPECT_EQ(options.cleanup.min_fragment_fraction, 0.1);
    EXPECT_EQ(options.tessellation.vote_levels, 2);
    EXPECT_EQ(options.tessellation.anisotropy_kind, "none");
    EXPECT_EQ(options.tessellation.anisotropy_axis, (std::array<double, 3>{0.0, 0.0, 1.0}));
    EXPECT_EQ(options.tessellation.anisotropy_normal, (std::array<double, 3>{0.0, 0.0, 1.0}));
    EXPECT_FALSE(options.tessellation.fit_volumes.enabled);
    EXPECT_EQ(options.tessellation.fit_volumes.tolerance, 0.02);
    EXPECT_EQ(options.tessellation.fit_volumes.max_iterations, 200);
    EXPECT_EQ(options.tessellation.fit_volumes.samples, "quadrature");
    EXPECT_EQ(options.tessellation.lloyd_iterations, 0);
    EXPECT_FALSE(options.tessellation.centroidal_power);
    EXPECT_EQ(options.seeds.size.distribution, "none");
    EXPECT_EQ(options.seeds.size.mean_diameter, 0.0);
    EXPECT_EQ(options.seeds.size.mean_volume, 0.0);
    EXPECT_EQ(options.seeds.size.sigma, 0.0);
    EXPECT_FALSE(options.seeds.size.polydisperse_placement);
    EXPECT_EQ(options.seeds.size.exclusion_factor, 0.5);
    EXPECT_EQ(options.seeds.johnson_mehl.rate, 0.0);
    EXPECT_EQ(options.seeds.johnson_mehl.growth_speed, 1.0);
    EXPECT_EQ(options.seeds.johnson_mehl.end_time, 0.0);
    EXPECT_EQ(options.seeds.johnson_mehl.num_nuclei, 0);
}

// Every new key is read into its field, including integers for real values (rate = 60) and the
// inline-table forms of anisotropy and fit_volumes.
TEST(MicrostructureOptions, Stage1bcKeysParse) {
    const auto options = FromString(R"(
[Seeds]
placement = "johnson_mehl"
[Seeds.Size]
distribution = "lognormal"
mean_volume = 0.002
sigma = 0.25
polydisperse_placement = true
exclusion_factor = 0.4
[Seeds.JohnsonMehl]
rate = 60
growth_speed = 2.5
end_time = 0.75
num_nuclei = 40
[Tessellation]
metric = "quadric"
sampling = "quadrature_vote"
vote_levels = 3
anisotropy = {kind = "lamellar", normal = [1, 1, 0], axis = [0, 1, 0]}
fit_volumes = {enabled = true, tolerance = 0.01, max_iterations = 50, samples = "element"}
lloyd_iterations = 4
centroidal_power = true
[Cleanup]
fragments = "merge"
keep_component = "seed"
min_grain_elements = 3
min_fragment_elements = 5
min_fragment_fraction = 0
)");
    EXPECT_EQ(options.seeds.placement, ms::SeedPlacement::JOHNSON_MEHL);
    EXPECT_EQ(options.seeds.size.distribution, "lognormal");
    EXPECT_EQ(options.seeds.size.mean_volume, 0.002);
    EXPECT_EQ(options.seeds.size.sigma, 0.25);
    EXPECT_TRUE(options.seeds.size.polydisperse_placement);
    EXPECT_EQ(options.seeds.size.exclusion_factor, 0.4);
    EXPECT_EQ(options.seeds.johnson_mehl.rate, 60.0);
    EXPECT_EQ(options.seeds.johnson_mehl.growth_speed, 2.5);
    EXPECT_EQ(options.seeds.johnson_mehl.end_time, 0.75);
    EXPECT_EQ(options.seeds.johnson_mehl.num_nuclei, 40);
    EXPECT_EQ(options.tessellation.sampling, ms::SamplingMode::QUADRATURE_VOTE);
    EXPECT_EQ(options.tessellation.vote_levels, 3);
    EXPECT_EQ(options.tessellation.anisotropy_kind, "lamellar");
    EXPECT_EQ(options.tessellation.anisotropy_normal, (std::array<double, 3>{1.0, 1.0, 0.0}));
    EXPECT_EQ(options.tessellation.anisotropy_axis, (std::array<double, 3>{0.0, 1.0, 0.0}));
    EXPECT_TRUE(options.tessellation.fit_volumes.enabled);
    EXPECT_EQ(options.tessellation.fit_volumes.tolerance, 0.01);
    EXPECT_EQ(options.tessellation.fit_volumes.max_iterations, 50);
    EXPECT_EQ(options.tessellation.fit_volumes.samples, "element");
    EXPECT_EQ(options.tessellation.lloyd_iterations, 4);
    EXPECT_TRUE(options.tessellation.centroidal_power);
    EXPECT_EQ(options.cleanup.fragments, "merge");
    EXPECT_EQ(options.cleanup.keep_component, "seed");
    EXPECT_EQ(options.cleanup.min_grain_elements, 3);
    EXPECT_EQ(options.cleanup.min_fragment_elements, 5);
    EXPECT_EQ(options.cleanup.min_fragment_fraction, 0.0);
}

// A written global_tensor (inline or bare) means anisotropy kind "tensor" unless a kind is given
// explicitly, in which case the explicit kind wins (the tensor is then ignored by the labeler).
// Malformed new arrays (a 2-entry axis) and out-of-range integers throw while parsing.
TEST(MicrostructureOptions, AnisotropyKindPrecedence) {
    EXPECT_EQ(FromString("[Tessellation]\nglobal_tensor = [2, 1, 1, 0, 0, 0]\n")
                  .tessellation.anisotropy_kind,
              "tensor");
    EXPECT_EQ(FromString("[Tessellation]\nanisotropy = {global_tensor = [2, 1, 1, 0, 0, 0]}\n")
                  .tessellation.anisotropy_kind,
              "tensor");
    EXPECT_EQ(FromString("[Tessellation]\nanisotropy = {kind = \"columnar\"}\n"
                         "global_tensor = [2, 1, 1, 0, 0, 0]\n")
                  .tessellation.anisotropy_kind,
              "columnar");
    EXPECT_THROW(FromString("[Tessellation]\nanisotropy = {axis = [0, 1]}\n"),
                 std::invalid_argument);
    EXPECT_THROW(FromString("[Tessellation]\nvote_levels = 4294967298\n"), std::invalid_argument);
}

// The example files of the new features parse to the values they spell out and validate as
// written (all use auto meshes, so no input file is needed).
TEST(MicrostructureOptions, ParseStage1bcExamples) {
    const auto periodic = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/periodic_vote_auto.toml");
    EXPECT_EQ(periodic.tessellation.periodic, (std::array<bool, 3>{true, true, true}));
    EXPECT_EQ(periodic.tessellation.sampling, ms::SamplingMode::QUADRATURE_VOTE);
    EXPECT_EQ(periodic.tessellation.vote_levels, 2);
    EXPECT_EQ(periodic.cleanup.fragments, "split");
    EXPECT_TRUE(periodic.validate());

    const auto lognormal = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/lognormal_fit_auto.toml");
    EXPECT_EQ(lognormal.seeds.num_seeds, 0);
    EXPECT_EQ(lognormal.seeds.size.distribution, "lognormal");
    EXPECT_EQ(lognormal.seeds.size.mean_diameter, 0.3);
    EXPECT_EQ(lognormal.seeds.size.sigma, 0.3);
    EXPECT_TRUE(lognormal.seeds.size.polydisperse_placement);
    EXPECT_EQ(lognormal.tessellation.metric, ms::MetricType::LAGUERRE);
    EXPECT_TRUE(lognormal.tessellation.fit_volumes.enabled);
    EXPECT_EQ(lognormal.tessellation.fit_volumes.tolerance, 0.05);
    EXPECT_EQ(lognormal.cleanup.min_grain_elements, 2);
    EXPECT_TRUE(lognormal.validate());

    const auto columnar = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/columnar_auto.toml");
    EXPECT_EQ(columnar.tessellation.metric, ms::MetricType::QUADRIC);
    EXPECT_EQ(columnar.tessellation.anisotropy_kind, "columnar");
    EXPECT_EQ(columnar.tessellation.anisotropy_axis, (std::array<double, 3>{0.0, 0.0, 1.0}));
    EXPECT_TRUE(columnar.validate());

    const auto jm = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/johnson_mehl_auto.toml");
    EXPECT_EQ(jm.seeds.placement, ms::SeedPlacement::JOHNSON_MEHL);
    EXPECT_EQ(jm.seeds.johnson_mehl.rate, 60.0);
    EXPECT_EQ(jm.tessellation.metric, ms::MetricType::ADDITIVE);
    EXPECT_TRUE(jm.validate());

    const auto lloyd = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/lloyd_cvt_auto.toml");
    EXPECT_EQ(lloyd.tessellation.lloyd_iterations, 5);
    EXPECT_FALSE(lloyd.tessellation.fit_volumes.enabled);
    EXPECT_TRUE(lloyd.validate());

    // The pre-existing examples keep validating with their new comments
    const auto cube_tet = ms::MicrostructureOptions::parse_options_file(
        data_dir / "microstructure/voronoi_cube_tet.toml");
    EXPECT_EQ(cube_tet.cleanup.keep_component, "largest");
    EXPECT_EQ(cube_tet.cleanup.min_grain_elements, 1);
    EXPECT_EQ(cube_tet.cleanup.min_fragment_elements, 8);
    EXPECT_EQ(cube_tet.cleanup.min_fragment_fraction, 0.1);
}

// validate() must reject each impossible or contradictory combination of the new options. Each
// case changes the valid baseline (4 random Voronoi seeds on a 2^3 auto box) in one respect, so a
// false result can only come from that change; the valid variants show the checks are not too
// strict.
TEST(MicrostructureOptions, Stage1bcValidationFailures) {
    auto validate_with = [](auto&& mutate) {
        auto options = ValidAutoOptions();
        mutate(options);
        return options.validate();
    };
    using O = ms::MicrostructureOptions;

    // Spellings and ranges
    EXPECT_FALSE(validate_with([](O& o) { o.cleanup.keep_component = "first"; }));
    // What "auto" calls a tiny piece: a count >= 0 and a fraction in [0, 1] (NaN rejected)
    EXPECT_FALSE(validate_with([](O& o) { o.cleanup.min_fragment_elements = -1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.cleanup.min_fragment_fraction = 1.5; }));
    EXPECT_FALSE(validate_with([](O& o) { o.cleanup.min_fragment_fraction = -0.1; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.cleanup.min_fragment_fraction = std::numeric_limits<double>::quiet_NaN();
    }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.cleanup.min_fragment_elements = 0;
        o.cleanup.min_fragment_fraction = 1.0;
    }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.vote_levels = 4; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.vote_levels = -1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.anisotropy_kind = "fibrous"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.fit_volumes.samples = "nodes"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.fit_volumes.tolerance = 0.0; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.fit_volumes.max_iterations = -1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.lloyd_iterations = -2; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.size.distribution = "gamma"; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.size.sigma = -0.1; }));
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.size.mean_diameter = -1.0; }));

    // Columnar / lamellar need the Quadric metric and a nonzero direction
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.anisotropy_kind = "columnar"; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::QUADRIC;
        o.tessellation.anisotropy_kind = "lamellar";
        o.tessellation.anisotropy_normal = {0.0, 0.0, 0.0};
    }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::QUADRIC;
        o.tessellation.anisotropy_kind = "columnar";
        o.tessellation.anisotropy_axis = {1.0, 2.0, 3.0};
        // a semidefinite stored tensor is fine: it is not used for columnar grains
        o.tessellation.global_tensor = {1, 1, 0, 0, 0, 0};
    }));

    // Fitting needs weights; Lloyd needs Voronoi or Laguerre; centroidal power needs Lloyd and
    // Laguerre; neither works with periodic images
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.fit_volumes.enabled = true; }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::LAGUERRE;
        o.tessellation.fit_volumes.enabled = true;
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::ADDITIVE;
        o.tessellation.lloyd_iterations = 3;
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.tessellation.lloyd_iterations = 3;
        o.tessellation.centroidal_power = true; // Voronoi baseline
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::LAGUERRE;
        o.tessellation.centroidal_power = true; // no Lloyd steps
    }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.tessellation.metric = ms::MetricType::LAGUERRE;
        o.tessellation.lloyd_iterations = 3;
        o.tessellation.centroidal_power = true;
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.tessellation.periodic = {true, false, false};
        o.tessellation.lloyd_iterations = 2;
    }));

    // Seed count: zero seeds are fine with a mean size, not with both means given
    EXPECT_TRUE(validate_with([](O& o) {
        o.seeds.num_seeds = 0;
        o.seeds.size.mean_diameter = 0.5;
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.num_seeds = 0;
        o.seeds.size.mean_diameter = 0.5;
        o.seeds.size.mean_volume = 0.1;
    }));
    // from_source needs a seed table
    EXPECT_FALSE(validate_with([](O& o) { o.seeds.size.distribution = "from_source"; }));

    // Polydisperse placement: feasible exclusion factor (kappa^3 <= 0.3), a random placement,
    // and no competing fixed minimum distance
    EXPECT_TRUE(validate_with([](O& o) { o.seeds.size.polydisperse_placement = true; }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.size.polydisperse_placement = true;
        o.seeds.size.exclusion_factor = 0.7; // 0.343 > 0.3
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.size.polydisperse_placement = true;
        o.seeds.placement = ms::SeedPlacement::LATTICE;
    }));
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.size.polydisperse_placement = true;
        o.seeds.min_distance_factor = 0.5;
    }));

    // Johnson-Mehl: positive rate and speed, the Additive metric, and no size/fit/Lloyd
    auto jm = [](O& o) {
        o.seeds.num_seeds = 0;
        o.seeds.placement = ms::SeedPlacement::JOHNSON_MEHL;
        o.seeds.johnson_mehl.rate = 50.0;
        o.tessellation.metric = ms::MetricType::ADDITIVE;
    };
    EXPECT_TRUE(validate_with(jm));
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.seeds.johnson_mehl.rate = 0.0;
    }));
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.seeds.johnson_mehl.growth_speed = -1.0;
    }));
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.seeds.johnson_mehl.num_nuclei = -1;
    }));
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.tessellation.metric = ms::MetricType::VORONOI;
    }));
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.seeds.size.distribution = "lognormal";
    }));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.seeds.johnson_mehl.end_time = nan;
    }));
    // Johnson-Mehl coverage does not wrap around periodic faces
    EXPECT_FALSE(validate_with([&](O& o) {
        jm(o);
        o.tessellation.periodic = {false, true, false};
    }));

    // Orientations from the source need a seed table (generated seeds carry none)
    EXPECT_FALSE(validate_with([](O& o) { o.orientations.source = "from_source"; }));
    EXPECT_TRUE(validate_with([](O& o) {
        o.orientations.source = "from_source";
        o.seeds.num_seeds = 0;
        o.seeds.source = "file";
        o.seeds.placement = ms::SeedPlacement::FILE;
        o.seeds.file = data_dir / "grains.txt";
    }));

    // Polydisperse targets fill the whole material, so a sampling box is refused
    EXPECT_FALSE(validate_with([](O& o) {
        o.seeds.size.polydisperse_placement = true;
        o.seeds.use_sampling_box = true;
        o.seeds.sampling_box_lo = {0.0, 0.0, 0.0};
        o.seeds.sampling_box_hi = {0.5, 0.5, 0.5};
    }));

    // Periodic face matching tolerance in (0, 0.5)
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.periodic_tolerance = 0.0; }));
    EXPECT_FALSE(validate_with([](O& o) { o.tessellation.periodic_tolerance = 0.5; }));
    EXPECT_TRUE(validate_with([](O& o) { o.tessellation.periodic_tolerance = 1e-3; }));
    EXPECT_EQ(FromString("[Tessellation]\nperiodic_tolerance = 1e-4\n")
                  .tessellation.periodic_tolerance,
              1e-4);
    EXPECT_EQ(ms::MicrostructureOptions{}.tessellation.periodic_tolerance, 1e-6);
}

// [Output] grain_adjacency: off by default, so existing runs write no extra file;
// `grain_adjacency = true` switches the adjacency CSV on; both settings validate (a boolean has
// no invalid value); a value of another TOML type cannot be stored and throws while parsing, like
// the other wrongly typed keys in ParseErrors. A file without the key keeps the default.
TEST(MicrostructureOptions, GrainAdjacencyOutput) {
    EXPECT_FALSE(ms::MicrostructureOptions{}.output.grain_adjacency);
    EXPECT_FALSE(FromString("[Output]\nbasename = \"a\"\n").output.grain_adjacency);
    EXPECT_TRUE(FromString("[Output]\ngrain_adjacency = true\n").output.grain_adjacency);
    EXPECT_FALSE(FromString("[Output]\ngrain_adjacency = false\n").output.grain_adjacency);
    EXPECT_THROW(FromString("[Output]\ngrain_adjacency = \"yes\"\n"), std::exception);
    EXPECT_THROW(FromString("[Output]\ngrain_adjacency = 1\n"), std::exception);
    // The periodic example switches it on; the tet example writes the default explicitly
    EXPECT_TRUE(ms::MicrostructureOptions::parse_options_file(
                    data_dir / "microstructure/periodic_vote_auto.toml")
                    .output.grain_adjacency);
    EXPECT_FALSE(ms::MicrostructureOptions::parse_options_file(
                     data_dir / "microstructure/voronoi_cube_tet.toml")
                     .output.grain_adjacency);

    ms::MicrostructureOptions options = ValidAutoOptions();
    options.output.grain_adjacency = true;
    EXPECT_TRUE(options.validate());
    options.output.grain_adjacency = false;
    EXPECT_TRUE(options.validate());
}

// MPI must be initialized because the options code prints warnings through WARNING_0_OPT, which
// asks MPI for the rank.
int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
