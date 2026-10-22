#pragma once

#include "mfem.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @file microstructure_types.hpp
 * @brief Shared data contracts of the microstructure generator.
 *
 * These enums and structs are shared by all modules (seeding, geometry, labeling, IO); changing a
 * field affects every one of them, so keep them stable.
 *
 * Vocabulary used throughout the generator:
 * - *seed*: a point s_i (plus optional weight/shape/orientation) from which one grain grows;
 * - *tessellation*: the partition of space that gives every point x to the seed minimizing a
 *   distance-like function f_i(x); each element of the mesh is then labeled with the grain of
 *   its sample point(s);
 * - *grain id*: compact, 1-based (1..num_grains) after labeling and cleanup; seed indices are
 *   0-based positions in SeedSet.
 */

namespace exaconstit::microstructure {

/**
 * @brief Distance metric family. Every metric has the form f_i(x) = g_i(x) - w_i and
 * grain(x) = argmin_i f_i, with ties going to the lowest seed id (so the label does not depend
 * on the order seeds are visited or on the rank count).
 */
enum class MetricType {
    VORONOI,  ///< g_i = |x - s_i|^2, w_i = 0: planar grain boundaries halfway between seeds
    LAGUERRE, ///< g_i = |x - s_i|^2 with weight w_i (power diagram): planar, cells can be empty
    QUADRIC,  ///< g_i = (x - s_i)^T A_i (x - s_i) with weight w_i: quadric (curved) boundaries,
              ///< elongated grains
    ADDITIVE  ///< g_i = |x - s_i| with weight a_i (Apollonius / Johnson-Mehl): hyperbolic
              ///< boundaries
};

/**
 * @brief Where each element is sampled when it is labeled.
 */
enum class SamplingMode {
    CENTROID,       ///< One sample at the true volume centroid of the element
    QUADRATURE_VOTE ///< Majority vote over the element quadrature points
};

/**
 * @brief How the distance between a point and a seed is measured.
 */
enum class DistanceMode {
    EUCLIDEAN, ///< Straight-line distance, even across holes or notches in the material
    CONNECTED, ///< Euclidean labels, then grain pieces cut off by holes are reassigned through
               ///< element adjacency in the material
    GEODESIC   ///< Shortest path that stays inside the material
};

/**
 * @brief How seed positions are generated.
 */
enum class SeedPlacement {
    REJECTION,        ///< Uniform candidates in the bounding box, kept if inside the material
    ELEMENT_WEIGHTED, ///< Uniform-in-volume: pick an element with probability ~ its volume, then
                      ///< a point inside it
    LATTICE,          ///< Regular lattice (cubic, BCC, FCC, HCP) with optional random jitter
    JOHNSON_MEHL,     ///< Space-time Poisson nucleation (seeds appear at random times and grow)
    FILE              ///< Read from a seed table
};

/**
 * @brief Layout of the output mesh files.
 */
enum class MeshFormat {
    SERIAL,   ///< One serial MFEM mesh file
    PARALLEL, ///< One file per part, named with mfem::MakeParFilename (e.g. "basename.000003")
    BOTH      ///< Both of the above
};

/**
 * @brief Host structure-of-arrays seed container, replicated (identical) on every rank.
 *
 * "Structure of arrays" means one std::vector per attribute (x, y, z, weight, ...) instead of a
 * vector of seed structs, so each attribute can be copied to the device as one contiguous array.
 * Seed i is entry i (0-based) of every per-seed array; multi-component attributes are stored
 * seed-major (all components of seed 0, then seed 1, ...). Optional arrays are empty when unused.
 * Positions are in mesh coordinates and length units (after any units_scale/translate of a seed
 * table). Orientations are stored here as supplied; the generator normalizes them to unit length
 * when it writes per-grain orientations (convention, sign and component order kept).
 */
struct SeedSet {
    /** @brief Seed positions, one entry per seed (size N), mesh length units. */
    std::vector<double> x, y, z;
    /**
     * @brief w_i (Laguerre/Quadric, length^2) or a_i (Additive, length), size N; empty means all
     * zero.
     */
    std::vector<double> weight;
    /**
     * @brief Per-seed Quadric shape tensor A_i, 6 entries per seed: shape[6*i + k] with
     * k = 0..5 -> Axx Ayy Azz Ayz Axz Axy (symmetric 3x3, diagonal first). Size 6N or empty.
     */
    std::vector<double> shape;
    /**
     * @brief Orientation quaternion per seed: quat[4*i + k], k = 0..3 -> w x y z, crystal-to-
     * sample, passive, as supplied (normalized to unit length when written per grain). Size 4N or
     * empty.
     */
    std::vector<double> quat;
    /**
     * @brief Stable id of each seed from its source, size N.
     * Seed tables: the row id. Generated seeds: the candidate index (rejection-sampling candidate
     * k, lattice point index, or position in the element-weighted arrival order), strictly
     * increasing in seed order, so it is stable but not necessarily 0..N-1: candidates rejected
     * by the material test, the minimum distance or the clearance leave gaps. Random orientations
     * are keyed by it, so a seed keeps its orientation when other candidates are rejected.
     */
    std::vector<std::int64_t> source_id;
    /** @brief Optional phase id per seed (size N or empty). */
    std::vector<int> phase;
    /** @brief Reserved for hierarchical (parent/child) grain labels, size N or empty. */
    std::vector<std::int64_t> parent_id;
    /** @brief Reserved for Johnson-Mehl nucleation times, size N or empty. */
    std::vector<double> nucleation_time;

    /** @brief Number of seeds N (the length of x). Host only. */
    std::size_t Size() const {
        return x.size();
    }
};

/**
 * @brief Nearest and second-nearest seed for one query point (the result of a seed search).
 *
 * Plain data, usable inside device kernels. id2 = -1 (and f2 unspecified) when the second-best
 * result was not requested. The second-best can be used to detect near-ties (f2 - f1 small), i.e.
 * points close to a grain boundary.
 */
struct BestPair {
    std::int32_t id1; ///< 0-based seed index of the minimum of f
    std::int32_t id2; ///< 0-based seed index of the second minimum of f, or -1
    double f1;        ///< f_{id1}(x)
    double f2;        ///< f_{id2}(x)
};

/**
 * @brief Output of the labeling pipeline on the local ParMesh (this rank's mesh partition).
 *
 * Per-grain arrays are replicated on every rank and indexed by grain id - 1. Per-region arrays
 * are replicated too and indexed by region id - 1.
 *
 * Kinds of grains (Stage 1d):
 * - grains grown from a seed (random, lattice, Johnson-Mehl or seed-table seeds), including
 *   "filler" grains grown from random seeds placed outside a measured region of interest;
 * - grains read from a voxel map (one per feature id used by the mesh);
 * - one "passthrough" grain per passthrough mesh attribute ([Mesh] passthrough_attributes): a
 *   part of the mesh (e.g. a build plate) kept as a single grain with its own region. These are
 *   always the last grains, num_grains - (number of passthrough attributes) + 1 .. num_grains.
 *
 * Every row of grain_quat is a unit quaternion (norm 1 to machine precision).
 */
struct MicrostructureResult {
    /** @brief Per local element (0-based local element index), compact grain id 1..num_grains. */
    mfem::Array<int> element_grain;
    /** @brief Global number of grains (same on every rank). */
    int num_grains = 0;
    /** @brief Size 4 * num_grains: grain_quat[4*(g-1) + k], k -> w x y z, for grain g. */
    std::vector<double> grain_quat;
    /** @brief Size num_grains: region id (1..num_regions) of grain g at index g-1. */
    std::vector<int> grain_region;
    /**
     * @brief Size num_grains: stable source id of grain g at index g-1: the SeedSet::source_id of
     * its seed, the feature id of a voxel-map grain, a derived id (DeriveFragmentSourceId) for a
     * piece split off by the fragment cleanup, or -attribute for a passthrough grain.
     */
    std::vector<std::int64_t> grain_source_id;
    /** @brief Global number of material regions (same on every rank). */
    int num_regions = 1;
    /**
     * @brief Size num_grains: phase id (>= 0) of grain g at index g-1, or -1 when the grain has
     * no phase. Phases come from a seed table's phase column, from a voxel map's per-feature
     * phases (read whenever the file has them, in every region mode), from Regions.filler_phase
     * for filler grains, or from the synthetic phase fractions; -1 when no phase data exists and
     * for passthrough grains.
     */
    std::vector<int> grain_phase;
    /**
     * @brief Size num_grains: 1 when grain g (index g-1) grew from a random filler seed placed
     * outside the region of interest of a seed table (or was split off such a grain), else 0.
     * char rather than bool so the vector is a plain byte array.
     */
    std::vector<char> grain_filler;
    /**
     * @brief Size num_regions: region_phases[r-1] = the phase ids (ascending) whose grains are in
     * region r. Empty for a region without phase data (e.g. Regions.mode = "single" without
     * phases, or a region that holds only passthrough grains).
     */
    std::vector<std::vector<int>> region_phases;
    /**
     * @brief Size num_regions: 1 when region r (index r-1) is crystalline (its grains carry real
     * orientations), 0 when it is not (non-crystalline phases, or a passthrough-only region);
     * the grains of non-crystalline regions get the identity quaternion (1, 0, 0, 0).
     */
    std::vector<char> region_crystalline;
};

} // namespace exaconstit::microstructure
