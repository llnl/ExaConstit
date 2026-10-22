#pragma once

#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mpi.h>
#include <ostream>
#include <streambuf>
#include <string>
#include <vector>

/**
 * @file output_writer.hpp
 * @brief Writes a generated microstructure as the input files ExaConstit reads.
 *
 * The writer is the last step of the generator. It receives data that has already been gathered
 * (a serial mesh whose element attributes are the grain ids, per-grain tables, element-ordered
 * grain ids) and turns it into files. It does no labeling, and no MPI communication beyond what
 * the functions marked "collective" document. It depends only on ms_core and MFEM, not on
 * ms_mesh. Everything here runs on the host.
 *
 * Files and the ExaConstit code that reads each one (functions in
 * src/sim_state/simulation_state.cpp):
 *
 * | File                       | Writer                 | Read by (ExaConstit)                    |
 * |----------------------------|------------------------|-----------------------------------------|
 * | serial mesh (.mesh/.gz)    | WriteSerialMesh        | makeMesh, `[Mesh] type = "file"`, via   |
 * |                            |                        | `mfem::Mesh(floc)` (gzip auto-detected) |
 * | mesh parts (prefix.NNNNNN) | WritePartitionedMesh,  | readParallelMesh, `type = "parallel"`:  |
 * |                            | WriteParallelMesh      | rank r opens MakeParFilename(floc, r)   |
 * |                            |                        | with mfem::ifgzstream, then ParMesh     |
 * | orientation file (.ori)    | WriteOrientationFile   | LoadSharedOrientationData:              |
 * |                            |                        | `while (std::ifstream >> value)`        |
 * | region map                 | WriteRegionMap         | create_grains_to_map: getline, then     |
 * |                            |                        | `iss >> grain >> region` per line       |
 * | grain file                 | WriteGrainFile         | makeMesh (auto meshes only):            |
 * |                            |                        | `mfem::Vector::Load(std::ifstream, NE)` |
 * | grain table (.csv)         | WriteGrainTable        | nobody; per-grain summary for the user  |
 * | grain adjacency (.csv)     | WriteGrainAdjacency    | nobody; grain pairs sharing faces       |
 * | ParaView collection        | WriteParaView          | nobody; visualization                   |
 *
 * The tool does not write ExaConstit input decks (options.toml); users write their own and point
 * them at these files. Provenance (tool version, rng_seed, grain counts) goes into the mesh file
 * header comments (the `comments` argument of the mesh writers) and the rank-0 log.
 *
 * Conventions the files follow:
 * - Element attributes are the compact grain ids 1..N (N = number of grains).
 * - The orientation file, region map and grain file have no header line (ExaConstit reads every
 *   number in them). The grain table and grain adjacency CSVs have a column-name row.
 * - Numbers are written with 17 significant digits, enough for any double to read back
 *   bit-for-bit.
 * - Only the mesh files may be gzip compressed. The orientation file, region map and grain file
 *   are read by ExaConstit with a plain std::ifstream, so they are always plain text.
 *
 * Why ParMesh::Save and ParMesh::PrintAsOne are never used for meshes:
 * - ParMesh::Save writes each rank's piece with ParMesh::Print, which by default also writes the
 *   faces shared with neighbouring ranks as boundary elements (pmesh.hpp:726-733: "the parallel
 *   interface can be added as boundary", on by default). Read back, those faces become real
 *   boundaries, so the pieces are not a valid decomposition and boundary conditions would land on
 *   interior faces.
 * - ParMesh::PrintAsOne writes one file in which the pieces are disconnected and "both the
 *   element and boundary attributes are set to the processor number" (pmesh.hpp:750-755). That
 *   destroys the grain ids.
 *
 * Instead, serial meshes use Mesh::Print, and parallel decompositions use MeshPart::Print or
 * ParMesh::ParPrint, which write the "MFEM mesh v1.2" parallel format: one file per part that
 * contains the part's elements plus the list of vertices/faces it shares with other parts, so
 * the ParMesh can be rebuilt without inventing boundaries. MFEM picks the version line itself
 * (mesh.cpp:12285-12289): a mesh with named attribute sets is written as "MFEM mesh v1.3"
 * instead (v1.0 for a serial mesh, v1.2 for a part, become v1.3), and a nonconforming mesh (one
 * with hanging nodes from local refinement) is written in MFEM's separate NC mesh format. MFEM's
 * readers accept all of these; WriteParallelMesh rejects nonconforming meshes (see there).
 *
 * Errors: std::runtime_error for file problems (cannot open, short write) and
 * std::invalid_argument for inconsistent input. A write that fails part-way leaves the partial,
 * possibly corrupt file on disk; the writer does not delete it, so check for the exception before
 * using the output. A "collective" function is one that every rank of the communicator must call,
 * because ranks exchange data inside it. Collective functions here first combine an error flag
 * over all ranks (an MPI_Allreduce) and then throw on every rank together, so a failure on one
 * rank cannot leave the others waiting forever in a later MPI call.
 */

namespace exaconstit::microstructure {

/**
 * @brief Per-grain columns of the grain table CSV. Optional columns are empty when unused.
 *
 * Row g-1 (0-based) holds grain g (grain ids are 1-based).
 */
struct GrainTable {
    /** @brief Number of grains (rows); grain ids are 1..num_grains. */
    int num_grains = 0;
    /** @brief Required, 4 * num_grains values: grain g at [4(g-1) .. 4(g-1)+3], order w x y z. */
    std::vector<double> quat;
    /** @brief num_grains region ids (1-based), or empty. */
    std::vector<int> region;
    /** @brief num_grains seed source ids, or empty. */
    std::vector<std::int64_t> source_id;
    /** @brief num_grains global element counts, or empty. */
    std::vector<std::int64_t> num_elements;
    /** @brief num_grains grain volumes in mesh units cubed, or empty. */
    std::vector<double> volume;
    /** @brief 3 * num_grains grain centroids (x y z per grain, interleaved), or empty. */
    std::vector<double> centroid;
    /**
     * @brief num_grains neighbor counts, or empty: the number of distinct other grains that share
     * at least one element face with the grain.
     */
    std::vector<int> num_neighbors;
    /**
     * @brief num_grains target volumes in mesh units cubed, or empty: the size-control target of
     * the seed the grain grew from. Grains split off one seed by the fragment cleanup share that
     * seed's target, so compare the target with the sum of their volumes.
     */
    std::vector<double> target_volume;
    /**
     * @brief num_grains phase ids, or empty: the phase of each grain (from a seed table, a voxel
     * map or synthetic phase fractions; -1 for a grain without a phase, e.g. a passthrough
     * grain). MakeGrainTable fills it only when at least one grain has a phase (>= 0).
     */
    std::vector<int> phase;
    /**
     * @brief num_grains flags (0 or 1), or empty: 1 for a grain grown from a random filler seed
     * placed outside the measured region of interest of a seed table, or for a piece the
     * fragment cleanup split off such a grain (it inherits the flag). MakeGrainTable fills it
     * only when at least one grain is a filler grain.
     */
    std::vector<int> filler;
    /**
     * @brief num_grains voxel-map feature ids, or empty: the feature id (the grain id stored in
     * the voxel file) each grain was read from, -1 for a grain not read from the voxel map (a
     * passthrough grain). Filled by the caller for voxel-map runs only.
     */
    std::vector<std::int64_t> feature_id;
};

/**
 * @brief Pairs of face-sharing grains for the grain adjacency CSV. Row k describes the pair
 * (grain_a[k], grain_b[k]) with 1 <= grain_a < grain_b (1-based grain ids). Four columns of
 * equal length; rows are written in the given order (ComputeGrainStatistics produces them
 * sorted by (grain_a, grain_b)).
 */
struct GrainAdjacencyTable {
    /** @brief Smaller grain id of each pair (1-based). */
    std::vector<int> grain_a;
    /** @brief Larger grain id of each pair (1-based). */
    std::vector<int> grain_b;
    /** @brief Total area of the element faces the two grains share (mesh length units squared). */
    std::vector<double> shared_area;
    /** @brief Number of element faces the two grains share. */
    std::vector<std::int64_t> num_faces;
};

/**
 * @brief Builds a GrainTable from the per-grain arrays of a MicrostructureResult.
 *
 * Copies the orientation, region and source id columns. The phase column is copied only when
 * some grain has a phase (MicrostructureResult::grain_phase >= 0), and the filler column only
 * when some grain is a filler grain (grain_filler != 0), so runs without phases or filler seeds
 * keep the shorter table (a column is written only when it carries information). The volume,
 * centroid, element-count, neighbor, target-volume and feature-id columns are left empty for
 * the caller to fill in. Host only, not collective, never throws (apart from std::bad_alloc).
 */
GrainTable MakeGrainTable(const MicrostructureResult& result);

/**
 * @brief Writes a serial MFEM mesh at 17 significant digits. Not collective; host only.
 *
 * The format is "MFEM mesh v1.0", or v1.3 when the mesh has named attribute sets, or MFEM's NC
 * format for a nonconforming mesh (MFEM chooses; mfem::Mesh(file) reads all of them).
 *
 * The output is what ExaConstit reads for `[Mesh] type = "file"`. With compression > 0 the file
 * is gzip-compressed at that level and ".gz" is appended to the path if missing. With
 * compression == 0, or when MFEM was built without zlib (MFEM_USE_ZLIB undefined; a warning is
 * printed), a plain file is written and a trailing ".gz" is dropped from the name. MFEM's mesh
 * reader detects gzip by the file's first two bytes, so either kind loads.
 *
 * @param serial_with_attributes Serial mesh whose element attributes are the grain ids. A
 *        ParMesh is rejected, because its Print() writes only the local piece (plus the shared
 *        faces as boundary).
 * @param path Requested file name; missing parent directories are created.
 * @param compression gzip level 0-9 (0 = no compression).
 * @param comments Provenance text printed after the first line of the file; lines that do not
 *        start with '#' get a "# " prefix so the mesh reader skips them.
 * @return The path that was actually written (with ".gz" added or removed as described).
 * @throws std::invalid_argument for a ParMesh or a level outside 0-9;
 *         std::runtime_error if the file cannot be created or fully written.
 */
std::filesystem::path WriteSerialMesh(const mfem::Mesh& serial_with_attributes,
                                      const std::filesystem::path& path,
                                      int compression,
                                      const std::string& comments = "");

/**
 * @brief Splits a serial mesh into @p num_parts pieces and writes them in MFEM's parallel mesh
 * format ("MFEM mesh v1.2"), one file per piece. Collective over @p comm; host only.
 *
 * The output is what ExaConstit reads for `[Mesh] type = "parallel"`: a run with num_parts MPI
 * ranks has rank r read the piece mfem::MakeParFilename(prefix, r), i.e. prefix followed by r
 * zero-padded to 6 digits (globals.hpp:84).
 *
 * Steps:
 * 1. Rank 0 asks METIS (the graph-partitioning library MFEM uses to split a mesh into
 *    balanced, compact pieces) for the element-to-part map
 *    (Mesh::GeneratePartitioning(num_parts, 1)) and broadcasts it, so every rank cuts the same
 *    decomposition.
 * 2. Each rank builds an mfem::MeshPartitioner from the map and extracts (MeshPartitioner::
 *    ExtractPart) and writes (MeshPart::Print) the parts p = rank, rank + nranks, ... . Any
 *    num_parts therefore works with any number of writing ranks.
 * 3. Curved meshes keep their high-order Nodes: ExtractPart copies them into the part.
 *
 * The part names carry no ".gz" suffix even when compressed, because ExaConstit opens
 * MakeParFilename(floc, rank) exactly; mfem::ifgzstream, which it reads with, detects gzip by
 * content. MeshPart::Print takes no comments, so none can be added here.
 *
 * @param serial_with_attributes The same serial mesh on every rank (non-const only because the
 *        MeshPartitioner constructor takes Mesh&; the mesh is not modified).
 * @param num_parts Number of parts, 1 <= num_parts <= number of elements.
 * @param prefix Part-file prefix (may include directories, which rank 0 creates).
 * @param compression gzip level 0-9.
 * @param comm Communicator of the writing ranks (any size).
 * @return The names of all parts, in part order (identical on every rank).
 * @throws std::invalid_argument (on every rank, before any communication) for a ParMesh, a bad
 *         num_parts or level; std::runtime_error on every rank if METIS leaves a part empty or
 *         any rank fails to write one of its parts.
 */
std::vector<std::filesystem::path> WritePartitionedMesh(mfem::Mesh& serial_with_attributes,
                                                        int num_parts,
                                                        const std::string& prefix,
                                                        int compression,
                                                        MPI_Comm comm);

/**
 * @brief Writes a ParMesh as its own parallel decomposition: each rank writes its piece with
 * ParMesh::ParPrint to mfem::MakeParFilename(prefix, rank). Collective over the mesh
 * communicator; host only.
 *
 * Use this when the input already was a parallel decomposition, so the output parts match the
 * input parts and the result must again be run on the same number of ranks. File naming and
 * compression follow WritePartitionedMesh. Unlike MeshPart::Print, ParPrint accepts comments.
 *
 * Only conforming, non-NURBS meshes are accepted. For those two other kinds ParPrint does not
 * write the parallel v1.2 format (pmesh.cpp:6353-6364): a NURBS mesh is written as a plain
 * serial v1.0 piece without the shared-entity information, which readParallelMesh cannot
 * reassemble into a decomposition; a nonconforming mesh is written in the NC format, which the
 * microstructure pipeline does not support (MakeParMesh rejects nonconforming meshes too).
 *
 * @param comments Provenance text, formatted as for WriteSerialMesh.
 * @return The names of all parts, in rank order (identical on every rank).
 * @throws std::invalid_argument on every rank, before anything is written, for a level outside
 *         0-9, a NURBS mesh or a nonconforming mesh; std::runtime_error on every rank if any rank
 *         fails to write its part.
 */
std::vector<std::filesystem::path> WriteParallelMesh(const mfem::ParMesh& pmesh,
                                                     const std::string& prefix,
                                                     int compression,
                                                     const std::string& comments = "");

/**
 * @brief Writes the orientation file: one `w x y z` line per grain, 17 significant digits, no
 * header, never compressed. Not collective; host only.
 *
 * ExaConstit reads every number in the file in order (`while (f >> value)`) and takes grain g's
 * quaternion from values 4(g-1)..4(g-1)+3, so line g is grain g. The quaternions are written
 * exactly as given (crystal-to-sample, passive, ExaConstit's w x y z order); no conversion.
 *
 * @param grain_quat 4 values per grain, grain g at [4(g-1) .. 4(g-1)+3].
 * @throws std::invalid_argument if the size is not a multiple of 4; std::runtime_error on file
 *         errors.
 */
void WriteOrientationFile(const std::filesystem::path& path, const std::vector<double>& grain_quat);

/**
 * @brief Writes the region map: one `g r` line per grain g = 1..N, no header, never compressed.
 * Not collective; host only.
 *
 * ExaConstit reads it line by line as `grain region` pairs and throws if a grain present in the
 * mesh has no line. Regions are 1-based and select the `[[Materials]]` block (in file order).
 *
 * @param grain_region Region id of grain g at index g-1.
 * @throws std::runtime_error on file errors.
 */
void WriteRegionMap(const std::filesystem::path& path, const std::vector<int>& grain_region);

/**
 * @brief Writes the grain file of an "auto" mesh: one grain id per line, in serial element
 * order, never compressed. Not collective; host only.
 *
 * ExaConstit builds the auto mesh itself (exaconstit::MakeAutoHexMesh, lexicographic element
 * order x fastest, then y, then z) and reads exactly NE numbers from this file with
 * mfem::Vector::Load, assigning number e to element e. The ids must therefore be in that same
 * element order.
 *
 * @param serial_grain_ids n grain ids (1-based), element e at index e; may be null when n == 0.
 * @param n Number of elements.
 * @throws std::invalid_argument for a null pointer with n > 0; std::runtime_error on file errors.
 */
void WriteGrainFile(const std::filesystem::path& path, const int* serial_grain_ids, std::size_t n);

/** @brief Overload of WriteGrainFile for a std::vector (element e at index e). */
void WriteGrainFile(const std::filesystem::path& path, const std::vector<int>& serial_grain_ids);

/**
 * @brief Overload of WriteGrainFile for an mfem::Array (element e at index e). The host copy of
 * the data is read (HostRead), so device-resident arrays are copied back first.
 */
void WriteGrainFile(const std::filesystem::path& path, const mfem::Array<int>& serial_grain_ids);

/**
 * @brief Writes the grain table as CSV with a header row: grain_id, qw, qx, qy, qz, then every
 * non-empty optional column of @p table, in this order: region, source_id, num_elements, volume,
 * cx, cy, cz, num_neighbors, target_volume, phase, filler, feature_id (the last three were added
 * in Stage 1d after the existing columns, so scripts that read columns by position keep
 * working). Numbers use 17 significant digits. Not collective; host only.
 *
 * @throws std::invalid_argument if a column has the wrong length; std::runtime_error on file
 *         errors.
 */
void WriteGrainTable(const std::filesystem::path& path, const GrainTable& table);

/**
 * @brief Writes the grain adjacency table as CSV: a header row `grain_a,grain_b,shared_area,
 * num_faces`, then one row per pair in the order given (17 significant digits). Not collective;
 * host only.
 *
 * @throws std::invalid_argument if the columns differ in length or a row does not have
 *         1 <= grain_a < grain_b; std::runtime_error on file errors.
 */
void WriteGrainAdjacency(const std::filesystem::path& path, const GrainAdjacencyTable& table);

/**
 * @brief Writes a ParaView collection with the grain id (and the region id, when
 * @p grain_region is non-empty) as piecewise-constant element fields. Collective over the mesh
 * communicator; host only.
 *
 * The fields live in an order-0 L2 space (one value per element) and are written with
 * mfem::ParaViewDataCollection, which writes one .vtu piece per rank plus a .pvd/.pvtu index.
 *
 * @param pmesh Distributed mesh (non-const because the data collection takes a Mesh*).
 * @param element_grain Grain id (1-based) of each local element, size pmesh.GetNE().
 * @param grain_region Region of grain g at index g-1, or empty for no region field.
 * @param path Collection path: the parent directory is the ParaView prefix path, the file name
 *        the collection name (e.g. "out/sample_paraview" gives out/sample_paraview/...pvd).
 * @throws std::invalid_argument wrapped in std::runtime_error on every rank if any rank has a
 *         wrong element_grain size or a grain without a region; std::runtime_error on every rank
 *         if the data collection reports an error.
 */
void WriteParaView(mfem::ParMesh& pmesh,
                   const mfem::Array<int>& element_grain,
                   const std::vector<int>& grain_region,
                   const std::filesystem::path& path);

namespace detail {

/**
 * @brief Gzip-compresses everything @p write sends to its stream into @p sink. Internal to the
 * writer; declared here only so the tests can feed it a failing sink. Not collective; host only.
 *
 * A stream buffer (std::streambuf) is the object behind every C++ stream that actually moves the
 * bytes: a std::ofstream's buffer writes to a file, a std::stringbuf to memory. A callback is a
 * function passed in as an argument for this function to call; here @p write is called once to
 * produce the text.
 *
 * @p write receives a std::ostream with precision 17. The whole output becomes gzip data at the
 * given level (1-9). The function is exception-safe: if the sink stops accepting bytes (disk full,
 * closed pipe) the error is reported as std::runtime_error, never by std::terminate from a
 * destructor; an exception thrown by @p write itself propagates unchanged.
 *
 * @param sink Destination stream buffer (for files, std::ofstream::rdbuf()); it is not flushed
 *        or closed here.
 * @param level gzip level 1-9.
 * @param write Callback that produces the uncompressed text.
 * @param what Name used in error messages (normally the file path).
 * @throws std::runtime_error if the sink rejects data, zlib reports an error, or MFEM was built
 *         without zlib; std::invalid_argument for a level outside 1-9.
 */
void WriteGzip(std::streambuf& sink,
               int level,
               const std::function<void(std::ostream&)>& write,
               const std::string& what);

} // namespace detail

} // namespace exaconstit::microstructure
