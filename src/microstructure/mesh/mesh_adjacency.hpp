#pragma once

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @file mesh_adjacency.hpp
 * @brief Face adjacency of the elements of a distributed mesh, including neighbors owned by other
 * ranks, plus the exchange of per-element values with those neighbors.
 *
 * **Vocabulary.**
 * - Two elements are *face neighbors* when they share a whole face (a triangle or quadrilateral
 *   in 3D). Grains are "connected" in this sense: a grain is one piece when one can walk from any
 *   of its elements to any other through shared faces without leaving the grain.
 * - On a distributed `mfem::ParMesh` each rank owns its own *mesh partition*. A face on the
 *   border between two partitions is a *shared face*: its two elements live on different ranks.
 *   The element on the other side is called a *ghost element* here (MFEM calls it a "face
 *   neighbor element"): this rank does not own it, but it receives a copy of the element and of
 *   any per-element data it asks for.
 *
 * **What MeshAdjacency stores** (host arrays, built once, collective constructor):
 * - for every local element e, the list of its face neighbors in compressed-row form
 *   (CSR: all lists stored back to back in one array, with an offsets array marking where each
 *   element's list starts): entries `Neighbors()[k]` for `Offsets()[e] <= k < Offsets()[e+1]`.
 *   A neighbor index n < NE is a local element; n >= NE is ghost element g = n - NE;
 * - for every entry, the area of the shared face (quadrature on the face) and its local face
 *   index.
 *
 * Every interior face therefore appears twice among the local entries when both elements are
 * local (once from each side), and once on each of the two ranks when it is a shared face.
 * Boundary faces of the domain do not appear.
 *
 * **Exchanging data with ghost elements.** The ExchangeLabels / ExchangeKeys / ExchangeValues
 * functions send one value per local element to the neighboring ranks and return the values of
 * the ghost elements. They use MFEM's own machinery: the values are stored in a piecewise-
 * constant (L2, order 0, one degree of freedom per element) `mfem::ParGridFunction` and
 * `ParGridFunction::ExchangeFaceNbrData` (`fem/pgridfunc.cpp:209-268`) sends every value that a
 * neighboring rank needs, after which `FaceNbrData()` holds the ghost values. ExchangeKeys sends
 * both 32-bit halves of a key in one exchange (a two-component L2 field). The values are passed
 * in by the caller every time; element attributes cached in the mesh are never used, because
 * they go stale as soon as elements are relabeled.
 *
 * **Collective calls and their cost.** The constructor and the Exchange* functions are
 * *collective*: every rank of the mesh communicator must call them, in the same order, even a
 * rank that owns no element or has no shared faces. Each Exchange* call does one global
 * MPI_Allreduce of one integer (the input-size check, so a wrong size throws on every rank
 * instead of hanging), then point-to-point messages with the face-neighbor ranks only (MFEM's
 * exchange) and, with periodic pairing, point-to-point messages with the periodic partner ranks
 * only. A rank that skips a call leaves its neighbors waiting forever.
 *
 * **Periodic microstructures.** Optionally (PeriodicFaceParameters) boundary faces on opposite
 * planes of a periodic box are paired, so elements on the two sides become face neighbors and a
 * grain that wraps around the box is one connected grain. Pairs whose two elements live on
 * different ranks add *periodic ghosts*, exchanged with MPI_Isend / MPI_Irecv between the partner
 * ranks inside the same Exchange* calls. Two consequences for meshes only one or two elements
 * thick along a periodic axis: an element that is both a face-neighbor ghost and a periodic
 * partner appears under two ghost indices (one of each kind), and a same-rank periodic pair can
 * repeat an entry of an interior neighbor (two shared faces between the same two elements). Both
 * are correct for connectivity and face counting.
 *
 * Requirements: a conforming 3D ParMesh (no hanging nodes). Curved meshes (with Nodes) are
 * supported; face areas then use the curved face geometry.
 */

namespace exaconstit::microstructure {

/**
 * @brief Optional periodic face pairing of MeshAdjacency: which axes are periodic and the box.
 *
 * In a periodic microstructure a grain that leaves the box through the face x = lo continues on
 * the face x = hi. The mesh itself does not connect those faces, so without pairing a grain that
 * wraps around looks like two separate pieces. With pairing, a boundary face on the plane
 * x_d = lo_d and a boundary face on x_d = hi_d whose vertices coincide after shifting by the box
 * length L_d e_d (within the tolerance) are treated as one shared face: their two elements become
 * face neighbors (*periodic neighbors*).
 *
 * **Requirement and failure behavior.** The mesh must have matching faces on opposite periodic
 * planes (as a structured box mesh, or a mesh generated with periodic surface constraints, has).
 * If any face on a periodic plane has no unique partner, the constructor throws on every rank
 * with a message giving the number of unmatched faces (silently leaving them unpaired would
 * split or merge wrapped grains without notice).
 *
 * Host data; every rank must pass the same values (checked collectively).
 */
struct PeriodicFaceParameters {
    /** @brief periodic[d] is true when axis d (0 = x, 1 = y, 2 = z) is periodic. */
    std::array<bool, 3> periodic = {false, false, false};
    /** @brief Lower box corner (mesh length units), e.g. PeriodicBox::lo. */
    std::array<double, 3> lo = {0.0, 0.0, 0.0};
    /** @brief Upper box corner (mesh length units); hi[d] > lo[d] on every periodic axis. */
    std::array<double, 3> hi = {1.0, 1.0, 1.0};
    /**
     * @brief Matching tolerance as a fraction of the smallest edge length h_min of the faces on
     * the periodic planes (over all ranks): two faces match when every vertex of one, shifted by
     * L_d, lies within tol = relative_tolerance x h_min of a vertex of the other in every
     * coordinate. A face is on a periodic plane when all its vertices are within
     * relative_tolerance x (its own smallest edge) of the plane. tol is raised to at least
     * 1e-14 x the largest periodic box length, so the tolerance cells stay representable. Must be
     * in (0, 0.5).
     */
    double relative_tolerance = 1.0e-6;
};

/**
 * @brief What the periodic face pairing found. Global counts, identical on every rank.
 */
struct PeriodicFaceReport {
    /** @brief Boundary faces lying on a periodic plane (a face counts once per such plane). */
    std::int64_t plane_faces = 0;
    /** @brief Pairs of matched faces (each pair joins two elements as periodic neighbors). */
    std::int64_t matched_pairs = 0;
    /**
     * @brief Plane faces without a unique partner on the opposite plane. Always 0 after a
     * successful construction (the constructor throws otherwise).
     */
    std::int64_t unmatched_faces = 0;
    /** @brief The absolute matching tolerance used (mesh length units). */
    double tolerance = 0.0;
};

/**
 * @brief Face adjacency of a ParMesh's local elements, including ghost neighbors on other ranks,
 * with face areas and collective value exchange (see the file comment).
 *
 * Host data structure. The ParMesh must outlive the object and must not be refined or
 * repartitioned while it is in use.
 */
class MeshAdjacency {
public:
    /**
     * @brief Builds the adjacency of `pmesh`.
     *
     * Steps: calls `ParMesh::ExchangeFaceNbrData()` (receives the ghost elements); creates the
     * L2 order-0 finite element space used for value exchange; lists every local interior face
     * (both elements local) and every shared face (`ParMesh::GetNSharedFaces`,
     * `ParMesh::GetSharedFace`); integrates each face's area.
     *
     * **Collective** over the mesh communicator. Host only.
     *
     * @param pmesh           the distributed mesh (3D, conforming). Non-const because MFEM's face
     *                        neighbor exchange and finite element space construction modify
     *                        internal caches of the mesh; element data are not changed.
     * @param area_quad_order quadrature order for face areas; a negative value selects
     *                        2 p + 1, where p is the mesh order (1 without Nodes). Exact for
     *                        planar faces of any shape; for curved faces the area element
     *                        sqrt(det(J^T J)) is not a polynomial and the order is a good
     *                        approximation.
     * @throws std::invalid_argument on every rank if the mesh is not 3D or is nonconforming (both
     *         properties are the same on every rank).
     */
    explicit MeshAdjacency(mfem::ParMesh& pmesh, int area_quad_order = -1);

    /**
     * @brief Builds the adjacency of `pmesh` with periodic face pairing (see
     * PeriodicFaceParameters).
     *
     * In addition to the steps of the other constructor:
     *  1. the parameters are compared over all ranks (MPI_Allreduce MIN/MAX); a mismatch throws
     *     on every rank;
     *  2. every rank lists its boundary faces lying on a periodic plane, with their vertices;
     *  3. the lists are gathered on every rank (MPI_Allgatherv; memory proportional to the number
     *     of faces on the periodic planes), and the tolerance is set from the smallest gathered
     *     face edge;
     *  4. every rank matches each face on a lower plane to the face on the opposite upper plane
     *     whose vertices are the shifted ones, looking up candidates by centroid in an ordered
     *     map (std::map) of cells of size 2 tol, in the same global order, so all ranks find the
     *     same pairs; unmatched faces throw on every rank;
     *  5. each rank adds the pairs that involve its own elements. A partner on the same rank
     *     becomes a local neighbor; a partner on another rank becomes a *periodic ghost* (see
     *     NumGhostElements), whose values the Exchange* functions deliver point to point. A face
     *     paired with a face of its own element (a mesh one element thick along a periodic axis)
     *     adds nothing.
     *
     * **Collective** over the mesh communicator. With no periodic axis this is the same as the
     * other constructor (Exchange* then make no extra MPI call).
     *
     * @throws std::invalid_argument on every rank for a non-3D or nonconforming mesh, a
     *         non-positive box length on a periodic axis, a tolerance outside (0, 0.5), or
     *         parameters that differ between ranks; std::runtime_error on every rank if any face
     *         on a periodic plane has no unique partner.
     */
    MeshAdjacency(mfem::ParMesh& pmesh, const PeriodicFaceParameters& periodic,
                  int area_quad_order = -1);

    ~MeshAdjacency();
    MeshAdjacency(const MeshAdjacency&) = delete;
    MeshAdjacency& operator=(const MeshAdjacency&) = delete;

    /** @brief Number of local elements NE (this rank). */
    int NumElements() const {
        return m_num_elements;
    }

    /**
     * @brief Number of ghost elements NG: elements owned by other ranks that share a face with a
     * local element. Ghosts 0 .. NumFaceNeighborGhosts()-1 are MFEM's face-neighbor elements (in
     * MFEM's order); the remaining NumPeriodicGhosts() ones are periodic neighbors on other ranks
     * (ordered by owner rank, then the owner's local element id).
     */
    int NumGhostElements() const {
        return m_num_ghosts + m_num_periodic_ghosts;
    }

    /** @brief Ghosts across shared faces of the partition (MFEM face-neighbor elements). */
    int NumFaceNeighborGhosts() const {
        return m_num_ghosts;
    }

    /** @brief Ghosts that are periodic neighbors owned by other ranks (0 without pairing). */
    int NumPeriodicGhosts() const {
        return m_num_periodic_ghosts;
    }

    /** @brief True if periodic face pairing was requested for at least one axis. */
    bool HasPeriodicPairing() const {
        return m_periodic_pairing;
    }

    /** @brief Counts of the periodic face pairing (all zero without pairing). */
    const PeriodicFaceReport& PeriodicReport() const {
        return m_periodic_report;
    }

    /**
     * @brief CSR row offsets, size NE + 1: the neighbors of local element e are entries
     * Offsets()[e] .. Offsets()[e+1]-1 of Neighbors(), FaceAreas() and FaceIndices().
     */
    const mfem::Array<int>& Offsets() const {
        return m_offsets;
    }

    /**
     * @brief Neighbor of each entry, size Offsets()[NE]: n in [0, NE) is a local element,
     * n in [NE, NE + NG) is ghost element n - NE. Within one element's row the entries are sorted
     * by increasing n.
     */
    const mfem::Array<int>& Neighbors() const {
        return m_neighbors;
    }

    /**
     * @brief Area of the shared face of each entry (mesh length units squared). For a periodic
     * entry it is the area of this rank's own face on the periodic plane (equal to the partner's
     * up to round-off).
     */
    const mfem::Vector& FaceAreas() const {
        return m_face_areas;
    }

    /**
     * @brief Local (this rank's) face index of each entry, as used by Mesh::GetFaceElements. For
     * a periodic entry it is this rank's face on the periodic plane (the partner face belongs to
     * the neighbor).
     */
    const mfem::Array<int>& FaceIndices() const {
        return m_face_indices;
    }

    /** @brief True if neighbor index n (from Neighbors()) is a ghost element. */
    bool IsGhost(int n) const {
        return n >= m_num_elements;
    }

    /** @brief The mesh this adjacency describes. */
    mfem::ParMesh& GetParMesh() const {
        return *m_pmesh;
    }

    /** @brief The mesh communicator. */
    MPI_Comm GetComm() const {
        return m_pmesh->GetComm();
    }

    /**
     * @brief Sends one double per local element to the neighboring ranks and returns the values
     * of the ghost elements.
     *
     * **Collective** over the mesh communicator (see the file comment). Host function; MFEM may
     * run its packing kernel on the configured mfem::Device.
     *
     * @param local size NE: the value of each local element. Read on the host (HostRead).
     * @param ghost resized to NG: ghost[g] = the owner's value of ghost element g. The bits are
     *              copied exactly (no arithmetic is applied).
     * @throws std::invalid_argument if local.Size() != NE (checked on every rank before any
     *         communication, so all ranks throw together).
     */
    void ExchangeValues(const mfem::Vector& local, mfem::Vector& ghost) const;

    /**
     * @brief ExchangeValues for integer labels (grain or seed ids).
     *
     * The ints travel as doubles, which represent every int exactly.
     *
     * **Collective**. Host function.
     *
     * @param local size NE: label of each local element (HostRead).
     * @param ghost resized to NG: label of each ghost element (host memory).
     * @throws std::invalid_argument on every rank if local.Size() != NE on any rank.
     */
    void ExchangeLabels(const mfem::Array<int>& local, mfem::Array<int>& ghost) const;

    /**
     * @brief ExchangeValues for 64-bit keys (element keys, fragment ids).
     *
     * A double holds integers exactly only up to 2^53, so each key is split into its upper and
     * lower 32-bit halves; both halves travel in one exchange (a two-component field) and the key
     * is reassembled.
     *
     * **Collective** (one size check, one exchange). Host only.
     *
     * @param local size NE: key of each local element.
     * @param ghost resized to NG: key of each ghost element.
     * @throws std::invalid_argument on every rank if local.size() != NE on any rank.
     */
    void ExchangeKeys(const std::vector<std::uint64_t>& local,
                      std::vector<std::uint64_t>& ghost) const;

private:
    /** @brief One adjacency found while scanning faces (defined in the .cpp). */
    struct Entry;

    /** @brief Throws on every rank if `size` differs from NE on any rank (collective). */
    void CheckLocalSize(int size, const char* caller) const;

    /**
     * @brief The exchange itself, without the size check: `num_components` (1 or 2) values per
     * local element, stored component-major in `local` (local[c * NE + e]); `ghost` receives
     * ghost[c * NG + g]. Collective.
     */
    void ExchangeComponents(const double* local, int num_components,
                            std::vector<double>& ghost) const;

    /** @brief Shared constructor body; `periodic` may be null (no pairing). Collective. */
    void Build(int area_quad_order, const PeriodicFaceParameters* periodic);

    /**
     * @brief Finds the periodic face pairs and appends their adjacency entries; sets up the
     * periodic ghost exchange (see the periodic constructor). Collective.
     */
    void PairPeriodicFaces(const PeriodicFaceParameters& periodic, int quad_order,
                           std::vector<Entry>& entries);

    mfem::ParMesh* m_pmesh = nullptr;
    int m_num_elements = 0;
    int m_num_ghosts = 0;
    mfem::Array<int> m_offsets;
    mfem::Array<int> m_neighbors;
    mfem::Vector m_face_areas;
    mfem::Array<int> m_face_indices;

    /**
     * @brief Piecewise-constant (L2 order 0) spaces with 1 and 2 components per element, used
     * for exchange (index = number of components - 1).
     */
    std::unique_ptr<mfem::L2_FECollection> m_fec;
    std::array<std::unique_ptr<mfem::ParFiniteElementSpace>, 2> m_fes;
    /**
     * @brief Per number of components c+1: element_vdof[c][k * NE + e] = vector degree of freedom
     * of component k of local element e; ghost_vdof[c][k * NG + g] = index of component k of
     * face-neighbor ghost g in ParGridFunction::FaceNbrData().
     */
    std::array<std::vector<int>, 2> m_element_vdof;
    std::array<std::vector<int>, 2> m_ghost_vdof;

    // Periodic pairing (all empty / false without it)
    bool m_periodic_pairing = false;
    int m_num_periodic_ghosts = 0;
    PeriodicFaceReport m_periodic_report;
    /** @brief Partner ranks of the periodic exchange (ascending; never this rank). */
    std::vector<int> m_periodic_partners;
    /**
     * @brief Per partner (same order as m_periodic_partners): the local elements sent to it
     * (ascending local id), and the first periodic ghost and number of ghosts received from it.
     * Ghosts are numbered by (owner rank, owner's element), so each partner's block is
     * contiguous and in the order the partner sends.
     */
    std::vector<std::vector<int>> m_periodic_send_elements;
    std::vector<int> m_periodic_recv_begin, m_periodic_recv_count;
};

} // namespace exaconstit::microstructure
