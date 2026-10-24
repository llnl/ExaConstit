#pragma once

#include "microstructure/mesh/element_mask.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <math.h> // HUGE_VAL in device code (C math library)

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

/**
 * @file vertex_field.hpp
 * @brief A distance value and a seed label on every field node of the mesh (the element corners,
 * or every nodal point of a curved mesh), with the bookkeeping that the geodesic
 * (path-inside-the-material) labelers share.
 *
 * **What it is for.** In `geodesic` distance mode a grain grows from its seed along paths that
 * stay inside the material, so grains wrap around corners and voids instead of reaching across
 * them. The distance from the seeds is computed on a set of points of the mesh, the *field
 * nodes*. Every field node v carries
 *   - T(v): the shortest path length from the nearest seed (mesh length units; +infinity when no
 *     seed reaches the node, e.g. on a body of the mesh without a seed), and
 *   - label(v): the 0-based index of that seed in the SeedSet (-1 when T(v) = +infinity).
 * Two solvers fill this field: the graph geodesic (labeling/geodesic_graph.hpp, Dijkstra on the
 * cell edges and diagonals) and the Fast Iterative Method eikonal solver
 * (labeling/geodesic_fim.hpp). Both start from InitializeSources, exchange values between ranks
 * with ReduceShared*, and turn the field into element labels with LabelElementCentroids /
 * LabelElementsByVote. This class owns everything they have in common. For historical reasons
 * the accessors say "vertex" (NumVertices, Coordinates, ...); read "field node".
 *
 * **Cells.** The solvers never look at elements directly but at *cells*: small straight-sided
 * pieces whose corners are field nodes. A path runs in straight lines inside a cell, and a point
 * of an element is reached in a straight line from a corner of the cell that contains it.
 * FieldCellView (below) lists the cells of an element; there are two kinds of field:
 *
 *  1. **Corner field** (FieldNodes::CORNERS; the Stage 1e behavior). The field nodes are the mesh
 *     vertices and every element is one cell with its corners. Exact for straight-sided elements.
 *     For a curved element it replaces the element by the convex hull of its corners, so a path
 *     follows chords of length h (the element size) and may cut through a concave curved wall
 *     (the error is of the order of the chord sag, h^2 times the wall curvature).
 *
 *  2. **Nodal field** (FieldNodes::MESH_NODES; used automatically for curved meshes). The field
 *     nodes are the degrees of freedom ("dofs", the nodal points) of the mesh's own order-p
 *     coordinate space (the Nodes space, mesh Nodes = the coordinates of these points), and every
 *     element is split on the fly into p^3 virtual sub-cells spanned by its nodal lattice:
 *       - hex: the (p+1)^3 nodes form a lattice (i, j, k), 0 <= i, j, k <= p, in the element's
 *         reference coordinates; sub-hex (a, b, c), 0 <= a, b, c < p, has the 8 corners
 *         (a + di, b + dj, c + dk), di, dj, dk in {0, 1};
 *       - tet: the (p+1)(p+2)(p+3)/6 nodes form the lattice i + j + k <= p, which is split into p^3
 *         sub-tets by the standard "up tet / octahedron / down tet" decomposition (J. Bey,
 *         "Tetrahedral grid refinement", Computing 55, 1995; the same split as one red
 *         refinement step, applied to the whole lattice at once): for each lattice point
 *         (i, j, k) an upright tet when i + j + k <= p - 1, an octahedron cut into 4 tets around
 *         one of its diagonals when i + j + k <= p - 2, and an inverted tet when
 *         i + j + k <= p - 3 (count: C(p+2,3) + 4 C(p+1,3) + C(p,3) = p^3).
 *     The sub-cells are never stored (no refined mesh, no sub-cell table): FieldCellView derives
 *     them from the lattice index of a node and two small per-mesh lookup tables. Since all field
 *     nodes lie on the true (curved) geometry, paths follow chords of length h/p, i.e. the
 *     resolution of a mesh refined p times, without building that mesh.
 *     How the lattice maps to MFEM's dofs (verified in MFEM v4.10):
 *       - the element -> dof table of the Nodes space (FiniteElementSpace::GetElementToDofTable,
 *         fespace.hpp:1278, built by BuildElementToDofTable, fespace.cpp:356-395) lists each
 *         element's dofs in the finite element's own ("native") order, for H1 without signs;
 *       - the native index of lattice point (i, j, k) is lex[o], with o the position of (i, j, k)
 *         in lexicographic order (i fastest, then j, then k; for a tet only i + j + k <= p):
 *         NodalFiniteElement::GetLexicographicOrdering (fe_base.hpp:878), which for an H1 hex is
 *         H1_HexahedronElement's dof map (fe_h1.cpp:284-290 places node dof_map[o] at
 *         (cp[i], cp[j], cp[k]); NodalTensorFiniteElement copies it, fe_base.cpp:2750) and for an
 *         H1 tet the table built next to the nodes (fe_h1.cpp:644-732, idx(i,j,k) is that
 *         lexicographic position). Bernstein ("positive") elements give the same map through
 *         their GetDofMap (fe_pos.hpp:207, 251). An empty map means "identity";
 *       - the constructor checks that the map is one-to-one and that the reference node
 *         coordinates (FiniteElement::GetNodes) increase along every lattice direction, so a
 *         wrong ordering is caught instead of producing twisted sub-cells.
 *     Node coordinates: for a nodal basis (Gauss-Lobatto, closed uniform, ...) the Nodes value at
 *     a dof *is* the position of the nodal point. For a Bernstein basis the dofs are control
 *     points off the geometry, so the position is evaluated through the element map at the
 *     node's reference point (in the first local element holding the dof); the owner's value is
 *     then broadcast (see below), which is exact for the geometry but partition independent only
 *     to round-off when ranks evaluate in different elements.
 *
 * **Which field (FieldNodes::AUTO).** Straight-sided meshes (mesh/straight_sided.hpp:
 * IsStraightSided, which accepts Nodes that describe straight elements, e.g. after
 * Mesh::SetCurvature) use the corner field, bit for bit as before Wave 8. Curved meshes use the
 * nodal field. With an element mask only the active elements decide (a curved passthrough part
 * does not switch the material to the nodal field). CORNERS / MESH_NODES force a choice
 * (MESH_NODES on a mesh without Nodes is the corner field). The nodal field supports a single
 * geometry of the active elements, hex or tet (inactive elements may differ), and a
 * fixed-order continuous Nodes space that is an mfem::ParFiniteElementSpace (what ParMesh and
 * ParMesh::SetCurvature create, pmesh.cpp:278-288, 2049-2051).
 *
 * **Numbering and the rank exchange.** Corner field: field node v is mesh vertex v; no finite
 * element space is built. The exchange uses a GroupCommunicator over the mesh's own group
 * topology (ParMesh::gtopo) whose group table lists, for every shared-vertex group g >= 1, the
 * local vertices ParMesh::GroupVertex(g, j) in that order, exactly what an order-1 H1 space would
 * build for its vertex dofs (ParFiniteElementSpace::GetGroupComm, pfespace.cpp:392-415), so the
 * reductions are bit for bit those of the Stage 1e version; the owner of a vertex is the master
 * of its group (GroupTopology::IAmMaster), as GetLocalTDofNumber decides for that space
 * (pfespace.cpp:1720-1739, 2236-2257). Nodal field: no space is built; the field nodes are the
 * scalar dofs of the mesh's Nodes space, and the exchange uses
 * ParFiniteElementSpace::ScalarGroupComm of that space (pfespace.cpp:1129-1141: a new
 * GroupCommunicator on scalar dofs over the same group topology; its memory is proportional to the
 * shared dofs only). The space's own GroupComm() works on the 3 vector components
 * (pfespace.hpp:419-423, ConstructTrueDofs pfespace.cpp:2229-2232 builds it with ldof_type = 1), so
 * using it would need 3x larger exchange buffers in every round. A
 * GroupCommunicator is MFEM's object that knows which local dofs are copies of the same node on
 * other ranks ("shared" nodes) and can combine and redistribute their values.
 *
 * **Identical coordinates on every copy.** Both solvers compare sums of lengths exactly, so every
 * rank must compute the same bits for the same node. The constructor broadcasts the owning
 * rank's coordinates to every copy of a shared node (GroupCommunicator::Bcast, one array per
 * coordinate), so Coordinates() is identical on all copies by construction. The mesh itself
 * (vertices, Nodes) is never modified. Outside seeds are snapped with these coordinates too.
 * Coordinates are read from the Nodes vertex dofs (corner field with Nodes) or all Nodes dofs
 * (nodal field), not from Mesh::GetVertex, which MFEM documents as possibly stale when Nodes
 * exist (mesh.hpp:1426). Discontinuous Nodes (periodic meshes) are rejected: Stage 1e does not
 * combine geodesic distance with periodicity. Partition independence assumes that all copies of
 * a node hold the same coordinates before the broadcast (true for a ParMesh distributed from a
 * serial mesh); otherwise the owner's value, which depends on the partition, wins.
 *
 * **Connectivity tables (CSR).** "CSR" (compressed sparse row) stores a list of variable-length
 * lists in two flat arrays: `offsets` (one more entry than lists) and `items`; list i is
 * `items[offsets[i] .. offsets[i+1]-1]`. Two tables are kept, as mfem::Array<int> so device
 * kernels can read them (Read()):
 *   - element -> nodes: ElementVertexOffsets() (size NE+1), ElementVertices(); corner field: the
 *     element's vertices in MFEM's element vertex order (an array built here); nodal field: the
 *     element's dofs in native order, a no-copy view (mfem::Array::MakeRef on the Memory of the
 *     Nodes space's element -> dof Table, array.hpp:281, mem_manager.hpp:1130). The view is valid
 *     only while the Nodes space is unchanged (no refinement or Update() of the mesh while the
 *     field lives). Its first device Read() registers MFEM's table memory with the memory manager
 *     (Memory::MakeAlias registers an unregistered base, mem_manager.hpp:1130-1160), which is
 *     harmless for MFEM but means the table's host and device copies are then managed by MFEM;
 *   - nodal field only: the lattice point of the node of every incidence (an int per entry of
 *     the node -> element CSR), so the solvers never search an element's (p+1)^3 dofs;
 *   - node -> elements: VertexElementOffsets() (size NV+1), VertexElements(); the local elements
 *     that contain node v, in increasing element order (built here: the one connectivity array the
 *     nodal field needs, NV+1 + NE * (nodes per element) ints; the solvers walk it in every
 *     round, and searching the element table instead would cost a pass over all elements).
 * NE and NV are this rank's local element and field node counts; indices are 0-based and local.
 *
 * **Memory (nodal field).** Per field node: 3 coordinates, T, the source T0 (doubles), label,
 * source label, shared and owned markers (ints); the node -> element CSR; per element nothing
 * (the element -> node view aliases MFEM's table); the lookup tables are O(p^3) per mesh. No mesh,
 * sub-mesh or refined mesh is created.
 *
 * **Shared nodes and determinism.** A node on a partition boundary exists on every rank whose
 * elements touch it. After a solver has updated its local copies, ReduceShared() makes all copies
 * equal to the lexicographic minimum of (T, label) over the copies: the smallest T, and among
 * copies with that T the smallest seed index. "Lexicographic" means compared like words in a
 * dictionary: first by T, then by label. Since MFEM's group reduction is a plain element-wise
 * minimum, the pair is reduced in two steps (see ReduceShared). The result does not depend on
 * the order of the ranks or on which rank owns the node, so a solver built on it can be made
 * independent of the rank count. The cells of an element are fixed by the element alone (its
 * lattice is defined by its own reference coordinates, which come from the serial mesh's element
 * vertex order), so they are the same on every partition.
 *
 * **Where it runs.** Construction, source initialization and the rank exchanges run on the host
 * (MPI). The point labeling (LabelPoints and the element labelings built on it) is an mfem::forall
 * kernel (MFEM's parallel loop: a GPU kernel when the mfem::Device has a CUDA or HIP backend, a
 * host loop otherwise) that reads the CSR tables, coordinates and field through Read(), so it runs
 * wherever the mfem::Device runs kernels; for the nodal field, the reference coordinates of
 * physical points are found on the host first (mfem::InverseElementTransformation). Solvers may
 * update Distance() / Label() on the device (ReadWrite()); ReduceShared* moves them to the host
 * for the exchange (HostReadWrite()).
 *
 * **Typical use (a solver):**
 * @code
 *   VertexField field(pmesh);                                  // collective
 *   SourceReport src = field.InitializeSources(seeds, a_i);    // collective, T/label at sources
 *   const FieldCellView cells = field.CellView();              // device view of the cells
 *   do {
 *       ... update field.Distance() / field.Label() from the local cells ...
 *   } while (field.ReduceShared());                            // collective, any change?
 *   mfem::Array<int> element_seed;
 *   field.LabelElementCentroids(element_seed);                 // local, kernel
 * @endcode
 */

namespace exaconstit::microstructure {

/** @brief Which points carry the distance field (see the file comment, "Which field"). */
enum class FieldNodes {
    /** @brief Corners for straight-sided meshes, every Nodes dof for curved meshes (default). */
    AUTO,
    /** @brief The element corners only: one cell per element (the Stage 1e behavior). */
    CORNERS,
    /** @brief Every dof of the mesh's Nodes space, with p^3 sub-cells per element (curved
     * meshes; also usable on straight high-order meshes for an h/p resolution). A mesh without
     * Nodes falls back to CORNERS. */
    MESH_NODES
};

/**
 * @brief Device-callable description of the cells of every local element (see the file comment,
 * "Cells"): which field nodes span which cell, and how to find the cell holding a point.
 *
 * Plain data (pointers into the VertexField's arrays plus small fixed tables), copied by value
 * into mfem::forall kernels; valid while the VertexField lives. Obtained from
 * VertexField::CellView(); the pointers come from Read(), so they are device pointers when the
 * mfem::Device runs kernels on a GPU (use VertexField::HostCellView() in host code).
 *
 * **Incidence and neighbor bits.** The solvers store information per (node v, element e) pair
 * ("incidence"), e.g. which nodes of e produced v's distance. Those nodes are encoded as bits of
 * an int ("neighbor bits"):
 *   - corner field: bit j = the j-th vertex of e in element vertex order (j < 8);
 *   - nodal field: every cell containing v lies inside the 3 x 3 x 3 block of lattice points
 *     around v's lattice point L, so bit (di+1) + 3 (dj+1) + 9 (dk+1) means the node at
 *     L + (di, dj, dk), di, dj, dk in {-1, 0, 1} (27 bits; bit 13 is v itself).
 * NeighborNode turns a bit back into a node; NumNeighborBits(e) bounds the bits to scan.
 *
 * **Cell ids.** Corner field: 0 (the element itself). Nodal field: type * (p+1)^3 + anchor, where
 * anchor = a + (p+1) (b + (p+1) c) is the lattice point the cell type's offsets start from and
 * type is 0 for a hex; 0 (up tet), 1-4 (the 4 tets of the octahedron) or 5 (down tet) for a tet.
 * Reference lattice offsets per type are in `type_offset` (see vertex_field.cpp for the tables).
 */
struct FieldCellView {
    static constexpr int MAX_CELL_NODES = 8;      ///< corners of a cell (hex)
    static constexpr int MAX_CELL_TYPES = 6;      ///< cell types (tet lattice)
    static constexpr int NUM_LATTICE_BITS = 27;   ///< neighbor bits of the nodal field

    int curved = 0;         ///< 0: corner field, 1: nodal field
    int order = 1;          ///< p, the order of the nodal lattice (1 for the corner field)
    int stride = 2;         ///< p + 1
    int tet = 0;            ///< nodal field: 1 = tet lattice (i + j + k <= p), 0 = hex lattice
    int num_types = 0;      ///< nodal field: cell types (hex 1, tet 6)
    int nodes_per_cell = 0; ///< nodal field: corners per cell (hex 8, tet 4)
    /** @brief Lattice offsets (di, dj, dk) of corner s of cell type t from the cell's anchor. */
    int type_offset[MAX_CELL_TYPES][MAX_CELL_NODES][3] = {};
    const int* elem_offsets = nullptr;     ///< element -> node CSR offsets (NE+1)
    const int* elem_nodes = nullptr;       ///< element -> node CSR items
    const int* lattice_to_local = nullptr; ///< (p+1)^3: native index of lattice point, -1 if none
    const int* local_to_lattice = nullptr; ///< per native index: packed lattice point
    /** @brief Nodal field: packed lattice point of the node of each incidence k (an entry of the
     * node -> element CSR), so the solvers need not search the element's node list. */
    const int* incidence_lattice = nullptr;
    /** @brief Hex: reference coordinate of lattice index a along axis d, axis_points[3*a + d]. */
    const double* axis_points = nullptr;
    /** @brief Reference coordinates of native node l, ref_nodes[3*l + d]. */
    const double* ref_nodes = nullptr;

    /** @brief Packed lattice index i + (p+1) (j + (p+1) k). Host and device. */
    MFEM_HOST_DEVICE int Pack(int i, int j, int k) const {
        return i + stride * (j + stride * k);
    }

    /** @brief Whether (i, j, k) is a lattice point of the element. Host and device. */
    MFEM_HOST_DEVICE bool ValidLattice(int i, int j, int k) const {
        if (i < 0 || j < 0 || k < 0 || i > order || j > order || k > order) {
            return false;
        }
        return tet == 0 || i + j + k <= order;
    }

    /**
     * @brief Nodal field: the field node at lattice point (i, j, k) of element e, or -1 if
     * (i, j, k) is not a lattice point. Host and device.
     */
    MFEM_HOST_DEVICE int NodeAt(int e, int i, int j, int k) const {
        if (!ValidLattice(i, j, k)) {
            return -1;
        }
        return elem_nodes[elem_offsets[e] + lattice_to_local[Pack(i, j, k)]];
    }

    /**
     * @brief Nodal field: the packed lattice point of node v in element e (a search through the
     * element's node list, at most (p+1)^3 entries), or -1 if v is not a node of e. Corner field:
     * 0. Host and device. The solvers use the precomputed LatticeOfIncidence instead.
     */
    MFEM_HOST_DEVICE int IncidenceLattice(int v, int e) const {
        if (curved == 0) {
            return 0;
        }
        for (int k = elem_offsets[e]; k < elem_offsets[e + 1]; ++k) {
            if (elem_nodes[k] == v) {
                return local_to_lattice[k - elem_offsets[e]];
            }
        }
        return -1;
    }

    /**
     * @brief Lattice point of incidence k (entry k of the node -> element CSR: node v, element
     * VertexElements()[k]) from the precomputed table; 0 for the corner field. Host and device.
     */
    MFEM_HOST_DEVICE int LatticeOfIncidence(int k) const {
        return (curved != 0) ? incidence_lattice[k] : 0;
    }

    /** @brief Number of neighbor bits to scan for element e (see the struct comment). */
    MFEM_HOST_DEVICE int NumNeighborBits(int e) const {
        return (curved != 0) ? NUM_LATTICE_BITS : elem_offsets[e + 1] - elem_offsets[e];
    }

    /**
     * @brief The node of element e encoded by neighbor bit `bit` of the incidence whose node has
     * lattice point `lattice` (IncidenceLattice), or -1 if there is no such node (outside the
     * lattice). Host and device.
     */
    MFEM_HOST_DEVICE int NeighborNode(int e, int lattice, int bit) const {
        if (curved == 0) {
            return elem_nodes[elem_offsets[e] + bit];
        }
        if (lattice < 0) {
            return -1;
        }
        const int li = lattice % stride;
        const int lj = (lattice / stride) % stride;
        const int lk = lattice / (stride * stride);
        return NodeAt(e, li + bit % 3 - 1, lj + (bit / 3) % 3 - 1, lk + bit / 9 - 1);
    }

    /**
     * @brief The corner nodes of cell `cell` of element e, in the cell type's corner order.
     * Host and device.
     * @param nodes out, at least MAX_CELL_NODES entries.
     * @return the number of corners, 0 if `cell` is not a cell of the element.
     */
    MFEM_HOST_DEVICE int CellNodes(int e, int cell, int* nodes) const {
        if (curved == 0) {
            int n = 0;
            for (int k = elem_offsets[e]; k < elem_offsets[e + 1] && n < MAX_CELL_NODES; ++k) {
                nodes[n++] = elem_nodes[k];
            }
            return n;
        }
        const int lattice_size = stride * stride * stride;
        const int t = cell / lattice_size;
        const int anchor = cell % lattice_size;
        const int a = anchor % stride;
        const int b = (anchor / stride) % stride;
        const int c = anchor / (stride * stride);
        for (int s = 0; s < nodes_per_cell; ++s) {
            const int u = NodeAt(e, a + type_offset[t][s][0], b + type_offset[t][s][1],
                                 c + type_offset[t][s][2]);
            if (u < 0) {
                return 0;
            }
            nodes[s] = u;
        }
        return nodes_per_cell;
    }

    /**
     * @brief Calls visit(others, bits, n) once for every cell of element e that has node v as a
     * corner: `others` = the other n corners of the cell, `bits` = their neighbor bits (see the
     * struct comment). Corner field: one call (the element, its other vertices in element order,
     * bit = position). Nodal field: every (cell type, corner slot) whose cell placed with v at
     * that slot lies inside the lattice; a cell is visited once (v is at exactly one slot).
     *
     * Host and device; `visit` must be callable where the caller runs (a lambda defined inside a
     * kernel is). The arrays passed to visit live on the caller's stack.
     *
     * @param lattice the lattice point of v in e (LatticeOfIncidence or IncidenceLattice; ignored
     *        for the corner field).
     */
    template <typename Visit>
    MFEM_HOST_DEVICE void ForEachCellOfIncidence(int v, int e, int lattice, Visit&& visit) const {
        int others[MAX_CELL_NODES - 1];
        int bits[MAX_CELL_NODES - 1];
        if (curved == 0) {
            int n = 0;
            for (int j = elem_offsets[e]; j < elem_offsets[e + 1]; ++j) {
                const int u = elem_nodes[j];
                if (u != v && n < MAX_CELL_NODES - 1) {
                    others[n] = u;
                    bits[n] = j - elem_offsets[e];
                    ++n;
                }
            }
            visit(others, bits, n);
            return;
        }
        if (lattice < 0) {
            return;
        }
        const int li = lattice % stride;
        const int lj = (lattice / stride) % stride;
        const int lk = lattice / (stride * stride);
        for (int t = 0; t < num_types; ++t) {
            for (int s = 0; s < nodes_per_cell; ++s) {
                // Place the cell so that its corner s is v; keep it if every corner exists
                int n = 0;
                bool inside = true;
                for (int q = 0; q < nodes_per_cell && inside; ++q) {
                    if (q == s) {
                        continue;
                    }
                    const int di = type_offset[t][q][0] - type_offset[t][s][0];
                    const int dj = type_offset[t][q][1] - type_offset[t][s][1];
                    const int dk = type_offset[t][q][2] - type_offset[t][s][2];
                    const int u = NodeAt(e, li + di, lj + dj, lk + dk);
                    if (u < 0) {
                        inside = false;
                    } else {
                        others[n] = u;
                        bits[n] = (di + 1) + 3 * (dj + 1) + 9 * (dk + 1);
                        ++n;
                    }
                }
                if (inside) {
                    visit(others, bits, n);
                }
            }
        }
    }

    /**
     * @brief The cell of an element that holds the point with reference coordinates (x, y, z).
     *
     * Corner field: 0. Nodal hex: in each direction the lattice interval
     * [axis_points[a], axis_points[a+1]] holding the coordinate (clamped to the element, so a
     * point slightly outside goes to the nearest boundary cell). Nodal tet: the sub-tet whose
     * reference-space barycentric coordinates of the point have the largest minimum (> 0 means
     * strictly inside, so this picks the containing sub-tet and, for a point on a shared face or
     * slightly outside, the nearest one); ties go to the first cell in (anchor, type) order. The
     * result depends only on the reference coordinates, so the same point gets the same cell on
     * every rank. Host and device.
     */
    MFEM_HOST_DEVICE int LocateCell(double x, double y, double z) const {
        if (curved == 0) {
            return 0;
        }
        if (tet == 0) {
            const double xi[3] = {x, y, z};
            int anchor[3] = {0, 0, 0};
            for (int d = 0; d < 3; ++d) {
                int a = 0;
                while (a < order - 1 && xi[d] >= axis_points[3 * (a + 1) + d]) {
                    ++a;
                }
                anchor[d] = a;
            }
            return Pack(anchor[0], anchor[1], anchor[2]);
        }
        const int lattice_size = stride * stride * stride;
        int best_cell = 0;
        double best_score = -HUGE_VAL;
        for (int c = 0; c < order; ++c) {
            for (int b = 0; b < order; ++b) {
                for (int a = 0; a < order; ++a) {
                    for (int t = 0; t < num_types; ++t) {
                        // Corners of the candidate sub-tet in reference coordinates
                        const double* r[4];
                        bool inside = true;
                        for (int s = 0; s < 4 && inside; ++s) {
                            const int i = a + type_offset[t][s][0];
                            const int j = b + type_offset[t][s][1];
                            const int k = c + type_offset[t][s][2];
                            if (!ValidLattice(i, j, k)) {
                                inside = false;
                            } else {
                                r[s] = &ref_nodes[3 * lattice_to_local[Pack(i, j, k)]];
                            }
                        }
                        if (!inside) {
                            continue;
                        }
                        // Barycentric coordinates: solve [r1-r0 r2-r0 r3-r0] l = p - r0
                        // (Cramer's rule with the scalar triple products)
                        double m[3][3];
                        for (int d = 0; d < 3; ++d) {
                            m[d][0] = r[1][d] - r[0][d];
                            m[d][1] = r[2][d] - r[0][d];
                            m[d][2] = r[3][d] - r[0][d];
                        }
                        const double rhs[3] = {x - r[0][0], y - r[0][1], z - r[0][2]};
                        const double det =
                            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                        if (det == 0.0) {
                            continue;
                        }
                        double l[3];
                        for (int col = 0; col < 3; ++col) {
                            double mc[3][3];
                            for (int d = 0; d < 3; ++d) {
                                for (int q = 0; q < 3; ++q) {
                                    mc[d][q] = (q == col) ? rhs[d] : m[d][q];
                                }
                            }
                            l[col] = (mc[0][0] * (mc[1][1] * mc[2][2] - mc[1][2] * mc[2][1]) -
                                      mc[0][1] * (mc[1][0] * mc[2][2] - mc[1][2] * mc[2][0]) +
                                      mc[0][2] * (mc[1][0] * mc[2][1] - mc[1][1] * mc[2][0])) /
                                     det;
                        }
                        double score = 1.0 - l[0] - l[1] - l[2];
                        for (int q = 0; q < 3; ++q) {
                            score = (l[q] < score) ? l[q] : score;
                        }
                        if (score > best_score) {
                            best_score = score;
                            best_cell = t * lattice_size + Pack(a, b, c);
                        }
                    }
                }
            }
        }
        return best_cell;
    }
};

/**
 * @brief Where each seed starts its distance field (result of VertexField::InitializeSources).
 *
 * All per-seed vectors have one entry per seed (size N, seed index 0-based) and are identical on
 * every rank, except source_element_local and source_cell_local, which are rank-local.
 */
struct SourceReport {
    /**
     * @brief Element key (mesh/element_key.hpp) of each seed's source element: the element that
     * contains the seed (smallest key when the seed lies on a face, edge or vertex shared by
     * several elements), or, for a seed outside the material, an element around the field node
     * nearest to the seed (nearest node: ties by smallest VertexKey of its coordinates; then the
     * smallest element key among the elements sharing that node).
     */
    std::vector<std::uint64_t> source_element_key;
    /**
     * @brief 0-based local index of the source element on the rank that owns it, -1 on every
     * other rank. Exactly one rank holds a non-negative entry per seed.
     */
    std::vector<int> source_element_local;
    /**
     * @brief Cell id (FieldCellView) of the seed's source cell inside its source element on the
     * owning rank, -1 elsewhere. Corner field: 0. Nodal field: the sub-cell holding the seed's
     * reference coordinates, or for a snapped seed the first cell having the nearest node as a
     * corner.
     */
    std::vector<int> source_cell_local;
    /** @brief 1 for a seed that lies in no element and was moved ("snapped") to the nearest one. */
    std::vector<char> seed_outside;
    /** @brief Number of seeds with seed_outside = 1 (same on every rank). */
    int num_seeds_outside = 0;
};

/**
 * @brief Per-node distance and seed label on one rank's mesh partition, plus the node numbering,
 * cells, connectivity tables and rank exchange shared by the geodesic solvers (see the file
 * comment).
 *
 * Host object. Not copyable (it owns an MFEM communicator). The
 * ParMesh must outlive the VertexField and must not change (no refinement, no node motion) while
 * it is used.
 */
class VertexField {
public:
    /**
     * @brief Chooses the field nodes, builds their numbering, the CSR tables, the coordinates
     * and the shared/owned markers, and sets T = +infinity, label = -1 everywhere.
     *
     * **Collective** over the mesh communicator (straightness test, space and communicator
     * construction, coordinate broadcast). A rank with no elements takes part with NE = 0.
     *
     * @param pmesh conforming 3D ParMesh (any element types for the corner field; the vote
     *        sampling additionally needs a single hex or tet geometry). Its Nodes, if any, must be
     *        a continuous (H1) space.
     * @param nodes which field (default AUTO: corners for straight-sided meshes, nodal field
     *        for curved ones).
     * @param mask optional active elements (mesh/element_mask.hpp; copied): only active elements
     *        enter the node -> element table, so the solvers walk the cells of active elements
     *        only and a face to an inactive element is a free surface; nodes of inactive elements
     *        only are never reached (NodeUsed() = 0) and are ignored by the statistics and by the
     *        snapping of outside seeds; inactive elements get label -1 in LabelElementCentroids /
     *        LabelElementsByVote. The element -> node table stays complete. nullptr = every
     *        element (unchanged).
     * @throws std::runtime_error on every rank if the mesh is nonconforming (hanging vertices
     *         from local refinement: their values are not plain copies of one node), not 3D, or
     *         has discontinuous Nodes, or, for the nodal field, if the Nodes space is not a
     *         fixed-order ParFiniteElementSpace or the mesh is not all hexes or all tets.
     */
    explicit VertexField(mfem::ParMesh& pmesh, FieldNodes nodes = FieldNodes::AUTO,
                         const ElementMask* mask = nullptr);
    ~VertexField();
    VertexField(const VertexField&) = delete;
    VertexField& operator=(const VertexField&) = delete;

    /** @brief The mesh the field lives on. */
    mfem::ParMesh& Mesh() const {
        return m_pmesh;
    }
    /** @brief The mesh communicator (ParMesh::GetComm()); every collective call uses it. */
    MPI_Comm Comm() const;
    /** @brief True for the nodal field (every Nodes dof, sub-cells), false for corners. */
    bool UsesMeshNodes() const {
        return m_curved;
    }
    /** @brief Order p of the nodal lattice (1 for the corner field). */
    int LatticeOrder() const {
        return m_order;
    }
    /** @brief Number of local field nodes NV (corners or Nodes dofs). */
    int NumVertices() const {
        return m_num_vertices;
    }
    /** @brief Number of local elements NE. */
    int NumElements() const {
        return m_num_elements;
    }
    /** @brief True when local element e is active (always without a mask). */
    bool ElementActive(int e) const {
        return m_mask.IsActive(e);
    }
    /** @brief True when this rank holds a non-empty element mask. */
    bool HasMask() const {
        return m_mask.active.Size() != 0;
    }
    /**
     * @brief 1 for a node of at least one active element on some rank, 0 for a node of inactive
     * elements only (identical on every copy; all 1 without a mask). Size NV.
     */
    const mfem::Array<int>& NodeUsed() const {
        return m_node_used;
    }

    /** @brief Element -> node CSR offsets, size NE+1 (see the file comment). */
    const mfem::Array<int>& ElementVertexOffsets() const {
        return m_elem_vert_offsets;
    }
    /** @brief Element -> node CSR items: the local node ids of each element. */
    const mfem::Array<int>& ElementVertices() const {
        return m_elem_verts;
    }
    /** @brief Node -> element CSR offsets, size NV+1. */
    const mfem::Array<int>& VertexElementOffsets() const {
        return m_vert_elem_offsets;
    }
    /** @brief Node -> element CSR items: the local elements containing each node. */
    const mfem::Array<int>& VertexElements() const {
        return m_vert_elems;
    }
    /**
     * @brief Device view of the cells (pointers from Read(): valid wherever the mfem::Device
     * runs kernels). Not collective.
     */
    FieldCellView CellView() const;
    /** @brief The same view with host pointers (HostRead()), for host code. Not collective. */
    FieldCellView HostCellView() const;
    /**
     * @brief Node coordinates, size 3*NV, layout `xyz[3*v + d]` (d = 0, 1, 2 for x, y, z; mesh
     * length units). Bitwise identical on every rank that holds a copy of the node.
     */
    const mfem::Vector& Coordinates() const {
        return m_coords;
    }
    /**
     * @brief 1 for a node that also exists on another rank (it appears in a GroupCommunicator
     * group other than the rank-local group 0), else 0. Size NV.
     */
    const mfem::Array<int>& SharedVertexMarker() const {
        return m_shared;
    }
    /**
     * @brief 1 on exactly one rank per node (the rank owning its "true dof", i.e. the group
     * master), else 0. Size NV. Use it to count nodes globally without double counting.
     */
    const mfem::Array<int>& OwnedVertexMarker() const {
        return m_owned;
    }

    /** @brief T(v): distance from the nearest seed per local node, +infinity = not reached. */
    mfem::Vector& Distance() {
        return m_T;
    }
    /** @brief Read-only T(v). */
    const mfem::Vector& Distance() const {
        return m_T;
    }
    /** @brief label(v): 0-based seed index per local node, -1 = none. */
    mfem::Array<int>& Label() {
        return m_label;
    }
    /** @brief Read-only label(v). */
    const mfem::Array<int>& Label() const {
        return m_label;
    }
    /**
     * @brief Initial (source) values from the last InitializeSources: T0(v) = |v - s_i| - a_i for
     * the corners of each seed's source cell (lexicographic minimum over seeds), +infinity
     * elsewhere. Identical on all copies of a node. A solver keeps these nodes fixed or uses them
     * as boundary values; the graph labeler uses them to tell which nodes start a path.
     */
    const mfem::Vector& SourceDistance() const {
        return m_source_T;
    }
    /** @brief Seed index of SourceDistance() per node, -1 where SourceDistance() is +infinity. */
    const mfem::Array<int>& SourceLabel() const {
        return m_source_label;
    }

    /**
     * @brief Sets T = +infinity and label = -1 on every node, and clears the source values and
     * the per-element seed lists.
     * Not collective.
     */
    void Reset();

    /**
     * @brief Places the seeds on the mesh and sets the starting values of the field.
     *
     * 1. Source element of seed i: the element that contains s_i (ContainingElementKeys of
     *    seeding/seed_element_locator.hpp: BVH plus the exact point-in-element test, smallest
     *    element key on ties, reduced over ranks). For a seed in no element: the field node
     *    nearest to s_i (smallest squared distance, ties by smallest VertexKey of the node's
     *    coordinates, over all ranks, using Coordinates()), then the smallest key among the
     *    elements of every rank that contain that node (MPI_Allreduce(MIN)).
     * 2. Source cell inside the source element (owning rank only): corner field, the element;
     *    nodal field, the sub-cell holding the seed's reference coordinates
     *    (mfem::InverseElementTransformation, then FieldCellView::LocateCell), or for a snapped
     *    seed the first cell having the nearest node as a corner.
     * 3. Every corner v of seed i's source cell gets the candidate (|v - s_i| - a_i, i); a node
     *    touched by several seeds keeps the lexicographic minimum (smallest value, then smallest
     *    seed index).
     * 4. T and label are set to these values (+infinity / -1 elsewhere), ReduceShared() makes all
     *    copies agree, and the result is stored as SourceDistance() / SourceLabel().
     * 5. Each source element remembers its seeds (position, a_i and source cell) on the rank that
     *    owns it, so LabelPoints can use the straight line from the seed inside the source cell.
     * Previous field values are discarded.
     *
     * For the Voronoi metric a_i = 0 and T0 is the straight-line distance; for the Additive metric
     * the grain of seed i is the set where (distance - a_i) is smallest, so the offset is applied
     * once, at the source. Collective; host.
     *
     * @param seeds            replicated seed set, N >= 1, finite positions (mesh length units).
     * @param additive_weights a_i per seed (length units), size N, or empty for all zero.
     * @param context          optional shared locator (MaterialContext) built with the same mask
     *                         as the field, for the containment test; nullptr = build one over
     *                         the field's mask.
     * @return the source elements and cells (see SourceReport).
     * @throws std::invalid_argument (on every rank, the inputs being replicated) for an empty seed
     *         set, a non-finite position or weight, or a weight array of the wrong size;
     *         std::runtime_error on every rank if the mesh has no element on any rank.
     */
    SourceReport InitializeSources(const SeedSet& seeds,
                                   const std::vector<double>& additive_weights,
                                   const MaterialContext* context = nullptr);

    /**
     * @brief Makes all copies of every shared node equal to the lexicographic minimum of
     * (T, label) over the copies.
     *
     * Steps (MFEM's GroupCommunicator, communication.hpp:534 Reduce, :435 Bcast; the reduce
     * gathers each group's copies on the group's master rank and applies the operation there,
     * Bcast then sends the master's value back to every copy; the Min operation is
     * communication.cpp:1548):
     *  1. Min-reduce plus Bcast of T: every copy holds the smallest T.
     *  2. On each rank, label(v) = INT_MAX wherever the local T differed from that smallest T
     *     (this copy lost and must not contribute its label).
     *  3. Min-reduce plus Bcast of the labels: the smallest label among the copies that had the
     *     smallest T.
     * Unshared nodes are not touched.
     *
     * **Collective** (MPI_Allreduce for the return value). Host (moves T and label to the host).
     *
     * @param changed optional, resized to NV: changed[v] = 1 where T(v) or label(v) changed on
     *        this rank, else 0 (a solver restarts its local sweep from these nodes).
     * @return true if any node changed on any rank (same value on every rank).
     */
    bool ReduceShared(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Step 1 of ReduceShared only: every copy of a shared node gets the smallest T over
     * the copies; labels are left as they are. For solvers that settle T first and labels later.
     * Collective. Host.
     * @param changed optional, resized to NV: 1 where T(v) changed on this rank.
     * @return true if any T changed on any rank.
     */
    bool ReduceSharedDistance(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Min-reduce plus Bcast of the labels only (T is not used or changed): every copy of a
     * shared node gets the smallest label over the copies. Collective. Host.
     * @param changed optional, resized to NV: 1 where label(v) changed on this rank.
     * @return true if any label changed on any rank.
     */
    bool ReduceSharedLabels(mfem::Array<int>* changed = nullptr);

    /**
     * @brief Min-reduce plus Bcast of an arbitrary per-node array (not the field's own T or
     * label): every copy of a shared node gets the smallest value over the copies. Unshared
     * nodes keep their value. Used by solvers that need a per-node quantity other than T to
     * agree across ranks (the FIM label pass reduces the smallest recomputed update value).
     *
     * **Collective** (every rank must call it, also ranks without nodes). Host (moves `values`
     * to the host, HostReadWrite()). The same GroupCommunicator Reduce(Min) + Bcast pair as
     * ReduceShared.
     *
     * @param values per-node values, size NV (any finite or infinite doubles, no NaN).
     * @throws std::invalid_argument if values.Size() != NV (thrown on that rank only, before any
     *         communication; callers pass arrays sized from NumVertices(), so all ranks agree).
     */
    void ReduceSharedMinimum(mfem::Vector& values) const;

    /**
     * @brief Labels arbitrary points of a range of local elements from the field.
     *
     * Point i belongs to local element e = elem_begin + i / points_per_element and to the cell c
     * of e that holds it (corner field: the element; nodal field: the sub-cell holding the point's
     * reference coordinates, found by inverting the element map on the host). It is reached in a
     * straight line from one of c's corners, or, when c is the source cell of seed j, directly
     * from the seed. Its distance and label are the lexicographic minimum of
     *   (T(v) + |p - v|, label(v))   over the corners v of c, and
     *   (|p - s_j| - a_j, j)          over the seeds j whose source cell is c,
     * i.e. the smallest arrival distance, then the smallest seed index. The direct term makes two
     * seeds in the same or neighboring cells compare by straight lines instead of by detours
     * through the corners. A point whose cell has no reached corner and no seed gets distance
     * +infinity and label -1.
     *
     * Not collective (element-local). An mfem::forall kernel over the points (runs where the
     * mfem::Device runs kernels; for the nodal field the inversion before it runs on the host);
     * the field must be consistent across ranks first (after a ReduceShared that returned false)
     * for the result to be partition independent.
     *
     * @param elem_begin         first 0-based local element.
     * @param elem_end           one past the last; 0 <= elem_begin <= elem_end <= NE.
     * @param points_per_element points per element, >= 1 (unused when the range is empty).
     * @param x, y, z            point coordinates, (elem_end - elem_begin) * points_per_element
     *                           entries each, element-major: entry (e - elem_begin) *
     *                           points_per_element + q is point q of element e (the
     *                           QuadraturePointSamples layout of mesh/mesh_sampler.hpp).
     * @param label              resized to the number of points: 0-based seed index or -1.
     * @param distance           optional, resized likewise: the arrival distance.
     * @throws std::invalid_argument for an invalid range, points_per_element < 1 with a
     *         non-empty range, or coordinate arrays of the wrong size.
     */
    void LabelPoints(int elem_begin,
                     int elem_end,
                     int points_per_element,
                     const mfem::Vector& x,
                     const mfem::Vector& y,
                     const mfem::Vector& z,
                     mfem::Array<int>& label,
                     mfem::Vector* distance = nullptr) const;

    /**
     * @brief Element label = LabelPoints at the element's volume centroid (ComputeElementSamples
     * of mesh/mesh_sampler.hpp, the same centroid the Euclidean centroid labeler uses).
     *
     * Not collective. Host setup plus device kernels.
     *
     * @param element_label    resized to NE: 0-based seed index, -1 for an unreached element.
     * @param element_distance optional, resized to NE: arrival distance at the centroid.
     */
    void LabelElementCentroids(mfem::Array<int>& element_label,
                               mfem::Vector* element_distance = nullptr) const;

    /**
     * @brief Element label by a volume-weighted vote of subcell points, the geodesic counterpart
     * of LabelElementsByQuadratureVote (labeling/quadrature_vote.hpp).
     *
     * The voting points are VoteRule(geometry, levels) evaluated by a QuadraturePointSampler with
     * weights w_q detJ_q; every point is labeled by LabelPoints (for the nodal field the rule's
     * reference points locate the sub-cells directly, no inversion); each element goes to the
     * label with the largest total weight, exact ties to the lowest seed index (totals summed in
     * increasing point order, so identical on every partition). Points labeled -1 do not vote;
     * an element none of whose points is reached gets -1.
     *
     * Not collective. Host setup plus device kernels, in chunks of whole elements.
     *
     * @param levels               subdivision levels 0..3 (8^levels points per element).
     * @param max_points_per_chunk upper bound on points processed at once (>= 1).
     * @param element_label        resized to NE: winning seed index or -1.
     * @return number of local elements whose voting points carried more than one label.
     * @throws std::invalid_argument for levels outside 0..3, max_points_per_chunk < 1 or an
     *         element geometry other than hex/tet; std::runtime_error for a mixed-geometry mesh.
     */
    int LabelElementsByVote(int levels,
                            int max_points_per_chunk,
                            mfem::Array<int>& element_label) const;

private:
    /** @brief Corner field numbering (mesh vertices), CSR tables, coordinates, markers. */
    void BuildCornerField(const mfem::GridFunction* nodes);
    /** @brief Nodal field numbering (the Nodes space's dofs), lattice tables, CSR, markers. */
    void BuildNodalField(mfem::GridFunction& nodes);
    /** @brief Node -> element CSR as the transpose of the element -> node table. */
    void BuildNodeToElementTable();
    /** @brief The owner's coordinates on every copy of a shared node (Bcast per component). */
    void BroadcastCoordinates();
    /** @brief Shared marker from the communicator's group table (rows 1..). */
    void MarkSharedNodes();
    /**
     * @brief Nodal field: the sub-cell (FieldCellView cell id) holding each physical point of a
     * range of elements: the element map is inverted (host) and the reference point located with
     * FieldCellView::LocateCell. Same point layout as LabelPoints.
     */
    void PointCells(int elem_begin, int elem_end, int points_per_element, const mfem::Vector& x,
                    const mfem::Vector& y, const mfem::Vector& z, mfem::Array<int>& cells) const;
    /**
     * @brief LabelPoints with given sub-cells: `cells` holds point i's cell at index i, or at
     * i % points_per_element when `cells_repeat` (one rule for all elements); null = cell 0 (the
     * corner field's only cell).
     */
    void LabelPointsImpl(int elem_begin, int elem_end, int points_per_element,
                         const mfem::Vector& x, const mfem::Vector& y, const mfem::Vector& z,
                         const mfem::Array<int>* cells, bool cells_repeat,
                         mfem::Array<int>& label, mfem::Vector* distance) const;
    /** @brief Shared implementation of the three ReduceShared variants. */
    bool ReduceSharedImpl(bool reduce_T, bool reduce_labels, mfem::Array<int>* changed);

    mfem::ParMesh& m_pmesh;                          ///< the mesh (not owned)
    bool m_curved = false;                           ///< nodal field?
    int m_order = 1;                                 ///< lattice order p
    mfem::ParFiniteElementSpace* m_node_fes = nullptr;  ///< nodal field: the mesh's Nodes space
    /** @brief Exchange communicator: corner field, built over the mesh's shared-vertex groups;
     * nodal field, ScalarGroupComm() of the Nodes space. */
    std::unique_ptr<mfem::GroupCommunicator> m_owned_gcomm;
    const mfem::GroupCommunicator* m_gcomm = nullptr; ///< communicator used by the exchanges
    int m_num_vertices = 0;                          ///< NV
    int m_num_elements = 0;                          ///< NE
    mfem::Array<int> m_elem_vert_offsets;            ///< element -> node CSR offsets (NE+1)
    mfem::Array<int> m_elem_verts;                   ///< element -> node CSR items
    mfem::Array<int> m_vert_elem_offsets;            ///< node -> element CSR offsets (NV+1)
    mfem::Array<int> m_vert_elems;                   ///< node -> element CSR items
    mfem::Array<int> m_vert_elem_lattice;            ///< nodal field: lattice point per incidence
    mfem::Vector m_coords;                           ///< xyz[3*v + d]
    mfem::Array<int> m_shared;                       ///< 1 = node also on another rank
    mfem::Array<int> m_owned;                        ///< 1 = this rank is the node's master
    ElementMask m_mask;                              ///< copy of the mask (empty = all active)

    /** The stored mask, or nullptr without one (the convention of mesh/element_mask.hpp). */
    const ElementMask* MaskOrNull() const {
        return HasMask() ? &m_mask : nullptr;
    }
    mfem::Array<int> m_node_used;                    ///< 1 = node of an active element (global)
    mfem::Vector m_T;                                ///< T(v)
    mfem::Array<int> m_label;                        ///< label(v)
    mfem::Vector m_source_T;                         ///< T0(v)
    mfem::Array<int> m_source_label;                 ///< seed of T0(v)
    mfem::Array<int> m_elem_src_offsets;             ///< element -> own seeds CSR offsets (NE+1)
    mfem::Array<int> m_elem_src_seeds;               ///< seed index of each entry
    mfem::Array<int> m_elem_src_cell;                ///< source cell id of each entry
    mfem::Vector m_elem_src_data;                    ///< per entry: x, y, z, a_j (4 doubles)
    // Nodal field lookup tables (FieldCellView); empty for the corner field
    int m_tet = 0;                                   ///< 1 = tet lattice
    mfem::Array<int> m_lattice_to_local;             ///< (p+1)^3, -1 = not a lattice point
    mfem::Array<int> m_local_to_lattice;             ///< per native index
    mfem::Vector m_axis_points;                      ///< hex: 3*(p+1) reference coordinates
    mfem::Vector m_ref_nodes;                        ///< 3 * nodes per element
};

} // namespace exaconstit::microstructure
