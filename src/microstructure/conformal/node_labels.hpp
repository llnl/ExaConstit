#pragma once

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <utility>
#include <vector>

#include "microstructure/labeling/vertex_field.hpp"

/**
 * @file node_labels.hpp
 * @brief Which grains meet at every mesh node, and how many surfaces (grain boundaries and the
 * outer boundary of the mesh) the node must stay on: the first step of the conformal fit.
 *
 * **What it is for.** The conformal fit moves the mesh nodes so that element faces lie on the
 * grain boundaries. Before a node can be moved it has to be known what it must stay on: a node
 * inside a grain is free, a node between two grains must stay on their boundary surface, a node
 * where three grains meet must stay on the line where the three boundaries meet (a *triple line*),
 * and a node where four grains meet is pinned to a point (a *quad point*). A node on the outer
 * boundary of the mesh must in addition stay on that boundary: on a face of a box, on one of its
 * edges, or at a corner. This file computes, for every node:
 *   - its *label set*: the sorted, distinct grain ids of the elements that contain it;
 *   - its *boundary surfaces*: how many distinct surfaces of the outer boundary meet at it
 *     (0 inside the mesh, 1 on a smooth part of the surface, 2 on a feature edge, 3 or more at a
 *     corner);
 *   - its *constraint count* (below) and whether that is more than a point in 3D can satisfy.
 *
 * **Which nodes.** The nodes the fitter will move: the degrees of freedom of the mesh's Nodes
 * space when the mesh has Nodes (every nodal point of a curved mesh, including those inside
 * element edges and faces), the vertices otherwise. (For a mesh without Nodes the fitter gives the
 * mesh order-1 Nodes, whose degrees of freedom are numbered like the vertices.) The numbering, the
 * coordinates and the exchange between ranks are those of a VertexField with
 * FieldNodes::MESH_NODES (labeling/vertex_field.hpp), so node v here is node v there.
 *
 * **Label sets.** (A *label* is the grain id of an element.) The label set of node v is the union
 * of the labels of all elements that contain v, over all ranks: each rank collects the labels of
 * its own elements, the copies of every shared node are exchanged
 * (VertexField::ExchangeSharedLists) and each copy keeps the sorted union of all copies, with the
 * number of elements of each label (summed over the copies). A union of sets and a sum of counts do
 * not depend on the order in which they arrive, so all copies end up identical and the result does
 * not depend on the rank count or the partition. There is no fixed limit on the size of a set. The
 * sets are stored as an *offsets array*: all lists one after another in Labels(), and the list of
 * node v starting at LabelOffsets()[v] (often called CSR, compressed sparse row, storage).
 *
 * **Boundary surfaces.** The outer boundary is the set of faces that belong to one element only
 * (faces shared between ranks are not boundary faces). For every boundary face its unit outward
 * normal is evaluated at the face center (through the face transformation, so it follows curved
 * Nodes). A node collects the normals of all boundary faces that contain it, again over all ranks
 * (each boundary face belongs to exactly one rank, so every copy receives the same bits). Two of
 * these faces lie on the *same surface* when their normals differ by at most the feature angle
 * (NodeLabelOptions::feature_angle_degrees), and "same surface" is extended through chains: the
 * surfaces at the node are the groups of normals connected by such steps (connected components,
 * which do not depend on the order of the normals). On a box this gives 1 on a face, 2 on an edge
 * and 3 at a corner; on a smoothly curved surface meshed with flat faces, neighboring faces
 * differ by less than the feature angle and the node sees 1 surface. Two opposite surfaces of a
 * one-element-thick plate count as 2. Each surface keeps a normal and whether it is planar at the
 * node (all its face normals agree), which the target computation (node_targets.hpp) uses to
 * keep nodes on flat boundaries exactly. The normals of a node are sorted before they are
 * grouped, so every copy computes the same bits.
 *
 * **Fixed labels (passthrough parts).** Labels listed in NodeLabelOptions::fixed_labels belong to
 * parts whose shape must not change (e.g. a build plate: a passthrough attribute of the
 * generator). Their elements still deform with the mesh, but the faces between an element with a
 * fixed label and an element with another label are treated exactly like the outer boundary:
 * they are *interface faces*, they add their normals to the boundary surfaces of their nodes (and
 * their triangles to the boundary patches, SurfaceFaces()), so nodes slide along them and never
 * cross them. A fixed label is then dropped from the label set of every node that also has a
 * non-fixed label; a node with fixed labels only keeps the smallest one. So a grain boundary
 * that meets the plate is a grain boundary node on a boundary surface, and a node on the plate's
 * top face inside one grain is a one-label node on a surface. Each interface face is counted
 * once over all ranks: by the rank of its non-fixed element (of the smaller label when both are
 * fixed), with the normal pointing out of that element.
 *
 * **Pinches.** A grain *pinches* at a node when its elements around the node form two or more
 * groups that do not share a face through the node: the grain touches itself only at the node, or
 * along an edge through it (a 2 x 2 checkerboard around an edge). Its boundaries there cannot be
 * fitted (the node would have to be on two sheets of the grain's boundary at once). For every
 * node each rank lists its elements around the node (element key, label) and the pairs of them
 * that share a face containing the node (ghost keys and labels from MeshAdjacency); the lists of
 * all copies are merged and every copy counts the face-connected groups per label, so Pinched()
 * is the same on every copy for any partition. Fixed labels are ignored.
 *
 * **Constraints.** Each surface a node must stay on removes one direction in which it can move:
 *   - grain constraints = (number of labels) - 1 (0 inside a grain, 1 on a grain boundary face,
 *     2 on a triple line, 3 at a quad point);
 *   - boundary constraints = min(boundary surfaces, 3);
 *   - constraints = grain constraints + boundary constraints.
 * A node with more than 3 constraints is *over-constrained*: in 3D no position satisfies all of
 * them in general (e.g. a quad point on the outer boundary, or a grain boundary that meets a box
 * corner). Such nodes are counted and flagged here; label repair (label_repair.hpp) decides what
 * to do with them.
 *
 * **Where it runs.** Host only (the conformal fit runs on the CPU), MPI parallel. The constructor
 * and SetElementLabels are collective over the mesh communicator; the accessors are local.
 */

namespace exaconstit::microstructure {

/** @brief Face neighbors of the elements (mesh/mesh_adjacency.hpp); declared here so this
 * header does not need to include it. */
class MeshAdjacency;

/** @brief Settings of NodeLabels. */
struct NodeLabelOptions {
    /**
     * @brief Two boundary faces at a node belong to the same surface when their normals differ by
     * at most this angle (degrees, in (0, 90)). A starting value, not a measured one: 90-degree
     * box edges are always features, and a curved surface needs more than 360 / 30 = 12 flat
     * faces around its circumference to be seen as one surface.
     */
    double feature_angle_degrees = 30.0;
    /** @brief Which nodes (see the file comment). MESH_NODES = the nodes the fitter moves. */
    FieldNodes nodes = FieldNodes::MESH_NODES;
    /**
     * @brief Labels of parts whose interfaces must stay in place (passthrough parts; see the
     * file comment), ascending. Empty: none.
     */
    std::vector<int> fixed_labels;
};

/** @brief How many grains meet at a node; the value is the number of labels - 1, capped at 4. */
enum class JunctionKind : int {
    INTERIOR = 0,    ///< one grain
    FACE = 1,        ///< two grains: on a grain boundary face
    TRIPLE_LINE = 2, ///< three grains
    QUAD_POINT = 3,  ///< four grains
    MULTI = 4        ///< five or more grains (always over-constrained)
};

/** @brief Number of JunctionKind values. */
inline constexpr int NUM_JUNCTION_KINDS = 5;
/** @brief Number of possible boundary constraint counts, 0..3 (more surfaces count as 3). */
inline constexpr int NUM_BOUNDARY_CONSTRAINTS = 4;

/** @brief Node counts over the whole mesh, for the log (each node counted once, on the rank that
 * owns it; the same on every rank). */
struct NodeLabelSummary {
    /** @brief count[kind][b]: nodes of JunctionKind `kind` with b boundary constraints. */
    std::array<std::array<std::int64_t, NUM_BOUNDARY_CONSTRAINTS>, NUM_JUNCTION_KINDS> count{};
    /** @brief Nodes with more than 3 constraints. */
    std::int64_t over_constrained = 0;
    /** @brief Largest label set of any node. */
    int max_labels = 0;
    /** @brief Total number of nodes. */
    std::int64_t num_nodes = 0;
};

/**
 * @brief The label set, boundary surfaces and constraint count of every node (see the file
 * comment). All results are identical on every copy of a shared node and do not depend on the
 * rank count or partition.
 */
class NodeLabels {
public:
    /**
     * @brief Builds the node numbering, the label sets, the boundary surfaces and the summary.
     *
     * **Collective** over the mesh communicator. A rank with no elements takes part.
     * @param pmesh         conforming 3D ParMesh (see VertexField for the supported meshes).
     * @param element_label per local element, its grain id (>= 1; e.g.
     *                      MicrostructureResult::element_grain). Any positive ids work; they need
     *                      not be compact.
     * @param options       feature angle and node choice.
     * @throws std::invalid_argument on every rank if element_label has the wrong size or a label
     *         < 1 on some rank, or the feature angle is not in (0, 90); std::runtime_error as
     *         the VertexField constructor.
     */
    NodeLabels(mfem::ParMesh& pmesh,
               const mfem::Array<int>& element_label,
               const NodeLabelOptions& options = {});
    /** @brief Defined in the source file, where MeshAdjacency is a complete type. */
    ~NodeLabels();
    /** @brief Not copyable: owns the node field and the adjacency. */
    NodeLabels(const NodeLabels&) = delete;
    /** @brief Not copyable. */
    NodeLabels& operator=(const NodeLabels&) = delete;

    /**
     * @brief Recomputes the label sets and the summary for new element labels on the same mesh
     * (label repair, label_repair.hpp). The node numbering and the boundary surfaces are kept.
     * **Collective**. Same requirements and errors as the constructor's element_label.
     */
    void SetElementLabels(const mfem::Array<int>& element_label);

    /** @brief Node numbering, coordinates, which nodes this rank owns, and the exchange between
     * the copies of shared nodes. */
    const VertexField& Field() const {
        return *m_field;
    }
    /** @brief Number of nodes on this rank (owned and shared copies). */
    int NumNodes() const {
        return m_field->NumVertices();
    }

    /** @brief Offsets of the label sets in Labels(), NumNodes() + 1 entries. */
    const mfem::Array<int>& LabelOffsets() const {
        return m_label_offsets;
    }
    /** @brief All label sets one after another: the labels of node v, ascending, are entries
     * LabelOffsets()[v] .. LabelOffsets()[v + 1] - 1. */
    const mfem::Array<int>& Labels() const {
        return m_labels;
    }
    /**
     * @brief Per entry of Labels(): how many elements (over all ranks) that contain the node have
     * that label. Label repair uses it to see whether switching an element removes a label from
     * a node.
     */
    const mfem::Array<int>& LabelCounts() const {
        return m_label_counts;
    }
    /** @brief The element labels the label sets were built from (per local element). */
    const mfem::Array<int>& ElementLabels() const {
        return m_element_label;
    }
    /** @brief Cosine of the feature angle: two face normals belong to one surface when their dot
     * product is at least this. */
    double FeatureCosine() const {
        return m_feature_cosine;
    }
    /** @brief The fixed labels (NodeLabelOptions::fixed_labels), ascending. */
    const std::vector<int>& FixedLabels() const {
        return m_fixed_labels;
    }
    /** @brief True when label l is one of the fixed labels. */
    bool IsFixedLabel(int l) const;
    /**
     * @brief Per local face (Mesh::GetNumFaces): 1 when this rank contributes the face as a
     * boundary surface face: an outer boundary face (one element, not shared) or an interface
     * face of a fixed label counted by this rank (see the file comment). Used for the boundary
     * patches (boundary_projector.hpp).
     */
    const std::vector<char>& SurfaceFaces() const {
        return m_surface_faces;
    }
    /** @brief True when a label of node v pinches there (see the file comment). */
    bool Pinched(int v) const {
        return m_pinched[v] != 0;
    }
    /** @brief Number of distinct labels at node v (>= 1). */
    int NumLabels(int v) const {
        return m_label_offsets[v + 1] - m_label_offsets[v];
    }
    /** @brief Distinct surfaces of the outer boundary at node v (0 = not on the boundary). */
    int BoundarySurfaces(int v) const {
        return m_boundary_surfaces[v];
    }
    /**
     * @brief Offsets of the boundary surfaces, NumNodes() + 1 entries: the surfaces of node v are
     * numbers SurfaceOffsets()[v] .. SurfaceOffsets()[v + 1] - 1 in SurfaceNormals() and
     * SurfacePlanar().
     */
    const mfem::Array<int>& SurfaceOffsets() const {
        return m_surface_offsets;
    }
    /**
     * @brief Unit outward normal of each surface, 3 doubles per surface (surface k at 3 k). For a
     * planar surface it is one of its face normals (exact axis normals stay exact); otherwise
     * the normalized sum of its face normals.
     */
    const mfem::Vector& SurfaceNormals() const {
        return m_surface_normals;
    }
    /** @brief Per surface: 1 when all its face normals at the node agree to 1e-12 (the surface
     * is flat there), else 0. */
    const mfem::Array<int>& SurfacePlanar() const {
        return m_surface_planar;
    }
    /** @brief Directions node v loses to the outer boundary: min(BoundarySurfaces(v), 3). */
    int BoundaryConstraints(int v) const {
        return m_boundary_surfaces[v] < 3 ? m_boundary_surfaces[v] : 3;
    }
    /** @brief Directions node v loses to grain boundaries: NumLabels(v) - 1. */
    int GrainConstraints(int v) const {
        return NumLabels(v) - 1;
    }
    /** @brief All constraints of node v: GrainConstraints(v) + BoundaryConstraints(v). */
    int Constraints(int v) const {
        return GrainConstraints(v) + BoundaryConstraints(v);
    }
    /** @brief Number of pinched nodes over all ranks (each node once). */
    std::int64_t NumPinched() const {
        return m_num_pinched;
    }
    /** @brief True when node v has more than 3 constraints (no point satisfies them all). */
    bool OverConstrained(int v) const {
        return Constraints(v) > 3;
    }
    /** @brief How many grains meet at node v. */
    JunctionKind Kind(int v) const;

    /** @brief The node counts over all ranks (the same on every rank). */
    const NodeLabelSummary& Summary() const {
        return m_summary;
    }

private:
    /** @brief Label sets and label counts of every node from the element labels. Collective. */
    void BuildLabelSets(const mfem::Array<int>& element_label);
    /** @brief Marks the faces this rank counts as boundary surface faces (outer boundary and
     * interfaces of fixed labels). Collective when there are fixed labels. */
    void FindSurfaceFaces();
    /** @brief Flags the nodes where a grain pinches. Collective. */
    void FindPinches();
    /** @brief Boundary surfaces (normals, planarity) of every node. Collective. */
    void BuildBoundarySurfaces(double feature_angle_degrees);
    /** @brief The global node counts. Collective. */
    void BuildSummary();

    std::unique_ptr<VertexField> m_field;       ///< node numbering, coordinates and exchange
    mfem::Array<int> m_element_label;           ///< grain of each local element (current labels)
    std::vector<int> m_fixed_labels;            ///< fixed labels, ascending
    double m_feature_cosine = 0.0;              ///< cosine of the feature angle
    std::vector<char> m_surface_faces;          ///< per local face: 1 = counted surface face
    mfem::Array<int> m_pinched;                 ///< per node: 1 = pinched
    std::unique_ptr<MeshAdjacency> m_adjacency; ///< face neighbors; built once (fixed topology)
    std::vector<std::uint64_t> m_keys;          ///< element keys (mesh/element_key.hpp)
    std::vector<std::uint64_t> m_ghost_keys;    ///< keys of the ghost elements
    /** @brief Interface faces counted here: (local face, local element they point out of). */
    std::vector<std::pair<int, int>> m_interface_faces;
    mfem::Array<int> m_label_offsets;     ///< see LabelOffsets()
    mfem::Array<int> m_labels;            ///< see Labels()
    mfem::Array<int> m_label_counts;      ///< see LabelCounts()
    mfem::Array<int> m_boundary_surfaces; ///< per node: number of boundary surfaces
    mfem::Array<int> m_surface_offsets;   ///< see SurfaceOffsets()
    mfem::Vector m_surface_normals;       ///< see SurfaceNormals()
    mfem::Array<int> m_surface_planar;    ///< see SurfacePlanar()
    NodeLabelSummary m_summary;           ///< see Summary()
    std::int64_t m_num_pinched = 0;       ///< see NumPinched()
};

} // namespace exaconstit::microstructure
