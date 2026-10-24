#pragma once

#include "microstructure/labeling/vertex_field.hpp"

#include "mfem.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mpi.h>

/**
 * @file node_labels.hpp
 * @brief Stage 2a of conformal fitting: which grains meet at every mesh node, and how many
 * surfaces (grain boundaries and the outer boundary of the mesh) the node must stay on.
 *
 * **What it is for.** Stage 2 moves the mesh nodes so that element faces lie on the grain
 * boundaries. Before a node can be moved it has to be known what it must stay on: a node inside a
 * grain is free, a node between two grains must stay on their boundary surface, a node where
 * three grains meet must stay on the line where the three boundaries meet (a *triple line*), and
 * a node where four grains meet is pinned to a point (a *quad point*). A node on the outer
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
 * **Label sets.** The label set of node v is the union of the labels of all elements that contain
 * v, over all ranks: each rank collects the labels of its own elements, the copies of every
 * shared node are exchanged (VertexField::ExchangeSharedLists) and each copy keeps the sorted
 * union of all copies. A union of sets does not depend on the order in which they arrive, so all
 * copies end up identical and the result does not depend on the rank count or the partition.
 * There is no fixed limit on the size of a set.
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
 * **Constraints.** Each surface a node must stay on removes one direction in which it can move:
 *   - grain constraints = (number of labels) - 1 (0 inside a grain, 1 on a grain boundary face,
 *     2 on a triple line, 3 at a quad point);
 *   - boundary constraints = min(boundary surfaces, 3);
 *   - constraints = grain constraints + boundary constraints.
 * A node with more than 3 constraints is *over-constrained*: in 3D no position satisfies all of
 * them in general (e.g. a quad point on the outer boundary, or a grain boundary that meets a box
 * corner). Such nodes are counted and flagged here; label repair (Stage 2c) decides what to do
 * with them.
 *
 * **Where it runs.** Host and MPI only (Stage 2 is CPU-only: TMOP fitting needs legacy
 * assembly).
 */

namespace exaconstit::microstructure {

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
};

/** @brief How many grains meet at a node (index = number of labels - 1, capped). */
enum class JunctionKind : int {
    INTERIOR = 0,    ///< one grain
    FACE = 1,        ///< two grains: on a grain boundary face
    TRIPLE_LINE = 2, ///< three grains
    QUAD_POINT = 3,  ///< four grains
    MULTI = 4        ///< five or more grains (always over-constrained)
};

/** @brief Number of JunctionKind values. */
inline constexpr int NUM_JUNCTION_KINDS = 5;
/** @brief Boundary constraint counts 0..3 (more surfaces are capped at 3). */
inline constexpr int NUM_BOUNDARY_CONSTRAINTS = 4;

/** @brief Global node counts (each node counted once, on the rank that owns it). */
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
    NodeLabels(mfem::ParMesh& pmesh, const mfem::Array<int>& element_label,
               const NodeLabelOptions& options = {});
    ~NodeLabels();
    NodeLabels(const NodeLabels&) = delete;
    NodeLabels& operator=(const NodeLabels&) = delete;

    /** @brief Node numbering, coordinates, owned / shared markers and the rank exchange. */
    const VertexField& Field() const {
        return *m_field;
    }
    /** @brief Number of local nodes. */
    int NumNodes() const {
        return m_field->NumVertices();
    }

    /** @brief Label set CSR offsets, size NumNodes() + 1. */
    const mfem::Array<int>& LabelOffsets() const {
        return m_label_offsets;
    }
    /** @brief Label set CSR items: the labels of node v, ascending, at
     * [LabelOffsets()[v], LabelOffsets()[v + 1]). */
    const mfem::Array<int>& Labels() const {
        return m_labels;
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
     * @brief Boundary surface CSR offsets, size NumNodes() + 1: the surfaces of node v are
     * [SurfaceOffsets()[v], SurfaceOffsets()[v + 1]).
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
    /** @brief 1 when all face normals of surface k agree to 1e-12 (flat at the node), else 0. */
    const mfem::Array<int>& SurfacePlanar() const {
        return m_surface_planar;
    }
    /** @brief min(BoundarySurfaces(v), 3). */
    int BoundaryConstraints(int v) const {
        return m_boundary_surfaces[v] < 3 ? m_boundary_surfaces[v] : 3;
    }
    /** @brief NumLabels(v) - 1. */
    int GrainConstraints(int v) const {
        return NumLabels(v) - 1;
    }
    /** @brief GrainConstraints(v) + BoundaryConstraints(v). */
    int Constraints(int v) const {
        return GrainConstraints(v) + BoundaryConstraints(v);
    }
    /** @brief More than 3 constraints. */
    bool OverConstrained(int v) const {
        return Constraints(v) > 3;
    }
    /** @brief How many grains meet at node v. */
    JunctionKind Kind(int v) const;

    /** @brief Global counts (same on every rank). */
    const NodeLabelSummary& Summary() const {
        return m_summary;
    }

private:
    void BuildLabelSets(const mfem::Array<int>& element_label);
    void BuildBoundarySurfaces(double feature_angle_degrees);
    void BuildSummary();

    std::unique_ptr<VertexField> m_field;
    mfem::Array<int> m_label_offsets;
    mfem::Array<int> m_labels;
    mfem::Array<int> m_boundary_surfaces;
    mfem::Array<int> m_surface_offsets;
    mfem::Vector m_surface_normals;
    mfem::Array<int> m_surface_planar;
    NodeLabelSummary m_summary;
};

} // namespace exaconstit::microstructure
