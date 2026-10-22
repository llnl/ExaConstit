#include "microstructure/mesh/mesh_adjacency.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/*
 * FaceArea: area of local face f, integrated with quadrature.
 *
 * Mesh::GetFaceTransformation (mesh.cpp:539, the const variant filling a user-owned
 * IsoparametricTransformation) maps the reference face (triangle or square) to the physical face,
 * using the vertices for straight-sided meshes and the Nodes for curved meshes. At each
 * quadrature point, ElementTransformation::Weight() is the area element: for a 3x2 Jacobian J it
 * is sqrt(det(J^T J)) (DenseMatrix::Weight, densemat.cpp:568-575), the ratio between physical
 * and reference area. Area = sum over points of (reference weight) * Weight().
 */
double FaceArea(const mfem::Mesh& mesh, int f, int quad_order) {
    mfem::IsoparametricTransformation transformation;
    mesh.GetFaceTransformation(f, &transformation);
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(mesh.GetFaceGeometry(f), quad_order);
    double area = 0.0;
    for (int q = 0; q < rule.GetNPoints(); q++) {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        transformation.SetIntPoint(&ip);
        area += ip.weight * transformation.Weight();
    }
    return area;
}

/** Centroid of local face f: the average of its vertex coordinates (host). */
std::array<double, 3> FaceCentroid(const mfem::Mesh& mesh, int f) {
    mfem::Array<int> vertices;
    mesh.GetFaceVertices(f, vertices);
    std::array<double, 3> c = {0.0, 0.0, 0.0};
    for (int v = 0; v < vertices.Size(); v++) {
        const double* x = mesh.GetVertex(vertices[v]);
        for (int d = 0; d < 3; d++) {
            c[static_cast<std::size_t>(d)] += x[d];
        }
    }
    for (double& coordinate : c) {
        coordinate /= static_cast<double>(vertices.Size());
    }
    return c;
}

/** Largest number of vertices of a face (quadrilateral). */
constexpr int MAX_FACE_VERTICES = 4;

/**
 * One boundary face on a periodic plane, as gathered from every rank. Plain doubles so the
 * records travel in one MPI_DOUBLE Allgatherv; the integer fields (axis, side, rank, element,
 * face, vertex count) are exact in a double (|value| < 2^53).
 */
struct PlaneFace {
    double axis;    ///< periodic axis d of the plane
    double side;    ///< 0: plane x_d = lo_d, 1: plane x_d = hi_d
    double c[3];    ///< face centroid (vertex average)
    double rank;    ///< owner rank
    double element; ///< owner's local element
    double face;    ///< owner's local face index
    double num_vertices;                ///< 3 or 4
    double vertex[3 * MAX_FACE_VERTICES]; ///< vertex coordinates, vertex[3 v + d]
    double min_edge;                    ///< smallest edge length of the face
};
constexpr int PLANE_FACE_DOUBLES = 22;
static_assert(sizeof(PlaneFace) == PLANE_FACE_DOUBLES * sizeof(double),
              "PlaneFace must be a plain block of doubles");

/** Smallest distance between consecutive vertices of a face (its smallest edge). */
double SmallestEdge(const mfem::Mesh& mesh, const mfem::Array<int>& vertices) {
    double smallest = std::numeric_limits<double>::max();
    const int n = vertices.Size();
    for (int v = 0; v < n; v++) {
        const double* a = mesh.GetVertex(vertices[v]);
        const double* b = mesh.GetVertex(vertices[(v + 1) % n]);
        double d2 = 0.0;
        for (int d = 0; d < 3; d++) {
            d2 += (a[d] - b[d]) * (a[d] - b[d]);
        }
        smallest = std::min(smallest, std::sqrt(d2));
    }
    return smallest;
}

/**
 * True if every vertex of `lower`, shifted by `shift` along `axis`, lies within `tol` (in every
 * coordinate) of some vertex of `upper`, and both faces have the same number of vertices.
 */
bool VerticesMatch(const PlaneFace& lower, const PlaneFace& upper, std::size_t axis,
                   double shift, double tol) {
    if (lower.num_vertices != upper.num_vertices) {
        return false;
    }
    const int n = static_cast<int>(lower.num_vertices);
    for (int v = 0; v < n; v++) {
        double p[3] = {lower.vertex[3 * v], lower.vertex[3 * v + 1], lower.vertex[3 * v + 2]};
        p[axis] += shift;
        bool found = false;
        for (int w = 0; w < n && !found; w++) {
            bool close = true;
            for (int d = 0; d < 3; d++) {
                close = close && std::fabs(upper.vertex[3 * w + d] - p[d]) <= tol;
            }
            found = close;
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

} // namespace

/** One face adjacency found while scanning faces: element `element` sees `neighbor`. */
struct MeshAdjacency::Entry {
    int element;  ///< local element
    int neighbor; ///< local element (< NE) or NE + ghost index
    int face;     ///< local face index
    double area;  ///< face area
};

/*
 * MeshAdjacency constructor
 *
 * Steps:
 *  1. Checks: 3D and conforming. Both are global properties of the ParMesh (identical on every
 *     rank), so every rank throws together and no rank is left waiting in step 2.
 *  2. Ghost elements: ParMesh::ExchangeFaceNbrData() (pmesh.cpp:2079) sends every rank a copy of
 *     the elements on the far side of its shared faces (face_nbr_elements) and records, for each
 *     shared face, which ghost it touches: afterwards Mesh::GetFaceElements(lface, &e1, &e2)
 *     returns e2 = -1 - g for ghost g (pmesh.cpp:2508). It exchanges only with neighboring ranks,
 *     but every rank must call it.
 *  3. Exchange spaces: L2 order-0 ParFiniteElementSpaces with one and two components (one
 *     degree of freedom per element and component; two components carry both halves of a
 *     64-bit key); ParFiniteElementSpace::ExchangeFaceNbrData() builds the table of the ghost
 *     elements' degrees of freedom (face_nbr_element_dof), read with GetFaceNbrElementVDofs. The
 *     constructor of a ParFiniteElementSpace is itself collective (it numbers the global degrees
 *     of freedom).
 *  4. Faces: a face f with GetFaceElements(f) -> (e1, e2), e2 >= 0, is interior to this rank and
 *     gives two entries (e1 sees e2 and e2 sees e1). Shared faces are visited separately through
 *     ParMesh::GetSharedFace(sf) (the local index of shared face sf, pmesh.cpp:3193); they give
 *     one entry (the local element sees the ghost). Testing e2 < 0 alone would not do: after
 *     step 2, e2 = -1 is ghost 0 for a shared face, but also marks a boundary face.
 *     (4c, periodic constructor only: matched faces on opposite periodic planes, see
 *     PairPeriodicFaces.)
 *  5. CSR: count the entries per element, prefix-sum the counts into offsets, then fill each
 *     element's row, sorted by neighbor index.
 */
MeshAdjacency::MeshAdjacency(mfem::ParMesh& pmesh, int area_quad_order) : m_pmesh(&pmesh) {
    Build(area_quad_order, nullptr);
}

/* Periodic constructor: the same build with face pairing (see PairPeriodicFaces). */
MeshAdjacency::MeshAdjacency(mfem::ParMesh& pmesh,
                             const PeriodicFaceParameters& periodic,
                             int area_quad_order)
    : m_pmesh(&pmesh) {
    Build(area_quad_order, &periodic);
}

/*
 * Build: the constructor steps listed above; step 4c (periodic pairs) only when `periodic` asks
 * for at least one axis.
 */
void MeshAdjacency::Build(int area_quad_order, const PeriodicFaceParameters* periodic) {
    mfem::ParMesh& pmesh = *m_pmesh;
    // Step 1: requirements (global properties, so all ranks agree)
    if (pmesh.Dimension() != 3 || pmesh.SpaceDimension() != 3) {
        throw std::invalid_argument("MeshAdjacency: the mesh must be 3D");
    }
    if (pmesh.Nonconforming()) {
        throw std::invalid_argument("MeshAdjacency: nonconforming meshes are not supported");
    }
    m_num_elements = pmesh.GetNE();

    // Step 2: ghost elements across shared faces
    pmesh.ExchangeFaceNbrData();
    m_num_ghosts = pmesh.GetNFaceNeighborElements();

    // Step 3: piecewise-constant exchange spaces (1 and 2 components) and their dof maps.
    // Component k of element e is vdof k * ndofs + dof (byNODES ordering).
    m_fec = std::make_unique<mfem::L2_FECollection>(0, 3);
    mfem::Array<int> vdofs;
    for (int c = 0; c < 2; c++) {
        const int num_components = c + 1;
        auto& fes = m_fes[static_cast<std::size_t>(c)];
        fes = std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, m_fec.get(), num_components,
                                                            mfem::Ordering::byNODES);
        fes->ExchangeFaceNbrData();
        std::vector<int>& element_vdof = m_element_vdof[static_cast<std::size_t>(c)];
        std::vector<int>& ghost_vdof = m_ghost_vdof[static_cast<std::size_t>(c)];
        element_vdof.assign(static_cast<std::size_t>(num_components * m_num_elements), 0);
        ghost_vdof.assign(static_cast<std::size_t>(num_components * m_num_ghosts), 0);
        for (int e = 0; e < m_num_elements; e++) {
            fes->GetElementVDofs(e, vdofs);
            for (int k = 0; k < num_components; k++) {
                element_vdof[static_cast<std::size_t>(k * m_num_elements + e)] = vdofs[k];
            }
        }
        for (int g = 0; g < m_num_ghosts; g++) {
            mfem::DofTransformation doftrans;
            fes->GetFaceNbrElementVDofs(g, vdofs, doftrans);
            for (int k = 0; k < num_components; k++) {
                ghost_vdof[static_cast<std::size_t>(k * m_num_ghosts + g)] = vdofs[k];
            }
        }
    }

    // Quadrature order for face areas: 2 p + 1 for mesh order p
    int quad_order = area_quad_order;
    if (quad_order < 0) {
        int mesh_order = 1;
        if (pmesh.GetNodes() != nullptr) {
            mesh_order = pmesh.GetNodes()->FESpace()->FEColl()->GetOrder();
        }
        quad_order = 2 * mesh_order + 1;
    }

    // Step 4a: faces interior to this rank (both elements local): two entries per face
    std::vector<Entry> entries;
    for (int f = 0; f < pmesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        if (e1 >= 0 && e2 >= 0) {
            const double area = FaceArea(pmesh, f, quad_order);
            entries.push_back({e1, e2, f, area});
            entries.push_back({e2, e1, f, area});
        }
    }

    // Step 4b: shared faces: the local element (e1) sees ghost g, stored as NE + g
    for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++) {
        const int f = pmesh.GetSharedFace(sf);
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        const int ghost = -1 - e2;
        if (e1 < 0 || ghost < 0 || ghost >= m_num_ghosts) {
            // Cannot happen for a conforming mesh after ExchangeFaceNbrData; a local, rank-only
            // failure here would desynchronize the collective calls that follow, so abort.
            mfem::mfem_error("MeshAdjacency: inconsistent shared face data");
        }
        entries.push_back({e1, m_num_elements + ghost, f, FaceArea(pmesh, f, quad_order)});
    }

    // Step 4c: periodic face pairs (collective; identical decision on every rank)
    if (periodic != nullptr) {
        for (const bool axis_periodic : periodic->periodic) {
            m_periodic_pairing = m_periodic_pairing || axis_periodic;
        }
    }
    if (m_periodic_pairing) {
        PairPeriodicFaces(*periodic, quad_order, entries);
    }

    // Step 5: CSR rows sorted by (element, neighbor)
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) {
                  if (a.element != b.element) {
                      return a.element < b.element;
                  }
                  return a.neighbor < b.neighbor;
              });
    const int num_entries = static_cast<int>(entries.size());
    m_offsets.SetSize(m_num_elements + 1);
    m_offsets = 0;
    for (const Entry& entry : entries) {
        m_offsets[entry.element + 1] += 1; // count per element, shifted by one
    }
    for (int e = 0; e < m_num_elements; e++) {
        m_offsets[e + 1] += m_offsets[e]; // running sum: start of each row
    }
    m_neighbors.SetSize(num_entries);
    m_face_indices.SetSize(num_entries);
    m_face_areas.SetSize(num_entries);
    for (int k = 0; k < num_entries; k++) {
        // entries are sorted by element, so entry k is already at its CSR position
        m_neighbors[k] = entries[static_cast<std::size_t>(k)].neighbor;
        m_face_indices[k] = entries[static_cast<std::size_t>(k)].face;
        m_face_areas[k] = entries[static_cast<std::size_t>(k)].area;
    }
}

/*
 * PairPeriodicFaces (collective)
 *
 * Steps:
 *  1. Checks on this rank's parameters (positive, finite box lengths on the periodic axes; a
 *     tolerance in (0, 0.5)), then the same parameters on every rank: MPI_Allreduce MIN and MAX
 *     of (lo, hi, flags, tolerance) must agree, otherwise every rank throws (they all see the
 *     same minima and maxima).
 *  2. Local plane faces: a boundary face is a face with no second element that is not a shared
 *     face of the partition (after ExchangeFaceNbrData, e2 = -1 marks both boundary faces and
 *     ghost 0, so the shared faces are excluded explicitly). It lies on the lower plane of axis d
 *     when all its vertices are within relative_tolerance x (its smallest edge) of lo_d (upper
 *     plane: of hi_d). Measuring against the face's own size makes this test independent of the
 *     box size and of the matching tolerance, which is only known after step 3.
 *  3. MPI_Allgatherv of all plane-face records (counts first with MPI_Allgather): every rank
 *     holds the same global list, ordered by rank and then by local face order. The matching
 *     tolerance is tol = max(relative_tolerance x smallest edge of all listed faces, 1e-14 x
 *     the largest periodic box length); the lower bound keeps (x - lo) / (2 tol) far below the
 *     range of long long.
 *  4. Matching, done identically on every rank over the global list: for each axis the upper
 *     faces are filed by centroid in an ordered map (std::map) keyed by integer cell coordinates
 *     of cells of size 2 tol. Each lower face, in list order, looks at the 27 cells around its
 *     centroid shifted by +L_d; a candidate matches when its centroid is within tol and every
 *     shifted vertex coincides with one of its vertices within tol. The face is paired when
 *     exactly one candidate matches and that candidate is not paired yet; everything else is
 *     unmatched. Unmatched faces (identical count on every rank) throw on every rank.
 *  5. Entries for pairs that involve this rank: a partner on this rank is a local neighbor (one
 *     entry from each side, like an interior face); a partner on rank r != this rank is a
 *     periodic ghost. Ghosts are numbered in increasing (r, partner's local element) order, after
 *     MFEM's face-neighbor ghosts, and a ghost seen through several faces gets one number.
 *  6. Exchange plan: rank r needs from this rank exactly the local elements paired with r's
 *     elements; because the pairing is symmetric, the list this rank sends to r, sorted by
 *     element id, lines up with the ghost block r receives from this rank, sorted by the same
 *     ids. Only partner ranks exchange messages.
 */
void MeshAdjacency::PairPeriodicFaces(const PeriodicFaceParameters& periodic, int quad_order,
                                      std::vector<Entry>& entries) {
    mfem::ParMesh& pmesh = *m_pmesh;
    const MPI_Comm comm = GetComm();
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // Step 1a: this rank's parameters
    std::array<double, 3> length = {0.0, 0.0, 0.0};
    double largest = 0.0;
    for (std::size_t d = 0; d < 3; d++) {
        length[d] = periodic.hi[d] - periodic.lo[d];
        if (periodic.periodic[d]) {
            if (!(length[d] > 0.0) || !std::isfinite(length[d])) {
                throw std::invalid_argument("MeshAdjacency: the periodic box must have a positive, "
                                            "finite length on every periodic axis");
            }
            largest = std::max(largest, length[d]);
        }
    }
    const double rel_tol = periodic.relative_tolerance;
    if (!(rel_tol > 0.0 && rel_tol < 0.5)) {
        throw std::invalid_argument("MeshAdjacency: the periodic relative_tolerance must be in "
                                    "(0, 0.5)");
    }

    // Step 1b: identical parameters on every rank
    double mine[10];
    for (std::size_t d = 0; d < 3; d++) {
        mine[d] = periodic.lo[d];
        mine[3 + d] = periodic.hi[d];
        mine[6 + d] = periodic.periodic[d] ? 1.0 : 0.0;
    }
    mine[9] = rel_tol;
    double lowest[10];
    double highest[10];
    MPI_Allreduce(mine, lowest, 10, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(mine, highest, 10, MPI_DOUBLE, MPI_MAX, comm);
    for (int k = 0; k < 10; k++) {
        if (lowest[k] != highest[k]) {
            throw std::invalid_argument("MeshAdjacency: the periodic face parameters (box, axes, "
                                        "tolerance) differ between ranks");
        }
    }

    // Step 2: this rank's boundary faces on periodic planes
    std::vector<char> is_shared(static_cast<std::size_t>(pmesh.GetNumFaces()), 0);
    for (int sf = 0; sf < pmesh.GetNSharedFaces(); sf++) {
        is_shared[static_cast<std::size_t>(pmesh.GetSharedFace(sf))] = 1;
    }
    std::vector<PlaneFace> local_faces;
    mfem::Array<int> vertices;
    for (int f = 0; f < pmesh.GetNumFaces(); f++) {
        int e1 = -1;
        int e2 = -1;
        pmesh.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 >= 0 || is_shared[static_cast<std::size_t>(f)]) {
            continue; // interior or shared face
        }
        pmesh.GetFaceVertices(f, vertices);
        if (vertices.Size() > MAX_FACE_VERTICES) {
            continue; // not a triangle or quadrilateral (cannot occur for hex/tet meshes)
        }
        const double edge = SmallestEdge(pmesh, vertices);
        const double plane_tol = rel_tol * edge;
        const std::array<double, 3> c = FaceCentroid(pmesh, f);
        for (std::size_t d = 0; d < 3; d++) {
            if (!periodic.periodic[d]) {
                continue;
            }
            bool on_lo = true;
            bool on_hi = true;
            for (int v = 0; v < vertices.Size(); v++) {
                const double x = pmesh.GetVertex(vertices[v])[d];
                on_lo = on_lo && std::fabs(x - periodic.lo[d]) <= plane_tol;
                on_hi = on_hi && std::fabs(x - periodic.hi[d]) <= plane_tol;
            }
            if (!on_lo && !on_hi) {
                continue;
            }
            PlaneFace record{};
            record.axis = static_cast<double>(d);
            record.side = on_lo ? 0.0 : 1.0;
            for (std::size_t k = 0; k < 3; k++) {
                record.c[k] = c[k];
            }
            record.rank = static_cast<double>(rank);
            record.element = static_cast<double>(e1);
            record.face = static_cast<double>(f);
            record.num_vertices = static_cast<double>(vertices.Size());
            for (int v = 0; v < vertices.Size(); v++) {
                const double* x = pmesh.GetVertex(vertices[v]);
                for (int k = 0; k < 3; k++) {
                    record.vertex[3 * v + k] = x[k];
                }
            }
            record.min_edge = edge;
            local_faces.push_back(record);
        }
    }

    // Step 3: the global list of plane faces on every rank (counts in doubles), and tol
    const int local_count = static_cast<int>(local_faces.size()) * PLANE_FACE_DOUBLES;
    std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    long long total = 0;
    for (int r = 0; r < nranks; r++) {
        displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
        total += counts[static_cast<std::size_t>(r)];
    }
    if (total > std::numeric_limits<int>::max()) {
        // Same total on every rank, so every rank throws
        throw std::invalid_argument("MeshAdjacency: too many faces on the periodic planes");
    }
    std::vector<PlaneFace> all_faces(static_cast<std::size_t>(total / PLANE_FACE_DOUBLES));
    MPI_Allgatherv(reinterpret_cast<const double*>(local_faces.data()), local_count, MPI_DOUBLE,
                   reinterpret_cast<double*>(all_faces.data()), counts.data(), displs.data(),
                   MPI_DOUBLE, comm);
    double smallest_edge = largest;
    for (const PlaneFace& face : all_faces) {
        smallest_edge = std::min(smallest_edge, face.min_edge);
    }
    const double tol = std::max(rel_tol * smallest_edge, 1.0e-14 * largest);
    m_periodic_report.tolerance = tol;

    // Step 4: match lower faces to upper faces, axis by axis, in global list order
    const double cell = 2.0 * tol;
    auto cell_of = [&](const double* x) {
        std::array<long long, 3> key{};
        for (std::size_t d = 0; d < 3; d++) {
            key[d] = static_cast<long long>(std::floor((x[d] - periodic.lo[d]) / cell));
        }
        return key;
    };
    std::vector<std::pair<std::size_t, std::size_t>> pairs; // (lower, upper) indices
    std::int64_t unmatched = 0;
    for (std::size_t d = 0; d < 3; d++) {
        if (!periodic.periodic[d]) {
            continue;
        }
        // Upper faces of this axis, filed by centroid cell
        std::map<std::array<long long, 3>, std::vector<std::size_t>> upper_cells;
        std::vector<char> upper_used(all_faces.size(), 0);
        std::int64_t num_upper = 0;
        for (std::size_t k = 0; k < all_faces.size(); k++) {
            if (all_faces[k].axis == static_cast<double>(d) && all_faces[k].side == 1.0) {
                upper_cells[cell_of(all_faces[k].c)].push_back(k);
                num_upper++;
            }
        }
        // Each lower face looks for exactly one upper face with the shifted vertices
        std::int64_t matched_here = 0;
        for (std::size_t k = 0; k < all_faces.size(); k++) {
            const PlaneFace& lower = all_faces[k];
            if (lower.axis != static_cast<double>(d) || lower.side != 0.0) {
                continue;
            }
            double target[3] = {lower.c[0], lower.c[1], lower.c[2]};
            target[d] += length[d];
            const std::array<long long, 3> center = cell_of(target);
            int found = 0;
            std::size_t partner = 0;
            for (long long a = -1; a <= 1; a++) {
                for (long long b = -1; b <= 1; b++) {
                    for (long long g = -1; g <= 1; g++) {
                        const auto it =
                            upper_cells.find({center[0] + a, center[1] + b, center[2] + g});
                        if (it == upper_cells.end()) {
                            continue;
                        }
                        for (const std::size_t u : it->second) {
                            bool close = true;
                            for (std::size_t c = 0; c < 3; c++) {
                                close = close && std::fabs(all_faces[u].c[c] - target[c]) <= tol;
                            }
                            if (close && VerticesMatch(lower, all_faces[u], d, length[d], tol)) {
                                found++;
                                partner = u;
                            }
                        }
                    }
                }
            }
            if (found == 1 && !upper_used[partner]) {
                upper_used[partner] = 1;
                pairs.emplace_back(k, partner);
                matched_here++;
            } else {
                unmatched++; // no partner, an ambiguous match, or a partner already taken
            }
        }
        unmatched += num_upper - matched_here;
    }
    m_periodic_report.plane_faces = static_cast<std::int64_t>(all_faces.size());
    m_periodic_report.matched_pairs = static_cast<std::int64_t>(pairs.size());
    m_periodic_report.unmatched_faces = unmatched;
    if (unmatched > 0) {
        // Identical counts on every rank: every rank throws
        throw std::runtime_error(
            "MeshAdjacency: periodic face pairing found " + std::to_string(unmatched) + " of " +
            std::to_string(all_faces.size()) +
            " faces on the periodic planes without a unique partner on the opposite plane "
            "(matching tolerance " + std::to_string(tol) +
            "). The mesh must have matching faces on opposite periodic planes; use a periodic "
            "mesh, adjust the periodic tolerance, or turn the periodic axes off.");
    }

    // Step 5: entries of the pairs that involve this rank; remote partners become ghosts
    struct Pending {
        int element;  ///< local element
        int face;     ///< local face
        int partner_rank;
        int partner_element;
    };
    std::vector<Pending> pending;
    for (const auto& pair : pairs) {
        const PlaneFace& a = all_faces[pair.first];
        const PlaneFace& b = all_faces[pair.second];
        const PlaneFace* sides[2][2] = {{&a, &b}, {&b, &a}};
        for (const auto& side : sides) {
            const PlaneFace& mine_face = *side[0];
            const PlaneFace& other = *side[1];
            if (static_cast<int>(mine_face.rank) != rank) {
                continue;
            }
            const int element = static_cast<int>(mine_face.element);
            const int other_rank = static_cast<int>(other.rank);
            const int other_element = static_cast<int>(other.element);
            if (other_rank == rank && other_element == element) {
                continue; // the element is its own periodic neighbor: no connection to add
            }
            pending.push_back(
                {element, static_cast<int>(mine_face.face), other_rank, other_element});
        }
    }
    // Number the remote partners in increasing (rank, element) order
    std::map<std::pair<int, int>, int> ghost_index;
    for (const Pending& p : pending) {
        if (p.partner_rank != rank) {
            ghost_index[{p.partner_rank, p.partner_element}] = 0;
        }
    }
    int next_ghost = 0;
    for (auto& entry : ghost_index) {
        entry.second = next_ghost++;
    }
    m_num_periodic_ghosts = next_ghost;
    for (const Pending& p : pending) {
        const double area = FaceArea(pmesh, p.face, quad_order);
        if (p.partner_rank == rank) {
            entries.push_back({p.element, p.partner_element, p.face, area});
        } else {
            const int g = ghost_index[{p.partner_rank, p.partner_element}];
            entries.push_back({p.element, m_num_elements + m_num_ghosts + g, p.face, area});
        }
    }

    // Step 6: exchange plan with the partner ranks only. Receive: the ghosts of each partner form
    // one contiguous block (map order). Send: the distinct local elements paired with the
    // partner's elements, ascending.
    std::map<int, std::vector<int>> send_lists;
    for (const Pending& p : pending) {
        if (p.partner_rank != rank) {
            send_lists[p.partner_rank].push_back(p.element);
        }
    }
    std::map<int, std::pair<int, int>> recv_blocks; // rank -> (first ghost, count)
    for (const auto& entry : ghost_index) {
        auto it = recv_blocks.find(entry.first.first);
        if (it == recv_blocks.end()) {
            recv_blocks[entry.first.first] = {entry.second, 1};
        } else {
            it->second.second++;
        }
    }
    m_periodic_partners.clear();
    m_periodic_send_elements.clear();
    m_periodic_recv_begin.clear();
    m_periodic_recv_count.clear();
    for (auto& item : send_lists) {
        std::vector<int>& list = item.second;
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        // Symmetric pairing: a rank this rank sends to also sends to this rank
        const auto block = recv_blocks.at(item.first);
        m_periodic_partners.push_back(item.first);
        m_periodic_send_elements.push_back(list);
        m_periodic_recv_begin.push_back(block.first);
        m_periodic_recv_count.push_back(block.second);
    }
}

MeshAdjacency::~MeshAdjacency() = default;

/*
 * CheckLocalSize: every rank compares the caller's array size with its element count, and the
 * pass/fail flags are combined with MPI_Allreduce (MAX of 0/1 = "did any rank fail?"), so either
 * all ranks throw or none does. Throwing on one rank only would leave the other ranks waiting in
 * the exchange that follows. Called once per public Exchange* call.
 */
void MeshAdjacency::CheckLocalSize(int size, const char* caller) const {
    const int local_bad = (size != m_num_elements) ? 1 : 0;
    int any_bad = 0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, GetComm());
    if (any_bad != 0) {
        throw std::invalid_argument(std::string("MeshAdjacency::") + caller +
                                    ": the input must have one value per local element " +
                                    (local_bad ? "(wrong size on this rank)"
                                               : "(wrong size on another rank)"));
    }
}

/*
 * ExchangeComponents
 *
 * Steps:
 *  1. Store the local values in an L2 order-0 ParGridFunction with `num_components` components
 *     (component k of element e at its recorded vector dof).
 *  2. ParGridFunction::ExchangeFaceNbrData (pgridfunc.cpp:209-268): every rank packs the values
 *     its neighbors need, sends them with MPI_Isend, and receives its ghost values into
 *     FaceNbrData() with MPI_Irecv; both are completed with MPI_Waitall before it returns.
 *  3. Copy each face-neighbor ghost's components out of FaceNbrData().
 *  4. With periodic pairing: for every partner rank, MPI_Isend the components of the elements it
 *     needs and MPI_Irecv its block of periodic ghosts (num_components values per element,
 *     element-major), then MPI_Waitall. Only partner ranks are involved.
 */
void MeshAdjacency::ExchangeComponents(const double* local, int num_components,
                                       std::vector<double>& ghost) const {
    const std::size_t c = static_cast<std::size_t>(num_components - 1);
    const int ne = m_num_elements;
    const int ng = NumGhostElements();

    // Step 1: local values into the grid function
    mfem::ParGridFunction field(m_fes[c].get());
    {
        double* data = field.HostWrite();
        const std::vector<int>& vdof = m_element_vdof[c];
        for (int k = 0; k < num_components; k++) {
            for (int e = 0; e < ne; e++) {
                data[vdof[static_cast<std::size_t>(k * ne + e)]] =
                    local[static_cast<std::size_t>(k) * static_cast<std::size_t>(ne) +
                          static_cast<std::size_t>(e)];
            }
        }
    }

    // Step 2: neighbor exchange (point-to-point with the face-neighbor ranks)
    field.ExchangeFaceNbrData();

    // Step 3: face-neighbor ghost values, component-major
    ghost.assign(static_cast<std::size_t>(num_components * ng), 0.0);
    if (m_num_ghosts > 0) {
        const double* nbr = field.FaceNbrData().HostRead();
        const std::vector<int>& vdof = m_ghost_vdof[c];
        for (int k = 0; k < num_components; k++) {
            for (int g = 0; g < m_num_ghosts; g++) {
                ghost[static_cast<std::size_t>(k * ng + g)] =
                    nbr[vdof[static_cast<std::size_t>(k * m_num_ghosts + g)]];
            }
        }
    }

    // Step 4: periodic ghosts, point to point with the partner ranks
    if (!m_periodic_pairing || m_periodic_partners.empty()) {
        return;
    }
    const int tag = 7307;
    const std::size_t num_partners = m_periodic_partners.size();
    std::vector<std::vector<double>> send(num_partners);
    std::vector<std::vector<double>> recv(num_partners);
    std::vector<MPI_Request> requests(2 * num_partners);
    for (std::size_t p = 0; p < num_partners; p++) {
        const std::vector<int>& elements = m_periodic_send_elements[p];
        send[p].resize(elements.size() * static_cast<std::size_t>(num_components));
        for (std::size_t i = 0; i < elements.size(); i++) {
            for (int k = 0; k < num_components; k++) {
                send[p][i * static_cast<std::size_t>(num_components) +
                        static_cast<std::size_t>(k)] =
                    local[static_cast<std::size_t>(k) * static_cast<std::size_t>(ne) +
                          static_cast<std::size_t>(elements[i])];
            }
        }
        recv[p].resize(static_cast<std::size_t>(m_periodic_recv_count[p] * num_components));
        MPI_Irecv(recv[p].data(), static_cast<int>(recv[p].size()), MPI_DOUBLE,
                  m_periodic_partners[p], tag, GetComm(), &requests[2 * p]);
        MPI_Isend(send[p].data(), static_cast<int>(send[p].size()), MPI_DOUBLE,
                  m_periodic_partners[p], tag, GetComm(), &requests[2 * p + 1]);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    for (std::size_t p = 0; p < num_partners; p++) {
        for (int i = 0; i < m_periodic_recv_count[p]; i++) {
            const int g = m_num_ghosts + m_periodic_recv_begin[p] + i;
            for (int k = 0; k < num_components; k++) {
                ghost[static_cast<std::size_t>(k * ng + g)] =
                    recv[p][static_cast<std::size_t>(i * num_components + k)];
            }
        }
    }
}

/* ExchangeValues: size check, then one single-component exchange (bits copied exactly). */
void MeshAdjacency::ExchangeValues(const mfem::Vector& local, mfem::Vector& ghost) const {
    CheckLocalSize(local.Size(), "ExchangeValues");
    std::vector<double> values;
    ExchangeComponents(local.HostRead(), 1, values);
    ghost.SetSize(static_cast<int>(values.size()));
    double* out = ghost.HostWrite();
    for (std::size_t g = 0; g < values.size(); g++) {
        out[g] = values[g];
    }
}

/* ExchangeLabels: ints converted to doubles (exact for every int), exchanged, converted back. */
void MeshAdjacency::ExchangeLabels(const mfem::Array<int>& local, mfem::Array<int>& ghost) const {
    CheckLocalSize(local.Size(), "ExchangeLabels");
    std::vector<double> local_values(static_cast<std::size_t>(m_num_elements));
    const int* in = local.HostRead();
    for (int e = 0; e < m_num_elements; e++) {
        local_values[static_cast<std::size_t>(e)] = static_cast<double>(in[e]);
    }
    std::vector<double> values;
    ExchangeComponents(local_values.data(), 1, values);
    ghost.SetSize(static_cast<int>(values.size()));
    int* out = ghost.HostWrite();
    for (std::size_t g = 0; g < values.size(); g++) {
        out[g] = static_cast<int>(values[g]);
    }
}

/*
 * ExchangeKeys: a 64-bit key does not fit exactly in a double (53-bit mantissa), so the upper
 * 32 bits (key >> 32) and the lower 32 bits (key & 0xFFFFFFFF) travel as the two components of
 * one exchange, each exact, and the ghost key is reassembled as (upper << 32) | lower.
 */
void MeshAdjacency::ExchangeKeys(const std::vector<std::uint64_t>& local,
                                 std::vector<std::uint64_t>& ghost) const {
    CheckLocalSize(static_cast<int>(local.size()), "ExchangeKeys");
    const std::size_t ne = static_cast<std::size_t>(m_num_elements);
    std::vector<double> halves(2 * ne);
    for (std::size_t e = 0; e < ne; e++) {
        halves[e] = static_cast<double>(local[e] >> 32);
        halves[ne + e] = static_cast<double>(local[e] & 0xFFFFFFFFULL);
    }
    std::vector<double> values;
    ExchangeComponents(halves.data(), 2, values);
    const std::size_t ng = static_cast<std::size_t>(NumGhostElements());
    ghost.assign(ng, 0);
    for (std::size_t g = 0; g < ng; g++) {
        const auto hi = static_cast<std::uint64_t>(values[g]);
        const auto lo = static_cast<std::uint64_t>(values[ng + g]);
        ghost[g] = (hi << 32) | lo;
    }
}

} // namespace exaconstit::microstructure
