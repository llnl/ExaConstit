#include "microstructure/mesh/preflight.hpp"

#include "microstructure/mesh/mesh_sampler.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace exaconstit::microstructure {

namespace {

/*
 * ElementPoints / GetElementPoints: derivatives of an element's shape functions at the quadrature points
 * and at the nodal points plus corners, per distinct FiniteElement (MFEM shares one object per
 * geometry and order in a collection, so a mesh has one or a few entries). dshape(k, j, q) is
 * d(shape k)/d(xi_j) at point q, the layout FiniteElement::CalcDShape writes (ndof x dim).
 */
struct ElementPoints {
    mfem::DenseTensor quad_dshape;
    mfem::DenseTensor node_dshape;
};

const ElementPoints& GetElementPoints(std::map<const mfem::FiniteElement*, ElementPoints>& cache,
                                      const mfem::FiniteElement& fe,
                                      const mfem::IntegrationRule& quad) {
    auto it = cache.find(&fe);
    if (it != cache.end()) {
        return it->second;
    }
    const int ndof = fe.GetDof();
    const int dim = fe.GetDim();
    auto fill = [&](const std::vector<const mfem::IntegrationPoint*>& points,
                    mfem::DenseTensor& out) {
        out.SetSize(ndof, dim, static_cast<int>(points.size()));
        for (std::size_t q = 0; q < points.size(); q++) {
            fe.CalcDShape(*points[q], out(static_cast<int>(q)));
        }
    };
    ElementPoints entry;
    std::vector<const mfem::IntegrationPoint*> points;
    for (int q = 0; q < quad.GetNPoints(); q++) {
        points.push_back(&quad.IntPoint(q));
    }
    fill(points, entry.quad_dshape);
    points.clear();
    const mfem::IntegrationRule& nodes = fe.GetNodes();
    for (int q = 0; q < nodes.GetNPoints(); q++) {
        points.push_back(&nodes.IntPoint(q));
    }
    const mfem::IntegrationRule* corners = mfem::Geometries.GetVertices(fe.GetGeomType());
    for (int q = 0; q < corners->GetNPoints(); q++) {
        points.push_back(&corners->IntPoint(q));
    }
    fill(points, entry.node_dshape);
    return cache.emplace(&fe, std::move(entry)).first->second;
}

/*
 * MinDetJ: minimum over the points q of det(X dshape(q)), X = the element's nodal coordinates
 * (sdim x ndof, IsoparametricTransformation::GetPointMat). This is the Jacobian
 * IsoparametricTransformation::EvalJacobian forms (PointMat times the shape derivatives), and
 * Weight() is its determinant for a square Jacobian, so the sign shows inversion.
 */
double MinDetJ(const mfem::DenseMatrix& X, const mfem::DenseTensor& dshape) {
    mfem::DenseMatrix J(X.Height(), dshape.SizeJ());
    double min_det = std::numeric_limits<double>::max();
    for (int q = 0; q < dshape.SizeK(); q++) {
        mfem::Mult(X, dshape(q), J);
        min_det = std::min(min_det, static_cast<double>(J.Det()));
    }
    return min_det;
}

} // namespace

/*
 * CheckMesh (local part + communicator)
 *
 * Steps:
 *  1. Element geometries: OR together one bit per element geometry found locally.
 *  2. Jacobians: for each local element, evaluate detJ at every quadrature point of
 *     DefaultSampleQuadOrder and at the element's own nodal points plus its corners
 *     (FiniteElement::GetNodes and Geometries.GetVertices), and record elements whose minimum is
 *     not positive. The extra points catch elements that fold between quadrature points: a
 *     curved element pinched near a corner, or a badly distorted trilinear hex, can have
 *     detJ > 0 at every quadrature point and detJ < 0 near a corner. Elements caught only by the
 *     extra points are also counted separately (num_folded_between_points).
 *     Why the nodal points: detJ is a polynomial in the reference coordinates (degree up to
 *     3p - 1 per direction for a hex), so no finite point set proves it positive; the nodal
 *     points are the natural sample of an order-p element, since they are where the element's
 *     geometry is defined. They include the corners, edges and faces for every closed H1 basis
 *     (Gauss-Lobatto, closed uniform, Bernstein, whose GetNodes are uniform points), and the
 *     corners, where folds of curved elements usually start, are added explicitly for any other
 *     basis. The set has (p + 1)^3 points for a hex instead of the (2p + 1)^3 of a lattice
 *     twice as fine (64 instead of 343 at p = 3). A fold narrower than the node spacing can be
 *     missed.
 *     Cost: the shape-function derivatives at the quadrature and nodal points depend only on the
 *     finite element, so they are computed once per distinct element (GetElementPoints) and
 *     J = (nodal coordinates) x (derivatives) is a small matrix product per point, instead of
 *     re-evaluating the basis at every point of every element as
 *     IsoparametricTransformation::SetIntPoint does.
 *  3. Mesh-level flags: dimension, nonconforming, discontinuous (periodic) Nodes.
 *  4. Combine everything over `comm` with MPI_Allreduce so every rank gets the same report.
 *
 * Step 2 is a plain host loop rather than the device-capable sampler: preflight runs once per
 * mesh and needs per-element ids of the offending elements, so a GPU path buys nothing.
 * `IsoparametricTransformation::Weight()` returns the signed determinant for square (3x3)
 * Jacobians, so inverted elements show up as negative values. The const overload
 * `Mesh::GetElementTransformation(int, IsoparametricTransformation*) const` (`mesh.hpp:1779`)
 * lets this function take a const mesh.
 *
 * Every rank, including one with no elements, makes the same four MPI_Allreduce calls in the
 * same order, which is what makes this function safely collective.
 */
PreflightReport CheckMesh(const mfem::Mesh& mesh, MPI_Comm comm) {
    PreflightReport report;
    const int ne = mesh.GetNE();

    // Step 1: bit g of the mask is set if some local element has geometry type g.
    int local_geometry_mask = 0;
    for (int e = 0; e < ne; e++) {
        const int geometry = static_cast<int>(mesh.GetElementGeometry(e));
        local_geometry_mask |= 1 << geometry;
    }

    // Step 2: minimum detJ per element over the quadrature points and the nodal points plus
    // corners. `!(min > 0.0)` also flags NaN determinants as bad.
    const int quad_order = DefaultSampleQuadOrder(mesh); // local, no MPI
    double local_min_detj = std::numeric_limits<double>::max();
    std::int64_t local_folded = 0;
    std::map<const mfem::FiniteElement*, ElementPoints> cache;
    mfem::IsoparametricTransformation trans;
    for (int e = 0; e < ne; e++) {
        mesh.GetElementTransformation(e, &trans);
        const mfem::Geometry::Type geom = mesh.GetElementGeometry(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, quad_order);
        const ElementPoints& points = GetElementPoints(cache, *trans.GetFE(), ir);

        // Step 2a: the quadrature points the samplers use; 2b: the nodal points and corners
        const double quad_min = MinDetJ(trans.GetPointMat(), points.quad_dshape);
        const double node_min = MinDetJ(trans.GetPointMat(), points.node_dshape);

        const double elem_min = std::min(quad_min, node_min);
        local_min_detj = std::min(local_min_detj, elem_min);
        if (!(elem_min > 0.0)) {
            report.local_bad_elements.push_back(e);
            if (quad_min > 0.0) {
                local_folded++; // positive at every quadrature point, folded in between
            }
        }
    }

    // Step 3: mesh-level properties. A rank with no elements counts as 3D so it cannot fail
    // the dimension check on its own. Periodic meshes from Mesh::MakePeriodic carry L2
    // (discontinuous) Nodes, detected with FiniteElementSpace::IsDGSpace() (`fespace.hpp:1519`).
    const bool local_is_3d = (mesh.Dimension() == 3 && mesh.SpaceDimension() == 3) || ne == 0;
    const bool local_nonconforming = mesh.Nonconforming();
    const bool local_discontinuous =
        mesh.GetNodes() != nullptr && mesh.GetNodes()->FESpace()->IsDGSpace();

    // Step 4: global reductions, one MPI_Allreduce per reduction operator.
    //  - flags (0/1, MAX = "true on any rank"): [not 3D, nonconforming, discontinuous]
    //  - geometry mask: bitwise OR
    //  - minimum detJ: MIN
    //  - [bad element count, element count, folded-between-points count]: SUM in 64-bit so very
    //    large meshes cannot overflow
    int flags[3] = {local_is_3d ? 0 : 1, local_nonconforming ? 1 : 0, local_discontinuous ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, flags, 3, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_geometry_mask, &report.geometry_mask, 1, MPI_INT, MPI_BOR, comm);
    MPI_Allreduce(&local_min_detj, &report.min_detj, 1, MPI_DOUBLE, MPI_MIN, comm);
    std::int64_t counts[3] = {static_cast<std::int64_t>(report.local_bad_elements.size()),
                              static_cast<std::int64_t>(ne), local_folded};
    MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_INT64_T, MPI_SUM, comm);

    report.is_3d = flags[0] == 0;
    report.conforming = flags[1] == 0;
    report.continuous_nodes = flags[2] == 0;
    report.num_bad_elements = counts[0];
    report.num_elements = counts[1];
    report.num_folded_between_points = counts[2];

    // A single geometry means exactly one bit is set in the mask. For a nonzero integer m,
    // m & (m - 1) clears the lowest set bit, so it is zero exactly when m has one bit set. An
    // empty global mesh has mask 0 and fails.
    const int mask = report.geometry_mask;
    report.single_geometry = mask != 0 && (mask & (mask - 1)) == 0;
    return report;
}

/* ParMesh overload: the local part of a ParMesh is a Mesh, and its communicator is known. */
PreflightReport CheckMesh(const mfem::ParMesh& pmesh) {
    return CheckMesh(static_cast<const mfem::Mesh&>(pmesh), pmesh.GetComm());
}

/*
 * DescribePreflightFailures
 *
 * Builds one bullet line per failed check. For mixed geometries it lists the geometry names
 * from the mask (`mfem::Geometry::Name`, e.g. "Cube", "Tetrahedron"). For inverted elements it
 * lists up to 10 local ids from this rank; the global count and minimum detJ are identical on
 * all ranks.
 */
std::string DescribePreflightFailures(const PreflightReport& report) {
    if (report.Passed()) {
        return {};
    }
    std::ostringstream msg;
    msg << "Mesh preflight failed:";
    if (!report.is_3d) {
        msg << "\n  - the mesh is not a 3D volume mesh";
    }
    if (!report.single_geometry) {
        if (report.num_elements == 0) {
            msg << "\n  - the mesh has no elements";
        } else {
            // Name every geometry whose bit is set in the mask.
            msg << "\n  - the mesh mixes element geometries (";
            bool first = true;
            for (int g = 0; g < mfem::Geometry::NumGeom; g++) {
                if ((report.geometry_mask & (1 << g)) != 0) {
                    if (!first) {
                        msg << ", ";
                    }
                    msg << mfem::Geometry::Name[g];
                    first = false;
                }
            }
            msg << "); ExaConstit needs a single element geometry";
        }
    }
    if (!report.conforming) {
        msg << "\n  - the mesh is nonconforming; ExaConstit needs a conforming mesh";
    }
    if (!report.continuous_nodes) {
        msg << "\n  - the mesh has discontinuous (L2) nodes, i.e. it is periodic; periodic "
               "meshes are not supported";
    }
    if (report.num_bad_elements > 0) {
        msg << "\n  - " << report.num_bad_elements << " of " << report.num_elements
            << " elements have detJ <= 0 at a quadrature point or a nodal point (min "
               "detJ = "
            << report.min_detj << ")";
        if (report.num_folded_between_points > 0) {
            msg << "; " << report.num_folded_between_points
                << " of them are positive at every quadrature point and fold between them";
        }
        if (!report.local_bad_elements.empty()) {
            msg << "; local element ids on this rank:";
            const std::size_t max_shown = 10;
            const std::size_t n_show = std::min(report.local_bad_elements.size(), max_shown);
            for (std::size_t i = 0; i < n_show; i++) {
                msg << " " << report.local_bad_elements[i];
            }
            if (report.local_bad_elements.size() > n_show) {
                msg << " ...";
            }
        }
    }
    return msg.str();
}

/* ThrowOnPreflightFailure: consistent across ranks because Passed() uses only reduced fields. */
void ThrowOnPreflightFailure(const PreflightReport& report) {
    if (!report.Passed()) {
        throw std::runtime_error(DescribePreflightFailures(report));
    }
}

} // namespace exaconstit::microstructure
