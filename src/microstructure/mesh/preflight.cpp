#include "microstructure/mesh/preflight.hpp"

#include "microstructure/mesh/mesh_sampler.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace exaconstit::microstructure {

/*
 * CheckMesh (local part + communicator)
 *
 * Steps:
 *  1. Element geometries: OR together one bit per element geometry found locally.
 *  2. Jacobians: for each local element, evaluate detJ at every quadrature point of
 *     DefaultSampleQuadOrder and record elements whose minimum is not positive.
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

    // Step 2: minimum detJ per element over the quadrature points. `!(elem_min > 0.0)` also
    // flags NaN determinants as bad.
    const int quad_order = DefaultSampleQuadOrder(mesh); // local, no MPI
    double local_min_detj = std::numeric_limits<double>::max();
    mfem::IsoparametricTransformation trans;
    for (int e = 0; e < ne; e++) {
        mesh.GetElementTransformation(e, &trans);
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(mesh.GetElementGeometry(e), quad_order);
        double elem_min = std::numeric_limits<double>::max();
        for (int q = 0; q < ir.GetNPoints(); q++) {
            trans.SetIntPoint(&ir.IntPoint(q));
            elem_min = std::min(elem_min, static_cast<double>(trans.Weight()));
        }
        local_min_detj = std::min(local_min_detj, elem_min);
        if (!(elem_min > 0.0)) {
            report.local_bad_elements.push_back(e);
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
    //  - [bad element count, element count]: SUM in 64-bit so very large meshes cannot overflow
    int flags[3] = {local_is_3d ? 0 : 1, local_nonconforming ? 1 : 0, local_discontinuous ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, flags, 3, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&local_geometry_mask, &report.geometry_mask, 1, MPI_INT, MPI_BOR, comm);
    MPI_Allreduce(&local_min_detj, &report.min_detj, 1, MPI_DOUBLE, MPI_MIN, comm);
    std::int64_t counts[2] = {static_cast<std::int64_t>(report.local_bad_elements.size()),
                              static_cast<std::int64_t>(ne)};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, comm);

    report.is_3d = flags[0] == 0;
    report.conforming = flags[1] == 0;
    report.continuous_nodes = flags[2] == 0;
    report.num_bad_elements = counts[0];
    report.num_elements = counts[1];

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
            << " elements have detJ <= 0 at a quadrature point (min detJ = " << report.min_detj
            << ")";
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
