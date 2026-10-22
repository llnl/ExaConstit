#include "microstructure/mesh/straight_sided.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace exaconstit::microstructure {

namespace {

/*
 * ContinuousNodes: the mesh's Nodes when they live in a continuous (H1) space, else null. Only
 * continuous Nodes can be straight-sided (file comment); for them every element numbers its
 * vertex dofs first, in the element's vertex order, which ElementCornerPoints relies on.
 */
const mfem::GridFunction* ContinuousNodes(const mfem::Mesh& mesh) {
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (nodes == nullptr) {
        return nullptr;
    }
    if (nodes->FESpace()->FEColl()->GetContType() != mfem::FiniteElementCollection::CONTINUOUS) {
        return nullptr;
    }
    return nodes;
}

} // namespace

/*
 * ElementCornerPoints
 *
 * Without Nodes: mfem::Mesh::GetPointMatrix (the vertex coordinates, column per vertex).
 * With continuous Nodes: GetElementVDofs lists all x dofs, then all y dofs, then all z dofs of
 * the element (byNODES within the element, mesh.cpp:376-388 reads them the same way); the first
 * n_corners dofs of each component are the vertex dofs.
 */
void ElementCornerPoints(const mfem::Mesh& mesh, int e, mfem::DenseMatrix& corners) {
    const mfem::GridFunction* nodes = ContinuousNodes(mesh);
    if (nodes == nullptr) {
        mesh.GetPointMatrix(e, corners);
        return;
    }
    const int sdim = mesh.SpaceDimension();
    const int num_corners = mesh.GetElement(e)->GetNVertices();
    mfem::Array<int> vdofs;
    nodes->FESpace()->GetElementVDofs(e, vdofs);
    const int num_dofs = vdofs.Size() / sdim;
    const double* data = nodes->HostRead();
    corners.SetSize(sdim, num_corners);
    for (int d = 0; d < sdim; ++d) {
        for (int k = 0; k < num_corners; ++k) {
            // A negative vdof marks a sign flip; vertex dofs never carry one, but decode it anyway
            const int vdof = vdofs[k + d * num_dofs];
            const double value = data[vdof >= 0 ? vdof : -1 - vdof];
            corners(d, k) = (vdof >= 0) ? value : -value;
        }
    }
}

/*
 * StraightElementTransformation: the same steps as Mesh::GetElementTransformation for a mesh
 * without Nodes (mesh.cpp:361-373), with the corner points taken from ElementCornerPoints.
 */
void StraightElementTransformation(const mfem::Mesh& mesh, int e,
                                   mfem::IsoparametricTransformation& trans) {
    trans.Attribute = mesh.GetAttribute(e);
    trans.ElementNo = e;
    trans.ElementType = mfem::ElementTransformation::ELEMENT;
    trans.mesh = &mesh;
    trans.Reset();
    ElementCornerPoints(mesh, e, trans.GetPointMat());
    trans.SetFE(mfem::Mesh::GetTransformationFEforElementType(mesh.GetElementType(e)));
}

/*
 * VertexPoint: GetVertexDofs gives the vertex's (single, scalar) dof of the Nodes space, and
 * DofToVDof the entry of each coordinate component (it handles both byNODES and byVDIM).
 */
void VertexPoint(const mfem::Mesh& mesh, int v, double* x) {
    const int sdim = mesh.SpaceDimension();
    const mfem::GridFunction* nodes = ContinuousNodes(mesh);
    if (nodes == nullptr) {
        const double* coords = mesh.GetVertex(v);
        for (int d = 0; d < sdim; ++d) {
            x[d] = coords[d];
        }
        return;
    }
    mfem::Array<int> dofs;
    nodes->FESpace()->GetVertexDofs(v, dofs);
    const double* data = nodes->HostRead();
    for (int d = 0; d < sdim; ++d) {
        x[d] = data[nodes->FESpace()->DofToVDof(dofs[0], d)];
    }
}

/*
 * StraightFaceTransformation: the steps of Mesh::GetFaceTransformation for a mesh without Nodes
 * (face number, FACE type, the corner point matrix from GetFaceVertices, and the
 * order-1 element of the face type), with the corners from VertexPoint.
 */
void StraightFaceTransformation(const mfem::Mesh& mesh, int f,
                                mfem::IsoparametricTransformation& trans) {
    // MFEM copies the face element's attribute, which is not reachable through the public Mesh
    // interface; it plays no part in the geometry, so the default face attribute 1 is used
    trans.Attribute = 1;
    trans.ElementNo = f;
    trans.ElementType = mfem::ElementTransformation::FACE;
    trans.mesh = &mesh;
    trans.Reset();
    mfem::Array<int> vertices;
    mesh.GetFaceVertices(f, vertices);
    const int sdim = mesh.SpaceDimension();
    mfem::DenseMatrix& pm = trans.GetPointMat();
    pm.SetSize(sdim, vertices.Size());
    double x[3];
    for (int j = 0; j < vertices.Size(); ++j) {
        VertexPoint(mesh, vertices[j], x);
        for (int d = 0; d < sdim; ++d) {
            pm(d, j) = x[d];
        }
    }
    trans.SetFE(mfem::Mesh::GetTransformationFEforElementType(mesh.GetFaceElementType(f)));
}

/*
 * StraightSidedLocal
 *
 * Steps (rules in the header):
 *  1. No Nodes -> true; discontinuous Nodes -> false.
 *  2. Continuous Nodes of order 1 in every local element -> true (nothing to test).
 *  3. Otherwise, per element: the corner points (ElementCornerPoints) and the element's nodal
 *     points (the Nodes values at its dofs), plus the reference positions of those dofs
 *     (FiniteElement::GetNodes). Evaluate the multilinear corner map at each reference position
 *     (shape functions of Mesh::GetTransformationFEforElementType, the order-1 element MFEM uses
 *     for meshes without Nodes) and compare with the nodal point, component by component. The
 *     first point outside the tolerance makes the answer false.
 */
bool StraightSidedLocal(const mfem::Mesh& mesh, double rel_tol) {
    // Step 1
    if (mesh.GetNodes() == nullptr) {
        return true;
    }
    const mfem::GridFunction* nodes = ContinuousNodes(mesh);
    if (nodes == nullptr) {
        return false;
    }
    const mfem::FiniteElementSpace* fes = nodes->FESpace();
    const int ne = mesh.GetNE();

    // Step 2: order 1 everywhere (a fixed-order space, or every element of a variable one)
    bool all_order_one = true;
    if (fes->IsVariableOrder()) {
        for (int e = 0; e < ne && all_order_one; ++e) {
            all_order_one = (fes->GetElementOrder(e) == 1);
        }
    } else {
        all_order_one = (fes->FEColl()->GetOrder() == 1);
    }
    if (all_order_one) {
        return true;
    }

    // Step 3: compare every nodal point with the corner map
    const int sdim = mesh.SpaceDimension();
    const double* data = nodes->HostRead();
    mfem::DenseMatrix corners;
    mfem::Vector shape;
    mfem::Array<int> vdofs;
    for (int e = 0; e < ne; ++e) {
        ElementCornerPoints(mesh, e, corners);
        const int num_corners = corners.Width();

        // Element size and coordinate magnitude from the corners
        double size = 0.0;
        double magnitude = 0.0;
        for (int d = 0; d < sdim; ++d) {
            double lo = corners(d, 0);
            double hi = corners(d, 0);
            for (int k = 0; k < num_corners; ++k) {
                lo = std::min(lo, corners(d, k));
                hi = std::max(hi, corners(d, k));
                magnitude = std::max(magnitude, std::fabs(corners(d, k)));
            }
            size = std::max(size, hi - lo);
        }
        const double tol = rel_tol * size + 16.0 * DBL_EPSILON * magnitude;

        // Reference positions of the element's dofs and the corner-map shape functions
        const mfem::FiniteElement* fe = fes->GetFE(e);
        const mfem::IntegrationRule& ref_nodes = fe->GetNodes();
        const mfem::FiniteElement* linear =
            mfem::Mesh::GetTransformationFEforElementType(mesh.GetElementType(e));
        shape.SetSize(linear->GetDof());
        fes->GetElementVDofs(e, vdofs);
        const int num_dofs = vdofs.Size() / sdim;
        for (int k = 0; k < num_dofs; ++k) {
            linear->CalcShape(ref_nodes.IntPoint(k), shape);
            for (int d = 0; d < sdim; ++d) {
                double mapped = 0.0;
                for (int c = 0; c < num_corners; ++c) {
                    mapped += shape(c) * corners(d, c);
                }
                const int vdof = vdofs[k + d * num_dofs];
                const double value = (vdof >= 0) ? data[vdof] : -data[-1 - vdof];
                if (!(std::fabs(value - mapped) <= tol)) {
                    return false;
                }
            }
        }
    }
    return true;
}

/*
 * IsStraightSided: StraightSidedLocal, then MPI_Allreduce(MIN) over 0/1 so that one curved
 * element anywhere makes every rank answer false.
 */
bool IsStraightSided(const mfem::Mesh& mesh, MPI_Comm comm, double rel_tol) {
    const int local = StraightSidedLocal(mesh, rel_tol) ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
    return global == 1;
}

} // namespace exaconstit::microstructure

