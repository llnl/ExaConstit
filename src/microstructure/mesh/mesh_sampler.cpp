#include "microstructure/mesh/mesh_sampler.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/**
 * @brief Owns the geometric factors and everything they were computed from.
 *
 * `GeometricFactors` is computed from a node GridFunction and an integration rule. For a mesh
 * without Nodes the temporary order-1 space (`fec`, `fes`, `nodes`) is owned here; for a mesh
 * with Nodes those three stay null and the mesh's own Nodes are used.
 * `ir` points into MFEM's global rule table `mfem::IntRules` and is not owned.
 *
 * The member order matters. C++ destroys members in reverse declaration order, so each member
 * must be declared after everything it points to:
 *  - `fes` points to `fec`, and `nodes` points to `fes`;
 *  - `factors` must stay declared last, so it is destroyed first, before `nodes`, `fes` and
 *    `fec`. In this MFEM version it keeps only the mesh and rule pointers
 *    (`mesh.hpp:3093-3094`), but it is conceptually a view of data computed from them and must
 *    never outlive them.
 */
struct SampleFactors {
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    std::unique_ptr<mfem::FiniteElementSpace> fes;
    std::unique_ptr<mfem::GridFunction> nodes;
    const mfem::IntegrationRule* ir = nullptr;
    std::unique_ptr<mfem::GeometricFactors> factors;
};

/**
 * @brief Throws unless the mesh is a 3D volume mesh with a single local element geometry.
 *
 * Both samplers index `GeometricFactors::X` assuming 3 coordinates per point, and
 * GeometricFactors uses one integration rule (hence one geometry) for all elements. Local
 * check only; host only.
 */
void RequireSampleableMesh(const mfem::Mesh& mesh, const char* caller) {
    if (mesh.SpaceDimension() != 3 || mesh.Dimension() != 3) {
        throw std::runtime_error(std::string(caller) + ": only 3D volume meshes are supported");
    }
    if (mesh.GetNumGeometries(mesh.Dimension()) > 1) {
        throw std::runtime_error(std::string(caller) +
                                 ": mixed element geometries are not supported (run CheckMesh)");
    }
}

/**
 * @brief Builds the geometric factors (quadrature-point coordinates and detJ) of all local
 *        elements.
 *
 * Steps:
 *  1. Pick the quadrature rule for the (single) element geometry.
 *  2. If the mesh has Nodes (curved or high-order mesh), build `GeometricFactors` from them.
 *  3. Otherwise the geometry is defined by the vertices only. Copy the vertex coordinates into
 *     a temporary order-1 H1 GridFunction (`Mesh::GetNodes(GridFunction&)`, `mesh.cpp:6977`)
 *     and build the factors from that. `Mesh::GetGeometricFactors` would instead call
 *     `EnsureNodes()` (`mesh.cpp:896`), which permanently attaches Nodes to the mesh.
 *
 * The factor arrays are allocated in MFEM's device memory type and filled by MFEM's
 * QuadratureInterpolator, on the GPU when the `mfem::Device` has a GPU backend.
 *
 * Precondition: `mesh.GetNE() > 0` (the element geometry is read from element 0) and the mesh
 * passed RequireSampleableMesh. `order` is the quadrature order (already resolved, >= 0).
 * Not collective; host only.
 */
SampleFactors BuildSampleFactors(mfem::Mesh& mesh, int order) {
    SampleFactors sf;

    // Step 1: one rule for all elements; RequireSampleableMesh ensured one geometry.
    sf.ir = &mfem::IntRules.Get(mesh.GetElementGeometry(0), order);
    const int flags = mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::DETERMINANTS;

    if (mesh.GetNodes() != nullptr) {
        // Step 2: high-order or curved mesh, use its own Nodes.
        sf.factors = std::make_unique<mfem::GeometricFactors>(*mesh.GetNodes(), *sf.ir, flags);
    } else {
        // Step 3: vertex-only mesh. byNODES ordering stores all x values, then all y, then all
        // z, which is what Mesh::GetNodes(GridFunction&) fills.
        sf.fec = std::make_unique<mfem::H1_FECollection>(1, mesh.Dimension());
        sf.fes = std::make_unique<mfem::FiniteElementSpace>(
            &mesh, sf.fec.get(), mesh.SpaceDimension(), mfem::Ordering::byNODES);
        sf.nodes = std::make_unique<mfem::GridFunction>(sf.fes.get());
        mesh.GetNodes(*sf.nodes);
        sf.factors = std::make_unique<mfem::GeometricFactors>(*sf.nodes, *sf.ir, flags);
    }
    return sf;
}

/** @brief Resolves the user quadrature order: negative means DefaultSampleQuadOrder(mesh). */
int ResolveQuadOrder(const mfem::Mesh& mesh, int quad_order) {
    if (quad_order < 0) {
        return DefaultSampleQuadOrder(mesh);
    }
    return quad_order;
}

} // namespace

/*
 * DefaultSampleQuadOrder
 *
 * Finds the polynomial order p of the mesh geometry and returns 4p - 1 (see the header for why
 * that is exact).
 *  - No Nodes: the geometry is the vertex positions, i.e. order 1.
 *  - Fixed-order Nodes space: p is the order of its finite element collection.
 *  - Variable-order Nodes space: p is the largest order among this rank's elements.
 *
 * We deliberately avoid `FESpace()->GetMaxElementOrder()`: it is virtual, and the
 * ParFiniteElementSpace override does an MPI_Allreduce for variable-order spaces
 * (`pfespace.cpp:5081-5093`). That would make this function, and the samplers that call it,
 * collective, and a rank with no elements (which returns early from the samplers) would leave
 * the others waiting forever. The local maximum is still exact on every local element.
 */
int DefaultSampleQuadOrder(const mfem::Mesh& mesh) {
    int max_order = 1;
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (nodes != nullptr) {
        const mfem::FiniteElementSpace* fes = nodes->FESpace();
        // Fixed order (the usual case), and the fallback for a variable-order space on a rank
        // with no elements.
        max_order = fes->FEColl()->GetOrder();
        if (fes->IsVariableOrder() && fes->GetNE() > 0) {
            max_order = 0;
            for (int e = 0; e < fes->GetNE(); e++) {
                max_order = std::max(max_order, fes->GetElementOrder(e));
            }
        }
    }
    return 4 * max_order - 1;
}


/*
 * ComputeElementSamples
 *
 * Steps:
 *  1. Validate the mesh, resolve the quadrature order, and size the outputs (one entry per
 *     local element). This happens before the empty-rank early return so every rank does the
 *     same checks.
 *  2. Build the geometric factors for all local elements (BuildSampleFactors).
 *  3. Get pointers for the kernel:
 *     - `Read()` returns a read-only pointer to a Vector's data in the memory where kernels run
 *       (GPU memory with a CUDA/HIP Device, host memory otherwise), copying it there first if
 *       the valid copy is elsewhere.
 *     - `Write()` returns a pointer in the same place for data that will be completely
 *       overwritten, so nothing is copied.
 *     - `mfem::Reshape(ptr, n1, n2, ...)` wraps a raw pointer as a multi-dimensional array
 *       (an `mfem::DeviceTensor`) indexed as `A(i, j, k)` with the first index varying fastest.
 *       So `X(q, d, e)` is `X[q + NQ*(d + 3*e)]`, exactly the byNODES layout of
 *       GeometricFactors, and `detJ(q, e)` is `detJ[q + NQ*e]`. The tensor does not own the
 *       memory.
 *  4. `mfem::forall(n, body)` runs `body(i)` for i = 0..n-1: as a GPU kernel when the Device
 *     has a CUDA/HIP backend, otherwise as a host loop. `MFEM_HOST_DEVICE` marks the lambda as
 *     callable from GPU code, and `[=]` captures the pointers and tensors by value (a GPU kernel
 *     cannot use references to host stack variables). One iteration per element e: accumulate
 *     w_q * detJ_q (volume) and w_q * detJ_q * x_q (first moment) over the element's quadrature
 *     points, then divide the moment by the volume to get the centroid.
 */
void ComputeElementSamples(mfem::Mesh& mesh, int quad_order, ElementSamples& out) {
    // Step 1: checks, quadrature order, output sizes.
    RequireSampleableMesh(mesh, "ComputeElementSamples");
    const int order = ResolveQuadOrder(mesh, quad_order);
    const int ne = mesh.GetNE();
    out.cx.SetSize(ne);
    out.cy.SetSize(ne);
    out.cz.SetSize(ne);
    out.volume.SetSize(ne);
    if (ne == 0) {
        return;
    }

    // Step 2: quadrature-point coordinates and Jacobian determinants of all local elements.
    const SampleFactors sf = BuildSampleFactors(mesh, order);
    const int nq = sf.ir->GetNPoints();

    // Step 3: kernel-side views. `sf` owns the factor data until the end of this function.
    const auto X = mfem::Reshape(sf.factors->X.Read(), nq, 3, ne);
    const auto detJ = mfem::Reshape(sf.factors->detJ.Read(), nq, ne);
    const mfem::real_t* w = sf.ir->GetWeights().Read();
    mfem::real_t* cx = out.cx.Write();
    mfem::real_t* cy = out.cy.Write();
    mfem::real_t* cz = out.cz.Write();
    mfem::real_t* vol = out.volume.Write();

    // Step 4: one iteration per element.
    mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e) {
        mfem::real_t elem_volume = 0.0;
        mfem::real_t moment_x = 0.0;
        mfem::real_t moment_y = 0.0;
        mfem::real_t moment_z = 0.0;
        for (int q = 0; q < nq; q++) {
            const mfem::real_t w_detj = w[q] * detJ(q, e);
            elem_volume += w_detj;
            moment_x += w_detj * X(q, 0, e);
            moment_y += w_detj * X(q, 1, e);
            moment_z += w_detj * X(q, 2, e);
        }
        vol[e] = elem_volume;
        cx[e] = moment_x / elem_volume;
        cy[e] = moment_y / elem_volume;
        cz[e] = moment_z / elem_volume;
    });
}

/*
 * ComputeQuadraturePointSamples
 *
 * Steps:
 *  1. Validate the mesh and the element range; resolve the quadrature order. A rank with no
 *     elements returns empty Vectors.
 *  2. Build the geometric factors for all local elements (they cannot be restricted to a
 *     range), then size the outputs to num_qp * (elem_end - elem_begin).
 *  3. Get Read()/Write() pointers and Reshape views as in ComputeElementSamples.
 *  4. One mfem::forall iteration per output entry i: i = (e - elem_begin) * nq + q, so
 *     e = elem_begin + i / nq and q = i % nq. Copy the point's coordinates out of the
 *     GeometricFactors array and store w_q * detJ_q as its weight.
 */
void ComputeQuadraturePointSamples(mfem::Mesh& mesh,
                                   int quad_order,
                                   int elem_begin,
                                   int elem_end,
                                   QuadraturePointSamples& out) {
    // Step 1: checks and the empty-rank case.
    RequireSampleableMesh(mesh, "ComputeQuadraturePointSamples");
    const int order = ResolveQuadOrder(mesh, quad_order);
    const int ne = mesh.GetNE();
    if (elem_begin < 0 || elem_end < elem_begin || elem_end > ne) {
        throw std::runtime_error("ComputeQuadraturePointSamples: invalid element range [" +
                                 std::to_string(elem_begin) + ", " + std::to_string(elem_end) +
                                 ") for " + std::to_string(ne) + " local elements");
    }

    out.elem_begin = elem_begin;
    out.elem_end = elem_end;
    out.num_qp = 0;
    if (ne == 0) {
        out.x.SetSize(0);
        out.y.SetSize(0);
        out.z.SetSize(0);
        out.weight.SetSize(0);
        return;
    }

    // Step 2: factors for every local element; outputs for the requested range only. MFEM
    // indexes with int, so the output size must fit in an int.
    const SampleFactors sf = BuildSampleFactors(mesh, order);
    const int nq = sf.ir->GetNPoints();
    out.num_qp = nq;
    const int num_out = nq * (elem_end - elem_begin);
    out.x.SetSize(num_out);
    out.y.SetSize(num_out);
    out.z.SetSize(num_out);
    out.weight.SetSize(num_out);
    if (num_out == 0) {
        return;
    }

    // Step 3: kernel-side views over the factor data (owned by `sf`) and the outputs.
    const auto X = mfem::Reshape(sf.factors->X.Read(), nq, 3, ne);
    const auto detJ = mfem::Reshape(sf.factors->detJ.Read(), nq, ne);
    const mfem::real_t* w = sf.ir->GetWeights().Read();
    mfem::real_t* qx = out.x.Write();
    mfem::real_t* qy = out.y.Write();
    mfem::real_t* qz = out.z.Write();
    mfem::real_t* qw = out.weight.Write();
    const int begin = elem_begin;

    // Step 4: one iteration per (element, quadrature point) pair of the range.
    mfem::forall(num_out, [=] MFEM_HOST_DEVICE(int i) {
        const int e = begin + i / nq; // local element id
        const int q = i % nq;         // quadrature point within the element
        qx[i] = X(q, 0, e);
        qy[i] = X(q, 1, e);
        qz[i] = X(q, 2, e);
        qw[i] = w[q] * detJ(q, e);
    });
}

} // namespace exaconstit::microstructure
