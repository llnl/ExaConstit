#include "microstructure/mesh/mesh_sampler.hpp"

#include "mfem.hpp"
#include "mfem/linalg/kernels.hpp" // mfem::kernels::Det (not included by mfem.hpp)

#include <algorithm>
#include <limits>
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
 * `ir` points to the rule (an entry of MFEM's global table `mfem::IntRules`, or a caller's rule
 * that outlives the factors) and is not owned.
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
 *        elements, at the points of `rule`.
 *
 * Steps:
 *  1. Keep a pointer to the rule (one rule for all elements: the mesh has one geometry).
 *  2. If the mesh has Nodes (curved or high-order mesh), build `GeometricFactors` from them.
 *  3. Otherwise the geometry is defined by the vertices only. Copy the vertex coordinates into
 *     a temporary order-1 H1 GridFunction (`Mesh::GetNodes(GridFunction&)`, `mesh.cpp:6977`)
 *     and build the factors from that. `Mesh::GetGeometricFactors` would instead call
 *     `EnsureNodes()` (`mesh.cpp:896`), which permanently attaches Nodes to the mesh.
 *
 * The factor arrays are allocated in MFEM's device memory type and filled by MFEM's
 * QuadratureInterpolator, on the GPU when the `mfem::Device` has a GPU backend.
 *
 * Precondition: `mesh.GetNE() > 0` and the mesh passed RequireSampleableMesh. `rule` is a rule
 * on the reference element of the mesh's geometry and must outlive the returned factors.
 * Not collective; host only.
 */
SampleFactors BuildSampleFactors(mfem::Mesh& mesh, const mfem::IntegrationRule& rule) {
    SampleFactors sf;

    // Step 1: one rule for all elements; RequireSampleableMesh ensured one geometry.
    sf.ir = &rule;
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

/**
 * @brief Evaluates the points of one integration rule element range by element range, keeping
 * only per-element node data for the whole local mesh (QuadraturePointSampler's storage).
 *
 * Why not GeometricFactors: it evaluates every local element at once (it has no element range),
 * so it holds 4 values per point for the whole mesh, e.g. 4 * 512 doubles = 16 KiB per element
 * for a 512-point rule. Here only the inputs of that evaluation are kept for the whole mesh:
 *  - the element-wise node coordinates (the "E-vector" of the node space, from its element
 *    restriction with NATIVE dof ordering: E[d + nd (c + 3 e)] = coordinate c of node d of
 *    element e, restriction.cpp ElementRestriction::Mult), 3 nd values per element;
 *  - the shape-function tables of the rule on the reference element (DofToQuad in FULL mode,
 *    fe_base.cpp:373-460: B[q + nq d] = phi_d(xi_q), G[q + nq (k + 3 d)] = d phi_d / d xi_k).
 * Points are then evaluated for one range at a time (Evaluate).
 *
 * For a mesh without Nodes the temporary order-1 node space is owned here (fec, fes, nodes); for a
 * mesh with Nodes those stay null and the mesh's own Nodes are used. `fes` points to `fec` and
 * `nodes` to `fes`, so they are declared in that order (destroyed in reverse). `maps` points into
 * a cache inside the node space's finite element (owned by `fec`, or by the mesh's Nodes
 * collection), keyed by the rule's address: the rule must outlive this object.
 */
struct PointEvaluator {
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    std::unique_ptr<mfem::FiniteElementSpace> fes;
    std::unique_ptr<mfem::GridFunction> nodes;
    const mfem::IntegrationRule* ir = nullptr;
    const mfem::DofToQuad* maps = nullptr;
    mfem::Vector element_nodes; ///< E-vector, nd x 3 x ne
    int nd = 0;                 ///< nodes (dofs) per element
    int ne = 0;                 ///< local elements
};

/**
 * @brief Builds the PointEvaluator (precondition: mesh.GetNE() > 0, RequireSampleableMesh passed).
 *
 * Steps:
 *  1. The node GridFunction: the mesh's Nodes, or a temporary order-1 H1 copy of the vertices
 *     (Mesh::GetNodes(GridFunction&), mesh.cpp:6977; Mesh::EnsureNodes() would permanently attach
 *     Nodes to the mesh).
 *  2. The FULL-mode shape tables of the rule for the node space's element (GetTypicalFE: one
 *     geometry and one order; variable-order node spaces are refused).
 *  3. The E-vector: the element restriction (NATIVE ordering, the dof order of the FULL tables)
 *     applied to the node coordinates, in the memory where the mfem::Device runs kernels.
 * Not collective; host function (the restriction runs on the mfem::Device backend).
 */
void BuildPointEvaluator(mfem::Mesh& mesh, const mfem::IntegrationRule& rule, PointEvaluator& pe) {
    // Step 1: node coordinates
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (nodes == nullptr) {
        pe.fec = std::make_unique<mfem::H1_FECollection>(1, mesh.Dimension());
        pe.fes = std::make_unique<mfem::FiniteElementSpace>(
            &mesh, pe.fec.get(), mesh.SpaceDimension(), mfem::Ordering::byNODES);
        pe.nodes = std::make_unique<mfem::GridFunction>(pe.fes.get());
        mesh.GetNodes(*pe.nodes);
        nodes = pe.nodes.get();
    }
    const mfem::FiniteElementSpace& space = *nodes->FESpace();
    if (space.IsVariableOrder() || space.GetVDim() != 3) {
        throw std::runtime_error("QuadraturePointSampler: the mesh nodes must be a fixed-order "
                                 "space with 3 components");
    }

    // Step 2: shape-function values and reference gradients at the rule's points
    pe.ir = &rule;
    pe.maps = &space.GetTypicalFE()->GetDofToQuad(rule, mfem::DofToQuad::FULL);
    pe.nd = pe.maps->ndof;
    pe.ne = mesh.GetNE();

    // Step 3: element-wise node coordinates
    const mfem::Operator* restriction =
        space.GetElementRestriction(mfem::ElementDofOrdering::NATIVE);
    pe.element_nodes.SetSize(3 * pe.nd * pe.ne, mfem::Device::GetDeviceMemoryType());
    restriction->Mult(*nodes, pe.element_nodes);
}

/**
 * @brief Coordinates and weights w_q detJ_q of the points of local elements [begin, end).
 *
 * One mfem::forall iteration per output entry i = (e - begin) * nq + q:
 *   x_c   = sum_d phi_d(xi_q) X_{d,c}                       (position)
 *   D_ck  = sum_d X_{d,c} d phi_d / d xi_k                  (Jacobian dx_c / dxi_k)
 *   detJ  = det(D)
 * with the same loop order and the same operations as MFEM's non-tensor interpolation kernel
 * (qinterp/eval.hpp, ImplEval3D: value sum b * x, Jacobian sum x * g, kernels::Det<3> on the
 * column-major D), so the result is bitwise what GeometricFactors gives for tets. For hexes
 * GeometricFactors uses the tensor-product (sum-factorized) kernel instead, which adds the same
 * terms in a different order, so the results agree to round-off only.
 * MFEM indexes with int, so the range's point count must fit in an int. Host function launching
 * one kernel.
 */
void EvaluatePoints(const PointEvaluator& pe, int begin, int end, QuadraturePointSamples& out) {
    const int nq = pe.ir->GetNPoints();
    if (static_cast<long long>(nq) * (end - begin) > std::numeric_limits<int>::max()) {
        throw std::runtime_error("ComputeQuadraturePointSamples: too many points in the element "
                                 "range for an int index");
    }
    const int num_out = nq * (end - begin);
    out.num_qp = nq;
    out.elem_begin = begin;
    out.elem_end = end;
    out.x.SetSize(num_out);
    out.y.SetSize(num_out);
    out.z.SetSize(num_out);
    out.weight.SetSize(num_out);
    if (num_out == 0) {
        return;
    }

    // Kernel-side views: shape tables, element nodes, rule weights and the outputs
    const int nd = pe.nd;
    const auto B = mfem::Reshape(pe.maps->B.Read(), nq, nd);
    const auto G = mfem::Reshape(pe.maps->G.Read(), nq, 3, nd);
    const auto X = mfem::Reshape(pe.element_nodes.Read(), nd, 3, pe.ne);
    const mfem::real_t* w = pe.ir->GetWeights().Read();
    mfem::real_t* qx = out.x.Write();
    mfem::real_t* qy = out.y.Write();
    mfem::real_t* qz = out.z.Write();
    mfem::real_t* qw = out.weight.Write();

    mfem::forall(num_out, [=] MFEM_HOST_DEVICE(int i) {
        const int e = begin + i / nq; // local element id
        const int q = i % nq;         // point within the element

        // Position: shape-function interpolation of the element's node coordinates
        mfem::real_t pos[3] = {0.0, 0.0, 0.0};
        for (int d = 0; d < nd; ++d) {
            const mfem::real_t b = B(q, d);
            for (int c = 0; c < 3; c++) {
                pos[c] += b * X(d, c, e);
            }
        }

        // Jacobian D[c + 3 k] = dx_c / dxi_k (column-major), then its determinant
        mfem::real_t jac[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int d = 0; d < nd; ++d) {
            const mfem::real_t wx = G(q, 0, d);
            const mfem::real_t wy = G(q, 1, d);
            const mfem::real_t wz = G(q, 2, d);
            for (int c = 0; c < 3; c++) {
                const mfem::real_t node_c = X(d, c, e);
                jac[c + 3 * 0] += node_c * wx;
                jac[c + 3 * 1] += node_c * wy;
                jac[c + 3 * 2] += node_c * wz;
            }
        }
        const mfem::real_t det = mfem::kernels::Det<3>(jac);

        qx[i] = pos[0];
        qy[i] = pos[1];
        qz[i] = pos[2];
        qw[i] = w[q] * det;
    });
}

/** @brief An empty range result (a rank without elements): no points, num_qp = 0. */
void SetEmptySamples(int begin, int end, QuadraturePointSamples& out) {
    out.num_qp = 0;
    out.elem_begin = begin;
    out.elem_end = end;
    out.x.SetSize(0);
    out.y.SetSize(0);
    out.z.SetSize(0);
    out.weight.SetSize(0);
}

/** @brief Throws unless 0 <= begin <= end <= ne (an element range of the local mesh). */
void RequireElementRange(int begin, int end, int ne, const char* caller) {
    if (begin < 0 || end < begin || end > ne) {
        throw std::runtime_error(std::string(caller) + ": invalid element range [" +
                                 std::to_string(begin) + ", " + std::to_string(end) + ") for " +
                                 std::to_string(ne) + " local elements");
    }
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
    const SampleFactors sf =
        BuildSampleFactors(mesh, mfem::IntRules.Get(mesh.GetElementGeometry(0), order));
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
 * ComputeQuadraturePointSamples (quadrature-order overload)
 *
 * Validates the mesh, resolves the quadrature order, and calls the rule overload with MFEM's
 * standard rule of that order for the mesh's element geometry (mfem::IntRules.Get, whose rules
 * live in a global table for the whole program). A rank with no elements returns empty Vectors
 * (num_qp = 0) without looking up a rule, since there is no element to read the geometry from.
 */
void ComputeQuadraturePointSamples(mfem::Mesh& mesh,
                                   int quad_order,
                                   int elem_begin,
                                   int elem_end,
                                   QuadraturePointSamples& out) {
    RequireSampleableMesh(mesh, "ComputeQuadraturePointSamples");
    const int order = ResolveQuadOrder(mesh, quad_order);
    const int ne = mesh.GetNE();
    RequireElementRange(elem_begin, elem_end, ne, "ComputeQuadraturePointSamples");
    if (ne == 0) {
        SetEmptySamples(elem_begin, elem_end, out);
        return;
    }
    ComputeQuadraturePointSamples(mesh, mfem::IntRules.Get(mesh.GetElementGeometry(0), order),
                                  elem_begin, elem_end, out);
}

/*
 * ComputeQuadraturePointSamples (rule overload)
 *
 * Steps:
 *  1. Validate the mesh and the element range.
 *  2. Evaluate the rule's points on the requested range only (QuadraturePointSampler, which keeps
 *     per-element node data and evaluates a range at a time).
 */
void ComputeQuadraturePointSamples(mfem::Mesh& mesh,
                                   const mfem::IntegrationRule& rule,
                                   int elem_begin,
                                   int elem_end,
                                   QuadraturePointSamples& out) {
    // Step 1: checks (also done by the sampler; repeated here so the range error names this
    // function)
    RequireSampleableMesh(mesh, "ComputeQuadraturePointSamples");
    RequireElementRange(elem_begin, elem_end, mesh.GetNE(), "ComputeQuadraturePointSamples");

    // Step 2: points of the requested range only
    const QuadraturePointSampler sampler(mesh, rule);
    sampler.Sample(elem_begin, elem_end, out);
}

/*
 * QuadraturePointSampler
 *
 * Factors (declared in the header, defined here) wraps the file-local PointEvaluator, so the
 * header does not expose MFEM's interpolation setup. The constructor checks the mesh and, on a
 * rank with elements, gathers the element node coordinates and the shape tables once
 * (BuildPointEvaluator); Sample() evaluates one range (EvaluatePoints). On a rank without
 * elements Sample() returns empty Vectors with num_qp = 0, exactly like the quadrature-order
 * overload always did.
 */
struct QuadraturePointSampler::Factors {
    PointEvaluator pe;
};

QuadraturePointSampler::QuadraturePointSampler(mfem::Mesh& mesh,
                                               const mfem::IntegrationRule& rule)
    : m_mesh(mesh), m_rule(rule) {
    RequireSampleableMesh(mesh, "QuadraturePointSampler");
    if (mesh.GetNE() > 0) {
        m_factors = std::make_unique<Factors>();
        BuildPointEvaluator(mesh, rule, m_factors->pe);
    }
}

QuadraturePointSampler::~QuadraturePointSampler() = default;

int QuadraturePointSampler::NumPointsPerElement() const {
    if (!m_factors) {
        return 0;
    }
    return m_rule.GetNPoints();
}

void QuadraturePointSampler::Sample(int elem_begin, int elem_end,
                                    QuadraturePointSamples& out) const {
    const int ne = m_mesh.GetNE();
    RequireElementRange(elem_begin, elem_end, ne, "QuadraturePointSampler::Sample");
    if (!m_factors) {
        SetEmptySamples(elem_begin, elem_end, out); // no local elements
        return;
    }
    EvaluatePoints(m_factors->pe, elem_begin, elem_end, out);
}

} // namespace exaconstit::microstructure
