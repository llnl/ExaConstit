/*
 * Implementation of periodic seed images (periodic_images.hpp): wrap the seeds into the periodic
 * box, append copies shifted by whole box lengths, and map labels of copies back to the original
 * seeds.
 *
 * Terms used in this file:
 * - Image: a copy of a seed shifted by a whole number of box lengths along the periodic axes. In
 *   the infinite periodic microstructure every seed exists at all these positions.
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it (here: MPI_Allreduce of the mesh bounding box).
 * - mfem::forall: MFEM's parallel loop; it runs as a GPU kernel when the mfem::Device has a CUDA
 *   or HIP backend and as a plain host loop otherwise.
 */
#include "microstructure/geometry/periodic_images.hpp"

#include "microstructure/geometry/metric_functors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/**
 * @brief Moves coordinate x into [lo, lo + length) by a whole number of box lengths.
 *
 * A coordinate already in [lo, lo + length) is returned bitwise unchanged (tested explicitly:
 * the division below could round up to 1 for a coordinate within an ulp of the upper face).
 * Otherwise x - floor((x - lo) / length) * length; round-off can land that exactly on
 * lo + length (or a hair below lo), and those two cases are snapped back explicitly so the
 * result is always inside [lo, lo + length). Host only.
 */
double WrapIntoBox(double x, double lo, double length) {
    if (x >= lo && x < lo + length) {
        return x;
    }
    const double shifts = std::floor((x - lo) / length);
    double wrapped = x - shifts * length;
    if (wrapped >= lo + length) {
        wrapped -= length;
    }
    if (wrapped < lo) {
        wrapped = lo;
    }
    return wrapped;
}

/** @brief Coordinate array d (0 = x, 1 = y, 2 = z) of the seed set. Host only. */
const std::vector<double>& SeedCoordinate(const SeedSet& seeds, int d) {
    if (d == 0) {
        return seeds.x;
    }
    if (d == 1) {
        return seeds.y;
    }
    return seeds.z;
}

/**
 * @brief Appends the per-seed data of input seed i to `out` (all optional arrays that are
 * non-empty in `in`), with position (x, y, z). Host only.
 */
void AppendSeed(const SeedSet& in, std::size_t i, double x, double y, double z, SeedSet& out) {
    out.x.push_back(x);
    out.y.push_back(y);
    out.z.push_back(z);
    if (!in.weight.empty()) {
        out.weight.push_back(in.weight[i]);
    }
    if (!in.shape.empty()) {
        for (std::size_t k = 0; k < 6; ++k) {
            out.shape.push_back(in.shape[6 * i + k]);
        }
    }
    if (!in.quat.empty()) {
        for (std::size_t k = 0; k < 4; ++k) {
            out.quat.push_back(in.quat[4 * i + k]);
        }
    }
    if (!in.source_id.empty()) {
        out.source_id.push_back(in.source_id[i]);
    }
    if (!in.phase.empty()) {
        out.phase.push_back(in.phase[i]);
    }
    if (!in.parent_id.empty()) {
        out.parent_id.push_back(in.parent_id[i]);
    }
    if (!in.nucleation_time.empty()) {
        out.nucleation_time.push_back(in.nucleation_time[i]);
    }
}

/**
 * @brief Throws unless every non-empty optional array of `seeds` has the size N (or 6N / 4N)
 * that AppendSeed indexes, so a malformed set cannot read out of bounds. Host only.
 */
void RequireConsistentSeedArrays(const SeedSet& seeds) {
    const std::size_t n = seeds.Size();
    auto check = [n](std::size_t size, std::size_t per_seed, const char* name) {
        if (size != 0 && size != per_seed * n) {
            throw std::invalid_argument(std::string("BuildPeriodicImages: SeedSet::") + name +
                                        " has the wrong size");
        }
    };
    if (seeds.y.size() != n || seeds.z.size() != n) {
        throw std::invalid_argument("BuildPeriodicImages: SeedSet x, y, z sizes differ");
    }
    check(seeds.weight.size(), 1, "weight");
    check(seeds.shape.size(), 6, "shape");
    check(seeds.quat.size(), 4, "quat");
    check(seeds.source_id.size(), 1, "source_id");
    check(seeds.phase.size(), 1, "phase");
    check(seeds.parent_id.size(), 1, "parent_id");
    check(seeds.nucleation_time.size(), 1, "nucleation_time");
}

} // namespace

/*
 * PeriodicBoxFromMesh
 *
 * Algorithm:
 *  1. Local minimum and maximum of each coordinate over this rank's node coordinates:
 *     - a mesh with Nodes (curved, high-order, or topologically periodic): every node dof of the
 *       Nodes GridFunction, component c of dof i read through FiniteElementSpace::DofToVDof
 *       (which handles both the byNODES and the byVDIM storage order);
 *     - otherwise the vertices (Mesh::GetVertex, mesh.hpp:1419).
 *     Why Nodes first: in a topologically periodic MFEM mesh (Mesh::MakePeriodic) the vertices
 *     on opposite faces are merged into one, so the vertices alone end one element short of the
 *     far face (x in [0, 0.75] for a 4-element unit box); its discontinuous (L2) Nodes keep every
 *     element's own copy of its corners and span the full box. Starting values +inf / -inf make
 *     a rank without nodes neutral in the reduction.
 *  2. MPI_Allreduce with MPI_MIN on the minima and MPI_MAX on the maxima: afterwards every rank
 *     holds the bounding box of the whole mesh. MIN and MAX do not depend on the order in which
 *     ranks contribute, so the box is the same for every rank count and partition.
 *  3. Check the extents (identical on every rank, so every rank throws together).
 * Mesh::GetBoundingBox is not used: it refines elements and is local only.
 */
PeriodicBox PeriodicBoxFromMesh(const mfem::Mesh& local_mesh,
                                MPI_Comm comm,
                                const std::array<bool, 3>& periodic) {
    // Step 1: local extent over the vertices
    const double inf = std::numeric_limits<double>::infinity();
    double local_lo[3] = {inf, inf, inf};
    double local_hi[3] = {-inf, -inf, -inf};
    const int sdim = local_mesh.SpaceDimension();
    if (sdim != 3) {
        // Every rank holds the same kind of mesh, so this throws on every rank together
        throw std::invalid_argument("PeriodicBoxFromMesh: only 3D meshes are supported");
    }
    const mfem::GridFunction* nodes = local_mesh.GetNodes();
    if (nodes != nullptr) {
        // Node dofs of the Nodes GridFunction (host copy)
        const mfem::FiniteElementSpace* space = nodes->FESpace();
        const double* data = nodes->HostRead();
        for (int i = 0; i < space->GetNDofs(); ++i) {
            for (int d = 0; d < 3; ++d) {
                const double c = data[space->DofToVDof(i, d)];
                local_lo[d] = std::min(local_lo[d], c);
                local_hi[d] = std::max(local_hi[d], c);
            }
        }
    } else {
        for (int v = 0; v < local_mesh.GetNV(); ++v) {
            const double* coords = local_mesh.GetVertex(v);
            for (int d = 0; d < 3; ++d) {
                local_lo[d] = std::min(local_lo[d], coords[d]);
                local_hi[d] = std::max(local_hi[d], coords[d]);
            }
        }
    }

    // Step 2: global box
    double global_lo[3];
    double global_hi[3];
    MPI_Allreduce(local_lo, global_lo, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local_hi, global_hi, 3, MPI_DOUBLE, MPI_MAX, comm);

    // Step 3: validate and fill
    PeriodicBox box;
    box.periodic = periodic;
    for (int d = 0; d < 3; ++d) {
        if (!(global_hi[d] >= global_lo[d]) || !std::isfinite(global_lo[d]) ||
            !std::isfinite(global_hi[d])) {
            throw std::invalid_argument(
                "PeriodicBoxFromMesh: the mesh has no nodes or non-finite coordinates");
        }
        if (periodic[d] && !(global_hi[d] > global_lo[d])) {
            throw std::invalid_argument("PeriodicBoxFromMesh: periodic axis " +
                                        std::to_string(d) + " has zero extent");
        }
        box.lo[d] = global_lo[d];
        box.hi[d] = global_hi[d];
    }
    return box;
}

/*
 * BuildPeriodicImages
 *
 * Algorithm:
 *  1. Validate the inputs and resolve, per periodic axis, the shift range n_d in {-k..k}
 *     (k = params.layers) and the margin (negative = k box lengths, which keeps every image).
 *     Non-periodic axes have the single shift n_d = 0.
 *  2. Wrap every seed into the box on the periodic axes (WrapIntoBox) and store the wrapped
 *     originals first, so original[i] = i and the original seeds keep their indices.
 *  3. Loop over all shift vectors n (n_x slowest, then n_y, then n_z; n = 0 skipped because
 *     those are the originals) and, for each, over the seeds in index order. The image position
 *     is the wrapped position + n_d L_d. It is kept if it lies within the margin of the box on
 *     every periodic axis; then all per-seed data is copied and original[k] = i is recorded.
 *     The order is fixed by the loops, so every rank builds the identical list.
 *
 * The count is checked against 2^31 - 1 before allocating, because the seed search and the label
 * arrays use int indices.
 */
PeriodicImages BuildPeriodicImages(const SeedSet& seeds,
                                   const PeriodicBox& box,
                                   const PeriodicImageParameters& params) {
    // Step 1a: input checks
    const std::size_t num_seeds = seeds.Size();
    if (num_seeds == 0) {
        throw std::invalid_argument("BuildPeriodicImages: the seed set is empty");
    }
    if (num_seeds > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("BuildPeriodicImages: more than 2^31 - 1 seeds");
    }
    RequireConsistentSeedArrays(seeds);
    if (params.layers < 1) {
        throw std::invalid_argument("BuildPeriodicImages: layers must be >= 1");
    }
    for (std::size_t i = 0; i < num_seeds; ++i) {
        if (!std::isfinite(seeds.x[i]) || !std::isfinite(seeds.y[i]) ||
            !std::isfinite(seeds.z[i])) {
            throw std::invalid_argument("BuildPeriodicImages: seed " + std::to_string(i) +
                                        " has a non-finite position");
        }
    }

    // Step 1b: per-axis box length, shift range and margin
    double length[3] = {0.0, 0.0, 0.0};
    int max_shift[3] = {0, 0, 0};
    double margin[3] = {0.0, 0.0, 0.0};
    std::int64_t copies_per_seed = 1;
    for (int d = 0; d < 3; ++d) {
        if (!box.periodic[d]) {
            continue;
        }
        length[d] = box.hi[d] - box.lo[d];
        if (!(length[d] > 0.0) || !std::isfinite(length[d])) {
            throw std::invalid_argument("BuildPeriodicImages: periodic axis " +
                                        std::to_string(d) +
                                        " needs a positive, finite box length");
        }
        max_shift[d] = params.layers;
        if (params.margin < 0.0) {
            margin[d] = params.layers * length[d]; // default: keep every image (exact)
        } else {
            margin[d] = params.margin;
        }
        copies_per_seed *= 2 * static_cast<std::int64_t>(params.layers) + 1;
    }
    const std::int64_t max_total = copies_per_seed * static_cast<std::int64_t>(num_seeds);
    if (copies_per_seed > std::numeric_limits<int>::max() ||
        max_total > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "BuildPeriodicImages: seeds x image copies would exceed 2^31 - 1; reduce layers or "
            "use a margin");
    }

    PeriodicImages result;
    result.num_original = static_cast<int>(num_seeds);
    result.original.reserve(static_cast<std::size_t>(max_total));

    // Step 2: wrapped originals at indices 0..N-1
    std::vector<double> wrapped[3];
    for (int d = 0; d < 3; ++d) {
        const std::vector<double>& coord = SeedCoordinate(seeds, d);
        wrapped[d] = coord;
        if (box.periodic[d]) {
            for (std::size_t i = 0; i < num_seeds; ++i) {
                wrapped[d][i] = WrapIntoBox(coord[i], box.lo[d], length[d]);
            }
        }
    }
    for (std::size_t i = 0; i < num_seeds; ++i) {
        AppendSeed(seeds, i, wrapped[0][i], wrapped[1][i], wrapped[2][i], result.seeds);
        result.original.push_back(static_cast<int>(i));
    }

    // Step 3: images, by shift vector then by seed index
    for (int nx = -max_shift[0]; nx <= max_shift[0]; ++nx) {
        for (int ny = -max_shift[1]; ny <= max_shift[1]; ++ny) {
            for (int nz = -max_shift[2]; nz <= max_shift[2]; ++nz) {
                if (nx == 0 && ny == 0 && nz == 0) {
                    continue; // the originals, already stored
                }
                const int shift[3] = {nx, ny, nz};
                for (std::size_t i = 0; i < num_seeds; ++i) {
                    double pos[3];
                    bool keep = true;
                    for (int d = 0; d < 3; ++d) {
                        pos[d] = wrapped[d][i] + shift[d] * length[d];
                        // Margin filter only on periodic axes (non-periodic shifts are 0)
                        if (box.periodic[d] &&
                            (pos[d] < box.lo[d] - margin[d] || pos[d] > box.hi[d] + margin[d])) {
                            keep = false;
                        }
                    }
                    if (keep) {
                        AppendSeed(seeds, i, pos[0], pos[1], pos[2], result.seeds);
                        result.original.push_back(static_cast<int>(i));
                    }
                }
            }
        }
    }
    result.num_images = static_cast<int>(result.seeds.Size()) - result.num_original;
    return result;
}

/*
 * MapImageLabelsToOriginal
 *
 * element_seed[e] = original[element_seed[e]] for every local element, -1 left as is. The indices
 * are first range-checked on the host (a bad index would read outside the table inside the
 * kernel). The table is copied into an mfem::Array so the kernel can read it through Read()
 * (device memory on a GPU backend); element_seed is updated in place through ReadWrite().
 */
void MapImageLabelsToOriginal(const PeriodicImages& images, mfem::Array<int>& element_seed) {
    const int num_elements = element_seed.Size();
    const int table_size = static_cast<int>(images.original.size());
    if (num_elements == 0) {
        return;
    }

    // Range check on the host
    const int* host_seed = element_seed.HostRead();
    for (int e = 0; e < num_elements; ++e) {
        if (host_seed[e] < -1 || host_seed[e] >= table_size) {
            throw std::invalid_argument("MapImageLabelsToOriginal: element " +
                                        std::to_string(e) + " has seed index " +
                                        std::to_string(host_seed[e]) +
                                        ", outside the image set");
        }
    }

    // Device-visible copy of the image -> original table
    mfem::Array<int> table(table_size);
    for (int k = 0; k < table_size; ++k) {
        table[k] = images.original[static_cast<std::size_t>(k)];
    }
    const int* original = table.Read();
    int* seed = element_seed.ReadWrite();
    mfem::forall(num_elements, [=] MFEM_HOST_DEVICE(int e) {
        if (seed[e] >= 0) {
            seed[e] = original[seed[e]];
        }
    });
}

/*
 * RequiredImageLayers
 *
 * Implements the bound documented in the header:
 *  1. No periodic axis, or a metric whose value grows with each coordinate difference separately
 *     (Voronoi, Laguerre, Additive): 1 layer.
 *  2. Quadric: collect the tensors (global, or each per-seed tensor). A tensor whose off-diagonal
 *     entries are all exactly zero is axis-aligned and needs 1 layer. Otherwise its eigenvalue
 *     range comes from ComputeQuadricFactor (metric_functors.hpp; the same mu_min / mu_max the
 *     seed search uses for its own bounds), and k = ceil(sqrt(mu_max / mu_min) * R / L_min),
 *     with a relative safety of 1e-12 against round-off in the square root and division.
 *  R^2 adds (L_d / 2)^2 over periodic axes and E_d^2 over non-periodic axes, where E_d is the
 *  extent of the box and all seed coordinates together (a seed may sit outside the mesh box on a
 *  non-periodic axis, and the query points are inside the box).
 */
int RequiredImageLayers(const PeriodicBox& box,
                        MetricType metric,
                        const std::array<double, 6>& global_tensor,
                        const SeedSet& seeds) {
    // Step 1: trivial cases
    bool any_periodic = false;
    for (int d = 0; d < 3; ++d) {
        any_periodic = any_periodic || box.periodic[d];
    }
    if (!any_periodic || metric != MetricType::QUADRIC) {
        return 1;
    }

    // Geometry of the bound: R^2 and the smallest periodic box length
    double r_squared = 0.0;
    double min_length = std::numeric_limits<double>::infinity();
    for (int d = 0; d < 3; ++d) {
        if (box.periodic[d]) {
            const double length = box.hi[d] - box.lo[d];
            min_length = std::min(min_length, length);
            r_squared += 0.25 * length * length;
        } else {
            const std::vector<double>& coord = SeedCoordinate(seeds, d);
            double lo = box.lo[d];
            double hi = box.hi[d];
            for (double c : coord) {
                lo = std::min(lo, c);
                hi = std::max(hi, c);
            }
            r_squared += (hi - lo) * (hi - lo);
        }
    }
    if (!(min_length > 0.0) || !std::isfinite(min_length)) {
        throw std::invalid_argument("RequiredImageLayers: a periodic box length is not positive");
    }
    const double radius = std::sqrt(r_squared);

    // Step 2: worst tensor
    const bool per_seed = !seeds.shape.empty();
    std::size_t num_tensors = 1;
    if (per_seed) {
        num_tensors = seeds.shape.size() / 6;
    }
    int layers = 1;
    for (std::size_t t = 0; t < num_tensors; ++t) {
        double tensor[6];
        for (std::size_t k = 0; k < 6; ++k) {
            if (per_seed) {
                tensor[k] = seeds.shape[6 * t + k];
            } else {
                tensor[k] = global_tensor[k];
            }
        }
        const bool axis_aligned = (tensor[3] == 0.0 && tensor[4] == 0.0 && tensor[5] == 0.0);
        if (axis_aligned) {
            continue; // separable: 1 layer is exact (validity of the tensor is the index's job)
        }
        double factor[9];
        double mu_min = 0.0;
        double mu_max = 0.0;
        ComputeQuadricFactor(tensor, factor, mu_min, mu_max);
        if (!(mu_min > 0.0)) {
            throw std::invalid_argument(
                "RequiredImageLayers: a semidefinite quadric tensor that is not aligned with the "
                "box axes has no finite number of periodic image layers that is guaranteed exact");
        }
        const double needed =
            std::sqrt(mu_max / mu_min) * radius / min_length * (1.0 + 1.0e-12);
        if (needed > static_cast<double>(std::numeric_limits<int>::max() / 4)) {
            throw std::invalid_argument("RequiredImageLayers: the quadric tensor is too "
                                        "anisotropic for periodic images");
        }
        layers = std::max(layers, static_cast<int>(std::ceil(needed)));
    }
    return layers;
}

} // namespace exaconstit::microstructure
