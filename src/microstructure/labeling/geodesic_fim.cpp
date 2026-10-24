/*
 * Implementation of the Fast Iterative Method geodesic labeler (geodesic_fim.hpp): sources ->
 * synchronous (Jacobi) eikonal update rounds with distance exchanges -> upwind sets -> label rounds
 * with label exchanges -> element labels.
 *
 * Terms used in this file:
 * - Collective: a function every rank of the communicator must call, because the ranks exchange
 *   data inside it.
 * - mfem::forall: MFEM's parallel loop; a GPU kernel when the mfem::Device has a CUDA or HIP
 *   backend, a plain host loop otherwise. MFEM_HOST_DEVICE marks code callable inside it.
 * - Jacobi (synchronous) round: every vertex computes its new value from the values of the
 *   previous round only; new values are written to a second array and copied back afterwards, so
 *   the order in which vertices are processed cannot change the result.
 * - Candidate: one way of reaching vertex v through one of its elements: through another vertex a
 *   of the element, through the open segment between two other vertices, or through the interior
 *   of the triangle of three other vertices (labeling/eikonal_local_solver.hpp).
 * - Active vertex: a vertex that has to be recomputed this round because a vertex of one of its
 *   elements changed in the previous round.
 * - Upwind vertices of v: the vertices of the candidate(s) that produce v's final distance, with
 *   a smaller distance than v (the direction the front came from).
 * - Incidence: one (vertex, element) pair, i.e. one entry k of the vertex -> element CSR table.
 */
#include "microstructure/labeling/geodesic_fim.hpp"

#include "microstructure/labeling/eikonal_local_solver.hpp"
#include "microstructure/labeling/fim_scan.hpp"
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief +infinity, the "not reached" distance (host constant, captured by value in kernels). */
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief Label of a reached vertex whose label is not known yet (larger than any seed index). */
constexpr int UNSET_LABEL = INT_MAX;

/** @brief Largest number of vertices of an element (linear hex); 7 "other" vertices per element
 * (the gathered corner arrays of IncidenceCandidates have 7 entries). */
constexpr int MAX_ELEMENT_VERTICES = 8;

/**
 * @brief The smallest candidate of node v inside element e, over every cell of e that has v as a
 * corner, together with the neighbor bits (FieldCellView) of the nodes supporting it.
 *
 * Corner field: one cell, the element, with the other vertices in element order and bit = their
 * position (ScanCell of labeling/fim_scan.hpp on the gathered corners). Nodal field: every
 * lattice sub-cell around v (up to 8 sub-hexes or 24 sub-tets), each
 * scanned like an element; the results are combined like candidates of one scan (a strictly
 * smaller value replaces the best and its mask, an equal finite value adds its mask), so the
 * value is the minimum over all sub-cell candidates and the mask the union of the supports that
 * attain it, independent of the order in which the cells are visited.
 *
 * `lattice` is v's lattice point in e (FieldCellView::LatticeOfIncidence of the incidence).
 * Host and device (called inside the mfem::forall sweep).
 */
MFEM_HOST_DEVICE inline double IncidenceCandidates(const FieldCellView& view, int v, int e,
                                                   int lattice, const double* xyz,
                                                   const double* t, int* mask) {
    const double inf = EikonalInfinity();
    const double* xv = &xyz[3 * v];
    double best = inf;
    int m = 0;
    const bool curved = view.curved != 0;
    view.ForEachCellOfIncidence(v, e, lattice, [&](const int* others, const int* bits, int n) {
        int cell_mask = 0;
        // Gather the cell's other corners; hexes scan their far faces only (CellFarMasks)
        const double* xo[7];
        double to[7];
        for (int i = 0; i < n; ++i) {
            xo[i] = &xyz[3 * others[i]];
            to[i] = t[others[i]];
        }
        int far[7];
        const bool hex = CellFarMasks(curved, bits, n, far);
        const double value = ScanCell(xv, xo, to, bits, hex ? far : nullptr, n, &cell_mask);
        if (value < best) {
            best = value;
            m = cell_mask;
        } else if (value == best && value < inf) {
            m |= cell_mask;
        }
    });
    *mask = m;
    return best;
}

/**
 * @brief One synchronous distance sweep, the only place where candidate values are computed.
 *
 * For every vertex v and every element of v (incidence k of the vertex -> element table): if a
 * node the element's candidates of v read has changed[u] = 1 (it changed in the previous round,
 * locally or through the exchange), cand_value[k] = the element's smallest candidate and
 * cand_mask[k] = the nodes supporting it (IncidenceCandidates), computed from T only (the previous
 * round's values); otherwise the stored cand_value / cand_mask are kept, since they still equal
 * what a recomputation would give (none of their inputs changed since they were stored). Then
 * t_new[v] = min(T(v), min over k of cand_value[k]). After the last round the stored values are
 * therefore the candidates of the converged T for every vertex, and the label pass reads them
 * instead of recomputing them: one binary, one code path per value, so the label pass can never
 * disagree with the distances by a rounding bit (e.g. from multiply-add fusion done differently in
 * two kernels).
 *
 * The nodes checked per incidence are the element's neighbor nodes of v (FieldCellView: every
 * vertex of the element for the corner field; the 3 x 3 x 3 lattice block around v for the nodal
 * field, a superset of its cell neighbors, which costs extra checks only). Skipping unchanged
 * elements saves most of the work of the late rounds, where only a few nodes of each active
 * neighborhood still change (a hex scan is 63 local solves; the check is at most 27 reads).
 *
 * One mfem::forall iteration per vertex (gathers over the CSR tables, no atomics). Not
 * collective. cand_value / cand_mask must be sized to the number of incidences and initialized
 * (+infinity / 0) before the first sweep.
 */
void CandidateSweep(const VertexField& field, const mfem::Array<int>& changed,
                    mfem::Vector& cand_value, mfem::Array<int>& cand_mask, mfem::Vector& t_new) {
    const int nv = field.NumVertices();
    t_new.SetSize(nv);
    t_new.UseDevice(true);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const FieldCellView view = field.CellView();
    const double* xyz = field.MetricCoordinates().Read();
    const double* t = field.Distance().Read();
    const int* chg = changed.Read();
    // ReadWrite: unchanged incidences keep their stored candidates
    double* cvalue = cand_value.ReadWrite();
    int* cmask = cand_mask.ReadWrite();
    double* out = t_new.Write();
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        double best = t[v];
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            const int e = ve_items[k];
            const int lattice = view.LatticeOfIncidence(k);
            bool dirty = false;
            for (int j = 0; j < view.NumNeighborBits(e) && !dirty; ++j) {
                const int u = view.NeighborNode(e, lattice, j);
                dirty = u >= 0 && chg[u] != 0;
            }
            if (dirty) {
                // Smallest candidate of the element (over its cells), stored per incidence
                int mask = 0;
                cvalue[k] = IncidenceCandidates(view, v, e, lattice, xyz, t, &mask);
                cmask[k] = mask;
            }
            best = (cvalue[k] < best) ? cvalue[k] : best;
        }
        out[v] = best;
    });
}

/**
 * @brief Smallest stored candidate of every reached vertex, including the source value T0(v) at
 * source vertices: m[v] = min(T0(v), min over k of cand_value[k]); +infinity for unreached
 * vertices. Comparisons only (no arithmetic), so it adds no rounding. One mfem::forall iteration
 * per vertex. Not collective (the caller reduces m over the ranks).
 */
void SmallestCandidate(const VertexField& field, const mfem::Vector& cand_value,
                       mfem::Vector& m) {
    const int nv = field.NumVertices();
    m.SetSize(nv);
    m.UseDevice(true);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const double* t = field.Distance().Read();
    const double* t0 = field.SourceDistance().Read();
    const int* label0 = field.SourceLabel().Read();
    const double* cvalue = cand_value.Read();
    double* out = m.Write();
    const double inf = INF;
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        if (!(t[v] < inf)) {
            out[v] = inf;
            return;
        }
        double best = (label0[v] >= 0) ? t0[v] : inf;
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            best = (cvalue[k] < best) ? cvalue[k] : best;
        }
        out[v] = best;
    });
}

/**
 * @brief Upwind sets and starting labels for the label rounds, from the stored candidates.
 * (Neighbor bits are turned back into nodes with FieldCellView::NeighborNode.)
 *
 * For every reached vertex v and incidence k (element e of v):
 *  1. upwind_mask[k] = cand_mask[k] if cand_value[k] == m[v] (the element holds a candidate
 *     attaining the global minimum), restricted to the vertices u of e with T(u) < T(v); 0
 *     otherwise;
 *  2. arrival through each remaining u: a(u) = T(u) + |v - u| (EikonalVertexValue, in the metric
 *     coordinates); incidence_best[k] = the smallest a(u) of the incidence (+infinity if none) and
 *     the mask keeps only the vertices attaining it; local_best[v] = the smallest over v's
 *     incidences. The caller makes local_best the minimum over the ranks and FilterUpwind keeps
 *     only the incidences attaining it.
 * Why step 2: the support of a face candidate can span two grains (a face crossing a grain
 * boundary), and taking the smallest seed index over all of it would let the lowest-numbered seed
 * leak across boundaries and grow downstream. The support vertex whose own front reaches v first
 * in a straight line carries the right seed (exact ties still go to the lowest index).
 * A source vertex attaining m (T0(v) == m[v]) starts with its seed's label and keeps no upwind
 * vertices (they can only tie it); other reached vertices start at UNSET_LABEL; unreached
 * vertices get -1 and empty masks. Every a(u) is computed here only, so values compared later
 * (FilterUpwind, across ranks) come from one kernel and agree bitwise. One mfem::forall
 * iteration per vertex. Not collective.
 */
void UpwindSets(const VertexField& field, const mfem::Vector& m, const mfem::Vector& cand_value,
                const mfem::Array<int>& cand_mask, mfem::Array<int>& upwind_mask,
                mfem::Vector& incidence_best, mfem::Vector& local_best, mfem::Array<int>& label) {
    const int nv = field.NumVertices();
    const int num_incidences = field.VertexElements().Size();
    upwind_mask.SetSize(num_incidences);
    incidence_best.SetSize(num_incidences);
    incidence_best.UseDevice(true);
    local_best.SetSize(nv);
    local_best.UseDevice(true);
    label.SetSize(nv);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const FieldCellView view = field.CellView();
    const double* xyz = field.MetricCoordinates().Read();
    const double* t = field.Distance().Read();
    const double* t0 = field.SourceDistance().Read();
    const int* label0 = field.SourceLabel().Read();
    const double* mv = m.Read();
    const double* cvalue = cand_value.Read();
    const int* cmask = cand_mask.Read();
    int* masks = upwind_mask.Write();
    double* ibest = incidence_best.Write();
    double* vbest = local_best.Write();
    int* lab = label.Write();
    const double inf = INF;
    const int unset = UNSET_LABEL;
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        const double tv = t[v];
        const bool source = tv < inf && label0[v] >= 0 && t0[v] == mv[v];
        if (!(tv < inf) || source) {
            for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
                masks[k] = 0;
                ibest[k] = inf;
            }
            vbest[v] = inf;
            lab[v] = source ? label0[v] : -1;
            return;
        }
        lab[v] = unset;
        double best_v = inf;
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            const int e = ve_items[k];
            int mask = (cvalue[k] == mv[v]) ? cmask[k] : 0;
            // Keep only nodes strictly upwind (smaller T), so the dependencies form no loop, and
            // among them those whose straight-line arrival at v is smallest
            const int lattice = view.LatticeOfIncidence(k);
            double best_k = inf;
            int keep = 0;
            for (int j = 0; j < view.NumNeighborBits(e); ++j) {
                if ((mask & (1 << j)) == 0) {
                    continue;
                }
                const int u = view.NeighborNode(e, lattice, j);
                if (u < 0 || !(t[u] < tv)) {
                    continue;
                }
                const double arrival = EikonalVertexValue(&xyz[3 * v], &xyz[3 * u], t[u]);
                if (arrival < best_k) {
                    best_k = arrival;
                    keep = 1 << j;
                } else if (arrival == best_k) {
                    keep |= 1 << j;
                }
            }
            masks[k] = keep;
            ibest[k] = best_k;
            best_v = (best_k < best_v) ? best_k : best_v;
        }
        vbest[v] = best_v;
    });
}

/**
 * @brief Keeps the upwind vertices of the incidences whose smallest arrival equals the global
 * smallest arrival of their vertex (best[v], the rank minimum of UpwindSets' local_best): masks of
 * the other incidences are cleared. Comparisons only (of values computed by UpwindSets). One
 * mfem::forall iteration per vertex. Not collective.
 */
void FilterUpwind(const VertexField& field, const mfem::Vector& incidence_best,
                  const mfem::Vector& best, mfem::Array<int>& upwind_mask) {
    const int nv = field.NumVertices();
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const double* ibest = incidence_best.Read();
    const double* vbest = best.Read();
    int* masks = upwind_mask.ReadWrite();
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            if (!(ibest[k] == vbest[v])) {
                masks[k] = 0;
            }
        }
    });
}

/**
 * @brief One synchronous label sweep: label_new[v] = min(label[v], labels of v's upwind vertices),
 * reading label only (previous round); the upwind vertices are the set neighbor bits of each
 * incidence, decoded by FieldCellView::NeighborNode. One mfem::forall iteration per vertex. Not
 * collective.
 */
void LabelSweep(const VertexField& field, const mfem::Array<int>& upwind_mask,
                mfem::Array<int>& label_new) {
    const int nv = field.NumVertices();
    label_new.SetSize(nv);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const FieldCellView view = field.CellView();
    const int* masks = upwind_mask.Read();
    const int* lab = field.Label().Read();
    int* out = label_new.Write();
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        int best = lab[v];
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            const int mask = masks[k];
            if (mask == 0) {
                continue;
            }
            const int e = ve_items[k];
            const int lattice = view.LatticeOfIncidence(k);
            for (int j = 0; j < view.NumNeighborBits(e); ++j) {
                // UpwindSets clears bits without a node; the u >= 0 test keeps this sweep safe on
                // its own
                const int u = ((mask & (1 << j)) != 0) ? view.NeighborNode(e, lattice, j) : -1;
                if (u >= 0) {
                    best = (lab[u] < best) ? lab[u] : best;
                }
            }
        }
        out[v] = best;
    });
}

/**
 * @brief Diagonal of the bounding box of the field nodes in the metric coordinates
 * (VertexField::MetricCoordinates), over all ranks: min/max reductions, which are exact, so every
 * partition gets the same bits. 0 for a field without nodes. Collective. Host.
 */
double MetricDiameter(const VertexField& field, MPI_Comm comm) {
    double bounds[6] = {INF, INF, INF, INF, INF, INF}; // min x y z, -max x y z
    const double* xyz = field.MetricCoordinates().HostRead();
    for (int v = 0; v < field.NumVertices(); ++v) {
        for (int d = 0; d < 3; ++d) {
            bounds[d] = std::min(bounds[d], xyz[3 * v + d]);
            bounds[3 + d] = std::min(bounds[3 + d], -xyz[3 * v + d]);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, bounds, 6, MPI_DOUBLE, MPI_MIN, comm);
    double sum = 0.0;
    for (int d = 0; d < 3; ++d) {
        const double extent = (bounds[d] < INF) ? (-bounds[3 + d] - bounds[d]) : 0.0;
        sum += extent * extent;
    }
    return std::sqrt(sum);
}

/** @brief Global sum of a local count (MPI_Allreduce). Collective. */
std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    long long value = static_cast<long long>(local);
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return static_cast<std::int64_t>(value);
}

} // namespace

/*
 * LabelGeodesicFim
 *
 * Steps (see the header for the algorithm and the partition-independence argument):
 *  1. Parameter checks (identical on every rank: no communication needed to throw together), and
 *     the element-size check (collective: every rank learns whether any rank has an element with
 *     more than 8 vertices).
 *  2. VertexField (collective) and InitializeSources (collective).
 *  3. Phase 1, distance rounds: CandidateSweep -> copy back and note local changes ->
 *     ReduceSharedDistance (marks vertices changed by the exchange) -> stop when nothing changed
 *     on any rank. A new value is taken only when it is a first arrival or improves T by more
 *     than the stopping tolerance (GeodesicParameters::tolerance times the metric diameter).
 *     Round 1 treats every reached vertex as "changed", so the elements around the sources are
 *     scanned.
 *  4. Phase 2, labels (from the candidates stored by the last sweeps, never recomputed):
 *     SmallestCandidate -> ReduceSharedMinimum (global m per vertex) ->
 *     UpwindSets -> ReduceSharedMinimum (global smallest arrival) -> FilterUpwind ->
 *     rounds of LabelSweep + copy back + ReduceSharedLabels until no label changes.
 *     Vertices left at UNSET_LABEL (degenerate meshes only) become -1 and are counted.
 *  5. Element labels (centroid or vote) and the global statistics.
 * The copy-back and change detection loops run on the host: the exchange that follows needs the
 * values on the host anyway (ReduceShared* uses HostReadWrite).
 *
 * TODO (performance, large meshes / GPU): per distance or label round the solver pays
 *   - two MPI_Allreduce calls (AnyRank of the local changes, and the one inside ReduceShared*);
 *     they could be merged into one;
 *   - host loops over all vertices (copy-back, change detection, merging the exchange marker),
 *     although only a few vertices change in the late rounds;
 *   - on a GPU, about three vertex-sized host/device copies (T or labels to the host for the
 *     exchange and back, plus the changed marker), which an on-device copy-back and a device
 *     exchange buffer for the shared vertices only would avoid;
 *   - phase arrays kept alive after their phase (t_new, changed, the stored candidates
 *     cand_value / cand_mask during the label rounds, label_new), which could be released early.
 */
GeodesicLabels LabelGeodesicFim(mfem::ParMesh& pmesh, MPI_Comm comm, const SeedSet& seeds,
                                bool additive, const GeodesicParameters& params) {
    // Step 1: parameters (seeds are validated by InitializeSources)
    if (params.max_rounds < 1) {
        throw std::invalid_argument("LabelGeodesicFim: max_rounds must be >= 1");
    }
    if (!(params.tolerance >= 0.0 && params.tolerance < 1.0)) {
        throw std::invalid_argument("LabelGeodesicFim: tolerance must be in [0, 1)");
    }
    if (params.sampling == SamplingMode::QUADRATURE_VOTE) {
        if (params.vote_levels < 0 || params.vote_levels > 3) {
            throw std::invalid_argument("LabelGeodesicFim: vote_levels must be 0..3");
        }
        if (params.max_points_per_chunk < 1) {
            throw std::invalid_argument("LabelGeodesicFim: max_points_per_chunk must be >= 1");
        }
    }
    if (additive && !seeds.weight.empty() && seeds.weight.size() != seeds.Size()) {
        throw std::invalid_argument("LabelGeodesicFim: seeds.weight must be empty or have one "
                                    "entry per seed");
    }
    {
        int same_comm = MPI_UNEQUAL;
        MPI_Comm_compare(comm, pmesh.GetComm(), &same_comm);
        if (same_comm != MPI_IDENT && same_comm != MPI_CONGRUENT) {
            throw std::invalid_argument("LabelGeodesicFim: comm must be the mesh communicator");
        }
    }
    {
        bool too_many_vertices = false;
        for (int e = 0; e < pmesh.GetNE(); ++e) {
            too_many_vertices = too_many_vertices ||
                                pmesh.GetElement(e)->GetNVertices() > MAX_ELEMENT_VERTICES;
        }
        ThrowIfAnyRank<std::runtime_error>(
            comm, too_many_vertices,
            "LabelGeodesicFim: elements with more than 8 vertices are not supported");
    }

    // Step 2: vertex field and sources
    GeodesicLabels result;
    result.field =
        std::make_unique<VertexField>(pmesh, params.field_nodes, ContextMask(params.context));
    VertexField& field = *result.field;
    field.SetMetricStretch(params.stretch); // identity = plain lengths (Voronoi, Additive)
    const std::vector<double> no_weights;
    result.sources =
        field.InitializeSources(seeds, additive ? seeds.weight : no_weights, params.context);
    result.report.seeds_outside_material = result.sources.num_seeds_outside;
    const int nv = field.NumVertices();
    const double tolerance = params.tolerance * MetricDiameter(field, comm);

    // Step 3: phase 1, distance rounds. changed[v] = 1 for the vertices whose T changed in the
    // previous round (locally or through the exchange); at the start: every reached vertex.
    // (Host pointers from HostWrite/HostRead everywhere, never operator[]: operator[] bypasses
    // MFEM's host/device validity flags, which matters when the kernels run on a GPU.)
    mfem::Array<int> changed(nv);
    {
        const double* t = field.Distance().HostRead();
        int* chg = changed.HostWrite();
        for (int v = 0; v < nv; ++v) {
            chg[v] = (t[v] < INF) ? 1 : 0;
        }
    }
    mfem::Array<int> exchange_changed;
    mfem::Vector t_new;
    // Stored candidates per (vertex, element) incidence, filled by CandidateSweep and read by the
    // label pass (+infinity / 0 = not computed yet: no reached vertex nearby so far)
    const int num_incidences = field.VertexElements().Size();
    mfem::Vector cand_value(num_incidences);
    cand_value.UseDevice(true);
    cand_value = INF;
    mfem::Array<int> cand_mask(num_incidences);
    cand_mask = 0;
    for (int round = 1;; ++round) {
        // Synchronous update from the previous round's T (elements with a changed node only)
        CandidateSweep(field, changed, cand_value, cand_mask, t_new);
        // Copy back and note local changes (bitwise comparison; T only decreases)
        bool local_change = false;
        {
            const double* tn = t_new.HostRead();
            double* t = field.Distance().HostReadWrite();
            int* chg = changed.HostWrite();
            // A value is taken when it is the first arrival or improves T by more than the
            // tolerance; smaller improvements are dropped (see "Stopping tolerance")
            for (int v = 0; v < nv; ++v) {
                const bool c = !(t[v] < INF) ? (tn[v] < INF) : (tn[v] < t[v] - tolerance);
                chg[v] = c ? 1 : 0;
                local_change = local_change || c;
                if (c) {
                    t[v] = tn[v];
                }
            }
        }
        // Exchange: every copy of a shared vertex takes the smallest T over the ranks
        const bool exchange_change = field.ReduceSharedDistance(&exchange_changed);
        {
            const int* xchg = exchange_changed.HostRead();
            int* chg = changed.HostReadWrite();
            for (int v = 0; v < nv; ++v) {
                chg[v] = (chg[v] != 0 || xchg[v] != 0) ? 1 : 0;
            }
        }
        const bool any_change = AnyRank(local_change, comm) || exchange_change;
        result.report.distance_rounds = round;
        if (!any_change) {
            break;
        }
        // any_change is global, so every rank takes this branch together
        ThrowIfAnyRank<std::runtime_error>(
            comm, round >= params.max_rounds,
            "LabelGeodesicFim: max_rounds exceeded in the distance rounds");
    }

    // Step 4: phase 2, labels. Global smallest candidate per vertex, upwind sets, label rounds.
    mfem::Vector m;
    SmallestCandidate(field, cand_value, m);
    field.ReduceSharedMinimum(m);
    mfem::Array<int> upwind_mask;
    mfem::Array<int> label_start;
    mfem::Vector incidence_best;
    mfem::Vector arrival_best;
    UpwindSets(field, m, cand_value, cand_mask, upwind_mask, incidence_best, arrival_best,
               label_start);
    field.ReduceSharedMinimum(arrival_best);
    FilterUpwind(field, incidence_best, arrival_best, upwind_mask);
    {
        const int* ls = label_start.HostRead();
        int* lab = field.Label().HostWrite();
        for (int v = 0; v < nv; ++v) {
            lab[v] = ls[v];
        }
    }
    mfem::Array<int> label_new;
    for (int round = 1;; ++round) {
        LabelSweep(field, upwind_mask, label_new);
        bool local_change = false;
        {
            const int* ln = label_new.HostRead();
            int* lab = field.Label().HostReadWrite();
            for (int v = 0; v < nv; ++v) {
                local_change = local_change || ln[v] != lab[v];
                lab[v] = ln[v];
            }
        }
        const bool exchange_change = field.ReduceSharedLabels();
        const bool any_change = AnyRank(local_change, comm) || exchange_change;
        result.report.label_rounds = round;
        if (!any_change) {
            break;
        }
        ThrowIfAnyRank<std::runtime_error>(
            comm, round >= params.max_rounds,
            "LabelGeodesicFim: max_rounds exceeded in the label rounds");
    }
    // A reached vertex left without a label (degenerate mesh only) becomes -1 and is counted
    std::int64_t unlabeled = 0;
    std::int64_t unreached = 0;
    {
        const double* t = field.Distance().HostRead();
        int* lab = field.Label().HostReadWrite();
        const int* owned = field.OwnedVertexMarker().HostRead();
        // Without a mask every node counts (as before masks existed); with one, only nodes of
        // an active element (NodeUsed)
        const int* used = field.NodeUsed().HostRead();
        const bool masked = ContextMask(params.context) != nullptr &&
                            ContextMask(params.context)->active.Size() != 0;
        for (int v = 0; v < nv; ++v) {
            if (lab[v] == UNSET_LABEL) {
                lab[v] = -1;
                unlabeled += owned[v];
            }
            if (!(t[v] < INF) && (!masked || used[v] != 0)) {
                unreached += owned[v];
            }
        }
    }

    // Step 5: element labels and statistics
    std::int64_t split = 0;
    if (params.sampling == SamplingMode::QUADRATURE_VOTE) {
        split = field.LabelElementsByVote(params.vote_levels, params.max_points_per_chunk,
                                          result.element_seed);
    } else {
        field.LabelElementCentroids(result.element_seed);
    }
    // An element labeled -1 is "unreached" when none of its vertices has a finite T, otherwise
    // "unlabeled" (a reached element that only saw unlabeled vertices). Element-local data only.
    std::int64_t unreached_elements = 0;
    std::int64_t unlabeled_elements = 0;
    {
        const int* es = result.element_seed.HostRead();
        const double* t_final = field.Distance().HostRead();
        const int* ev_offsets = field.ElementVertexOffsets().HostRead();
        const int* ev_items = field.ElementVertices().HostRead();
        for (int e = 0; e < result.element_seed.Size(); ++e) {
            if (es[e] >= 0 || !field.ElementActive(e)) {
                continue; // labeled, or inactive (placeholder -1, not material)
            }
            bool any_reached = false;
            for (int j = ev_offsets[e]; j < ev_offsets[e + 1]; ++j) {
                any_reached = any_reached || t_final[ev_items[j]] < INF;
            }
            unlabeled_elements += any_reached ? 1 : 0;
            unreached_elements += any_reached ? 0 : 1;
        }
    }
    result.report.unreached_vertices = GlobalSum(unreached, comm);
    result.report.unlabeled_vertices = GlobalSum(unlabeled, comm);
    result.report.unreached_elements = GlobalSum(unreached_elements, comm);
    result.report.unlabeled_elements = GlobalSum(unlabeled_elements, comm);
    result.report.split_elements = GlobalSum(split, comm);
    return result;
}

} // namespace exaconstit::microstructure
