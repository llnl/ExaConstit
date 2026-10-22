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
#include "microstructure/util/collective_errors.hpp"

#include "mfem.hpp"

#include <climits>
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

/** @brief Largest number of vertices of an element (linear hex); 7 "other" vertices per element. */
constexpr int MAX_ELEMENT_VERTICES = 8;

/**
 * @brief Scans every candidate of vertex v inside one element and returns the smallest candidate
 * value together with the element vertices supporting the candidates that attain it.
 *
 * The candidates (see the file comment) are, over the element's other vertices o_0 .. o_{n-1}:
 *   - every vertex: T(o_i) + |v - o_i|                        (7 for a hex, 3 for a tet)
 *   - every pair:   the open-segment minimizer (o_i, o_j)       (21 for a hex, 3 for a tet)
 *   - every triple: the triangle-interior minimizer (o_i, o_j, o_k) (35 for a hex, 1 for a tet)
 * Their minimum is the minimum over all triangles of other vertices of the full-face update, since
 * every segment and vertex of such a triangle is a candidate of its own. Each candidate function
 * sorts its vertices by coordinates, so the values are the same bits whatever the element's vertex
 * order (see eikonal_local_solver.hpp).
 *
 * Single pass: a strictly smaller value replaces the running best and its mask, an exactly equal
 * value adds its vertices to the mask. Each candidate value is computed exactly once, here, and
 * both the distance update and the label pass use the stored result (see CandidateSweep).
 *
 * Host and device (called inside the mfem::forall sweep).
 *
 * @param xv     coordinates of v (3 doubles).
 * @param others local vertex ids of the other vertices of the element (n entries).
 * @param pos    position (0 .. 7) of each of them in the element's vertex list (n entries).
 * @param n      number of other vertices, 0 .. 7.
 * @param xyz    vertex coordinates, xyz[3*u + d].
 * @param t      distances T(u) per local vertex (+infinity = not reached).
 * @param mask   out: bit pos[i] set when o_i belongs to a candidate whose value equals the returned
 *               minimum (exact comparison); 0 when the minimum is +infinity.
 * @return the smallest candidate value (+infinity if no other vertex is reached).
 */
MFEM_HOST_DEVICE inline double ScanElementCandidates(const double* xv,
                                                     const int* others,
                                                     const int* pos,
                                                     int n,
                                                     const double* xyz,
                                                     const double* t,
                                                     int* mask) {
    const double inf = EikonalInfinity();
    double best = inf;
    int m = 0;
    // Through a single vertex
    for (int i = 0; i < n; ++i) {
        const int a = others[i];
        const double value = EikonalVertexValue(xv, &xyz[3 * a], t[a]);
        const int support = 1 << pos[i];
        if (value < best) {
            best = value;
            m = support;
        } else if (value == best && value < inf) {
            m |= support;
        }
    }
    // Through the open segment between two vertices
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int a = others[i];
            const int b = others[j];
            const double value = EikonalSegmentValue(xv, &xyz[3 * a], &xyz[3 * b], t[a], t[b]);
            const int support = (1 << pos[i]) | (1 << pos[j]);
            if (value < best) {
                best = value;
                m = support;
            } else if (value == best && value < inf) {
                m |= support;
            }
        }
    }
    // Through the interior of the triangle of three vertices
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                const int a = others[i];
                const int b = others[j];
                const int c = others[k];
                const double value = EikonalTriangleInteriorValue(
                    xv, &xyz[3 * a], &xyz[3 * b], &xyz[3 * c], t[a], t[b], t[c]);
                const int support = (1 << pos[i]) | (1 << pos[j]) | (1 << pos[k]);
                if (value < best) {
                    best = value;
                    m = support;
                } else if (value == best && value < inf) {
                    m |= support;
                }
            }
        }
    }
    *mask = m;
    return best;
}

/**
 * @brief Collects the other vertices of element e (all of e's vertices except v) and their
 * positions in e's vertex list. Host and device.
 * @return the number of other vertices (at most MAX_ELEMENT_VERTICES - 1; the caller has checked
 *         that no element has more than MAX_ELEMENT_VERTICES vertices).
 */
MFEM_HOST_DEVICE inline int GatherOtherVertices(int v, int e, const int* ev_offsets,
                                                const int* ev_items, int* others, int* pos) {
    int n = 0;
    for (int j = ev_offsets[e]; j < ev_offsets[e + 1]; ++j) {
        const int u = ev_items[j];
        if (u != v && n < MAX_ELEMENT_VERTICES - 1) {
            others[n] = u;
            pos[n] = j - ev_offsets[e];
            ++n;
        }
    }
    return n;
}

/**
 * @brief Marks the vertices to recompute in the next distance round: active[v] = 1 if any vertex
 * of any element of v (v included) has changed[u] = 1. One mfem::forall iteration per vertex
 * (gather over the CSR tables, no atomics). Not collective.
 */
void MarkActive(const VertexField& field, const mfem::Array<int>& changed,
                mfem::Array<int>& active) {
    const int nv = field.NumVertices();
    active.SetSize(nv);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const int* ev_offsets = field.ElementVertexOffsets().Read();
    const int* ev_items = field.ElementVertices().Read();
    const int* chg = changed.Read();
    int* act = active.Write();
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        int a = 0;
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1] && a == 0; ++k) {
            const int e = ve_items[k];
            for (int j = ev_offsets[e]; j < ev_offsets[e + 1]; ++j) {
                if (chg[ev_items[j]] != 0) {
                    a = 1;
                    break;
                }
            }
        }
        act[v] = a;
    });
}

/**
 * @brief One synchronous distance sweep, the only place where candidate values are computed.
 *
 * For every active vertex v and every element of v (incidence k of the vertex -> element table):
 * cand_value[k] = the element's smallest candidate and cand_mask[k] = the element vertices
 * supporting it (ScanElementCandidates), computed from T only (the previous round's values);
 * then t_new[v] = min(T(v), min over k of cand_value[k]). Inactive vertices copy T(v) and keep
 * their stored cand_value / cand_mask, which still equal what a recomputation would give (none of
 * the inputs changed since they were stored). After the last round the stored values are
 * therefore the candidates of the converged T for every vertex, and the label pass reads them
 * instead of recomputing them: one binary, one code path per value, so the label pass can never
 * disagree with the distances by a rounding bit (e.g. from multiply-add fusion done differently in
 * two kernels).
 *
 * One mfem::forall iteration per vertex (gathers over the CSR tables, no atomics). Not
 * collective. cand_value / cand_mask must be sized to the number of incidences and initialized
 * (+infinity / 0) before the first sweep.
 */
void CandidateSweep(const VertexField& field, const mfem::Array<int>& active,
                    mfem::Vector& cand_value, mfem::Array<int>& cand_mask, mfem::Vector& t_new) {
    const int nv = field.NumVertices();
    t_new.SetSize(nv);
    t_new.UseDevice(true);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const int* ev_offsets = field.ElementVertexOffsets().Read();
    const int* ev_items = field.ElementVertices().Read();
    const double* xyz = field.Coordinates().Read();
    const double* t = field.Distance().Read();
    const int* act = active.Read();
    // ReadWrite: inactive vertices keep their stored candidates
    double* cvalue = cand_value.ReadWrite();
    int* cmask = cand_mask.ReadWrite();
    double* out = t_new.Write();
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        double best = t[v];
        if (act[v] != 0) {
            const double* xv = &xyz[3 * v];
            int others[MAX_ELEMENT_VERTICES - 1];
            int pos[MAX_ELEMENT_VERTICES - 1];
            // Smallest candidate of every element of v, stored per incidence
            for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
                const int e = ve_items[k];
                const int n = GatherOtherVertices(v, e, ev_offsets, ev_items, others, pos);
                int mask = 0;
                const double value = ScanElementCandidates(xv, others, pos, n, xyz, t, &mask);
                cvalue[k] = value;
                cmask[k] = mask;
                best = (value < best) ? value : best;
            }
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
 *
 * For every reached vertex v and incidence k (element e of v): upwind_mask[k] = cand_mask[k] if
 * cand_value[k] == m[v] (the element holds a candidate attaining the global minimum), restricted
 * to the vertices u of e with T(u) < T(v); 0 otherwise. label[v] starts at the source label when
 * v is a source vertex attaining m (T0(v) == m[v]), at UNSET_LABEL otherwise; unreached vertices
 * get -1 and empty masks. Comparisons only. One mfem::forall iteration per vertex. Not collective.
 */
void UpwindSets(const VertexField& field, const mfem::Vector& m, const mfem::Vector& cand_value,
                const mfem::Array<int>& cand_mask, mfem::Array<int>& upwind_mask,
                mfem::Array<int>& label) {
    const int nv = field.NumVertices();
    const int num_incidences = field.VertexElements().Size();
    upwind_mask.SetSize(num_incidences);
    label.SetSize(nv);
    if (nv == 0) {
        return;
    }
    const int* ve_offsets = field.VertexElementOffsets().Read();
    const int* ve_items = field.VertexElements().Read();
    const int* ev_offsets = field.ElementVertexOffsets().Read();
    const int* ev_items = field.ElementVertices().Read();
    const double* t = field.Distance().Read();
    const double* t0 = field.SourceDistance().Read();
    const int* label0 = field.SourceLabel().Read();
    const double* mv = m.Read();
    const double* cvalue = cand_value.Read();
    const int* cmask = cand_mask.Read();
    int* masks = upwind_mask.Write();
    int* lab = label.Write();
    const double inf = INF;
    const int unset = UNSET_LABEL;
    mfem::forall(nv, [=] MFEM_HOST_DEVICE(int v) {
        const double tv = t[v];
        if (!(tv < inf)) {
            for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
                masks[k] = 0;
            }
            lab[v] = -1;
            return;
        }
        // Source vertex that attains the smallest candidate: starts with its seed's label
        lab[v] = (label0[v] >= 0 && t0[v] == mv[v]) ? label0[v] : unset;
        for (int k = ve_offsets[v]; k < ve_offsets[v + 1]; ++k) {
            const int e = ve_items[k];
            int mask = (cvalue[k] == mv[v]) ? cmask[k] : 0;
            // Keep only vertices strictly upwind (smaller T), so the dependencies form no loop
            for (int j = 0; j < ev_offsets[e + 1] - ev_offsets[e]; ++j) {
                if ((mask & (1 << j)) != 0 && !(t[ev_items[ev_offsets[e] + j]] < tv)) {
                    mask &= ~(1 << j);
                }
            }
            masks[k] = mask;
        }
    });
}

/**
 * @brief One synchronous label sweep: label_new[v] = min(label[v], labels of v's upwind vertices),
 * reading label only (previous round). One mfem::forall iteration per vertex. Not collective.
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
    const int* ev_offsets = field.ElementVertexOffsets().Read();
    const int* ev_items = field.ElementVertices().Read();
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
            for (int j = 0; j < ev_offsets[e + 1] - ev_offsets[e]; ++j) {
                if ((mask & (1 << j)) != 0) {
                    const int u_label = lab[ev_items[ev_offsets[e] + j]];
                    best = (u_label < best) ? u_label : best;
                }
            }
        }
        out[v] = best;
    });
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
 *  3. Phase 1, distance rounds: MarkActive -> CandidateSweep -> copy back and note local changes ->
 *     ReduceSharedDistance (marks vertices changed by the exchange) -> stop when nothing changed
 *     on any rank. Round 1 treats every reached vertex as "changed", so the vertices around the
 *     sources are active.
 *  4. Phase 2, labels (from the candidates stored by the last sweeps, never recomputed):
 *     SmallestCandidate -> ReduceSharedMinimum (global m per vertex) ->
 *     UpwindSets -> rounds of LabelSweep + copy back + ReduceSharedLabels until no label changes.
 *     Vertices left at UNSET_LABEL (degenerate meshes only) become -1 and are counted.
 *  5. Element labels (centroid or vote) and the global statistics.
 * The copy-back and change detection loops run on the host: the exchange that follows needs the
 * values on the host anyway (ReduceShared* uses HostReadWrite).
 *
 * TODO (performance, large meshes / GPU): per distance or label round the solver pays
 *   - two MPI_Allreduce calls (AnyRank of the local changes, and the one inside ReduceShared*);
 *     they could be merged into one;
 *   - host loops over all vertices (copy-back, change detection, merging the exchange marker),
 *     although only the active vertices change;
 *   - on a GPU, about three vertex-sized host/device copies (T or labels to the host for the
 *     exchange and back, plus the changed marker), which an on-device copy-back and a device
 *     exchange buffer for the shared vertices only would avoid;
 *   - phase arrays kept alive after their phase (t_new, active, changed, the stored candidates
 *     cand_value / cand_mask during the label rounds, label_new), which could be released early.
 */
GeodesicLabels LabelGeodesicFim(mfem::ParMesh& pmesh, MPI_Comm comm, const SeedSet& seeds,
                                bool additive, const GeodesicParameters& params) {
    // Step 1: parameters (seeds are validated by InitializeSources)
    if (params.max_rounds < 1) {
        throw std::invalid_argument("LabelGeodesicFim: max_rounds must be >= 1");
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
    result.field = std::make_unique<VertexField>(pmesh);
    VertexField& field = *result.field;
    const std::vector<double> no_weights;
    result.sources = field.InitializeSources(seeds, additive ? seeds.weight : no_weights);
    result.report.seeds_outside_material = result.sources.num_seeds_outside;
    const int nv = field.NumVertices();

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
    mfem::Array<int> active;
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
        // Synchronous update of the active vertices from the previous round's T
        MarkActive(field, changed, active);
        CandidateSweep(field, active, cand_value, cand_mask, t_new);
        // Copy back and note local changes (bitwise comparison; T only decreases)
        bool local_change = false;
        {
            const double* tn = t_new.HostRead();
            double* t = field.Distance().HostReadWrite();
            int* chg = changed.HostWrite();
            for (int v = 0; v < nv; ++v) {
                const bool c = tn[v] != t[v];
                chg[v] = c ? 1 : 0;
                local_change = local_change || c;
                t[v] = tn[v];
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
    UpwindSets(field, m, cand_value, cand_mask, upwind_mask, label_start);
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
        for (int v = 0; v < nv; ++v) {
            if (lab[v] == UNSET_LABEL) {
                lab[v] = -1;
                unlabeled += owned[v];
            }
            if (!(t[v] < INF)) {
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
            if (es[e] >= 0) {
                continue;
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
