/*
 * Spring-balanced pseudo-elastic fitting of the mesh nodes (conformal_fitter.hpp).
 */
#include "microstructure/conformal/conformal_fitter.hpp"

#include "microstructure/util/collective_errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

namespace exaconstit::microstructure {

namespace {

/** @brief Spring factor of nodes on triple lines and quad points (relative to face nodes). */
constexpr double JUNCTION_SPRING = 0.1;
/** @brief Elements within this margin above the floor also back off their springs. */
constexpr double BACKOFF_MARGIN = 0.05;
/** @brief Scaled Jacobian below which an element is stiffened. */
constexpr double STIFFEN_BELOW = 0.6;
/** @brief Cap of the stiffness factor. */
constexpr double MAX_STIFFNESS = 1e3;
/** @brief Bulk quality guard: at most this share of elements below floor + BULK_MARGIN. */
constexpr double BULK_SHARE = 0.01;
constexpr double BULK_MARGIN = 0.2;

bool IsAxis(const double n[3]) {
    int zeros = 0;
    for (int d = 0; d < 3; ++d) {
        zeros += (n[d] == 0.0) ? 1 : 0;
    }
    return zeros == 2;
}

std::int64_t GlobalSum(std::int64_t local, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT64_T, MPI_SUM, comm);
    return local;
}

/** @brief Node positions in node order (3 per node) from the Nodes grid function. */
void GatherPositions(const mfem::GridFunction& nodes, const mfem::FiniteElementSpace& fes, int nv,
                     std::vector<double>& out) {
    out.resize(3 * static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        for (int d = 0; d < 3; ++d) {
            out[3 * static_cast<std::size_t>(v) + d] = nodes(fes.DofToVDof(v, d));
        }
    }
}

/**
 * @brief Quad points: several nodes can share one 4-label set and so one target point; only the
 * node closest to it stays fitted. The owned candidates' (labels, squared distance, position) are
 * gathered on every rank and the smallest (distance, then position) wins, so every copy decides
 * the same way. Clears marker[v] for the losers.
 */
void KeepClosestQuadPointNodes(const NodeLabels& labels, const NodeTargets& targets,
                               mfem::Array<bool>& marker) {
    const VertexField& field = labels.Field();
    const MPI_Comm comm = field.Comm();
    const int nv = labels.NumNodes();
    const double* xyz = field.Coordinates().HostRead();
    const double* t = targets.Targets().HostRead();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const int* loff = labels.LabelOffsets().HostRead();
    const int* litems = labels.Labels().HostRead();
    const auto is_quad = [&](int v) { return marker[v] && labels.NumLabels(v) == 4; };
    std::vector<double> mine;
    for (int v = 0; v < nv; ++v) {
        if (!is_quad(v) || owned[v] == 0) {
            continue;
        }
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            d2 += (t[3 * v + d] - xyz[3 * v + d]) * (t[3 * v + d] - xyz[3 * v + d]);
        }
        mine.insert(mine.end(), litems + loff[v], litems + loff[v] + 4);
        mine.push_back(d2);
        mine.insert(mine.end(), xyz + 3 * v, xyz + 3 * v + 3);
    }
    int count = static_cast<int>(mine.size());
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r) - 1] + counts[static_cast<std::size_t>(r) - 1];
    }
    std::vector<double> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(mine.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, comm);
    std::map<std::array<int, 4>, std::array<double, 4>> best; // labels -> (d2, x, y, z)
    for (std::size_t k = 0; k + 8 <= all.size(); k += 8) {
        const std::array<int, 4> key = {static_cast<int>(all[k]), static_cast<int>(all[k + 1]),
                                        static_cast<int>(all[k + 2]), static_cast<int>(all[k + 3])};
        const std::array<double, 4> val = {all[k + 4], all[k + 5], all[k + 6], all[k + 7]};
        const auto it = best.find(key);
        if (it == best.end() || val < it->second) {
            best[key] = val;
        }
    }
    for (int v = 0; v < nv; ++v) {
        if (!is_quad(v)) {
            continue;
        }
        const std::array<int, 4> key = {litems[loff[v]], litems[loff[v] + 1], litems[loff[v] + 2],
                                        litems[loff[v] + 3]};
        const auto it = best.find(key);
        marker[v] = it != best.end() && it->second[1] == xyz[3 * v] &&
                    it->second[2] == xyz[3 * v + 1] && it->second[3] == xyz[3 * v + 2];
    }
}

} // namespace

/*
 * FitConformal
 *  1. Checks; quality of the input mesh.
 *  2. Fitted nodes (FIT, one per quad point), target positions, fixed components.
 *  3. Rounds of spring-balanced elastic solves with local back-off and quality stiffening, kept
 *     while above the floor and improving (file comment).
 *  4. Deviation and quality of the result.
 */
ConformalFitReport FitConformal(mfem::ParMesh& pmesh, const NodeTargets& targets,
                                const ConformalFitOptions& options) {
    const MPI_Comm comm = pmesh.GetComm();
    const NodeLabels& labels = targets.Labels();
    const VertexField& field = labels.Field();

    // Step 1
    auto* nodes = dynamic_cast<mfem::ParGridFunction*>(pmesh.GetNodes());
    ThrowIfAnyRank(comm, nodes == nullptr || &field.Mesh() != &pmesh,
                   "FitConformal: the mesh needs Nodes (a ParGridFunction; call "
                   "ParMesh::SetCurvature(1)) and must be the mesh of the targets");
    mfem::ParFiniteElementSpace& fes = *nodes->ParFESpace();
    ThrowIfAnyRank(comm, !field.UsesMeshNodes() || field.NumVertices() != fes.GetNDofs(),
                   "FitConformal: the node labels must use the mesh Nodes "
                   "(FieldNodes::MESH_NODES)");
    ThrowIfAnyRank(comm,
                   !(options.initial_spring > 0.0) || options.max_rounds < 1 ||
                       options.max_attempts < 1 || options.min_improvement < 0.0 ||
                       options.stiffening_exponent < 0.0 || options.stiffness_smoothing < 0 ||
                       !(options.poisson_ratio >= 0.0 && options.poisson_ratio < 0.5),
                   "FitConformal: invalid options");
    ConformalFitReport report;
    report.quality_before = ComputeMeshQuality(pmesh);
    const int nv = labels.NumNodes();
    const int ne = pmesh.GetNE();
    const double floor = (pmesh.GetTypicalElementGeometry() == mfem::Geometry::CUBE)
                             ? MESH_QUALITY_FLOOR_HEX
                             : MESH_QUALITY_FLOOR_TET;
    const mfem::ParGridFunction x0(*nodes);

    // Step 2
    mfem::Array<bool> marker(nv);
    for (int v = 0; v < nv; ++v) {
        marker[v] = targets.Status(v) == TargetStatus::FIT;
    }
    KeepClosestQuadPointNodes(labels, targets, marker);
    const double* t = targets.Targets().HostRead();
    const int* owned = field.OwnedVertexMarker().HostRead();
    const int* surf_off = labels.SurfaceOffsets().HostRead();
    const double* surf_normal = labels.SurfaceNormals().HostRead();
    const int* surf_planar = labels.SurfacePlanar().HostRead();
    const int* loff = labels.LabelOffsets().HostRead();
    const int* litems = labels.Labels().HostRead();
    std::vector<char> fixed(3 * static_cast<std::size_t>(nv), 0);
    std::int64_t fit_nodes = 0;
    std::int64_t fixed_nodes = 0;
    for (int v = 0; v < nv; ++v) {
        const auto vs = 3 * static_cast<std::size_t>(v);
        bool all_fixed = false;
        for (int s = surf_off[v]; s < surf_off[v + 1]; ++s) {
            const double* n = surf_normal + 3 * s;
            if (surf_planar[s] != 0 && IsAxis(n)) {
                for (int d = 0; d < 3; ++d) {
                    fixed[vs + d] |= (n[d] != 0.0) ? 1 : 0;
                }
            } else {
                all_fixed = all_fixed || !marker[v];
            }
        }
        if (all_fixed) {
            fixed[vs] = fixed[vs + 1] = fixed[vs + 2] = 1;
        }
        fit_nodes += (marker[v] && owned[v]) ? 1 : 0;
        fixed_nodes += (fixed[vs] && fixed[vs + 1] && fixed[vs + 2] && owned[v]) ? 1 : 0;
    }
    report.fit_nodes = GlobalSum(fit_nodes, comm);
    report.fixed_nodes = GlobalSum(fixed_nodes, comm);

    // Deviation of the fitted nodes from their grain boundaries at the current positions
    std::vector<double> here;
    const auto deviation = [&](double& max_dev, double& mean_dev) {
        GatherPositions(*nodes, fes, nv, here);
        std::vector<NodeLabelQuery> queries;
        for (int v = 0; v < nv; ++v) {
            if (!marker[v]) {
                continue;
            }
            NodeLabelQuery q;
            q.node = v;
            q.num_labels = labels.NumLabels(v);
            for (int k = 0; k < std::min(q.num_labels, NODE_QUERY_LABELS); ++k) {
                q.labels[k] = litems[loff[v] + k];
            }
            q.position = &here[3 * static_cast<std::size_t>(v)];
            queries.push_back(q);
        }
        const std::vector<NodeTargetResult> res = targets.Evaluate(queries);
        double sums[2] = {0.0, 0.0};
        max_dev = 0.0;
        for (std::size_t i = 0; i < queries.size(); ++i) {
            const int v = queries[i].node;
            if (res[i].status != TargetStatus::FIT || owned[v] == 0) {
                continue;
            }
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                d2 += (res[i].target[d] - queries[i].position[d]) *
                      (res[i].target[d] - queries[i].position[d]);
            }
            const double dev = std::sqrt(d2) / targets.MeshSize(v);
            max_dev = std::max(max_dev, dev);
            sums[0] += dev;
            sums[1] += 1.0;
        }
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &max_dev, 1, MPI_DOUBLE, MPI_MAX, comm);
        mean_dev = (sums[1] > 0.0) ? sums[0] / sums[1] : 0.0;
    };
    double unused_max = 0.0;
    deviation(unused_max, report.initial_deviation);

    // Step 3
    if (report.fit_nodes > 0) {
        mfem::L2_FECollection l2(0, 3);
        mfem::ParFiniteElementSpace l2_fes(&pmesh, &l2);
        mfem::ParGridFunction stiffness(&l2_fes);
        mfem::ParGridFunction quality_factor(&l2_fes);
        quality_factor = 1.0;
        mfem::Vector multiplier(nv);
        multiplier = 1.0;
        const int* en_off = field.ElementVertexOffsets().HostRead();
        const int* en = field.ElementVertices().HostRead();
        std::vector<double> sj(static_cast<std::size_t>(ne), 1.0);

        // Essential (fixed) true dofs; they keep their input value (zero displacement)
        mfem::Array<int> ess_marker(fes.GetVSize());
        ess_marker = 0;
        for (int v = 0; v < nv; ++v) {
            for (int d = 0; d < 3; ++d) {
                ess_marker[fes.DofToVDof(v, d)] = fixed[3 * static_cast<std::size_t>(v) + d];
            }
        }
        mfem::Array<int> ess_tdof_marker;
        mfem::Array<int> ess_tdof;
        fes.GetRestrictionMatrix()->BooleanMult(ess_marker, ess_tdof_marker);
        mfem::FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof);

        // Stiffness for the next solve: quality factor, spread to the neighbors
        const auto update_stiffness = [&]() {
            for (int e = 0; e < ne; ++e) {
                stiffness(e) = quality_factor(e);
            }
            for (int pass = 0; pass < options.stiffness_smoothing; ++pass) {
                mfem::Vector nodal(nv);
                nodal = 0.0;
                for (int e = 0; e < ne; ++e) {
                    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                        nodal(en[k]) = std::max(nodal(en[k]), stiffness(e));
                    }
                }
                nodal.Neg(); // maximum over the copies as the minimum of the negatives
                field.ReduceSharedMinimum(nodal);
                nodal.Neg();
                for (int e = 0; e < ne; ++e) {
                    double sum = 0.0;
                    for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                        sum += nodal(en[k]);
                    }
                    stiffness(e) = std::max(stiffness(e), sum / (en_off[e + 1] - en_off[e]));
                }
            }
        };

        // One spring-balanced elastic solve on the input geometry; sets the Nodes to x0 + u
        mfem::ParGridFunction u(&fes);
        const double nu = options.poisson_ratio;
        const auto solve = [&](double c) {
            *nodes = x0;
            update_stiffness();
            mfem::GridFunctionCoefficient e_coeff(&stiffness);
            mfem::ProductCoefficient mu(1.0 / (2.0 * (1.0 + nu)), e_coeff);
            mfem::ProductCoefficient lambda(nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e_coeff);
            mfem::ParBilinearForm a(&fes);
            a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));
            a.Assemble();
            // Springs and their loads on the local vdofs, then on the true dofs (owned values
            // only, so the spring of a shared node counts once)
            mfem::ParGridFunction k_local(&fes);
            mfem::ParGridFunction load_local(&fes);
            k_local = 0.0;
            load_local = 0.0;
            for (int v = 0; v < nv; ++v) {
                if (!marker[v]) {
                    continue;
                }
                const double k = c * multiplier(v) * targets.MeshSize(v) *
                                 (labels.NumLabels(v) >= 3 ? JUNCTION_SPRING : 1.0);
                for (int d = 0; d < 3; ++d) {
                    const int vd = fes.DofToVDof(v, d);
                    k_local(vd) = k;
                    load_local(vd) = k * (t[3 * v + d] - x0(vd));
                }
            }
            u = 0.0;
            mfem::ParLinearForm rhs(&fes);
            rhs = 0.0;
            mfem::HypreParMatrix A;
            mfem::Vector X;
            mfem::Vector B;
            a.FormLinearSystem(ess_tdof, u, rhs, A, X, B);
            mfem::Vector k_true(fes.GetTrueVSize());
            mfem::Vector load_true(fes.GetTrueVSize());
            fes.GetRestrictionMatrix()->Mult(k_local, k_true);
            fes.GetRestrictionMatrix()->Mult(load_local, load_true);
            for (const int i : ess_tdof) {
                k_true(i) = 0.0;
                load_true(i) = 0.0;
            }
            mfem::SparseMatrix diag(fes.GetTrueVSize());
            for (int i = 0; i < fes.GetTrueVSize(); ++i) {
                diag.Set(i, i, k_true(i));
            }
            diag.Finalize();
            mfem::HypreParMatrix P(comm, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets(), &diag);
            std::unique_ptr<mfem::HypreParMatrix> AP(mfem::ParAdd(&A, &P));
            B += load_true;
            mfem::HypreBoomerAMG amg(*AP);
            amg.SetSystemsOptions(3, true);
            amg.SetPrintLevel(0);
            mfem::CGSolver cg(comm);
            cg.SetRelTol(1e-10);
            cg.SetMaxIter(500);
            cg.SetPrintLevel(0);
            cg.SetPreconditioner(amg);
            cg.SetOperator(*AP);
            cg.Mult(B, X);
            a.RecoverFEMSolution(X, rhs, u);
            for (int i = 0; i < nodes->Size(); ++i) {
                (*nodes)(i) = x0(i) + u(i);
            }
            ++report.solves;
        };

        mfem::Vector good(x0);
        double c = options.initial_spring;
        double last_mean = report.initial_deviation;
        for (int round = 0; round < options.max_rounds; ++round) {
            bool ok = false;
            for (int attempt = 0; attempt < options.max_attempts && !ok; ++attempt) {
                solve(c);
                double worst = HUGE_VAL;
                bool inverted = false;
                for (int e = 0; e < ne; ++e) {
                    double ar = 0.0;
                    double det = 0.0;
                    ElementQuality(pmesh, e, sj[static_cast<std::size_t>(e)], ar, det);
                    worst = std::min(worst, sj[static_cast<std::size_t>(e)]);
                    inverted = inverted || det <= 0.0;
                }
                ok = !AnyRank(inverted || worst < floor, comm);
                // Stiffen the distorted elements (never softened again) and, if the solve failed,
                // halve the springs of the nodes of the elements near the floor
                mfem::Vector next(multiplier);
                for (int e = 0; e < ne; ++e) {
                    const double q = sj[static_cast<std::size_t>(e)];
                    if (options.stiffening_exponent > 0.0 && q < STIFFEN_BELOW) {
                        const double ratio = STIFFEN_BELOW / std::max(q, 0.01);
                        const double f =
                            std::min(MAX_STIFFNESS, std::pow(ratio, options.stiffening_exponent));
                        quality_factor(e) = std::max(quality_factor(e), f);
                    }
                    if (!ok && q < floor + BACKOFF_MARGIN) {
                        for (int k = en_off[e]; k < en_off[e + 1]; ++k) {
                            if (marker[en[k]]) {
                                next(en[k]) = 0.5 * multiplier(en[k]);
                            }
                        }
                    }
                }
                field.ReduceSharedMinimum(next);
                multiplier = next;
            }
            if (!ok) {
                break;
            }
            // Bulk quality guard: a round that leaves more than 1 % of the elements below
            // floor + 0.2 is not kept (quality first: the small extra fit is not worth it)
            double below[2] = {0.0, static_cast<double>(ne)};
            for (int e = 0; e < ne; ++e) {
                below[0] += (sj[static_cast<std::size_t>(e)] < floor + BULK_MARGIN) ? 1.0 : 0.0;
            }
            MPI_Allreduce(MPI_IN_PLACE, below, 2, MPI_DOUBLE, MPI_SUM, comm);
            if (below[0] > BULK_SHARE * below[1]) {
                break;
            }
            good = *nodes;
            report.rounds = round + 1;
            report.final_spring = c;
            double max_dev = 0.0;
            double mean_dev = 0.0;
            deviation(max_dev, mean_dev);
            // Weak early springs improve little, so the stop rule applies from the fourth round
            if (round >= 3 && mean_dev > (1.0 - options.min_improvement) * last_mean) {
                break;
            }
            last_mean = mean_dev;
            c *= 2.0;
        }
        *nodes = good;
    }

    // Step 4
    deviation(report.max_deviation, report.mean_deviation);
    report.quality_after = ComputeMeshQuality(pmesh);
    return report;
}

} // namespace exaconstit::microstructure
