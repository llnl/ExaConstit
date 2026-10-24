/*
 * The geodesic metric (geodesic_metric.hpp): tensor -> stretch, and the automatic elongation of
 * the regularized columnar / lamellar metrics from the extents of the active material.
 */
#include "microstructure/labeling/geodesic_metric.hpp"

#include "microstructure/geometry/anisotropy_tensors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

namespace {

/**
 * @brief Two unit vectors u, v that complete the unit vector n to an orthonormal basis. u is
 * built from the coordinate axis least aligned with n, so the choice is stable and identical on
 * every rank (replicated inputs). Host.
 */
void CompleteBasis(const std::array<double, 3>& n, std::array<double, 3>& u,
                   std::array<double, 3>& v) {
    int axis = 0;
    for (int c = 1; c < 3; ++c) {
        if (std::fabs(n[c]) < std::fabs(n[axis])) {
            axis = c;
        }
    }
    std::array<double, 3> e = {0.0, 0.0, 0.0};
    e[axis] = 1.0;
    // u = e - (e . n) n, normalized; v = n x u
    const double en = n[axis];
    u = {e[0] - en * n[0], e[1] - en * n[1], e[2] - en * n[2]};
    const double lu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    u = {u[0] / lu, u[1] / lu, u[2] / lu};
    v = {n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]};
}

/** @brief x rounded up to two significant digits (x > 0). Host. */
double RoundUpTwoDigits(double x) {
    const double scale = std::pow(10.0, std::floor(std::log10(x)) - 1.0);
    return std::ceil(x / scale - 1e-9) * scale;
}

} // namespace

/*
 * AutoGeodesicElongation: the extents along n (the axis or normal) and two perpendicular
 * directions u, v over the corner vertices of the active elements, as min/max reductions
 * (MPI_MIN / MPI_MAX are exact, so every partition gets the same bits). A rank without active
 * elements contributes +inf / -inf.
 */
double AutoGeodesicElongation(bool columnar, const std::array<double, 3>& direction,
                              mfem::ParMesh& pmesh, MPI_Comm comm, int num_seeds,
                              const ElementMask* mask, bool* clamped) {
    if (num_seeds < 1) {
        throw std::invalid_argument("AutoGeodesicElongation: num_seeds must be >= 1");
    }
    const std::array<double, 3> n = detail::UnitDirection(direction, "AutoGeodesicElongation");
    std::array<double, 3> u{};
    std::array<double, 3> v{};
    CompleteBasis(n, u, v);
    const std::array<const std::array<double, 3>*, 3> dirs = {&n, &u, &v};

    // lo[d] = min over active corners of dir_d . x; hi[d] = max (sent as -max to use MPI_MIN)
    const double inf = std::numeric_limits<double>::infinity();
    double bounds[6] = {inf, inf, inf, inf, inf, inf};
    mfem::Array<int> verts;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        if (!ElementIsActive(mask, e)) {
            continue;
        }
        pmesh.GetElementVertices(e, verts);
        for (int j = 0; j < verts.Size(); ++j) {
            const double* x = pmesh.GetVertex(verts[j]);
            for (int d = 0; d < 3; ++d) {
                const std::array<double, 3>& dir = *dirs[static_cast<std::size_t>(d)];
                const double p = dir[0] * x[0] + dir[1] * x[1] + dir[2] * x[2];
                bounds[d] = std::min(bounds[d], p);
                bounds[3 + d] = std::min(bounds[3 + d], -p);
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, bounds, 6, MPI_DOUBLE, MPI_MIN, comm);
    double extent[3];
    for (int d = 0; d < 3; ++d) {
        extent[d] = std::isfinite(bounds[d]) ? (-bounds[3 + d] - bounds[d]) : 0.0;
    }

    const double count = static_cast<double>(num_seeds);
    double k = GEODESIC_MIN_AUTO_ELONGATION;
    if (columnar) {
        // Column width if the seeds shared the bounding-box cross-section
        const double width = std::sqrt(extent[1] * extent[2] / count);
        if (width > 0.0 && extent[0] > 0.0) {
            k = 10.0 * extent[0] / width;
        }
    } else {
        // Layer thickness and the in-plane diameter
        const double thickness = extent[0] / count;
        const double diameter = std::sqrt(extent[1] * extent[1] + extent[2] * extent[2]);
        if (thickness > 0.0 && diameter > 0.0) {
            k = 10.0 * diameter / thickness;
        }
    }
    k = RoundUpTwoDigits(std::max(k, GEODESIC_MIN_AUTO_ELONGATION));
    const bool limited = k > GEODESIC_MAX_ELONGATION;
    if (clamped != nullptr) {
        *clamped = limited;
    }
    return limited ? GEODESIC_MAX_ELONGATION : k;
}

/*
 * ResolveGeodesicMetric: identity unless the metric is Quadric; for Quadric by anisotropy kind
 * (see the header). The elongation range is checked here too, so a direct caller gets the same
 * message as validate().
 */
GeodesicMetric ResolveGeodesicMetric(const TessellationOptions& tessellation,
                                     mfem::ParMesh& pmesh, MPI_Comm comm, int num_seeds,
                                     const ElementMask* mask) {
    GeodesicMetric metric;
    if (tessellation.metric != MetricType::QUADRIC) {
        return metric;
    }
    const std::string& kind = tessellation.anisotropy_kind;
    if (kind == "none" || kind == "tensor") {
        metric.stretch = MetricStretchFromTensor(tessellation.global_tensor);
        return metric;
    }
    if (kind == "columnar" || kind == "lamellar") {
        const bool columnar = (kind == "columnar");
        const std::array<double, 3>& direction =
            columnar ? tessellation.anisotropy_axis : tessellation.anisotropy_normal;
        double k = tessellation.geodesic_elongation;
        if (k == 0.0) {
            k = AutoGeodesicElongation(columnar, direction, pmesh, comm, num_seeds, mask,
                                       &metric.clamped);
            metric.automatic = true;
        } else if (!std::isfinite(k) || k < 1.0 || k > GEODESIC_MAX_ELONGATION) {
            throw std::invalid_argument("ResolveGeodesicMetric: geodesic elongation must be in "
                                        "[1, 1e4] (or 0 = automatic)");
        }
        metric.elongation = k;
        metric.stretch = columnar ? ColumnarStretch(direction, k) : LamellarStretch(direction, k);
        return metric;
    }
    if (kind == "per_seed") {
        return metric; // one tensor per seed: the weighted solver maps per seed, the field not
    }
    throw std::invalid_argument("ResolveGeodesicMetric: unknown anisotropy kind \"" + kind +
                                "\"");
}

} // namespace exaconstit::microstructure
