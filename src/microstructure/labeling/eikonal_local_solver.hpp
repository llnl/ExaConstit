#pragma once

#include "mfem.hpp"

#include <math.h>

/**
 * @file eikonal_local_solver.hpp
 * @brief The local update of the Fast Iterative Method (FIM): the arrival distance at a mesh vertex
 * v computed from known arrival distances on a triangle, a segment or a single vertex of one of
 * v's elements. Device-callable (MFEM_HOST_DEVICE, C math functions only), used inside the
 * mfem::forall sweeps of labeling/geodesic_fim.cpp.
 *
 * **The problem.** T(x) is the shortest path length from the seeds through the material, the
 * solution of the eikonal equation |grad T| = 1. Suppose T is known at the vertices a, b, c of a
 * triangle and varies linearly in between (T_lin). A path that reaches v by crossing the triangle
 * at the point p arrives at time T_lin(p) + |v - p| (it reaches p at T_lin(p), then goes straight
 * to v at unit speed). The best such path gives
 *     T_new(v) = min over p in the triangle of  T_lin(p) + |v - p|.                         (1)
 * This is the local solver of Fu, Kirby & Whitaker, "A fast iterative method for solving the
 * eikonal equation on tetrahedral domains", SIAM J. Sci. Comput. 35(5), 2013, written here as a
 * minimization over the face. The function being minimized is convex in p (a linear function plus
 * a distance), so (1) has a single minimum value, which is either at an interior stationary point
 * of the triangle or on its boundary (an edge or a vertex).
 *
 * **Interior minimizer, derived in plain terms.** Write the triangle with one corner x0 and edge
 * vectors e1 = x1 - x0, e2 = x2 - x0, a point as p = x0 + l1 e1 + l2 e2 (l1, l2 >= 0,
 * l1 + l2 <= 1; l1, l2 are barycentric-like coordinates) and w = v - x0. Then
 *     f(l) = T0 + l1 d1 + l2 d2 + |w - l1 e1 - l2 e2|,   d1 = T1 - T0, d2 = T2 - T0.
 * Let M be the 2x2 "Gram" matrix of the edges, M_ij = e_i . e_j, and l0 = M^-1 (e1.w, e2.w) the
 * coordinates of the foot point of v in the triangle's plane; h = |w - l0_1 e1 - l0_2 e2| is the
 * height of v above the plane. With mu = l0 - l the remaining vector from p to v is
 * r = (in-plane part E mu) + (normal part of height h), so |r|^2 = mu^T M mu + h^2. Setting the
 * gradient of f to zero gives d = M mu / |r|, i.e. mu = |r| M^-1 d. Substituting into |r|^2:
 *     |r|^2 = |r|^2 q + h^2,   q = d^T M^-1 d   =>   |r| = h / sqrt(1 - q).
 * q is the squared slope of T_lin along the face. If q >= 1 the face is "too steep" (the front
 * moves along the face at least as fast as it can travel to v), no interior stationary point
 * exists, and the minimum is on the boundary. Otherwise l = l0 - |r| M^-1 d, and it is accepted if
 * it lies strictly inside the triangle (l1 > 0, l2 > 0, l1 + l2 < 1). The value is then evaluated
 * directly from f(l), so it is the arrival time of an actual path through the computed point.
 *
 * **Segment (edge) minimizer.** The same derivation in one dimension: p = x0 + l e, e = x1 - x0,
 * w = v - x0, M = e.e, l0 = (e.w)/M, h = |w - l0 e|, d = T1 - T0, q = d^2 / M; if q < 1 then
 * l = l0 - (h / sqrt(1 - q)) d / M, accepted for 0 < l < 1.
 *
 * **Vertex.** T(a) + |v - a| (the path goes through the vertex a itself).
 *
 * **Full face.** (1) = the smallest of the interior value, the three segment values and the three
 * vertex values (EikonalTriangleUpdate).
 *
 * **Infinite inputs (+infinity = not reached yet).** T_lin is +infinity at every point with a
 * positive weight on an unreached vertex, so (1) restricted to the reached vertices is exact: the
 * interior candidate needs all three values finite, a segment both of its values, a vertex its own
 * value; a candidate with a missing value returns +infinity. No infinity enters the arithmetic, so
 * no NaN can appear.
 *
 * **Same bits on every rank.** Floating-point results depend on the order of the operations, and
 * the same triangle appears with different local vertex numbers (and in different elements) on
 * different ranks. The segment and triangle functions therefore first sort their vertices by
 * coordinates (lexicographic: x, then y, then z), so the computation for a given geometric (v,
 * face) pair is always carried out in the same order. Vertex coordinates are bitwise identical on
 * every copy (VertexField::Coordinates). All ranks run the same binary, and the FIM solver
 * computes every candidate value in exactly one kernel (geodesic_fim.cpp, CandidateSweep), so a
 * given (v, face) pair always goes through the same machine instructions and gives the same bits.
 * This does not rely on MICROSTRUCTURE_STRICT_FP (off by default): with multiply-add fusion
 * allowed, the compiler may fuse differently in two *different* copies of the code (two kernels
 * that inline these functions), which is why the solver never recomputes a value in a second
 * kernel and compares it with the first. A test or tool that calls these functions from its own
 * code may therefore differ from the library in the last bit unless both use STRICT_FP.
 *
 * **Memory layout.** Points are passed as pointers to 3 doubles (x, y, z) in mesh length units,
 * e.g. &xyz[3*v] into VertexField::Coordinates(). Distances are in the same units; they may be
 * negative (Additive metric: T0 = |v - s| - a_i).
 */

namespace exaconstit::microstructure {

/**
 * @brief The "not reached" distance, +infinity (HUGE_VAL of math.h is +infinity on IEEE-754
 * machines and is usable in device code). Host and device.
 */
MFEM_HOST_DEVICE inline double EikonalInfinity() {
    return HUGE_VAL;
}

/** @brief Dot product of two 3-vectors. Host and device. */
MFEM_HOST_DEVICE inline double EikonalDot(const double* p, const double* q) {
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
}

/**
 * @brief Lexicographic "p before q" on coordinates (x first, then y, then z), the canonical vertex
 * order of the segment and triangle updates. Host and device.
 */
MFEM_HOST_DEVICE inline bool EikonalLexLess(const double* p, const double* q) {
    if (p[0] != q[0]) {
        return p[0] < q[0];
    }
    if (p[1] != q[1]) {
        return p[1] < q[1];
    }
    return p[2] < q[2];
}

/**
 * @brief Vertex candidate: arrival at v through vertex a, T(a) + |v - a|.
 *
 * Host and device.
 * @param xv, xa coordinates (3 doubles each).
 * @param ta     T(a), +infinity if a is not reached.
 * @return the candidate, +infinity if ta is +infinity.
 */
MFEM_HOST_DEVICE inline double EikonalVertexValue(const double* xv, const double* xa, double ta) {
    const double inf = EikonalInfinity();
    if (!(ta < inf)) {
        return inf;
    }
    const double d[3] = {xv[0] - xa[0], xv[1] - xa[1], xv[2] - xa[2]};
    return ta + sqrt(EikonalDot(d, d));
}

/**
 * @brief Segment candidate: the minimum of T_lin(p) + |v - p| over the *open* segment (a, b) when
 * it is attained at an interior point of the segment (see the file comment), else +infinity (the
 * endpoints are covered by EikonalVertexValue).
 *
 * The two endpoints are sorted by coordinates first, so (a, b) and (b, a) give the same bits.
 * Host and device.
 * @param xv, xa, xb coordinates (3 doubles each); a != b.
 * @param ta, tb     T(a), T(b), +infinity if not reached.
 * @return the candidate, or +infinity if a value is missing, the segment is too steep (q >= 1) or
 *         the minimizer is not strictly inside the segment.
 */
MFEM_HOST_DEVICE inline double EikonalSegmentValue(const double* xv,
                                                   const double* xa,
                                                   const double* xb,
                                                   double ta,
                                                   double tb) {
    const double inf = EikonalInfinity();
    if (!(ta < inf) || !(tb < inf)) {
        return inf;
    }
    // Canonical order: x0 is the lexicographically smaller endpoint
    const double* x0 = xa;
    const double* x1 = xb;
    double t0 = ta;
    double t1 = tb;
    if (EikonalLexLess(xb, xa)) {
        x0 = xb;
        x1 = xa;
        t0 = tb;
        t1 = ta;
    }
    // Edge vector e, vector w from x0 to v, foot point l0 of v on the line, height h
    const double e[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
    const double w[3] = {xv[0] - x0[0], xv[1] - x0[1], xv[2] - x0[2]};
    const double m = EikonalDot(e, e);
    if (!(m > 0.0)) {
        return inf; // coincident endpoints (degenerate mesh)
    }
    const double l0 = EikonalDot(e, w) / m;
    const double n[3] = {w[0] - l0 * e[0], w[1] - l0 * e[1], w[2] - l0 * e[2]};
    const double h2 = EikonalDot(n, n);
    // Slope of T along the edge: q = d^2 / |e|^2 must be below 1 for an interior stationary point
    const double d = t1 - t0;
    const double q = d * d / m;
    if (!(q < 1.0)) {
        return inf;
    }
    // Stationary point l = l0 - |r| d / M with |r| = h / sqrt(1 - q)
    const double r = sqrt(h2 / (1.0 - q));
    const double l = l0 - r * d / m;
    if (!(l > 0.0 && l < 1.0)) {
        return inf;
    }
    // Evaluate f(l) directly: T_lin(p) + |v - p|
    const double rv[3] = {w[0] - l * e[0], w[1] - l * e[1], w[2] - l * e[2]};
    return t0 + l * d + sqrt(EikonalDot(rv, rv));
}

/**
 * @brief Triangle-interior candidate: the minimum of T_lin(p) + |v - p| over the triangle (a, b, c)
 * when it is attained strictly inside the triangle (closed form, see the file comment), else
 * +infinity (the boundary is covered by the segment and vertex candidates).
 *
 * The three vertices are sorted by coordinates first, so any permutation gives the same bits.
 * A nearly collinear triangle (Gram determinant <= 1e-12 |e1|^2 |e2|^2) has no usable interior
 * and returns +infinity; its segments still cover it. Host and device.
 * @param xv, xa, xb, xc coordinates (3 doubles each); a, b, c distinct.
 * @param ta, tb, tc     T at a, b, c; +infinity if not reached.
 * @return the candidate or +infinity.
 */
MFEM_HOST_DEVICE inline double EikonalTriangleInteriorValue(const double* xv,
                                                            const double* xa,
                                                            const double* xb,
                                                            const double* xc,
                                                            double ta,
                                                            double tb,
                                                            double tc) {
    const double inf = EikonalInfinity();
    if (!(ta < inf) || !(tb < inf) || !(tc < inf)) {
        return inf;
    }
    // Canonical order x0 < x1 < x2 (three compare-and-swap steps of a sorting network)
    const double* p[3] = {xa, xb, xc};
    double t[3] = {ta, tb, tc};
    for (int pass = 0; pass < 3; ++pass) {
        const int i = (pass == 1) ? 1 : 0; // compare (0,1), (1,2), (0,1)
        if (EikonalLexLess(p[i + 1], p[i])) {
            const double* tmp_p = p[i];
            p[i] = p[i + 1];
            p[i + 1] = tmp_p;
            const double tmp_t = t[i];
            t[i] = t[i + 1];
            t[i + 1] = tmp_t;
        }
    }
    const double* x0 = p[0];
    // Edge vectors from x0, vector w from x0 to v
    const double e1[3] = {p[1][0] - x0[0], p[1][1] - x0[1], p[1][2] - x0[2]};
    const double e2[3] = {p[2][0] - x0[0], p[2][1] - x0[1], p[2][2] - x0[2]};
    const double w[3] = {xv[0] - x0[0], xv[1] - x0[1], xv[2] - x0[2]};
    // Gram matrix M of the edges and its determinant
    const double m00 = EikonalDot(e1, e1);
    const double m01 = EikonalDot(e1, e2);
    const double m11 = EikonalDot(e2, e2);
    const double det = m00 * m11 - m01 * m01;
    if (!(det > 1e-12 * m00 * m11)) {
        return inf; // (nearly) collinear vertices: no interior
    }
    // Foot point of v in the plane: l0 = M^-1 (e1.w, e2.w); height^2 h2 = |w - E l0|^2
    const double b0 = EikonalDot(e1, w);
    const double b1 = EikonalDot(e2, w);
    const double l0a = (m11 * b0 - m01 * b1) / det;
    const double l0b = (m00 * b1 - m01 * b0) / det;
    const double n[3] = {w[0] - l0a * e1[0] - l0b * e2[0], w[1] - l0a * e1[1] - l0b * e2[1],
                         w[2] - l0a * e1[2] - l0b * e2[2]};
    const double h2 = EikonalDot(n, n);
    // Slope of T_lin on the face: g = M^-1 d, q = d . g (squared in-plane gradient of T_lin)
    const double d1 = t[1] - t[0];
    const double d2 = t[2] - t[0];
    const double g1 = (m11 * d1 - m01 * d2) / det;
    const double g2 = (m00 * d2 - m01 * d1) / det;
    const double q = d1 * g1 + d2 * g2;
    if (!(q < 1.0)) {
        return inf; // too steep: the minimum is on the boundary
    }
    // Stationary point l = l0 - |r| g with |r| = h / sqrt(1 - q); keep it if strictly inside
    const double r = sqrt(h2 / (1.0 - q));
    const double la = l0a - r * g1;
    const double lb = l0b - r * g2;
    if (!(la > 0.0 && lb > 0.0 && la + lb < 1.0)) {
        return inf;
    }
    // Evaluate f(l) directly: T_lin(p) + |v - p|
    const double rv[3] = {w[0] - la * e1[0] - lb * e2[0], w[1] - la * e1[1] - lb * e2[1],
                          w[2] - la * e1[2] - lb * e2[2]};
    return t[0] + la * d1 + lb * d2 + sqrt(EikonalDot(rv, rv));
}

/** @brief Result of a full-face update: value and which vertices support the minimizer. */
struct EikonalUpdate {
    /** @brief min over the face of T_lin(p) + |v - p|; +infinity if no vertex is reached. */
    double value;
    /**
     * @brief Bit mask of the face vertices the minimizer depends on, in the caller's argument
     * order: bit 0 = a, bit 1 = b, bit 2 = c (7 = interior point, two bits = a point on an edge,
     * one bit = a vertex). When several candidates attain the minimum exactly, their masks are
     * combined. 0 when value is +infinity.
     */
    int support;
};

/**
 * @brief Full-face local update (1): the minimum over the closed triangle (a, b, c) of
 * T_lin(p) + |v - p|, i.e. the smallest of the interior, the three segment and the three vertex
 * candidates, with the support of the minimizer.
 *
 * Host and device. Symmetric: permuting (a, b, c) together with (ta, tb, tc) permutes the support
 * bits and leaves the value bitwise unchanged.
 * @param xv, xa, xb, xc coordinates (3 doubles each); a, b, c distinct.
 * @param ta, tb, tc     T at a, b, c; +infinity if not reached.
 */
MFEM_HOST_DEVICE inline EikonalUpdate EikonalTriangleUpdate(const double* xv,
                                                            const double* xa,
                                                            const double* xb,
                                                            const double* xc,
                                                            double ta,
                                                            double tb,
                                                            double tc) {
    // The seven candidates and their support masks
    const double values[7] = {EikonalTriangleInteriorValue(xv, xa, xb, xc, ta, tb, tc),
                              EikonalSegmentValue(xv, xa, xb, ta, tb),
                              EikonalSegmentValue(xv, xb, xc, tb, tc),
                              EikonalSegmentValue(xv, xa, xc, ta, tc),
                              EikonalVertexValue(xv, xa, ta),
                              EikonalVertexValue(xv, xb, tb),
                              EikonalVertexValue(xv, xc, tc)};
    const int masks[7] = {7, 3, 6, 5, 1, 2, 4};
    // Smallest value, then the union of the masks of every candidate attaining it
    EikonalUpdate out;
    out.value = EikonalInfinity();
    out.support = 0;
    for (int k = 0; k < 7; ++k) {
        if (values[k] < out.value) {
            out.value = values[k];
        }
    }
    if (out.value < EikonalInfinity()) {
        for (int k = 0; k < 7; ++k) {
            if (values[k] == out.value) {
                out.support |= masks[k];
            }
        }
    }
    return out;
}

} // namespace exaconstit::microstructure
