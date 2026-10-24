#pragma once

#include "microstructure/labeling/eikonal_local_solver.hpp"

#include "mfem.hpp"

/**
 * @file fim_scan.hpp
 * @brief The per-cell candidate scan of the Fast Iterative Method (labeling/geodesic_fim.hpp),
 * in a header of its own so other FIM variants can run the same arithmetic. Host and device.
 */

namespace exaconstit::microstructure {

/**
 * @brief Scans every candidate of node v inside one cell (the element itself for the corner field,
 * a lattice sub-cell for the nodal field, see FieldCellView) and returns the smallest candidate
 * value together with the cell corners supporting the candidates that attain it.
 *
 * The candidates (see the file comment) are, over the cell's other corners o_0 .. o_{n-1}:
 *   - every vertex: T(o_i) + |v - o_i|                          (7 for a hex, 3 for a tet)
 *   - pairs:   the open-segment minimizer (o_i, o_j)            (15 for a hex, 3 for a tet)
 *   - triples: the triangle-interior minimizer (o_i, o_j, o_k)  (12 for a hex, 1 for a tet)
 * For a hex (far != nullptr) only the pairs and triples lying on a common face of the cell that
 * does not contain v (a "far face": every corner differs from v along the same axis) are scanned:
 * the 9 far edges and 6 far-face diagonals, and the 4 triangles of each of the 3 far faces (both
 * diagonal splits). Other cells scan every pair and triple. Their minimum is the minimum over the
 * scanned triangles of the full-face update, since every segment and vertex of such a triangle is
 * a candidate of its own. Each candidate function
 * sorts its vertices by coordinates, so the values are the same bits whatever the element's vertex
 * order (see eikonal_local_solver.hpp).
 *
 * Single pass: a strictly smaller value replaces the running best and its mask, an exactly equal
 * value adds its vertices to the mask. Each candidate value is computed exactly once, here, and
 * both the distance update and the label pass use the stored result (CandidateSweep in
 * geodesic_fim.cpp).
 *
 * Host and device (called inside the mfem::forall sweep).
 *
 * @param xv     coordinates of v (3 doubles).
 * @param xo     coordinates of the other corners of the cell (n pointers to 3 doubles).
 * @param to     T of the other corners (n entries, +infinity = not reached).
 * @param pos    neighbor bit of each of them (FieldCellView, n entries): the position 0 .. 7 in the
 *               element's vertex list for the corner field, the lattice offset code 0 .. 26 for
 *               the nodal field.
 * @param n      number of other corners, 0 .. 7.
 * @param far    hex cells: per other corner, bit d set when it differs from v along axis d (pairs
 *               and triples with no common bit are skipped); nullptr = scan every pair and triple.
 * @param mask   out: bit pos[i] set when o_i belongs to a candidate whose value equals the returned
 *               minimum (exact comparison); 0 when the minimum is +infinity.
 * @return the smallest candidate value (+infinity if no other vertex is reached).
 */
MFEM_HOST_DEVICE inline double ScanCell(const double* xv,
                                        const double* const* xo,
                                        const double* to,
                                        const int* pos,
                                        const int* far,
                                        int n,
                                        int* mask) {
    const double inf = EikonalInfinity();
    double best = inf;
    int m = 0;
    // Through a single vertex
    for (int i = 0; i < n; ++i) {
        const double value = EikonalVertexValue(xv, xo[i], to[i]);
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
            if (far != nullptr && (far[i] & far[j]) == 0) {
                continue;
            }
            const double value = EikonalSegmentValue(xv, xo[i], xo[j], to[i], to[j]);
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
                if (far != nullptr && (far[i] & far[j] & far[k]) == 0) {
                    continue;
                }
                const double value =
                    EikonalTriangleInteriorValue(xv, xo[i], xo[j], xo[k], to[i], to[j], to[k]);
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
 * @brief Far-face masks of the other corners of a hex cell: far[i] bit d is set when corner i
 * differs from v along axis d, so a pair or triple of corners lies on a face not containing v
 * exactly when their masks share a bit (ScanCell). Returns false (masks not filled) for a cell
 * that is not a hex (n != 7).
 *
 * Corner field (curved == false): bits[i] is the corner's position in the element's vertex list,
 * in MFEM's hex order (0,0,0) (1,0,0) (1,1,0) (0,1,0) and the same at z = 1; v's position is the
 * missing one (28 - the sum of the others). Nodal field: bits[i] is the lattice offset code
 * (di + 1) + 3 (dj + 1) + 9 (dk + 1) of the corner relative to v (FieldCellView), so the corner
 * differs from v along every axis with a nonzero offset. Host and device.
 */
MFEM_HOST_DEVICE inline bool CellFarMasks(bool curved, const int* bits, int n, int* far) {
    if (n != 7) {
        return false;
    }
    if (curved) {
        for (int i = 0; i < 7; ++i) {
            const int c = bits[i];
            far[i] = ((c % 3) != 1 ? 1 : 0) | (((c / 3) % 3) != 1 ? 2 : 0) |
                     ((c / 9) != 1 ? 4 : 0);
        }
        return true;
    }
    const int hex_bits[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    int sum = 0;
    for (int i = 0; i < 7; ++i) {
        sum += bits[i];
    }
    const int vb = hex_bits[28 - sum];
    for (int i = 0; i < 7; ++i) {
        far[i] = hex_bits[bits[i]] ^ vb;
    }
    return true;
}

} // namespace exaconstit::microstructure
