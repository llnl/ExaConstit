#include "microstructure/seeding/min_distance_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace exaconstit::microstructure {

namespace {
/** @brief Largest number of cells per axis; cell coordinates must fit in 21 bits of the key. */
constexpr std::int64_t MAX_CELLS_PER_AXIS = std::int64_t(1) << 20;
} // namespace

/*
 * MinDistanceGrid constructor
 *
 * Chooses the cell layout per axis:
 *  - n = floor(L / d) cells of size L / n >= d, so a point closer than d to a query can only be in
 *    the query's cell or a direct neighbor (the 3x3x3 block). Using exactly L / n (instead of d)
 *    makes the cells tile one period exactly, which periodic wrapping needs.
 *  - n is clamped to [1, 2^20]: at least one cell, and few enough that a cell coordinate fits in
 *    the 21-bit fields of the hash key. With a clamped n the cells are larger than needed, which
 *    is still correct (just more points per cell).
 */
MinDistanceGrid::MinDistanceGrid(const std::array<double, 3>& lo,
                                 const std::array<double, 3>& hi,
                                 double min_distance,
                                 const std::array<bool, 3>& periodic)
    : m_lo(lo), m_periodic(periodic), m_min_distance(min_distance),
      m_min_distance_sq(min_distance * min_distance) {
    if (!(min_distance > 0.0)) {
        throw std::invalid_argument("MinDistanceGrid: the minimum distance must be positive");
    }
    for (int d = 0; d < 3; d++) {
        m_length[d] = hi[d] - lo[d];
        if (!(m_length[d] > 0.0)) {
            throw std::invalid_argument("MinDistanceGrid: the box must have positive extent");
        }
        // Number of cells along this axis: as many as fit with size >= d, within [1, 2^20]
        const double fit = std::floor(m_length[d] / min_distance);
        const double clamped = std::min(std::max(fit, 1.0), double(MAX_CELLS_PER_AXIS));
        m_num_cells[d] = static_cast<std::int64_t>(clamped);
        m_cell_size[d] = m_length[d] / static_cast<double>(m_num_cells[d]);
    }
}

/*
 * CellCoordinate: integer cell index of one coordinate.
 *  - periodic axis: shift into [0, L) first (floor-based modulo, so negative offsets wrap too);
 *  - non-periodic axis: clamp to the first/last cell, so a point slightly outside the box is
 *    filed in the boundary cell (it can still only be close to points in the neighboring cells).
 */
std::int64_t MinDistanceGrid::CellCoordinate(double p, int d) const {
    double offset = p - m_lo[d];
    if (m_periodic[d]) {
        offset -= m_length[d] * std::floor(offset / m_length[d]);
    }
    std::int64_t c = static_cast<std::int64_t>(std::floor(offset / m_cell_size[d]));
    c = std::min(std::max(c, std::int64_t(0)), m_num_cells[d] - 1);
    return c;
}

/*
 * CellKey: packs (i, j, k), each in [0, 2^21), into bits [0,21), [21,42), [42,63) of one 64-bit
 * integer, which is the hash-map key of the cell.
 */
std::uint64_t MinDistanceGrid::CellKey(std::int64_t i, std::int64_t j, std::int64_t k) {
    return static_cast<std::uint64_t>(i) | (static_cast<std::uint64_t>(j) << 21) |
           (static_cast<std::uint64_t>(k) << 42);
}

/*
 * Distance with the minimum-image convention: along a periodic axis the offset dx is moved to the
 * nearest periodic copy by subtracting L * round(dx / L), so |dx| <= L / 2.
 */
double MinDistanceGrid::Distance(const std::array<double, 3>& p,
                                 const std::array<double, 3>& q) const {
    double sum = 0.0;
    for (int d = 0; d < 3; d++) {
        double dx = p[d] - q[d];
        if (m_periodic[d]) {
            dx -= m_length[d] * std::round(dx / m_length[d]);
        }
        sum += dx * dx;
    }
    return std::sqrt(sum);
}

/*
 * AnyNeighborPoint
 *
 * Steps:
 *  1. Find the query's cell (ci, cj, ck).
 *  2. List the neighbor cell coordinates per axis: c-1, c, c+1, wrapped on periodic axes and
 *     dropped when outside [0, n) on non-periodic axes. With fewer than 3 cells on a periodic
 *     axis, c-1 and c+1 can wrap onto the same cell; duplicates are removed so no cell is
 *     searched twice (that would only cost time, but it is easy to avoid).
 *  3. Visit the points filed in each of the (up to) 27 cells, stopping at the first one for
 *     which visit returns true.
 */
template <typename Visit>
bool MinDistanceGrid::AnyNeighborPoint(const std::array<double, 3>& p, Visit&& visit) const {
    if (m_points.empty()) {
        return false;
    }

    // Steps 1-2: neighbor cell coordinates per axis
    std::array<std::array<std::int64_t, 3>, 3> neighbors;
    std::array<int, 3> num_neighbors = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
        const std::int64_t c = CellCoordinate(p[d], d);
        for (std::int64_t offset = -1; offset <= 1; offset++) {
            std::int64_t n = c + offset;
            if (m_periodic[d]) {
                n = (n + m_num_cells[d]) % m_num_cells[d];
            } else if (n < 0 || n >= m_num_cells[d]) {
                continue;
            }
            bool duplicate = false;
            for (int k = 0; k < num_neighbors[d]; k++) {
                duplicate = duplicate || (neighbors[d][k] == n);
            }
            if (!duplicate) {
                neighbors[d][num_neighbors[d]++] = n;
            }
        }
    }

    // Step 3: the points filed in each neighbor cell
    for (int a = 0; a < num_neighbors[0]; a++) {
        for (int b = 0; b < num_neighbors[1]; b++) {
            for (int c = 0; c < num_neighbors[2]; c++) {
                const auto it = m_cells.find(CellKey(neighbors[0][a], neighbors[1][b],
                                                     neighbors[2][c]));
                if (it == m_cells.end()) {
                    continue;
                }
                for (const std::size_t idx : it->second) {
                    if (visit(idx)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

/*
 * SquaredDistanceTo: squared minimum-image distance (squared, so no sqrt per pair is needed for
 * the comparisons below).
 */
double MinDistanceGrid::SquaredDistanceTo(const std::array<double, 3>& p, std::size_t idx) const {
    double sum = 0.0;
    for (int d = 0; d < 3; d++) {
        double dx = p[d] - m_points[idx][d];
        if (m_periodic[d]) {
            dx -= m_length[d] * std::round(dx / m_length[d]);
        }
        sum += dx * dx;
    }
    return sum;
}

/*
 * HasNeighborCloserThan: any point of the 27 neighbor cells with squared minimum-image distance
 * < d^2 ("< d^2" means a pair at exactly d is allowed).
 */
bool MinDistanceGrid::HasNeighborCloserThan(const std::array<double, 3>& p) const {
    return AnyNeighborPoint(
        p, [&](std::size_t idx) { return SquaredDistanceTo(p, idx) < m_min_distance_sq; });
}

/*
 * OverlapsAny: like HasNeighborCloserThan, with the pair distance radius + r_q of the two
 * exclusion spheres instead of d. Correct because radius + r_q <= d = the cell size, so an
 * overlapping point lies in a neighbor cell.
 */
bool MinDistanceGrid::OverlapsAny(const std::array<double, 3>& p, double radius) const {
    return AnyNeighborPoint(p, [&](std::size_t idx) {
        const double reach = radius + m_radii[idx];
        return SquaredDistanceTo(p, idx) < reach * reach;
    });
}

/* Insert: append the point (exclusion radius d / 2) and file its index under its cell. */
void MinDistanceGrid::Insert(const std::array<double, 3>& p) {
    Insert(p, 0.5 * m_min_distance);
}

/* Insert with radius: append the point and its radius and file its index under its cell. */
void MinDistanceGrid::Insert(const std::array<double, 3>& p, double radius) {
    const std::size_t idx = m_points.size();
    m_points.push_back(p);
    m_radii.push_back(radius);
    const std::uint64_t key =
        CellKey(CellCoordinate(p[0], 0), CellCoordinate(p[1], 1), CellCoordinate(p[2], 2));
    m_cells[key].push_back(idx);
}

} // namespace exaconstit::microstructure
