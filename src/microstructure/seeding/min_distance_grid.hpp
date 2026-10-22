#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

/**
 * @file min_distance_grid.hpp
 * @brief Hash grid that answers "is any accepted seed closer than d to this point?" for random
 * sequential adsorption (RSA).
 *
 * **RSA in plain words.** Random sequential adsorption places seeds one at a time: a candidate
 * point is kept only if it is at least a distance d from every seed kept so far, otherwise it is
 * thrown away and the next candidate is tried. The result looks like a random packing of hard
 * spheres of diameter d, which gives grains of more uniform size than purely random (Poisson)
 * seeds. Because candidates are tested in a fixed global order (the candidate index), the outcome
 * is the same on every rank and for every rank count.
 *
 * **Hash grid in plain words.** Comparing a candidate with all N accepted seeds costs O(N) per
 * candidate. Instead, space is cut into cubic cells with edge length >= d and every accepted seed
 * is filed under the cell that contains it. A seed closer than d to the candidate can then only
 * sit in the candidate's cell or one of the 26 cells around it, so only those 27 cells are
 * searched. Only the occupied cells are stored, in a hash map keyed by the integer cell
 * coordinates (that is what "hash grid" means), so memory is O(N) however large the box is.
 *
 * **Periodicity.** Along a periodic axis of period L, the distance between two points is measured
 * to the nearest periodic copy ("minimum image" convention): dx is replaced by
 * dx - L * round(dx / L), so two seeds near opposite faces of the box count as close. The cells
 * then wrap around: cell n-1 is a neighbor of cell 0.
 *
 * **Polydisperse RSA.** With seeds of different sizes each seed i has its own exclusion radius
 * r_i, and two seeds may not be closer than r_i + r_j (their exclusion spheres must not overlap).
 * OverlapsAny / Insert(p, radius) implement that test; the cell size is then twice the largest
 * radius, so overlapping seeds still sit in neighboring cells.
 *
 * Host only (std:: containers); RSA is sequential by nature and seed counts are small compared
 * with element counts.
 */

namespace exaconstit::microstructure {

/**
 * @brief Hash grid of accepted seed positions for the RSA minimum-distance test.
 *
 * Usage:
 * @code
 *   MinDistanceGrid grid(lo, hi, d, periodic);
 *   if (!grid.HasNeighborCloserThan(p)) { grid.Insert(p); }
 * @endcode
 * Host only, not collective. Every rank builds an identical grid from the identical (replicated)
 * candidate stream, so every rank makes the same accept/reject decisions.
 */
class MinDistanceGrid {
public:
    /**
     * @brief Creates an empty grid over the box [lo, hi].
     *
     * @param lo,hi    corners of the box that contains every point that will be inserted or
     *                 queried (mesh length units); along a periodic axis [lo, hi] is also one
     *                 period, L = hi - lo. Points outside the box are allowed along non-periodic
     *                 axes (they are filed in the nearest boundary cell) and are wrapped back
     *                 into the period along periodic axes.
     * @param min_distance d > 0: points closer than d (strictly) are reported as too close.
     * @param periodic  periodic flag per axis x, y, z.
     * @throws std::invalid_argument if d <= 0 or hi <= lo along some axis.
     */
    MinDistanceGrid(const std::array<double, 3>& lo,
                    const std::array<double, 3>& hi,
                    double min_distance,
                    const std::array<bool, 3>& periodic);

    /**
     * @brief True if some inserted point is at distance < d from p (minimum image along
     * periodic axes). Host only.
     * @param p query point x, y, z (mesh length units).
     */
    bool HasNeighborCloserThan(const std::array<double, 3>& p) const;

    /** @brief Files p under its cell. Does not check the distance (call the query first). */
    void Insert(const std::array<double, 3>& p);

    /**
     * @brief Polydisperse test: true if some inserted point q, inserted with radius r_q, is at
     * distance < radius + r_q from p (minimum image along periodic axes), i.e. the spheres of
     * radius `radius` around p and r_q around q overlap. Host only.
     *
     * Precondition: radius and every inserted radius are <= d / 2, with d the constructor's
     * min_distance (the cell size), so an overlapping point can only be in the 27 neighbor
     * cells. Points inserted with Insert(p) (no radius) count with radius d / 2.
     */
    bool OverlapsAny(const std::array<double, 3>& p, double radius) const;

    /** @brief Files p with its exclusion radius (<= d / 2) for OverlapsAny. */
    void Insert(const std::array<double, 3>& p, double radius);

    /** @brief Number of inserted points. */
    std::size_t Size() const {
        return m_points.size();
    }

    /** @brief The minimum distance d this grid enforces. */
    double MinDistance() const {
        return m_min_distance;
    }

    /**
     * @brief Distance from p to q, with the minimum-image convention along periodic axes.
     * Public so tests can measure pair distances exactly the way the grid does.
     */
    double Distance(const std::array<double, 3>& p, const std::array<double, 3>& q) const;

private:
    /** @brief Integer cell coordinate of p along axis d, in [0, m_num_cells[d]). */
    std::int64_t CellCoordinate(double p, int d) const;

    /** @brief One 64-bit hash-map key from three cell coordinates (21 bits each). */
    static std::uint64_t CellKey(std::int64_t i, std::int64_t j, std::int64_t k);

    std::array<double, 3> m_lo;           ///< box lower corner
    std::array<double, 3> m_length;       ///< box edge lengths (= period on periodic axes)
    std::array<double, 3> m_cell_size;    ///< cell edge length per axis, >= d
    std::array<std::int64_t, 3> m_num_cells; ///< number of cells per axis, 1 .. 2^20
    std::array<bool, 3> m_periodic;       ///< periodic flag per axis
    double m_min_distance;                ///< d
    double m_min_distance_sq;             ///< d^2
    /**
     * @brief Calls visit(index) for every inserted point filed in p's cell or one of its
     * neighbor cells (wrapped on periodic axes, each cell once); stops and returns true as soon
     * as visit returns true.
     */
    template <typename Visit>
    bool AnyNeighborPoint(const std::array<double, 3>& p, Visit&& visit) const;

    /** @brief Squared minimum-image distance between p and inserted point idx. */
    double SquaredDistanceTo(const std::array<double, 3>& p, std::size_t idx) const;

    /** @brief Inserted points, in insertion order. */
    std::vector<std::array<double, 3>> m_points;
    /** @brief Exclusion radius of each inserted point (d / 2 for Insert(p) without a radius). */
    std::vector<double> m_radii;
    /** @brief Occupied cell -> indices into m_points. */
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> m_cells;
};

} // namespace exaconstit::microstructure
