// Shape energy of make_lattice_meshes.py and its gradient, parallel over elements (OpenMP).
//
// For every element corner (c; a, b, d) the corner matrix M has the columns x_a - x_c, x_b - x_c
// and x_d - x_c (times W^-1 for tets, so a regular tet scores 1). With det = det(M) regularized
// as dreg = (det + sqrt(det^2 + delta^2)) / 2, the corner measure is
//     mu = |M|_F^2 / (3 dreg^(2/3))
// (1 for a cube corner or a regular tet, larger for worse shapes), and the energy is the sum of
// mu^power over all corners. The gradient follows from
//     d mu / dM = 2 M / (3 dreg^(2/3)) - (2/3) (mu / dreg) (d dreg / d det) cof(M),
// d dreg / d det = (1 + det / sqrt(det^2 + delta^2)) / 2, pulled back through W^-1 and the
// edge vectors to the nodes. Each thread adds into its own gradient buffer, and the buffers are
// summed in thread order, so the result does not depend on scheduling.
//
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC lattice_energy.cpp -o liblattice_energy.so
#include <cmath>
#include <omp.h>
#include <vector>

extern "C" {

/**
 * Sum over elements and corners of mu^power, and its gradient with respect to the node
 * coordinates (added into grad, size 3 * num_nodes, which the caller zeroes).
 * @param num_elems  number of elements; nodes per element `nodes_per_elem` (4 or 8)
 * @param elems      element node indices, num_elems * nodes_per_elem
 * @param num_corners corners per element; corners: num_corners * 4 local indices (c, a, b, d)
 * @param x          node coordinates, 3 per node
 * @param winv       3 x 3 row-major matrix applied as M W^-1 (identity for hexes)
 * @param delta      determinant regularization (length^3)
 * @param power      exponent of mu
 * @return the energy sum (not divided by the number of corners)
 */
double lattice_energy(int num_elems,
                      int nodes_per_elem,
                      const long* elems,
                      int num_corners,
                      const int* corners,
                      int num_nodes,
                      const double* x,
                      const double* winv,
                      double delta,
                      double power,
                      double* grad) {
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> buffers(num_threads);
    std::vector<double> partial(num_threads, 0.0);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::vector<double>& g = buffers[tid];
        g.assign(3 * static_cast<std::size_t>(num_nodes), 0.0);
        double sum = 0.0;
#pragma omp for schedule(static)
        for (int e = 0; e < num_elems; ++e) {
            const long* en = elems + static_cast<std::size_t>(e) * nodes_per_elem;
            for (int k = 0; k < num_corners; ++k) {
                const int* cn = corners + 4 * k;
                const long nc = en[cn[0]];
                const long nodes[3] = {en[cn[1]], en[cn[2]], en[cn[3]]};
                // edge matrix Ed: column j = x_{nodes[j]} - x_c
                double ed[3][3];
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        ed[i][j] = x[3 * nodes[j] + i] - x[3 * nc + i];
                    }
                }
                // M = Ed W^-1
                double m[3][3];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        m[i][j] = ed[i][0] * winv[0 * 3 + j] + ed[i][1] * winv[1 * 3 + j] +
                                  ed[i][2] * winv[2 * 3 + j];
                    }
                }
                // cofactor matrix and determinant
                double cof[3][3];
                cof[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
                cof[0][1] = m[1][2] * m[2][0] - m[1][0] * m[2][2];
                cof[0][2] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
                cof[1][0] = m[0][2] * m[2][1] - m[0][1] * m[2][2];
                cof[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
                cof[1][2] = m[0][1] * m[2][0] - m[0][0] * m[2][1];
                cof[2][0] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
                cof[2][1] = m[0][2] * m[1][0] - m[0][0] * m[1][2];
                cof[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
                const double det = m[0][0] * cof[0][0] + m[0][1] * cof[0][1] + m[0][2] * cof[0][2];
                const double root = std::sqrt(det * det + delta * delta);
                const double dreg = 0.5 * (det + root);
                double frob = 0.0;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        frob += m[i][j] * m[i][j];
                    }
                }
                const double d23 = std::cbrt(dreg * dreg);
                const double mu = frob / (3.0 * d23);
                const double mup = (power == 1.0) ? mu : std::pow(mu, power);
                sum += mup;
                // d(mu^p)/dM
                const double dmu = (power == 1.0) ? 1.0 : power * mup / mu;
                const double ddreg = (root > 0.0) ? 0.5 * (1.0 + det / root) : 0.5;
                const double a = dmu * 2.0 / (3.0 * d23);
                const double b = dmu * (2.0 / 3.0) * (mu / dreg) * ddreg;
                double gm[3][3];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        gm[i][j] = a * m[i][j] - b * cof[i][j];
                    }
                }
                // d/dEd = dM W^-T
                double ge[3][3];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        ge[i][j] = gm[i][0] * winv[j * 3 + 0] + gm[i][1] * winv[j * 3 + 1] +
                                   gm[i][2] * winv[j * 3 + 2];
                    }
                }
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        g[3 * nodes[j] + i] += ge[i][j];
                        g[3 * nc + i] -= ge[i][j];
                    }
                }
            }
        }
        partial[tid] = sum;
    }
    double total = 0.0;
    for (int t = 0; t < num_threads; ++t) {
        total += partial[t];
        const std::vector<double>& g = buffers[t];
        for (std::size_t i = 0; i < g.size(); ++i) {
            grad[i] += g[i];
        }
    }
    return total;
}

} // extern "C"
