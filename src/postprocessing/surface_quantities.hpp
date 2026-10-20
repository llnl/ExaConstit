#pragma once

#include "mfem_expt/partial_qfunc.hpp"
#include "sim_state/simulation_state.hpp"

#include "mfem.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class PostProcessingFileManager;

/**
 * @brief Boundary faces of one or more mesh boundary attributes, with the tables needed to
 * evaluate element data at the face quadrature points
 *
 * @details Each selected face is integrated in the parametrization of its adjacent element
 * (elem1), so the orientation of the boundary element or a neighboring element never enters.
 * For every (element geometry, local face) pair that occurs, the following are precomputed on
 * the host once and then used by device kernels:
 * - the H1 shape values and reference gradients of elem1 at the face quadrature points, used
 *   for coordinates, normals (Nanson's formula) and nodal fields such as the displacement;
 * - a least-squares extrapolation matrix from elem1's volume quadrature points to the face
 *   quadrature points, using the degree-p L2 basis of the element (Q_p Gauss-Legendre for
 *   hexes, where it is exact interpolation, and P_p for tets);
 * - the outward reference normal of the local face and the reference face weights.
 *
 * MFEM's face restriction, face quadrature interpolator and face geometric factors are not
 * used: they only support tensor-product elements with closed bases, so they cannot handle
 * tets or the Gauss-Legendre layout of ExaConstit's quadrature data.
 *
 * Nothing in this class is specific to post-processing, so it can also serve future surface
 * terms such as traction boundary conditions.
 */
class BoundarySurface {
public:
    /**
     * @brief Select the local boundary faces of the given attributes and build the tables
     *
     * @param mesh Parallel mesh
     * @param fes H1 vector space of the mesh coordinates and displacement
     * @param boundary_ids Mesh boundary attributes; surface index s refers to boundary_ids[s]
     * @param order Polynomial order p of the mesh, which sets the 2p+1 volume and face rules
     */
    BoundarySurface(std::shared_ptr<mfem::ParMesh> mesh,
                    std::shared_ptr<mfem::ParFiniteElementSpace> fes,
                    const std::vector<int>& boundary_ids,
                    int order);

    /// Number of surfaces (boundary attributes)
    int GetNumSurfaces() const {
        return m_num_surfaces;
    }

    /// Number of selected faces on this rank
    int GetNumFaces() const {
        return m_face_elem.Size();
    }

    /// Number of face quadrature points over all selected faces on this rank
    int GetNumFacePoints() const {
        return m_num_face_points;
    }

    /// Surface index of each selected face
    const mfem::Array<int>& GetFaceSurface() const {
        return m_face_surface;
    }

    /// Offset of each face's quadrature points in face-point arrays (size GetNumFaces() + 1)
    const mfem::Array<int>& GetFacePointOffsets() const {
        return m_face_qoff;
    }

    /**
     * @brief Coordinates and outward area vectors at the face quadrature points
     *
     * @param coords L-vector of nodal coordinates on the H1 space given at construction
     * @param x Output, 3 values per face point
     * @param nda Output, 3 values per face point: n da = cof(J) n_ref dA_ref
     * @param da Output, 1 value per face point: the area element, |cof(J) n_ref| dA_ref
     *
     * Both nda and da carry the sign of the quadrature weight. Some rules have negative
     * weights (e.g. MFEM's order 3 triangle rule), so da must not be computed as |nda|.
     */
    void ComputeGeometry(const mfem::Vector& coords,
                         mfem::Vector& x,
                         mfem::Vector& nda,
                         mfem::Vector& da) const;

    /**
     * @brief Interpolate a vector H1 field to the face quadrature points
     *
     * @param field L-vector on the H1 space given at construction
     * @param values Output, 3 values per face point
     */
    void InterpolateH1(const mfem::Vector& field, mfem::Vector& values) const;

    /**
     * @brief Extrapolate quadrature data of each face's elem1 to the face quadrature points
     *
     * @param qf Quadrature function on the 2p+1 volume rule, global or region-partial. Faces
     *        whose elem1 is not in a partial function's space get zeros.
     * @param values Output, qf.GetVDim() values per face point
     */
    void ExtrapolateQF(const mfem::expt::PartialQuadratureFunction& qf, mfem::Vector& values) const;

private:
    /// Build (or find) the tables for an element geometry and local face; returns the table id
    int GetTable(mfem::Geometry::Type elem_geom, mfem::Geometry::Type face_geom, int local_face);

    std::shared_ptr<mfem::ParMesh> m_mesh;
    std::shared_ptr<mfem::ParFiniteElementSpace> m_fes;
    int m_order;
    int m_num_surfaces;
    int m_num_face_points = 0;

    /// L2 Gauss-Legendre collection whose basis spans the extrapolated polynomial space
    std::unique_ptr<mfem::L2_FECollection> m_l2_fec;
    /// (element geometry, local face) -> table id
    std::map<std::pair<int, int>, int> m_table_ids;

    // Per selected face
    mfem::Array<int> m_face_surface;
    mfem::Array<int> m_face_elem;
    mfem::Array<int> m_face_table;
    mfem::Array<int> m_face_qoff;
    mfem::Array<int> m_face_vdof_off;
    mfem::Array<int> m_vdofs;

    // Per table: sizes and offsets into the flat table vectors
    mfem::Array<int> m_tab_nfq;
    mfem::Array<int> m_tab_nvq;
    mfem::Array<int> m_tab_ndof;
    mfem::Array<int> m_tab_off_e;
    mfem::Array<int> m_tab_off_n;
    mfem::Array<int> m_tab_off_w;
    mfem::Array<double> m_tab_e;    ///< extrapolation, [nfq x nvq] row-major per table
    mfem::Array<double> m_tab_n;    ///< H1 shape values, [nfq x ndof] per table
    mfem::Array<double> m_tab_dn;   ///< H1 reference gradients, [nfq x ndof x 3] per table
    mfem::Array<double> m_tab_w;    ///< face weight times reference surface Jacobian, [nfq]
    mfem::Array<double> m_tab_nref; ///< outward unit reference normal, [3] per table
};

/**
 * @brief Per-surface force, displacement, area and normal output
 *
 * @details For each requested boundary attribute, at every output step:
 * - area = integral of da over the current surface;
 * - displacement = integral of u dA0 over the reference surface divided by its area A0;
 * - force = integral of the Cauchy traction sigma n da over the current surface, with sigma
 *   extrapolated from the adjacent element's quadrature points;
 * - normal = (integral of n da) / |integral of n da|, the normalized vector area. By Stokes'
 *   theorem it only depends on the boundary loop of the surface, so roughening of a free
 *   surface's interior does not change it;
 * - optional moment = integral of (x - x_c) x (sigma n da) about the current area centroid x_c.
 *
 * Results are written to one file per attribute through PostProcessingFileManager.
 */
class SurfaceQuantities {
public:
    SurfaceQuantities(std::shared_ptr<SimulationState> sim_state, const ExaOptions& options);

    /**
     * @brief Compute the surface quantities for the converged state and write them
     *
     * @param time Current simulation time
     * @param file_manager File manager providing paths and formatting
     *
     * Collective over MPI_COMM_WORLD.
     */
    void Update(double time, PostProcessingFileManager& file_manager);

    /// Whether output should happen at this step
    bool ShouldOutputAtStep(int step) const {
        return (step % m_output_frequency == 0);
    }

private:
    std::shared_ptr<SimulationState> m_sim_state;
    std::unique_ptr<BoundarySurface> m_surface;
    std::vector<int> m_boundary_ids;
    bool m_moments;
    int m_output_frequency;

    /// Face-point weights dA0 of the reference surface
    mfem::Vector m_dA0;
    /// Per-surface reference area
    std::vector<double> m_ref_area;
    /// Per-surface comment lines written at the top of each file
    std::vector<std::string> m_preambles;
    /// Per-surface flag set when the surface is closed and has no meaningful normal
    std::vector<bool> m_zero_normal;

    // Work vectors, sized to the number of face points or faces
    mfem::Vector m_x;
    mfem::Vector m_nda;
    mfem::Vector m_da;
    mfem::Vector m_disp;
    mfem::Vector m_stress;
    mfem::Vector m_face_sums;
};
