#pragma once

#include "mfem_expt/partial_qspace.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"

#include <string>

/**
 * @brief Lookup table telling each visualization value where its material data comes from
 *
 * Material quantities (stress, state variables, ...) are stored at the quadrature points of
 * each element of a material region. Visualization files need them on a discontinuous (L2)
 * finite element field instead. VizPointMap is the single table that connects the two, and it
 * is what lets the element-average ("zone") output and the per-quadrature-point output share
 * every projection kernel:
 *
 *   output_field(:, dest[k]) = f( material_data(:, source[k]) )     for k = 0 .. points-1
 *
 * - `source[k]` is a point index into a quadrature function of the region (multiply by the
 *   number of components per point to get the start of that point's data).
 * - `dest[k]` is a node (scalar degree of freedom) of the output L2 field. All output spaces use
 *   byVDIM ordering, so component c of that node lives at dest[k] * vdim + c.
 *
 * How the two output modes fill the table:
 * - ZONE: the source is the element-average quadrature function ("<name>_avg", one point per
 *   element) and the output space is L2 order 0 (one node per element). Every element gets
 *   exactly one entry: its average goes to its single node.
 * - QUADRATURE: the source is the end-of-step quadrature function ("<name>_end") and the output
 *   space is L2 of the mesh order using Gauss-Lobatto nodes, i.e. nodes sitting at the same
 *   reference locations as the H1 displacement nodes. Each node takes the value of the
 *   quadrature point closest to it in the element's reference coordinates. With the default
 *   integration order (2p + 1) on hexahedra this is a one-to-one match: p + 1 Gauss points and
 *   p + 1 Gauss-Lobatto nodes per direction, each node owning the Gauss point in its corner.
 *
 * Because each node is a straight copy of one quadrature point value (never a blend of several),
 * quantities that must stay bounded keep their bounds in the output: plastic strain stays
 * non-negative and quaternions can be re-normalized node by node. This is the problem with
 * letting the visualization tool interpolate quadrature data onto the H1 nodes itself.
 *
 * Both modes use the same construction: for each node of the output element, pick the closest
 * quadrature point of the source space. In zone mode the source space has a single point per
 * element, so the "closest" point is that point.
 *
 * Distances are measured in reference coordinates, so the table is valid for curved
 * (high-order) elements without looking at vertex positions.
 *
 * The table is built once on the host at setup; the projection kernels then read it on the
 * device through the mfem::Reshape views returned by ReadSource(), ReadDest(), ... Inside a kernel
 * the data on both sides is viewed the same way, e.g. for a stress copy:
 *
 *   const auto src = points.ReadSource();
 *   const auto dst = points.ReadDest();
 *   const auto stress = mfem::Reshape(stress_qf->Read(), 6, num_source_points);
 *   auto out = mfem::Reshape(stress_gf->ReadWrite(), 6, num_output_nodes);
 *   mfem::forall(points.GetNumPoints(), [=] MFEM_HOST_DEVICE(int k) {
 *       for (int i = 0; i < 6; i++) { out(i, dst(k)) = stress(i, src(k)); }
 *   });
 *
 * @ingroup ExaConstit_projections
 */
class VizPointMap {
public:
    VizPointMap() = default;

    /**
     * @brief Build the table for one material region
     *
     * @param location    Zone or quadrature output (decides which quadrature function to read)
     * @param source_qspace Quadrature space of the source data for this region. In zone mode this
     *                    is the one-point-per-element space of the "_avg" functions, in quadrature
     *                    mode the full integration space of the "_end" functions.
     * @param out_fes     Scalar (vdim = 1) L2 space the region is visualized on. Only its node
     *                    numbering and node locations are used, so the same table serves every
     *                    vector size built on the same mesh and collection.
     * @param elem_to_out Output mesh element for each local element of the region's partial
     *                    quadrature space (identity-like for one region, submesh element numbers
     *                    for several regions).
     */
    VizPointMap(VizFieldLocation location,
                const mfem::expt::PartialQuadratureSpace& source_qspace,
                const mfem::ParFiniteElementSpace& out_fes,
                const mfem::Array<int>& elem_to_out);

    /// Number of (source, dest) pairs owned by this rank for the region
    int GetNumPoints() const {
        return m_source.Size();
    }

    /// Number of local elements of the region on this rank
    int GetNumElements() const {
        return m_out_elem.Size();
    }

    /// Read-only view of an index table, usable inside mfem::forall kernels
    using IndexView = mfem::DeviceTensor<1, const int>;

    /// View: source quadrature point of each entry, source(k)
    IndexView ReadSource() const {
        return mfem::Reshape(m_source.Read(), m_source.Size());
    }

    /// View: output node of each entry, dest(k)
    IndexView ReadDest() const {
        return mfem::Reshape(m_dest.Read(), m_dest.Size());
    }

    /// View: output mesh element of each local element, out_elem(ie)
    IndexView ReadOutElement() const {
        return mfem::Reshape(m_out_elem.Read(), m_out_elem.Size());
    }

    /// View: entries of local element ie are [offsets(ie), offsets(ie + 1))
    IndexView ReadElementOffsets() const {
        return mfem::Reshape(m_elem_offsets.Read(), m_elem_offsets.Size());
    }

    /// Zone (element averages) or quadrature (nearest quadrature point) output
    VizFieldLocation GetLocation() const {
        return m_location;
    }

    /// The tables themselves, for checks on the host
    const mfem::Array<int>& GetSource() const {
        return m_source;
    }
    const mfem::Array<int>& GetDest() const {
        return m_dest;
    }
    const mfem::Array<int>& GetElementOffsets() const {
        return m_elem_offsets;
    }

    /**
     * @brief Name of the quadrature function holding the source data for a quantity
     *
     * @param base Quantity name without suffix, e.g. "cauchy_stress" or "state_var"
     * @return "<base>_avg" in zone mode, "<base>_end" in quadrature mode
     */
    std::string SourceName(const std::string& base) const {
        return base + ((m_location == VizFieldLocation::ZONE) ? "_avg" : "_end");
    }

private:
    VizFieldLocation m_location = VizFieldLocation::ZONE;
    /// Source quadrature point for each entry
    mfem::Array<int> m_source;
    /// Output node for each entry
    mfem::Array<int> m_dest;
    /// Output mesh element for each local element
    mfem::Array<int> m_out_elem;
    /// Start of each local element's entries (size = elements + 1)
    mfem::Array<int> m_elem_offsets;
};
