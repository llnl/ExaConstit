#include "postprocessing/viz_point_map.hpp"

#include <limits>

VizPointMap::VizPointMap(VizFieldLocation location,
                         const mfem::expt::PartialQuadratureSpace& source_qspace,
                         const mfem::ParFiniteElementSpace& out_fes,
                         const mfem::Array<int>& elem_to_out)
    : m_location(location) {
    const int nelems = source_qspace.GetNE();
    MFEM_VERIFY(elem_to_out.Size() == nelems,
                "VizPointMap: element map size does not match the quadrature space");

    // Offsets of each local element's quadrature points inside the region's (compressed)
    // quadrature function data. Built on the host, so make sure the host copies are current.
    const mfem::Array<int>& qp_offsets = source_qspace.Offsets(
        mfem::QSpaceOffsetStorage::COMPRESSED);
    qp_offsets.HostRead();
    m_out_elem = elem_to_out;
    m_out_elem.HostReadWrite();

    // First pass: count output nodes so every array is sized once.
    m_elem_offsets.SetSize(nelems + 1);
    m_elem_offsets[0] = 0;
    for (int ie = 0; ie < nelems; ie++) {
        m_elem_offsets[ie + 1] = m_elem_offsets[ie] + out_fes.GetFE(m_out_elem[ie])->GetDof();
    }

    const int npoints = m_elem_offsets[nelems];
    m_source.SetSize(npoints);
    m_dest.SetSize(npoints);

    // Second pass: for every output node, find the closest quadrature point of the same element.
    // Node and quadrature point locations are both in the element's reference coordinates, which
    // keeps this valid for curved elements.
    mfem::Array<int> elem_dofs;
    for (int ie = 0; ie < nelems; ie++) {
        const int oe = m_out_elem[ie];
        const mfem::FiniteElement* fe = out_fes.GetFE(oe);
        const mfem::IntegrationRule& nodes = fe->GetNodes();
        const mfem::IntegrationRule& ir = source_qspace.GetIntRule(ie);
        out_fes.GetElementDofs(oe, elem_dofs);

        const int nqpts = qp_offsets[ie + 1] - qp_offsets[ie];
        MFEM_VERIFY(nqpts == ir.GetNPoints(),
                    "VizPointMap: quadrature offsets do not match the integration rule");

        for (int jn = 0; jn < nodes.GetNPoints(); jn++) {
            const mfem::IntegrationPoint& node = nodes.IntPoint(jn);
            int closest = 0;
            double closest_dist = std::numeric_limits<double>::max();
            for (int iq = 0; iq < nqpts; iq++) {
                const mfem::IntegrationPoint& qp = ir.IntPoint(iq);
                const double dx = qp.x - node.x;
                const double dy = qp.y - node.y;
                const double dz = qp.z - node.z;
                const double dist = dx * dx + dy * dy + dz * dz;
                // Strict comparison: ties go to the lowest quadrature point index, so the table
                // is the same on every rank and every run.
                if (dist < closest_dist) {
                    closest_dist = dist;
                    closest = iq;
                }
            }
            const int k = m_elem_offsets[ie] + jn;
            m_source[k] = qp_offsets[ie] + closest;
            // L2 node numbers are never negative (no orientation sign as for H(curl)/H(div)).
            m_dest[k] = elem_dofs[jn];
        }
    }
}
