#pragma once

#include "mfem.hpp"

/**
 * @file element_mask.hpp
 * @brief Which local elements take part in microstructure generation (the "active" material).
 *
 * A mesh can hold parts that must not receive generated grains, for example a build plate that
 * becomes its own material region ([Mesh] passthrough_attributes). Instead of cutting those parts
 * out into a separate mesh (a copy of the mesh, which the mechanics hook must avoid on large
 * production meshes), every generation stage receives an element mask and simply skips the
 * elements that are not active:
 *  - seeding only places seeds inside active elements (point-in-material tests, element-weighted
 *    sampling, lattices, the ROI filler, Johnson-Mehl coverage);
 *  - labeling only labels active elements (centroid, vote, voxel, connected and geodesic modes);
 *  - a face between an active and an inactive element is treated like a free surface by the
 *    stages that follow material connectivity (fragment cleanup, connected mode, geodesic paths),
 *    but not by the surface clearance of seeding, which measures only true boundary faces of the
 *    mesh (the interface is bonded);
 *  - cell measures, Lloyd relaxation and the volume fit integrate over active elements only.
 *
 * Convention used by every function that accepts a mask: a `const ElementMask*` parameter, or a
 * `const MaterialContext*` (seeding/material_point_locator.hpp: the mask plus a point locator
 * built once over the active elements) for the stages that locate points, where nullptr (the
 * default) means "every element is active", which reproduces the behavior without a mask bit for
 * bit; an empty mask means the same. Inside device kernels the mask is passed as a raw
 * `const int*` from `active.Read()` (nullptr = every element active); use ElementMaskData() to
 * obtain it.
 *
 * Placeholders: labelers give inactive elements seed index -1 and grain id 0 (voxel features -1);
 * AttributeDomain::ToParent then writes the passthrough grain ids there.
 *
 * The mask describes this rank's local elements only. Decisions that must agree across ranks
 * (counts, volumes) are reduced by the caller, as for unmasked runs.
 */

namespace exaconstit::microstructure {

/**
 * @brief Active-element flags of the local elements of one mesh.
 *
 * `active[e]` is 1 when local element e takes part in generation and 0 when it does not
 * (e.g. a passthrough part). An empty array means every element is active. The size, when not
 * empty, must equal the number of local elements (mfem::Mesh::GetNE()) of the mesh it is used
 * with. Host data, readable on the device through active.Read().
 */
struct ElementMask {
    /** @brief Per local element: 1 = active, 0 = inactive. Empty = every element active. */
    mfem::Array<int> active;

    /** @brief True when local element `e` is active (host only). */
    bool IsActive(int e) const {
        return active.Size() == 0 || active[e] != 0;
    }
};

/** @brief True when local element `e` is active under `mask` (nullptr = every element). Host. */
inline bool ElementIsActive(const ElementMask* mask, int e) {
    return mask == nullptr || mask->IsActive(e);
}

/**
 * @brief Raw per-element flags for device kernels: nullptr when every element is active (no mask
 * or an empty one), otherwise `mask->active.Read(on_device)`. Inside a kernel test an element with
 * `flags == nullptr || flags[e] != 0`.
 */
inline const int* ElementMaskData(const ElementMask* mask, bool on_device) {
    if (mask == nullptr || mask->active.Size() == 0) {
        return nullptr;
    }
    return mask->active.Read(on_device);
}

/**
 * @brief Writes `placeholder` into values[e] for every inactive local element e (host loop over
 * HostReadWrite()); nothing happens without a mask. Used by the labelers to give inactive
 * elements their documented placeholder label (seed -1, grain 0).
 */
inline void SetInactiveValues(const ElementMask* mask, mfem::Array<int>& values, int placeholder) {
    if (mask == nullptr || mask->active.Size() == 0 || values.Size() == 0) {
        return;
    }
    int* data = values.HostReadWrite();
    for (int e = 0; e < values.Size(); ++e) {
        if (mask->active[e] == 0) {
            data[e] = placeholder;
        }
    }
}

} // namespace exaconstit::microstructure
