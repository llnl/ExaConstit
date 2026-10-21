#pragma once

#include "mfem.hpp"

#include <array>

/**
 * @file auto_mesh.hpp
 * @brief The `[Mesh] type = "auto"` box mesh, shared by makeMesh and the microstructure tool.
 *
 * MFEM only and header-only, so code outside exaconstit_static (the microstructure generator) can
 * reproduce exactly the mesh, element order and boundary attributes that ExaConstit builds. A
 * grain file written for an auto mesh is indexed by this element order, so both sides must use
 * these functions.
 */

namespace exaconstit {

/**
 * @brief Remaps the boundary attributes of an mfem::Mesh::MakeCartesian3D box to the
 * ExaConstit convention.
 *
 * MFEM numbers the six box faces 1 bottom (z = 0), 2 front (y = 0), 3 right (x = Lx), 4 back
 * (y = Ly), 5 left (x = 0), 6 top (z = Lz) (mfem/mesh/mesh.cpp:3816-3929, Mesh::Make3D).
 * ExaConstit's boundary conditions expect 1 bottom, 2 left, 3 front, 4 top, 5 right, 6 back, i.e.
 * attributes 1-3 are the minimum faces of z, x, y and 4-6 the maximum faces of z, x, y.
 * Any other attribute value is left unchanged.
 *
 * Also apply this before writing an auto mesh to a file, so a file mesh keeps the same boundary
 * attributes as the auto mesh it came from.
 *
 * Host only; serial (acts on an mfem::Mesh before it is partitioned), not collective.
 *
 * @param mesh box mesh from MakeAutoHexMesh; its boundary attributes are overwritten in place.
 */
inline void ApplyAutoMeshBoundaryAttributes(mfem::Mesh& mesh) {
    // Loop over boundary elements (faces on the box surface) and translate each MFEM face id
    for (int i = 0; i < mesh.GetNBE(); ++i) {
        switch (mesh.GetBdrAttribute(i)) {
        case 1:
            mesh.SetBdrAttribute(i, 1); // bottom, z = 0
            break;
        case 2:
            mesh.SetBdrAttribute(i, 3); // front, y = 0
            break;
        case 3:
            mesh.SetBdrAttribute(i, 5); // right, x = Lx
            break;
        case 4:
            mesh.SetBdrAttribute(i, 6); // back, y = Ly
            break;
        case 5:
            mesh.SetBdrAttribute(i, 2); // left, x = 0
            break;
        case 6:
            mesh.SetBdrAttribute(i, 4); // top, z = Lz
            break;
        }
    }
}

/**
 * @brief Builds the auto mesh: nxyz[0] x nxyz[1] x nxyz[2] trilinear hexahedra over the box
 * [0, mxyz[0]] x [0, mxyz[1]] x [0, mxyz[2]], in lexicographic element order.
 *
 * Lexicographic order (sfc_ordering = false in MakeCartesian3D) means element
 * e = i + nx * (j + ny * k) (0-based) is the i-th along x, j-th along y and k-th along z: x varies
 * fastest, then y, then z. This is the order a grain_file for an auto mesh is indexed by. MFEM's
 * default space-filling-curve order is not used; besides breaking that indexing, it also changed
 * the material response of a monotonic tension test with symmetric boundary conditions at 1%
 * strain.
 *
 * Boundary attributes are MFEM's; call ApplyAutoMeshBoundaryAttributes for ExaConstit's.
 * Host only; serial, not collective.
 *
 * @param nxyz number of elements along x, y, z (each > 0).
 * @param mxyz box edge lengths along x, y, z (each > 0), in mesh length units.
 * @return the serial mesh.
 */
inline mfem::Mesh MakeAutoHexMesh(const std::array<int, 3>& nxyz,
                                  const std::array<double, 3>& mxyz) {
    const bool sfc_ordering = false; // lexicographic element order, see above
    return mfem::Mesh::MakeCartesian3D(nxyz[0],
                                       nxyz[1],
                                       nxyz[2],
                                       mfem::Element::HEXAHEDRON,
                                       mxyz[0],
                                       mxyz[1],
                                       mxyz[2],
                                       sfc_ordering);
}

} // namespace exaconstit
