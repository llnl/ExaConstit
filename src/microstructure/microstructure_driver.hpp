#pragma once

#include "microstructure/microstructure_options.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

/**
 * @file microstructure_driver.hpp
 * @brief Entry point shared by the standalone tool and the mechanics hook.
 */

namespace exaconstit::microstructure {

/**
 * @brief Labels the elements of `pmesh` with grains and assigns orientations and regions.
 *
 * Pipeline: generate or read the seeds (replicated on every rank), label each local element with
 * the grain of its sample point(s) under the chosen metric, clean up fragments and tiny grains,
 * then assign orientations and regions per grain.
 *
 * Collective over `comm`: every rank of `comm` must call it with identical `options`. The result
 * does not depend on the number of ranks: element labels of the same element are identical for
 * any partition of the same mesh. Runs on the host; kernels inside (mfem::forall, and the Axom
 * BVH through DispatchAxomExec) run on the backend of the mfem::Device, which the caller must
 * have configured from `options.exec` with ConfigureMfemDevice (exec_dispatch.hpp).
 *
 * @param options validated options (MicrostructureOptions::validate() returned true).
 * @param pmesh   this rank's partition of the mesh. Its geometry and attributes are not
 *                modified, but MFEM builds caches on demand when element geometry is evaluated:
 *                the mesh keeps the GeometricFactors object (mesh.cpp:887-899), and the nodal
 *                FE space keeps a QuadratureInterpolator (its output layout set to byNODES) and
 *                an element restriction (mesh.cpp:15251-15261). Mesh::GetGeometricFactors is
 *                therefore a non-const member (mesh.hpp:1973), hence the non-const reference.
 * @param comm    communicator of `pmesh`.
 * @return per-local-element grain ids and the replicated per-grain tables (see
 *         MicrostructureResult).
 *
 * Not implemented yet.
 */
MicrostructureResult GenerateMicrostructure(const MicrostructureOptions& options,
                                            mfem::ParMesh& pmesh,
                                            MPI_Comm comm);

} // namespace exaconstit::microstructure
