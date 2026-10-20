#include "mfem_expt/partial_qfunc.hpp"

#include "mfem_expt/partial_qspace.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace mfem::expt {

/// Copy the data from @a qf.
// this is wrong we need to check and see if first the sizes are equal if so it's a simple
// copy. If not then we need to want to check and see if the meshes are equal,
// integration rules are same or integration rule are same then we can fill things up easy
// peasy
PartialQuadratureFunction& PartialQuadratureFunction::operator=(const QuadratureFunction& qf) {
    MFEM_ASSERT(qf.GetVDim() == vdim, "Vector dimensions don't match");
    MFEM_ASSERT(qf.GetSpaceShared()->GetSize() >= part_quad_space->GetSize(),
                "QuadratureSpace sizes aren't of equivalent sizes");

    if (qf.GetSpaceShared()->GetSize() == part_quad_space->GetSize()) {
        Vector::operator=(qf);
        return *this;
    } else {
        // Very basic check to see if the two spaces are roughly equivalent...
        // We would need to do a much more thorough job if we wanted to be 100% certain
        MFEM_ASSERT(qf.GetSpaceShared()->GetMeshShared()->GetNE() ==
                        part_quad_space->GetMeshShared()->GetNE(),
                    "QSpaces have mesh's with different # of elements");
        MFEM_ASSERT(qf.GetSpaceShared()->GetOrder() == part_quad_space->GetOrder(),
                    "QSpaces don't have the same integration order");
        // We now need to copy all of the relevant data over that we'll need
        auto l2g = part_quad_space->local2global.Read();
        auto loc_offsets = part_quad_space->offsets.Read();
        auto global_offsets = (part_quad_space->global_offsets.Size() > 1)
                                  ? part_quad_space->global_offsets.Read()
                                  : loc_offsets;
        auto qf_data = qf.Read();
        auto loc_data = this->ReadWrite();

        auto NE = part_quad_space->GetNE();
        // For now this is fine. Later on we might want to leverage like RAJA views and the
        // IndexLayout to make things even more performant. Additionally, we could look at using 2D
        // kernels if need be but probably overkill for now...
        const auto vdim_ = vdim;
        mfem::forall(NE, [=] MFEM_HOST_DEVICE(int ie) {
            const int global_idx = l2g[ie];
            const int global_offset_idx = global_offsets[global_idx];
            const int local_offset_idx = loc_offsets[ie];
            const int nqpts = loc_offsets[ie + 1] - local_offset_idx;
            const int npts = nqpts * vdim_;
            for (int jv = 0; jv < npts; jv++) {
                loc_data[local_offset_idx * vdim_ + jv] = qf_data[global_offset_idx * vdim_ + jv];
            }
        });
    }
    return *this;
}

/// Takes in a quadrature function and fill with either the values contained in this
/// class or the default value provided by users.
void PartialQuadratureFunction::FillQuadratureFunction(QuadratureFunction& qf, const bool fill) {
    if (qf.GetSpaceShared()->GetSize() == part_quad_space->GetSize()) {
        qf = *this;
    } else {
        // Very basic check to see if the two spaces are roughly equivalent...
        // We would need to do a much more thorough job if we wanted to be 100% certain
        MFEM_ASSERT(qf.GetVDim() == vdim, "Vector dimensions don't match");
        MFEM_ASSERT(qf.GetSpaceShared()->GetMeshShared()->GetNE() ==
                        part_quad_space->GetMeshShared()->GetNE(),
                    "QSpaces have mesh's with different # of elements");
        MFEM_ASSERT(qf.GetSpaceShared()->GetOrder() == part_quad_space->GetOrder(),
                    "QSpaces don't have the same integration order");
        // We now need to copy all of the relevant data over that we'll need
        auto l2g = part_quad_space->local2global.Read();
        auto offsets = part_quad_space->offsets.Read();
        auto global_offsets = (part_quad_space->global_offsets.Size() > 1)
                                  ? part_quad_space->global_offsets.Read()
                                  : offsets;
        // First set all values to default. Do this before taking the data pointer so the
        // pointer refers to whichever copy (host/device) the fill left valid.
        if (fill) {
            qf = default_value;
        }
        auto qf_data = qf.ReadWrite();
        auto loc_data = this->Read();
        auto NE = part_quad_space->GetNE();
        // Then copy our partial values to their proper places
        const auto vdim_ = vdim;
        mfem::forall(NE, [=] MFEM_HOST_DEVICE(int ie) {
            const int global_idx = l2g[ie];
            const int global_offset_idx = global_offsets[global_idx];
            const int local_offset_idx = offsets[ie];
            const int nqpts = offsets[ie + 1] - local_offset_idx;
            const int npts = nqpts * vdim_;
            for (int jv = 0; jv < npts; jv++) {
                qf_data[global_offset_idx * vdim_ + jv] = loc_data[local_offset_idx * vdim_ + jv];
            }
        });
    }
}

} // namespace mfem::expt