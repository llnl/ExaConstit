
#include "fem_operators/mechanics_operator_ext.hpp"

#include "fem_operators/mechanics_integrators.hpp"
#include "fem_operators/mechanics_operator.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "RAJA/RAJA.hpp"

MechOperatorJacobiSmoother::MechOperatorJacobiSmoother(const mfem::Vector& d,
                                                       const mfem::Array<int>& ess_tdofs,
                                                       const double dmpng)
    : mfem::Solver(d.Size()), ndofs(d.Size()), dinv(ndofs), damping(dmpng),
      ess_tdof_list(ess_tdofs), residual(ndofs) {
    Setup(d);
}

void MechOperatorJacobiSmoother::Setup(const mfem::Vector& diag) {
    residual.UseDevice(true);
    dinv.UseDevice(true);
    const double delta = damping;
    auto D = diag.Read();
    auto DI = dinv.Write();
    mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i) {
        DI[i] = delta / D[i];
    });
    auto I = ess_tdof_list.Read();
    mfem::forall(ess_tdof_list.Size(), [=] MFEM_HOST_DEVICE(int i) {
        DI[I[i]] = delta;
    });
}

void MechOperatorJacobiSmoother::Mult(const mfem::Vector& x, mfem::Vector& y) const {
    MFEM_ASSERT(x.Size() == ndofs, "invalid input vector");
    MFEM_ASSERT(y.Size() == ndofs, "invalid output vector");

    if (iterative_mode && oper) {
        oper->Mult(y, residual);         // r = A x
        subtract(x, residual, residual); // r = b - A x
    } else {
        residual = x;
        y.UseDevice(true);
        y = 0.0;
    }
    auto DI = dinv.Read();
    auto R = residual.Read();
    auto Y = y.ReadWrite();
    mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i) {
        Y[i] += DI[i] * R[i];
    });
}