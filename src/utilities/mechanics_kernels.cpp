#include "utilities/mechanics_kernels.hpp"

#include "mfem/general/forall.hpp"

namespace exaconstit {
namespace kernel {

// Updated implementation in mechanics_kernels.cpp
void GradCalc(const int nqpts,
              const int nelems,
              const int global_nelems,
              const int nnodes,
              const double* jacobian_data,
              const double* loc_grad_data,
              const double* field_data,
              double* field_grad_array,
              const int* const local2global) {
    const int DIM4 = 4;
    const int DIM3 = 3;
    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM4> perm4{{3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};
    std::array<RAJA::idx_t, DIM2> perm2{{1, 0}};

    const int dim = 3;
    const int space_dim2 = dim * dim;

    // Determine the size for input data views (global data)
    const int input_nelems = local2global ? global_nelems : nelems;

    // Set up RAJA views for input data (sized for global elements)
    RAJA::Layout<DIM4> layout_jacob_input = RAJA::make_permuted_layout(
        {{dim, dim, nqpts, input_nelems}}, perm4);
    RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> J_input(jacobian_data,
                                                                              layout_jacob_input);

    RAJA::Layout<DIM3> layout_field_input = RAJA::make_permuted_layout(
        {{nnodes, dim, input_nelems}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> field_input(
        field_data, layout_field_input);

    RAJA::Layout<DIM3> layout_loc_grad = RAJA::make_permuted_layout({{nnodes, dim, nqpts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> loc_grad_view(
        loc_grad_data, layout_loc_grad);

    // Set up RAJA views for output data (sized for local elements)
    RAJA::Layout<DIM4> layout_grad_output = RAJA::make_permuted_layout({{dim, dim, nqpts, nelems}},
                                                                       perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0>> field_grad_view(field_grad_array,
                                                                                layout_grad_output);

    RAJA::Layout<DIM2> layout_jinv = RAJA::make_permuted_layout({{dim, dim}}, perm2);

    // Process local elements (loop over nelems which is the local count)
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_local_elem) {
        // Map local element index to global element index for input data access
        const int i_global_elem = local2global ? local2global[i_local_elem] : i_local_elem;

        for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // Access input data using global element index
            const double J11 = J_input(0, 0, j_qpts, i_global_elem);
            const double J21 = J_input(1, 0, j_qpts, i_global_elem);
            const double J31 = J_input(2, 0, j_qpts, i_global_elem);
            const double J12 = J_input(0, 1, j_qpts, i_global_elem);
            const double J22 = J_input(1, 1, j_qpts, i_global_elem);
            const double J32 = J_input(2, 1, j_qpts, i_global_elem);
            const double J13 = J_input(0, 2, j_qpts, i_global_elem);
            const double J23 = J_input(1, 2, j_qpts, i_global_elem);
            const double J33 = J_input(2, 2, j_qpts, i_global_elem);

            const double detJ = J11 * (J22 * J33 - J32 * J23) - J21 * (J12 * J33 - J32 * J13) +
                                J31 * (J12 * J23 - J22 * J13);
            const double c_detJ = 1.0 / detJ;

            // Calculate adjugate matrix (inverse * determinant)
            const double A11 = c_detJ * ((J22 * J33) - (J23 * J32));
            const double A12 = c_detJ * ((J32 * J13) - (J12 * J33));
            const double A13 = c_detJ * ((J12 * J23) - (J22 * J13));
            const double A21 = c_detJ * ((J31 * J23) - (J21 * J33));
            const double A22 = c_detJ * ((J11 * J33) - (J13 * J31));
            const double A23 = c_detJ * ((J21 * J13) - (J11 * J23));
            const double A31 = c_detJ * ((J21 * J32) - (J31 * J22));
            const double A32 = c_detJ * ((J31 * J12) - (J11 * J32));
            const double A33 = c_detJ * ((J11 * J22) - (J12 * J21));

            const double A[space_dim2] = {A11, A21, A31, A12, A22, A32, A13, A23, A33};
            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0>> jinv_view(
                &A[0], layout_jinv);

            // Calculate field gradient - access input field data with global index
            // but write output data with local index
            for (int t = 0; t < dim; t++) {
                for (int s = 0; s < dim; s++) {
                    for (int r = 0; r < nnodes; r++) {
                        for (int q = 0; q < dim; q++) {
                            field_grad_view(
                                q, t, j_qpts, i_local_elem) += field_input(r, q, i_global_elem) *
                                                               loc_grad_view(r, s, j_qpts) *
                                                               jinv_view(s, t);
                        }
                    }
                }
            }
        }
    });
} // end GradCalc

void ElementAveragedShapeGradients(const int nqpts,
                                   const int nelems,
                                   const int nnodes,
                                   const double* jacobian_data,
                                   const double* loc_grad_data,
                                   const double* weights,
                                   double* bbar_data,
                                   const int* const local2global) {
    const std::ptrdiff_t jac_elem_stride = 9 * static_cast<std::ptrdiff_t>(nqpts);
    const int dshape_qpt_stride = 3 * nnodes;
    const int bbar_elem_stride = 3 * nnodes;
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_local) {
        // Input arrays are indexed by the global element, output by the local one.
        const int i_input = local2global ? local2global[i_local] : i_local;
        const double* jac_elem = jacobian_data + i_input * jac_elem_stride;
        double* bbar = bbar_data + static_cast<std::ptrdiff_t>(i_local) * bbar_elem_stride;
        for (int i = 0; i < bbar_elem_stride; ++i) {
            bbar[i] = 0.0;
        }
        double volume = 0.0;
        for (int q = 0; q < nqpts; ++q) {
            BBarAccumulateQpt(nnodes,
                              jac_elem + 9 * q,
                              loc_grad_data + q * dshape_qpt_stride,
                              weights[q],
                              bbar,
                              volume);
        }
        BBarFinalize(nnodes, volume, bbar);
    });
}

void ApplyBBarVelocityGradient(const int nqpts,
                               const int nelems,
                               const int nnodes,
                               const double* bbar_data,
                               const double* vel_data,
                               double* vel_grad_data,
                               const int* const local2global) {
    const int elem_stride = 3 * nnodes;
    const std::ptrdiff_t vgrad_elem_stride = 9 * static_cast<std::ptrdiff_t>(nqpts);
    mfem::forall(nelems, [=] MFEM_HOST_DEVICE(int i_local) {
        const int i_input = local2global ? local2global[i_local] : i_local;
        // tr_bar is element-constant: computed once, applied at every point.
        const double tr_bar = BBarTrace(
            nnodes,
            bbar_data + static_cast<std::ptrdiff_t>(i_local) * elem_stride,
            vel_data + static_cast<std::ptrdiff_t>(i_input) * elem_stride);
        double* vgrad_elem = vel_grad_data + i_local * vgrad_elem_stride;
        for (int q = 0; q < nqpts; ++q) {
            BBarCorrectVelocityGradient(tr_bar, vgrad_elem + 9 * q);
        }
    });
}
} // end namespace kernel
} // end namespace exaconstit