! Example batched UMAT (interface = "batched"), in Fortran.
!
! The arguments are those of the Abaqus UMAT, in the same order, with NPTS added first. Arrays
! get a last dimension NPTS and the per-point scalars become arrays of length NPTS, so point ip's
! data is STRESS(:,ip), DDSDDE(:,:,ip), DROT(:,:,ip), PNEWDT(ip), and so on. The loop over ip
! wraps the body of an ordinary UMAT. See umat_batched.cpp for the material (hypoelastic
! isotropic solid) and ExaConstit's src/umats/userumat.h for the full description.
!
! With OpenMP (compile with -fopenmp) the points run in parallel; the body must then not write to
! SAVE or COMMON data. On GPU runs the arrays are GPU arrays and the loop must run on the GPU
! (OpenMP target, CUDA Fortran, OpenACC, ...); the sizes (NPTS, NTENS, ...) are CPU memory.
!
! In the options: function_name = "umat_batched_f" (ExaConstit also finds the trailing
! underscore most Fortran compilers add).
!
! CMNAME: ExaConstit passes it as a plain pointer, without the hidden length argument a Fortran
! CHARACTER dummy expects, so (as in ExaConstit's built-in umat.f) it is declared as a number
! here; it carries no material name.

subroutine umat_batched_f(npts, stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt, drplde, &
                          drpldt, stran, dstran, time, dtime, temp, dtemp, predef, dpred, &
                          cmname, ndi, nshr, ntens, nstatv, props, nprops, coords, drot, &
                          pnewdt, celent, dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
    implicit none
    integer :: npts, ndi, nshr, ntens, nstatv, nprops, layer, kspt, kstep, kinc
    integer :: noel(npts), npt(npts)
    real(8) :: cmname
    real(8) :: stress(ntens, npts), statev(nstatv, npts), ddsdde(ntens, ntens, npts)
    real(8) :: ddsddt(ntens, npts), drplde(ntens, npts)
    real(8) :: stran(ntens, npts), dstran(ntens, npts)
    real(8) :: sse(npts), spd(npts), scd(npts), rpl(npts), drpldt(npts)
    real(8) :: temp(npts), dtemp(npts), predef(npts), dpred(npts)
    real(8) :: coords(3, npts), drot(3, 3, npts), pnewdt(npts), celent(npts)
    real(8) :: dfgrd0(3, 3, npts), dfgrd1(3, 3, npts)
    real(8) :: time(2), dtime, props(nprops)

    integer :: i, j, ip
    real(8) :: young, nu, lambda, shear, s(3, 3), rs(3, 3), srot(6)

    young = props(1)
    nu = props(2)
    lambda = young * nu / ((1.d0 + nu) * (1.d0 - 2.d0 * nu))
    shear = young / (2.d0 * (1.d0 + nu))

    !$omp parallel do private(i, j, s, rs, srot)
    do ip = 1, npts
        ! Elastic stiffness
        ddsdde(:, :, ip) = 0.d0
        do i = 1, 3
            ddsdde(1:3, i, ip) = lambda
            ddsdde(i, i, ip) = lambda + 2.d0 * shear
            ddsdde(i + 3, i + 3, ip) = shear
        end do

        ! Rotate the old stress (order 11, 22, 33, 12, 13, 23): S <- R S R^T
        s(1, :) = (/ stress(1, ip), stress(4, ip), stress(5, ip) /)
        s(2, :) = (/ stress(4, ip), stress(2, ip), stress(6, ip) /)
        s(3, :) = (/ stress(5, ip), stress(6, ip), stress(3, ip) /)
        rs = matmul(drot(:, :, ip), matmul(s, transpose(drot(:, :, ip))))
        srot = (/ rs(1, 1), rs(2, 2), rs(3, 3), rs(1, 2), rs(1, 3), rs(2, 3) /)

        ! Add C : DSTRAN
        do i = 1, ntens
            stress(i, ip) = srot(i)
            do j = 1, ntens
                stress(i, ip) = stress(i, ip) + ddsdde(i, j, ip) * dstran(j, ip)
            end do
        end do
    end do
    !$omp end parallel do
end subroutine umat_batched_f
