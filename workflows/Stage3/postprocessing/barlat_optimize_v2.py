#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Barlat Yld2004-18p parameter fitting with batched JAX kernels,
switchable equivalent plastic strain solvers, and Sachs-type orientation averaging.

Key improvements over v1:
- Sachs model: rotate sample-frame stress to crystal frames, solve material update
  in each crystal frame, then average EPS over orientations.
- Quaternion utilities for passive rotations (crystal-to-sample convention).
- Batched stress rotation and yield computation over all orientations.
- Configurable: use single-crystal or Sachs-averaged optimization.
"""

import os
import tempfile
import multiprocessing

# ============================================================================
# CPU Parallelism Configuration - MUST be set before importing JAX
# ============================================================================
NUM_CPU_CORES = int(os.environ.get("JAX_NUM_CORES", multiprocessing.cpu_count()))

# Let XLA use multiple threads for parallel loop execution
os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={NUM_CPU_CORES}")

# Disable BLAS threading (small matrices don't benefit)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ============================================================================

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# Print parallelism info
print(f"JAX configured with {NUM_CPU_CORES} intra-op threads")
print(f"JAX devices: {jax.devices()}")

# Persistent compilation cache
try:
    from jax.experimental import compilation_cache as cc
    cache_dir = os.environ.get("JAX_CACHE_DIR", os.path.join(tempfile.gettempdir(), "jax_cache"))
    cc.initialize_cache(cache_dir)
except Exception:
    pass

from scipy.optimize import minimize, differential_evolution
import scipy.stats as scist


# ----------------------------
# Global solver mode toggle
# Options: "closed", "newton", "bisection"
# ----------------------------
SOLVER_MODE = "closed"

# Sachs averaging toggle
USE_SACHS_AVERAGING = False

BETA_DEFAULT = 20.0
N_DEFAULT = 0.45
EPS0_DEFAULT = 0.0

# Choose optimizer: "nelder-mead" or "differential-evolution"
# differential-evolution uses parallel workers for function evaluations
OPTIMIZER = "nelder-mead" #"differential-evolution"

# ----------------------------
# Helpers for Voigt conversion and tensors
# ----------------------------
def voigtNotation(mat):
    return jnp.asarray([mat[0, 0], mat[1, 1], mat[2, 2], mat[1, 2], mat[0, 2], mat[0, 1]])

def matNotation(voigt):
    return jnp.asarray([
        [voigt[0], voigt[5], voigt[4]],
        [voigt[5], voigt[1], voigt[3]],
        [voigt[4], voigt[3], voigt[2]],
    ])

def matNotation_np(voigt):
    return np.asarray([
        [voigt[0], voigt[5], voigt[4]],
        [voigt[5], voigt[1], voigt[3]],
        [voigt[4], voigt[3], voigt[2]],
    ])

def effectiveTerm(mat):
    term1 = mat[0, 0] - mat[1, 1]
    term2 = mat[1, 1] - mat[2, 2]
    term3 = mat[2, 2] - mat[0, 0]
    term4 = mat[1, 2] * mat[1, 2] + mat[0, 2] * mat[0, 2] + mat[0, 1] * mat[0, 1]
    return np.sqrt(0.5 * (term1 * term1 + term2 * term2 + term3 * term3 + 6.0 * term4))

def full_4d_to_Voigt_2d_np(C):
    Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    Voigt = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i, j] = C[k, l, m, n]
    return Voigt

def ItenMat_np():
    Iten = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            delij = 1.0 if i == j else 0.0
            for k in range(3):
                delik = 1.0 if i == k else 0.0
                deljk = 1.0 if j == k else 0.0
                for l in range(3):
                    deljl = 1.0 if j == l else 0.0
                    delil = 1.0 if i == l else 0.0
                    delkl = 1.0 if k == l else 0.0
                    Iten[i, j, k, l] = 0.5 * (delik * deljl + delil * deljk) - (1.0 / 3.0) * delij * delkl
    return Iten

# Precompute deviatoric projector in Voigt form
_Iten_2d_np = full_4d_to_Voigt_2d_np(ItenMat_np())
_Iten_2d_np[3:6, 3:6] *= 2.0
ITEN_2D = jnp.asarray(_Iten_2d_np)


# ----------------------------
# Quaternion and Rotation Utilities
# ----------------------------
def quat_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix (passive convention).
    
    Args:
        q: Quaternion [q0, q1, q2, q3] where q0 is scalar component.
           Passive rotation: maps vectors from crystal frame to sample frame.
    
    Returns:
        R: 3x3 rotation matrix such that v_sample = R @ v_crystal
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    
    R = jnp.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def quat_to_rotation_matrix_np(q):
    """NumPy version of quaternion to rotation matrix."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def rotate_stress_sample_to_crystal(sig_voigt_sample, q):
    """
    Rotate stress tensor from sample frame to crystal frame.
    
    For passive rotation R (crystal->sample): sigma_crystal = R^T @ sigma_sample @ R
    
    Args:
        sig_voigt_sample: Stress in Voigt notation [6] in sample frame
        q: Quaternion [4] defining crystal orientation
    
    Returns:
        sig_voigt_crystal: Stress in Voigt notation [6] in crystal frame
    """
    R = quat_to_rotation_matrix(q)
    sig_mat_sample = matNotation(sig_voigt_sample)
    sig_mat_crystal = R.T @ sig_mat_sample @ R
    return voigtNotation(sig_mat_crystal)

def rotate_stress_sample_to_crystal_np(sig_voigt_sample, q):
    """NumPy version for debugging/verification."""
    R = quat_to_rotation_matrix_np(q)
    sig_mat_sample = matNotation_np(sig_voigt_sample)
    sig_mat_crystal = R.T @ sig_mat_sample @ R
    return np.array([sig_mat_crystal[0,0], sig_mat_crystal[1,1], sig_mat_crystal[2,2],
                     sig_mat_crystal[1,2], sig_mat_crystal[0,2], sig_mat_crystal[0,1]])

# Batched rotation: single stress to multiple crystal frames
_rotate_stress_single_to_all_orient = jax.vmap(
    rotate_stress_sample_to_crystal, in_axes=(None, 0)
)

# Batched rotation: multiple stresses to multiple crystal frames
# Result shape: [n_stress, n_orient, 6]
_rotate_stress_all_to_all_orient = jax.vmap(
    _rotate_stress_single_to_all_orient, in_axes=(0, None)
)

rotate_stress_batch_jit = jax.jit(_rotate_stress_all_to_all_orient)


# ----------------------------
# Quaternion file I/O
# ----------------------------
def read_quaternions_file(filepath, weights_filepath=None):
    """
    Read quaternions from a file.
    
    Expected format: whitespace-separated, one quaternion per line
    q0 q1 q2 q3
    where q0 is the scalar component.
    
    Args:
        filepath: Path to quaternion file
        weights_filepath: Optional path to orientation weights file (one weight per line)
    
    Returns:
        quaternions: np.array [n_orient, 4]
        weights: np.array [n_orient] (uniform if no weights file)
    """
    quaternions = np.loadtxt(filepath)
    if quaternions.ndim == 1:
        quaternions = quaternions.reshape(1, -1)
    
    n_orient = quaternions.shape[0]
    
    if weights_filepath is not None and os.path.exists(weights_filepath):
        weights = np.loadtxt(weights_filepath)
        if weights.size != n_orient:
            raise ValueError(f"Weights file has {weights.size} entries but quaternion file has {n_orient}")
    else:
        weights = np.ones(n_orient) / n_orient
    
    # Normalize quaternions
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / norms
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return quaternions, weights


# ----------------------------
# Hardening law
# ----------------------------
def hardSG(eps, Y0=120.0, beta=36.0, Ymax=640.0, n=0.45, eps0=0.0):
    powY = (1.0 + beta * (eps + eps0))
    Y = Y0 * powY ** n
    dYdeps = Y * beta * n / powY
    return (Y, dYdeps)

def hardSG_value(eps, Y0=120.0, beta=36.0, n=0.45, eps0=0.0):
    return Y0 * (1.0 + beta * (eps + eps0)) ** n


# ----------------------------
# Barlat Yld2004-18p yield function
# ----------------------------
def computeBarlatYieldFunc(sigv, L1, L2, a):
    sig_prime = L1 @ sigv
    sig_prime2 = L2 @ sigv
    e1 = jnp.linalg.eigvalsh(matNotation(sig_prime))
    e2 = jnp.linalg.eigvalsh(matNotation(sig_prime2))
    diffs = jnp.abs(e1[:, None] - e2[None, :])
    inner = 0.25 * jnp.sum(diffs ** a)
    return inner ** (1.0 / a)

def computeBarlatYieldFunc_np(sigv, L1, L2, a):
    sig_prime = L1.dot(sigv)
    sig_prime2 = L2.dot(sigv)
    e1 = np.linalg.eigvalsh(matNotation_np(sig_prime))
    e2 = np.linalg.eigvalsh(matNotation_np(sig_prime2))
    diffs = np.abs(e1[:, None] - e2[None, :])
    inner = 0.25 * np.sum(diffs ** a)
    return inner ** (1.0 / a)

_barlat_yield_batch = jax.vmap(computeBarlatYieldFunc, in_axes=(0, None, None, None))
barlat_yield_batch_jit = jax.jit(_barlat_yield_batch)


# ----------------------------
# Assemble L1 and L2 from parameter vector
# ----------------------------
def assemble_L_mats(x):
    dtype = x.dtype
    L1 = jnp.array([
        [0.0,   -1.0,  -1.0,   0.0,  0.0,  0.0],
        [-x[0],  0.0,  -x[1],  0.0,  0.0,  0.0],
        [-x[2], -x[3],  0.0,   0.0,  0.0,  0.0],
        [0.0,    0.0,   0.0,   x[4], 0.0,  0.0],
        [0.0,    0.0,   0.0,   0.0,  x[5], 0.0],
        [0.0,    0.0,   0.0,   0.0,  0.0,  x[6]],
    ], dtype=dtype)

    L2 = jnp.array([
        [0.0,   -x[7],  -x[8],  0.0,  0.0,  0.0],
        [-x[9],  0.0,   -x[10], 0.0,  0.0,  0.0],
        [-x[11], -x[12], 0.0,   0.0,  0.0,  0.0],
        [0.0,     0.0,   0.0,   x[13],0.0,  0.0],
        [0.0,     0.0,   0.0,   0.0,  x[14],0.0],
        [0.0,     0.0,   0.0,   0.0,  0.0,  x[15]],
    ], dtype=dtype)

    a  = x[16]
    Y0 = x[17]
    L1 = L1 @ ITEN_2D
    L2 = L2 @ ITEN_2D
    return L1, L2, a, Y0


# ----------------------------
# EPS solvers
# ----------------------------
def solve_eps_closed(y, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT):
    x = ((y / Y0) ** (1.0 / n) - 1.0) / beta - eps0
    return jnp.maximum(0.0, x)

def computeRJ_const(ysVal, x, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT):
    Y, dYdx = hardSG(x, Y0=Y0, beta=beta, n=n, eps0=eps0)
    residual = ysVal - Y
    jacobian = -dYdx
    return residual, jacobian

@jax.jit
def solve_eps_newton(y, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT, tol=1e-10, maxiter=100):
    x0 = solve_eps_closed(y, Y0, beta=beta, n=n, eps0=eps0)
    r0, j0 = computeRJ_const(y, x0, Y0, beta=beta, n=n, eps0=eps0)

    def cond_fun(state):
        x, r, j, i = state
        return (jnp.abs(r) > tol) & (i < maxiter)

    def body_fun(state):
        x, r, j, i = state
        j_safe = jnp.where(jnp.abs(j) < 1e-16, jnp.sign(j) * 1e-16, j)
        x_new = x - (r / j_safe)
        r_new, j_new = computeRJ_const(y, x_new, Y0, beta=beta, n=n, eps0=eps0)
        return (x_new, r_new, j_new, i + 1)

    x_final, _, _, _ = lax.while_loop(cond_fun, body_fun, (x0, r0, j0, 0))
    return jnp.maximum(0.0, x_final)

def solve_eps_newton_batch(ys, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT):
    return jax.vmap(lambda y: solve_eps_newton(y, Y0, beta=beta, n=n, eps0=eps0))(ys)

solve_eps_newton_batch_jit = jax.jit(solve_eps_newton_batch)

@jax.jit
def solve_eps_bisection(y, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT, maxiter=64):
    y0 = hardSG_value(0.0, Y0=Y0, beta=beta, n=n, eps0=eps0)
    def trivial_case():
        return jnp.array(0.0)
    def nontrivial_case():
        x_cf = solve_eps_closed(y, Y0, beta=beta, n=n, eps0=eps0)
        high0 = jnp.maximum(1e-12, x_cf)
        low = jnp.array(0.0)
        f_low = y - hardSG_value(low, Y0=Y0, beta=beta, n=n, eps0=eps0)

        def expand_high(state):
            high, f_high, i = state
            need_expand = f_high > 0.0
            high_new = jnp.where(need_expand, high * 2.0 + 1e-6, high)
            f_high_new = jnp.where(need_expand,
                                   y - hardSG_value(high_new, Y0=Y0, beta=beta, n=n, eps0=eps0),
                                   f_high)
            return (high_new, f_high_new, i + 1)

        def cond_expand(state):
            high, f_high, i = state
            return (f_high > 0.0) & (i < 32)

        f_high0 = y - hardSG_value(high0, Y0=Y0, beta=beta, n=n, eps0=eps0)
        high, f_high, _ = lax.while_loop(cond_expand, expand_high, (high0, f_high0, 0))

        def bisect_body(state):
            lo, hi, f_lo, f_hi, i = state
            mid = 0.5 * (lo + hi)
            f_mid = y - hardSG_value(mid, Y0=Y0, beta=beta, n=n, eps0=eps0)
            left = f_mid >= 0.0
            lo_new = jnp.where(left, mid, lo)
            hi_new = jnp.where(left, hi, mid)
            f_lo_new = jnp.where(left, f_mid, f_lo)
            f_hi_new = jnp.where(left, f_hi, f_mid)
            return (lo_new, hi_new, f_lo_new, f_hi_new, i + 1)

        def bisect_cond(state):
            lo, hi, f_lo, f_hi, i = state
            return i < maxiter

        lo, hi, _, _, _ = lax.while_loop(bisect_cond, bisect_body, (low, high, f_low, f_high, 0))
        return jnp.maximum(0.0, hi)

    return lax.cond(y <= y0, trivial_case, nontrivial_case)

def solve_eps_bisection_batch(ys, Y0, beta=BETA_DEFAULT, n=N_DEFAULT, eps0=EPS0_DEFAULT):
    return jax.vmap(lambda y: solve_eps_bisection(y, Y0, beta=beta, n=n, eps0=eps0))(ys)

solve_eps_bisection_batch_jit = jax.jit(solve_eps_bisection_batch)


# ----------------------------
# Standard (single-crystal) objective
# ----------------------------
def _barlat_objective_common(x, stressVoigt, eq_pl_strain, wsig, eps_solver):
    L1, L2, a, Y0 = assemble_L_mats(x)
    ys_vals = barlat_yield_batch_jit(stressVoigt, L1, L2, a)
    if eps_solver == "closed":
        eps_hat = solve_eps_closed(ys_vals, Y0)
    elif eps_solver == "newton":
        eps_hat = solve_eps_newton_batch_jit(ys_vals, Y0)
    elif eps_solver == "bisection":
        eps_hat = solve_eps_bisection_batch_jit(ys_vals, Y0)
    else:
        raise ValueError(f"Unknown eps_solver: {eps_solver}")
    eps_ref = jnp.maximum(eq_pl_strain, 1e-20)
    err_vec = jnp.sqrt(wsig) * jnp.abs(eps_hat / eps_ref - 1.0)
    return jnp.sum(err_vec)

def barlat_objective_closed(x, stressVoigt, eq_pl_strain, wsig):
    return _barlat_objective_common(x, stressVoigt, eq_pl_strain, wsig, "closed")

def barlat_objective_newton(x, stressVoigt, eq_pl_strain, wsig):
    return _barlat_objective_common(x, stressVoigt, eq_pl_strain, wsig, "newton")

def barlat_objective_bisection(x, stressVoigt, eq_pl_strain, wsig):
    return _barlat_objective_common(x, stressVoigt, eq_pl_strain, wsig, "bisection")

# Note: donate_argnums removed because these functions are called repeatedly
# during optimization with the same stress/strain arrays
barlat_objective_closed_jit = jax.jit(barlat_objective_closed)
barlat_objective_newton_jit = jax.jit(barlat_objective_newton)
barlat_objective_bisection_jit = jax.jit(barlat_objective_bisection)


# ----------------------------
# Sachs-averaged objective functions
# ----------------------------
def _sachs_objective_common(x, stress_crystal_3d, eq_pl_strain, wsig, 
                            weights_orient, eps_solver):
    """
    Sachs model objective: compute EPS in each crystal frame and take weighted average.
    
    Stress rotation is done OUTSIDE this function as a preprocessing step.
    
    Args:
        x: Parameter vector [18]
        stress_crystal_3d: Pre-rotated stress states [npoints, n_orient, 6]
        eq_pl_strain: Target EPS values [npoints]
        wsig: Weights for stress states [npoints]
        weights_orient: Orientation weights [n_orient], should sum to 1
        eps_solver: "closed", "newton", or "bisection"
    
    Returns:
        Scalar error
    """
    L1, L2, a, Y0 = assemble_L_mats(x)
    
    # Get dimensions
    npoints = stress_crystal_3d.shape[0]
    n_orient = stress_crystal_3d.shape[1]
    
    # Flatten for batched yield computation: [npoints * n_orient, 6]
    stress_flat = stress_crystal_3d.reshape(-1, 6)
    
    # Compute yield values
    ys_flat = barlat_yield_batch_jit(stress_flat, L1, L2, a)
    
    # Solve EPS
    if eps_solver == "closed":
        eps_flat = solve_eps_closed(ys_flat, Y0)
    elif eps_solver == "newton":
        eps_flat = solve_eps_newton_batch_jit(ys_flat, Y0)
    elif eps_solver == "bisection":
        eps_flat = solve_eps_bisection_batch_jit(ys_flat, Y0)
    else:
        raise ValueError(f"Unknown eps_solver: {eps_solver}")
    
    # Reshape: [npoints, n_orient]
    eps_all = eps_flat.reshape(npoints, n_orient)
    
    # Weighted average over orientations (Sachs average)
    eps_avg = jnp.sum(eps_all * weights_orient[None, :], axis=1)  # [npoints]
    
    # Compute error metric
    eps_ref = jnp.maximum(eq_pl_strain, 1e-20)
    err_vec = jnp.sqrt(wsig) * jnp.abs(eps_avg / eps_ref - 1.0)
    
    return jnp.sum(err_vec)

def sachs_objective_closed(x, stress_crystal_3d, eq_pl_strain, wsig, weights_orient):
    return _sachs_objective_common(x, stress_crystal_3d, eq_pl_strain, wsig, 
                                   weights_orient, "closed")

def sachs_objective_newton(x, stress_crystal_3d, eq_pl_strain, wsig, weights_orient):
    return _sachs_objective_common(x, stress_crystal_3d, eq_pl_strain, wsig, 
                                   weights_orient, "newton")

def sachs_objective_bisection(x, stress_crystal_3d, eq_pl_strain, wsig, weights_orient):
    return _sachs_objective_common(x, stress_crystal_3d, eq_pl_strain, wsig, 
                                   weights_orient, "bisection")

# JIT compile Sachs objectives (no quaternions needed - stress pre-rotated)
sachs_objective_closed_jit = jax.jit(sachs_objective_closed)
sachs_objective_newton_jit = jax.jit(sachs_objective_newton)
sachs_objective_bisection_jit = jax.jit(sachs_objective_bisection)


# ----------------------------
# Objective function selectors
# ----------------------------
def get_barlat_objective_jit():
    if SOLVER_MODE == "closed":
        return barlat_objective_closed_jit
    elif SOLVER_MODE == "newton":
        return barlat_objective_newton_jit
    elif SOLVER_MODE == "bisection":
        return barlat_objective_bisection_jit
    else:
        raise ValueError(f"Unknown SOLVER_MODE: {SOLVER_MODE}")

def get_sachs_objective_jit():
    if SOLVER_MODE == "closed":
        return sachs_objective_closed_jit
    elif SOLVER_MODE == "newton":
        return sachs_objective_newton_jit
    elif SOLVER_MODE == "bisection":
        return sachs_objective_bisection_jit
    else:
        raise ValueError(f"Unknown SOLVER_MODE: {SOLVER_MODE}")


# ----------------------------
# NumPy interface functions for optimizer
# ----------------------------
def barlat_optimize_np_fcn(x0, stressVoigt, eq_pl_strain, aparam, wsig):
    """Standard single-crystal objective (NumPy interface)."""
    x0j = jnp.asarray(x0, dtype=jnp.float64)
    stressVoigt_j = jnp.asarray(stressVoigt, dtype=jnp.float64)
    eq_pl_strain_j = jnp.asarray(eq_pl_strain, dtype=jnp.float64)
    wsig_j = jnp.asarray(wsig, dtype=jnp.float64)
    obj = get_barlat_objective_jit()
    err = obj(x0j, stressVoigt_j, eq_pl_strain_j, wsig_j)
    return float(err)

def sachs_optimize_np_fcn(x0, stress_crystal_3d, eq_pl_strain, aparam, wsig, 
                          weights_orient):
    """Sachs-averaged objective (NumPy interface). Stress must be pre-rotated."""
    x0j = jnp.asarray(x0, dtype=jnp.float64)
    stress_j = jnp.asarray(stress_crystal_3d, dtype=jnp.float64)
    eq_pl_strain_j = jnp.asarray(eq_pl_strain, dtype=jnp.float64)
    wsig_j = jnp.asarray(wsig, dtype=jnp.float64)
    weights_j = jnp.asarray(weights_orient, dtype=jnp.float64)
    obj = get_sachs_objective_jit()
    err = obj(x0j, stress_j, eq_pl_strain_j, wsig_j, weights_j)
    return float(err)


# ----------------------------
# Workflow functions
# ----------------------------
def job_directories(args, output_file_dir, df, nrves):
    rve_dirs = {}
    rve_time = {}
    rve_load_names = {}

    for irve in range(nrves):
        local_args = args.loc[irve]
        fdirs = os.path.abspath(output_file_dir)
        frve = local_args["rve_unique_name"]
        fdir_rve = os.path.join(fdirs, frve, "")
        fdiro = fdir_rve

        nruns = df[irve].shape[0]
        headers = list(df[irve].columns)
        headers.pop(0)
        sub_rve_dirs = {}
        sub_rve_time = {}
        sub_load_names = {}

        for iDir in range(nruns):
            rve_name = df[irve]["rve_unique_name"][iDir]
            load_dir_name = df[irve]["loading_name"][iDir]
            if "dt_file" in df[irve].columns:
                dt_file = df[irve]["dt_file"][iDir]
            else:
                dt_file = None
            temp_k = str(int(df[irve]["temperature"][iDir]))
            fdiron = fdiro
            fdironl = os.path.join(fdiron, load_dir_name+"_"+temp_k, "")
            if temp_k in sub_rve_dirs:  
                sub_rve_dirs[temp_k].append(fdironl)
                if dt_file is not None:
                    sub_rve_time[temp_k].append(dt_file)
                sub_load_names[temp_k].append(load_dir_name)
            else:
                sub_rve_dirs[temp_k] = [fdironl]
                if dt_file is not None:
                    sub_rve_time[temp_k] = [dt_file]
                else:
                    sub_rve_time[temp_k] = None
                sub_load_names[temp_k] = [load_dir_name]
                
        rve_dirs[frve] = sub_rve_dirs.copy()
        rve_time[frve] = sub_rve_time.copy()
        rve_load_names[frve] = sub_load_names.copy()
    
    return (rve_dirs, rve_time, rve_load_names)

def postprocessing_start(fdir_rve, tempk, ftime=None, loading_dir_names=None, 
                         strain_rate=0.001, quaternion_file=None, quat_weights_file=None):
    """
    Post-process simulation results and optionally load orientation data.
    
    Args:
        fdir_rve: List of directories for each loading condition
        tempk: Temperature in Kelvin
        ftime: Time file (optional)
        loading_dir_names: List of loading direction names
        strain_rate: Strain rate
        quaternion_file: Path to quaternion file for Sachs averaging (optional)
        quat_weights_file: Path to orientation weights file (optional)
    
    Returns:
        Dictionary with stress/strain data and optional orientation info
    """
    index_zz = 0
    if loading_dir_names is None:
        loading_dir_names = [
                "x_90_z_0", "x_0_y_90", "x_15_y_75", "x_30_y_60", "x_45_z_45", 
                "x_60_z_30", "x_75_z_15", "x_90_y_0", "x_15_z_75", "x_30_z_60",  
                "x_45_z_45", "x_60_z_30", "x_75_z_15", "y_15_z_75", "y_30_z_60",
                "y_45_z_45", "y_60_z_30", "y_75_z_15"]
        loading_dir_shear_names = ["shear_xy", "shear_xz", "shear_yz"]
        loading_dir_names.extend(loading_dir_shear_names)
    else:
        for i in range(len(loading_dir_names)):
            if "x_90_z_0" in loading_dir_names[i]:
                index_zz = i
                break

    ntests = len(loading_dir_names)
    weights = np.ones(ntests)
    r_include = [False] * len(loading_dir_names)
    properties = np.zeros((1, 19))
    
    print("Starting temperature: ", tempk)
    rve_name = os.path.basename(os.path.dirname(os.path.dirname(fdir_rve[index_zz])))
    print(rve_name)
    ext_name = rve_name+"_"+str(int(tempk))+"_"+loading_dir_names[index_zz]

    stress_fbname = "avg_stress_global.txt"
    plwork_fbname = "avg_pl_work_global.txt"
    eps_fbname = "avg_eq_pl_strain_global.txt"
    strain_fbname = "avg_euler_strain_global.txt"

    fz_dir = os.path.join(fdir_rve[index_zz], "results", ext_name, "")
    stress = np.loadtxt(os.path.join(fz_dir, stress_fbname))
    pl_work = np.loadtxt(os.path.join(fz_dir, plwork_fbname))[:, -1]
    eps = np.loadtxt(os.path.join(fz_dir, eps_fbname))[:, -1]
    strain = np.loadtxt(os.path.join(fz_dir, strain_fbname))[:, 4]
    time = stress[:, 0]
    stress = stress[:, 2:8]
    
    slope, intercept, r, p, se = scist.linregress(np.abs(strain[0:9]), np.abs(stress[0:9, 2]))
    stress_offset = slope * (np.abs(strain) - 0.002)
    
    plwork_driver = 0.0
    stress_exp = np.zeros((ntests, 6))
    eq_pl_strain = np.zeros((ntests))
    error = np.zeros((ntests))
    vMs = np.zeros(ntests)
    
    for j in range(2, time.size):
        if stress_offset[j] > np.abs(stress[j, 2]):
            sx1 = np.abs(strain[j - 1])
            sy1 = np.abs(stress[j - 1, 2])
            sx2 = np.abs(strain[j])
            sy2 = np.abs(stress[j, 2])
            oy1 = stress_offset[j - 1]
            oy2 = stress_offset[j]
            plwork_driver = pl_work[j]
            break
        
    YS = ((sx1 * oy2 - sx2 * oy1) * (sy1 - sy2) - (oy1 - oy2) * (sx1 * sy2 - sx2 * sy1)) / \
         ((sx1 - sx2) * (sy1 - sy2) - (sx1 - sx2) * (oy1 - oy2))
    print([YS, plwork_driver])
    
    iload = 0
    for load_dir in loading_dir_names:    
        ext_name = rve_name+"_"+str(int(tempk))+"_"+load_dir
        fl_dir = os.path.join(fdir_rve[iload], "results", ext_name, "")
        stress = np.loadtxt(os.path.join(fl_dir, stress_fbname))
        plwork = np.loadtxt(os.path.join(fl_dir, plwork_fbname))[:, -1]
        eps = np.loadtxt(os.path.join(fl_dir, eps_fbname))[:, -1]
        strain = np.loadtxt(os.path.join(fl_dir, strain_fbname))[:, 4]
        time = stress[:, 0]
        stress = stress[:, 2:8]
        
        dplwork = plwork - plwork_driver
        abs_dplwork = np.abs(dplwork)
        
        ind = np.argmin(abs_dplwork)
        tr_stress = 1.0/3.0 * np.sum(stress[ind, 0:3])
        stress_exp[iload, :] = stress[ind, :]
        stress_exp[iload, 0:3] -= tr_stress
        
        eq_pl_strain[iload] = eps[ind]
        sig = matNotation_np(stress_exp[iload, :])
        vMs[iload] = effectiveTerm(sig)
        print(rve_name, load_dir, ind, 100.0 * dplwork[ind]/plwork_driver, vMs[iload], eq_pl_strain[iload])
        error[iload] = 100.0 * dplwork[ind]/plwork_driver
        
        iload += 1

    for i in range(len(loading_dir_names)):
        if "x_90_z_0" in loading_dir_names[i]:
            r_include[i] = False
        if "x_90_y_0" in loading_dir_names[i]:
            r_include[i] = False
        if "x_0_y_90" in loading_dir_names[i]:
            r_include[i] = False

    # Build result dictionary
    df = {
        "vonMises": np.copy(vMs),
        "voigt_stress": np.copy(stress_exp),
        "equivalent_plastic_strain": np.copy(eq_pl_strain),
        "weights": np.copy(weights),
        "error": np.copy(error),
        "loading_dirs": np.copy(loading_dir_names),
        "index_zz": index_zz
    }
    
    # Load quaternions if provided (for Sachs averaging)
    if quaternion_file is not None and os.path.exists(quaternion_file):
        print(f"Loading orientations from: {quaternion_file}")
        quaternions, quat_weights = read_quaternions_file(quaternion_file, quat_weights_file)
        df["quaternions"] = quaternions
        df["orientation_weights"] = quat_weights
        df["use_sachs"] = True
        print(f"  Loaded {quaternions.shape[0]} orientations")
    else:
        df["quaternions"] = None
        df["orientation_weights"] = None
        df["use_sachs"] = False
    
    return df


# ----------------------------
# Utility: compute rotated stress array (for inspection)
# ----------------------------
def compute_rotated_stress_array(stress_sample, quaternions):
    """
    Compute the full 3D array of rotated stresses.
    
    Args:
        stress_sample: Sample frame stresses [npoints, 6]
        quaternions: Orientation quaternions [n_orient, 4]
    
    Returns:
        stress_crystal: Rotated stresses [npoints, n_orient, 6]
    """
    stress_j = jnp.asarray(stress_sample, dtype=jnp.float64)
    quat_j = jnp.asarray(quaternions, dtype=jnp.float64)
    return np.asarray(rotate_stress_batch_jit(stress_j, quat_j))


# ----------------------------
# Picklable objective classes for parallel optimization
# ----------------------------
class BarlatObjective:
    """Picklable objective function for standard (non-Sachs) optimization."""
    
    def __init__(self, stress_exp, eq_pl_strain, aparam, weights):
        self.stress_exp = stress_exp
        self.eq_pl_strain = eq_pl_strain
        self.aparam = aparam
        self.weights = weights
    
    def __call__(self, x):
        return barlat_optimize_np_fcn(
            x, self.stress_exp, self.eq_pl_strain, 
            self.aparam, self.weights
        )


class SachsObjective:
    """Picklable objective function for Sachs-averaged optimization."""
    
    def __init__(self, stress_crystal_3d, eq_pl_strain, aparam, weights, orientation_weights):
        self.stress_crystal_3d = stress_crystal_3d
        self.eq_pl_strain = eq_pl_strain
        self.aparam = aparam
        self.weights = weights
        self.orientation_weights = orientation_weights
    
    def __call__(self, x):
        return sachs_optimize_np_fcn(
            x, self.stress_crystal_3d, self.eq_pl_strain,
            self.aparam, self.weights, self.orientation_weights
        )


class ObjectiveWithTracking:
    """Wrapper that adds progress tracking to an objective function."""
    
    def __init__(self, objective, report_interval=100):
        self.objective = objective
        self.report_interval = report_interval
        self.eval_count = 0
        self.best_seen = float('inf')
    
    def __call__(self, x):
        err = self.objective(x)
        self.eval_count += 1
        if err < self.best_seen:
            self.best_seen = err
        if self.eval_count % self.report_interval == 0:
            print(f"Eval {self.eval_count:6d}: error = {err:.6e} (best = {self.best_seen:.6e})")
        return err


# ----------------------------
# Optimizer wrapper with multiple optimizer options
# ----------------------------
def optimize_wrapper(start_data, use_sachs=None, optimizer=OPTIMIZER):
    """
    Run optimization with either standard or Sachs-averaged objective.
    
    Args:
        start_data: Dictionary from postprocessing_start
        use_sachs: Override for Sachs mode (None = use start_data setting)
        optimizer: "nelder-mead" or "differential-evolution"
    """
    stress_exp = np.copy(start_data["voigt_stress"])
    eq_pl_strain = np.copy(start_data["equivalent_plastic_strain"])
    vMs = np.copy(start_data["vonMises"])
    weights = np.copy(start_data["weights"])
    index_zz = start_data['index_zz']
    
    # Determine if using Sachs averaging
    if use_sachs is None:
        use_sachs = start_data.get("use_sachs", False)
    
    # Precompute rotated stresses if using Sachs averaging
    stress_crystal_3d = None
    orientation_weights = None
    if use_sachs:
        quaternions = start_data.get("quaternions")
        orientation_weights = start_data.get("orientation_weights")
        if quaternions is None:
            raise ValueError("Sachs averaging requested but no quaternions loaded")
        print(f"Using Sachs averaging with {quaternions.shape[0]} orientations")
        
        # Precompute rotated stresses: [npoints, n_orient, 6]
        print("Precomputing rotated stresses...")
        stress_crystal_3d = compute_rotated_stress_array(stress_exp, quaternions)
        print(f"  Rotated stress array shape: {stress_crystal_3d.shape}")
    
    # Initial parameters
    x0 = np.zeros(18)
    x0[:] = 1.25
    aparam = 10.0
    x0[16] = aparam
    x0[17] = vMs[index_zz] * 0.95

    bounds = []
    for i in range(16):
        bounds.append((0.85, 3.0))
    bounds.append((4.0, 20.0))
    bounds.append((0.90*vMs[0], 1.05*vMs[0]))
    bounds = tuple(bounds)

    # Select objective function (use picklable classes for parallel optimizers)
    if use_sachs:
        objective = SachsObjective(
            stress_crystal_3d, eq_pl_strain, aparam, weights, orientation_weights
        )
    else:
        objective = BarlatObjective(
            stress_exp, eq_pl_strain, aparam, weights
        )

    # Warmup call to trigger JIT compilation before optimization
    print("Warming up JIT compilation...")
    _ = objective(x0)
    print("JIT warmup complete.")
    
    print("Initial error:")
    print(objective(x0))

    np.random.seed(0)
    
    # =========================================================================
    # Nelder-Mead optimizer
    # =========================================================================
    if optimizer.lower() == "nelder-mead":
        print("\nUsing Nelder-Mead optimizer")
        
        # Wrap objective with progress tracking
        objective_tracked = ObjectiveWithTracking(objective, report_interval=100)
        
        res = minimize(
            fun=objective_tracked,
            method="nelder-mead",
            x0=x0,
            bounds=bounds,
            options={
                "maxiter": 50000, 
                "disp": True, 
                "adaptive": True, 
                "fatol": 1e-10, 
                "xatol": 1e-10
            },
            tol=1e-10
        )
    
    # =========================================================================
    # Differential Evolution optimizer
    # =========================================================================
    elif optimizer.lower() == "differential-evolution":
        POPSIZE = 25
        print(f"\nUsing Differential Evolution optimizer")
        print(f"  Workers: {NUM_CPU_CORES} (parallel function evaluations)")
        print(f"  Population size: {POPSIZE} * {len(x0)} = {POPSIZE * len(x0)}")
        
        # Callback for per-generation progress
        generation_count = [0]
        
        def de_callback(xk, convergence):
            generation_count[0] += 1
            current_err = objective(xk)
            print(f"Generation {generation_count[0]:4d}: best error = {current_err:.6e}, convergence = {convergence:.6e}")
        
        # =====================================================================
        # DIFFERENTIAL EVOLUTION PARAMETER REFERENCE
        # =====================================================================
        #
        # STRATEGY: '{base}{n}{crossover}' - Mutation strategy
        #   Base vectors:
        #     'best' = mutate from best member (faster convergence)
        #     'rand' = mutate from random member (better exploration)
        #     'currenttobest' = blend current toward best
        #     'randtobest' = blend random toward best
        #   Number of difference vectors: 1 or 2 (2 = more diversity)
        #   Crossover: 'bin' (binomial) or 'exp' (exponential, for correlated params)
        #
        #   Options: 'best1bin' (default), 'best1exp', 'rand1bin', 'rand1exp',
        #            'randtobest1bin', 'randtobest1exp', 'currenttobest1bin',
        #            'currenttobest1exp', 'best2bin', 'best2exp', 'rand2bin', 'rand2exp'
        #
        #   Recommendations:
        #     - 'best1bin': Good default, balanced convergence/exploration
        #     - 'rand1bin': Better global search, avoids local minima
        #     - 'randtobest1bin': Balance between rand and best
        #     - 'rand2bin': Maximum exploration for highly multimodal problems
        #
        # MAXITER: Maximum generations (default: 1000)
        #   Total evaluations ≈ maxiter * popsize * len(x)
        #   Example: 1000 * 15 * 18 = 270,000 evaluations max
        #
        # POPSIZE: Population multiplier (default: 15)
        #   Actual population = popsize * len(x)
        #     5-10:  Fast, may miss global optimum
        #     15:    Default, good for most problems
        #     20-30: Thorough global search, slower
        #
        # TOL: Relative convergence tolerance (default: 0.01)
        #   Stops when std(population_energies)/mean < tol
        #     1e-3:  Loose, fast termination
        #     1e-6:  Moderate precision
        #     1e-10: High precision (current setting)
        #
        # MUTATION: Mutation constant F (default: (0.5, 1))
        #   Controls step size of mutations
        #   Can be float or tuple (min, max) for dithering
        #     (0.3, 0.7):  Fine-tuning, slower exploration
        #     (0.5, 1.0):  Balanced (recommended)
        #     (0.5, 1.5):  Aggressive exploration
        #     (0.8, 1.2):  Fast convergence on smooth problems
        #
        # RECOMBINATION: Crossover probability CR (default: 0.7)
        #   Probability each parameter comes from mutant vs parent
        #     0.3-0.5: Few params change, for correlated parameters
        #     0.7:     Default, balanced
        #     0.9:     Many params change, for separable problems
        #
        # WORKERS: Parallel function evaluations (default: 1)
        #     1:  Sequential
        #     -1: Use all CPU cores
        #     N:  Use N cores
        #   Note: Requires updating='deferred' when workers != 1
        #
        # UPDATING: Population update strategy (default: 'immediate')
        #     'immediate': Update as better members found (workers=1 only)
        #     'deferred':  Update after full generation (required for parallel)
        #
        # POLISH: Run L-BFGS-B refinement after DE (default: True)
        #     True:  Refine solution with local optimizer (recommended)
        #     False: Use raw DE result (for noisy/non-smooth functions)
        #
        # INIT: Population initialization (default: 'latinhypercube')
        #     'latinhypercube': Good space coverage (default)
        #     'sobol':          Better coverage (scipy >= 1.7)
        #     'halton':         Halton sequence
        #     'random':         Uniform random
        #
        # =====================================================================
        # RECOMMENDED CONFIGURATIONS
        # =====================================================================
        #
        # FAST EXPLORATION (approximate solution quickly):
        #   strategy='rand1bin', maxiter=500, popsize=10, tol=1e-6,
        #   mutation=(0.5, 1.0), recombination=0.7
        #
        # THOROUGH GLOBAL SEARCH (avoid local minima):
        #   strategy='rand2bin', maxiter=2000, popsize=25, tol=1e-10,
        #   mutation=(0.5, 1.5), recombination=0.6
        #
        # HIGH PRECISION REFINEMENT (when close to solution):
        #   strategy='best1bin', maxiter=1000, popsize=15, tol=1e-12,
        #   mutation=(0.3, 0.7), recombination=0.8
        #
        # BALANCED DEFAULT (current settings):
        #   strategy='best1bin', maxiter=1000, popsize=15, tol=1e-10,
        #   mutation=(0.5, 1.0), recombination=0.7
        #
        # =====================================================================
        
        res = differential_evolution(
            func=objective,
            bounds=bounds,
            # x0=x0,  # Use x0 as one member of initial population
            strategy='randtobest1bin',
            maxiter=1000,
            popsize=POPSIZE,  # Population = 15 * 18 = 270 members
            tol=1e-10,
            mutation=(0.5, 1.25),  # Dithered mutation for better exploration
            recombination=0.65,
            seed=0,
            disp=True,
            callback=de_callback,
            workers=NUM_CPU_CORES,  # Parallel function evaluations
            updating='deferred',  # Required for parallel workers
            polish=True,  # Refine with L-BFGS-B at end
            init='sobol',  # Good initial coverage
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Use 'nelder-mead' or 'differential-evolution'")

    # Print results
    print(f"\nOptimization complete")
    if hasattr(res, 'nfev'):
        print(f"  Function evaluations: {res.nfev}")
    if hasattr(res, 'nit'):
        print(f"  Iterations/generations: {res.nit}")
    print(f"Result: {res.x}")
    print("Final error: ")
    print(objective(np.copy(res.x)))

    return np.copy(res.x)


def calculate_barlat_start_data(args, output_file_dir, df, nrves, 
                                quaternion_file=None, quat_weights_file=None):
    """Calculate starting data with optional orientation loading."""
    rves_dirs, rves_times, rves_load_names = job_directories(args, output_file_dir, df, nrves)

    dfs = {}
    for rve_key in rves_dirs:
        rve_dirs = rves_dirs[rve_key]
        rve_times = rves_times[rve_key]
        rve_load_names = rves_load_names[rve_key]

        sub_dfs = {}
        for temp_key in rve_dirs:
            temp_rve_dirs = rve_dirs[temp_key]
            temp_rve_times = rve_times[temp_key]
            temp_rve_load_names = rve_load_names[temp_key]
            temp_k = int(temp_key)
            sub_dfs[temp_key] = postprocessing_start(
                temp_rve_dirs, temp_k, 
                ftime=temp_rve_times, 
                loading_dir_names=temp_rve_load_names,
                strain_rate=args['strain_rate'][0],
                quaternion_file=quaternion_file,
                quat_weights_file=quat_weights_file
            )

        dfs[rve_key] = sub_dfs.copy()

    return dfs


def optimize_function(start_data_dfs, use_sachs=None, optimizer=OPTIMIZER):
    """Run optimization for all RVEs and temperatures."""
    for rve_key in start_data_dfs:
        print("Starting optimization of RVE name: " + rve_key)
        rve_data = start_data_dfs[rve_key]
        for temp_key in rve_data:
            print("Starting temperature: " + temp_key)
            temp_rve_data = rve_data[temp_key]
            opt_params = optimize_wrapper(temp_rve_data, use_sachs=use_sachs, optimizer=optimizer)
            start_data_dfs[rve_key][temp_key]["barlat_params"] = np.copy(opt_params)
    return start_data_dfs


def postprocessing(args, output_file_dir, df, nrves, quaternion_file=None, 
                   quat_weights_file=None, use_sachs=None, optimizer=OPTIMIZER):
    """Full postprocessing pipeline."""
    start_data_dfs = calculate_barlat_start_data(
        args, output_file_dir, df, nrves,
        quaternion_file=quaternion_file,
        quat_weights_file=quat_weights_file
    )
    data_dfs = optimize_function(start_data_dfs, use_sachs=use_sachs, optimizer=optimizer)
    return data_dfs


def eps_calculations(x, stress_exp, eq_pl_strain, loading_dir_names):
    """Compute EPS for verification."""
    L1 = np.array([
        [0.0,   -1.0,  -1.0,   0.0,  0.0,  0.0],
        [-x[0],  0.0,  -x[1],  0.0,  0.0,  0.0],
        [-x[2], -x[3],  0.0,   0.0,  0.0,  0.0],
        [0.0,    0.0,   0.0,   x[4], 0.0,  0.0],
        [0.0,    0.0,   0.0,   0.0,  x[5], 0.0],
        [0.0,    0.0,   0.0,   0.0,  0.0,  x[6]],
    ], dtype=float)
    L2 = np.array([
        [0.0,   -x[7],  -x[8],  0.0,  0.0,  0.0],
        [-x[9],  0.0,   -x[10], 0.0,  0.0,  0.0],
        [-x[11], -x[12], 0.0,   0.0,  0.0,  0.0],
        [0.0,     0.0,   0.0,   x[13],0.0,  0.0],
        [0.0,     0.0,   0.0,   0.0,  x[14],0.0],
        [0.0,     0.0,   0.0,   0.0,  0.0,  x[15]],
    ], dtype=float)
    L1 = L1.dot(np.asarray(ITEN_2D))
    L2 = L2.dot(np.asarray(ITEN_2D))
    aparam = x[16]
    Y0 = x[17]

    eq_pl_strain_2 = np.zeros(stress_exp.shape[0])
    err = 0.0
    for i in range(stress_exp.shape[0]):
        ys = computeBarlatYieldFunc_np(stress_exp[i,:], L1, L2, aparam)
        eq_pl_strain_2[i] = max(0.0, ((ys / Y0) ** (1.0 / N_DEFAULT) - 1.0) / BETA_DEFAULT)
        err += np.sqrt((eq_pl_strain_2[i]/eq_pl_strain[i] - 1.0)**2)
        print([eq_pl_strain_2[i], eq_pl_strain[i]])
        print([np.sqrt((eq_pl_strain_2[i]/eq_pl_strain[i] - 1.0)**2), loading_dir_names[i]])
        print()
    print([np.mean(eq_pl_strain_2), np.mean(eq_pl_strain), err])


def eps_calculations_sachs(x, stress_exp, eq_pl_strain, loading_dir_names, 
                           quaternions, orientation_weights):
    """Compute Sachs-averaged EPS for verification."""
    L1 = np.array([
        [0.0,   -1.0,  -1.0,   0.0,  0.0,  0.0],
        [-x[0],  0.0,  -x[1],  0.0,  0.0,  0.0],
        [-x[2], -x[3],  0.0,   0.0,  0.0,  0.0],
        [0.0,    0.0,   0.0,   x[4], 0.0,  0.0],
        [0.0,    0.0,   0.0,   0.0,  x[5], 0.0],
        [0.0,    0.0,   0.0,   0.0,  0.0,  x[6]],
    ], dtype=float)
    L2 = np.array([
        [0.0,   -x[7],  -x[8],  0.0,  0.0,  0.0],
        [-x[9],  0.0,   -x[10], 0.0,  0.0,  0.0],
        [-x[11], -x[12], 0.0,   0.0,  0.0,  0.0],
        [0.0,     0.0,   0.0,   x[13],0.0,  0.0],
        [0.0,     0.0,   0.0,   0.0,  x[14],0.0],
        [0.0,     0.0,   0.0,   0.0,  0.0,  x[15]],
    ], dtype=float)
    L1 = L1.dot(np.asarray(ITEN_2D))
    L2 = L2.dot(np.asarray(ITEN_2D))
    aparam = x[16]
    Y0 = x[17]

    n_orient = quaternions.shape[0]
    eq_pl_strain_sachs = np.zeros(stress_exp.shape[0])
    err = 0.0
    
    for i in range(stress_exp.shape[0]):
        eps_orient = np.zeros(n_orient)
        for j in range(n_orient):
            # Rotate stress to crystal frame
            sig_crystal = rotate_stress_sample_to_crystal_np(stress_exp[i, :], quaternions[j, :])
            ys = computeBarlatYieldFunc_np(sig_crystal, L1, L2, aparam)
            eps_orient[j] = max(0.0, ((ys / Y0) ** (1.0 / N_DEFAULT) - 1.0) / BETA_DEFAULT)
        
        # Weighted average
        eq_pl_strain_sachs[i] = np.sum(eps_orient * orientation_weights)
        err += np.sqrt((eq_pl_strain_sachs[i]/eq_pl_strain[i] - 1.0)**2)
        print([eq_pl_strain_sachs[i], eq_pl_strain[i]])
        print([np.sqrt((eq_pl_strain_sachs[i]/eq_pl_strain[i] - 1.0)**2), loading_dir_names[i]])
        print()
    print([np.mean(eq_pl_strain_sachs), np.mean(eq_pl_strain), err])


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    frve_name = "rve_3.17"
    fdiro = os.path.abspath("/Users/carson16/Documents/exaconstit_crusher/workflow_test")
    fdir_rve = os.path.join(fdiro, frve_name, "")

    loading_dir_names = [
        "x_90_z_0", "x_0_y_90", "x_15_y_75", "x_30_y_60", "x_45_y_45", 
        "x_60_y_30", "x_75_y_15", "x_90_y_0", "x_15_z_75", "x_30_z_60",  
        "x_45_z_45", "x_60_z_30", "x_75_z_15", "y_15_z_75", "y_30_z_60",
        "y_45_z_45", "y_60_z_30", "y_75_z_15"
    ]
    loading_dir_shear_names = ["shear_xy", "shear_xz", "shear_yz"]
    loading_dir_names.extend(loading_dir_shear_names)

    tempk = 298.0

    ftime = os.path.join(fdir_rve, "custom_dt_fine2.txt")
    fdir_rves = []
    fdir_times = []
    for iload, load_name in enumerate(loading_dir_names):
        lname = load_name + "_" + str(int(tempk))
        fpath = os.path.join(fdir_rve, lname, "")
        print(fpath)
        fdir_rves.append(fpath)
        fdir_times.append(ftime)
    
    # Optional: path to quaternion file for Sachs averaging
    # Format: one quaternion per line, q0 q1 q2 q3 (q0 is scalar)
    quaternion_file = os.path.join(fdir_rve, "orientations.txt")
    quat_weights_file = os.path.join(fdir_rve, "orientation_weights.txt")  # optional
    
    start_data_dfs = {}
    start_data_dfs[frve_name] = {}
    start_data_dfs[frve_name][str(int(tempk))] = postprocessing_start(
        fdir_rves, tempk, fdir_times,
        quaternion_file=quaternion_file if os.path.exists(quaternion_file) else None,
        quat_weights_file=quat_weights_file if os.path.exists(quat_weights_file) else None
    )

    # Choose solver mode
    SOLVER_MODE = "closed"  # or "newton" or "bisection"
    
    # Choose optimizer: "nelder-mead" or "differential-evolution"
    # differential-evolution uses parallel workers for function evaluations
    OPTIMIZER = "differential-evolution"
    
    # Run optimization (will use Sachs if quaternions were loaded)
    data_dfs = optimize_function(start_data_dfs, optimizer=OPTIMIZER)