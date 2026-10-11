use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, PyResult};
use pyo3::prelude::PyModuleMethods;
 

// use crate::xtal_light_up;
pub(crate) mod math;
use math::*;


#[pyfunction]
#[pyo3(name = "strain_lattice2sample")]
fn strain_lattice2sample<'py>(
    lattice_orientations: &Bound<'_, PyArray3<f64>>,
    strains: &Bound<'_, PyArray3<f64>>,
) -> anyhow::Result<()> {
    let xtal_o = lattice_orientations.readonly();
    let xtal_orientations = xtal_o.as_slice()?;
    let mut xtal_s = strains.readwrite();
    let xtal_strains = xtal_s.as_slice_mut()?;

    // let xtal_ori_chunks = xtal_orientations.chunks_exact(4);
    // let mut xtal_strain_chunks = xtal_strains.chunks_exact_mut(6);
    // zip(xtal_orientations.chunks_exact(4)).
    // |(strain, _quats)|
    // xtal_strains.par_chunks_exact_mut(6).enumerate().for_each(|(i, strain)| { strain[0] = 0.0_f64;
    //   // Zipping quats isn't the most obvious if we wanted chunks of things
    //   // in parallel at least...
    //   // This at least gets us around the compiler issue
    //   let quats : &[f64] = {
    //         let start = i * 4;
    //         let end   = (i + 1) * 4; 
    //         &xtal_orientations[start..end]
    //     };
    // });

    xtal_strains.chunks_exact_mut(6).zip(xtal_orientations.chunks_exact(4)).for_each(|(strain_vec, quat)| {
        rotate_strain_to_sample(quat, strain_vec); 
    });
    Ok(())
}


#[pyfunction]
#[pyo3(name = "calc_taylor_factors")]
fn calc_taylor_factors<'py>(
    shear_rates: &Bound<'_, PyArray3<f64>>,
    taylor_factors: &Bound<'_, PyArray2<f64>>,
    eff_plastic_def_rate: &Bound<'_, PyArray2<f64>>,
) -> anyhow::Result<()> {
    let gamma_dots = shear_rates.readonly();
    let gammadots = gamma_dots.as_slice()?;
    let mut tay_factors = taylor_factors.readwrite();
    let tayfacs = tay_factors.as_slice_mut()?;
    let mut eff_pl_def_rate = eff_plastic_def_rate.readwrite();
    let eps_rate = eff_pl_def_rate.as_slice_mut()?;

    let eps_rate_gdots = eps_rate.iter_mut().zip(gammadots.chunks_exact(12));
    tayfacs.iter_mut().zip(eps_rate_gdots).for_each(|(tayfac, eps_rate_gdot)| {
        (*tayfac, *eps_rate_gdot.0)  = calc_taylor_factor(eps_rate_gdot.1);
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "calc_directional_stiffness")]
fn calc_directional_stiffness<'py>(
    cauchy_stress: &Bound<'_, PyArray3<f64>>,
    sample_strain: &Bound<'_, PyArray3<f64>>,
    directional_stiffness: &Bound<'_, PyArray2<f64>>,
) -> anyhow::Result<()> {

    let cauchy = cauchy_stress.readonly();
    let stress = cauchy.as_slice()?;
    let sstrain = sample_strain.readonly();
    let strain = sstrain.as_slice()?;

    let mut dir_stiffness = directional_stiffness.readwrite();
    let dir_stiff = dir_stiffness.as_slice_mut()?;

    let stress_strains = stress.chunks_exact(6).zip(strain.chunks_exact(6));

    dir_stiff.iter_mut().zip(stress_strains).for_each(|(dir_stiff, stress_strains)| {
        if f64::abs(stress_strains.1[2]) < f64::EPSILON {
            *dir_stiff = 0.0_f64
        }
        else {
            *dir_stiff = stress_strains.0[2] / stress_strains.1[2];
        }
    });
    Ok(())
}

#[pyfunction]
#[pyo3(name = "calc_lattice_strains")]
fn calc_lattice_strains<'py>(
    sample_strains: &Bound<'_, PyArray3<f64>>,
    sample_dir: &Bound<'_, PyArray1<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    lattice_strains: &Bound<'_, PyArray2<f64>>,
    lattice_volumes: &Bound<'_, PyArray2<f64>>,
    per_rank_update: bool
) -> anyhow::Result<()> {
    let num_elems: usize;
    let num_steps: usize;

    let s_in_fibers = in_fibers.readonly();
    {
        let in_fiber_av = s_in_fibers.as_array();
        (_, num_steps, num_elems) = in_fiber_av.dim();
    }

    let in_fiber_hkls = s_in_fibers.as_slice()?;
    let s_strains = sample_strains.readonly();
    let strains = s_strains.as_slice()?;

    let mut s_lattice_strains = lattice_strains.readwrite();
    let lat_strains = s_lattice_strains.as_slice_mut()?;

    let mut s_lattice_vols = lattice_volumes.readwrite();
    let lat_vols = s_lattice_vols.as_slice_mut()?;

    let s_elem_vols = elem_vols.readonly();
    let evs = s_elem_vols.as_slice()?;

    let s_sample_dir = sample_dir.readonly();
    let s_dir = s_sample_dir.as_slice()?;

    let project_vec = [s_dir[0] * s_dir[0], s_dir[1] * s_dir[1] , s_dir[2] * s_dir[2] , 2.0_f64 * s_dir[1] * s_dir[2] , 2.0_f64 * s_dir[0] * s_dir[2] , 2.0_f64 * s_dir[0] * s_dir[1]];

    let lat_strain_in_fiber_iter = lat_strains.chunks_exact_mut(num_steps).zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));

    let lat_vol_strain_in_fiber_iter = lat_vols.chunks_exact_mut(num_steps).zip(lat_strain_in_fiber_iter);

    for (lat_vols_hkl, (lat_strains_hkl, in_fiber_hkl)) in lat_vol_strain_in_fiber_iter.into_iter() {
        let ev_strains_iter = evs.chunks_exact(num_elems).zip(strains.chunks_exact(num_elems * 6));
        let ev_strains_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_strains_iter);

        for (istep, (in_fiber_hkl_step, (evs_step, strains_step))) in ev_strains_in_fiber_iter.enumerate() {
            let (total_lat_vol, inv_total_lat_vol) = calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);

            lat_vols_hkl[istep] += total_lat_vol;

            lat_strains_hkl[istep] += strains_step.chunks_exact(6)
            .zip(evs_step)
            .zip(in_fiber_hkl_step)
            .filter(|ev_strain_in_fiber| *ev_strain_in_fiber.1)
            .map(|((strain, ev), _)| {
                ev * inv_total_lat_vol * dot_prod::<6,f64>(&project_vec, strain)
            })
            .reduce(|lat_strain, lat_strain_elem|
                lat_strain + lat_strain_elem
            ).unwrap_or(0.0_f64);
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "calc_taylor_factors_lattice_fiber")]
fn calc_taylor_factors_lattice_fiber<'py>(
    shear_rates: &Bound<'_, PyArray3<f64>>,
    taylor_factors: &Bound<'_, PyArray2<f64>>,
    eff_plastic_def_rate: &Bound<'_, PyArray2<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    per_rank_update: bool
) -> anyhow::Result<()> {
    let num_elems: usize;
    let num_steps: usize;

    let s_in_fibers = in_fibers.readonly();
    {
        let in_fiber_av = s_in_fibers.as_array();
        (_, num_steps, num_elems) = in_fiber_av.dim();
    }
    let in_fiber_hkls = s_in_fibers.as_slice()?;
    let s_elem_vols = elem_vols.readonly();
    let evs = s_elem_vols.as_slice()?;

    let gamma_dots = shear_rates.readonly();
    let gammadots = gamma_dots.as_slice()?;
    let mut tay_factors = taylor_factors.readwrite();
    let tayfacs = tay_factors.as_slice_mut()?;
    let mut eff_pl_def_rate = eff_plastic_def_rate.readwrite();
    let eps_rate = eff_pl_def_rate.as_slice_mut()?;

    let lat_eps_in_fiber_iter = eps_rate.chunks_exact_mut(num_steps).zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));
    let ltf_leps_in_fiber_iter = tayfacs.chunks_exact_mut(num_steps).zip(lat_eps_in_fiber_iter);

    for (tay_fac_hkl, (eps_rate_hkl, in_fiber_hkl)) in ltf_leps_in_fiber_iter.into_iter() {
        let ev_gdot_iter = evs.chunks_exact(num_elems).zip(gammadots.chunks_exact(num_elems * 12));
        let ev_gdot_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_gdot_iter);
        for (istep, (in_fiber_hkl_step, (evs_step, gdots_step))) in ev_gdot_in_fiber_iter.enumerate() {
            let (_, inv_total_lat_vol) = calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);

            let (tay_fac_hkl_step, eps_rate_hkl_step) = gdots_step.chunks_exact(12)
            .zip(evs_step)
            .zip(in_fiber_hkl_step)
            .filter(|ev_gdot_in_fiber| *ev_gdot_in_fiber.1)
            .map(|((gdot, ev), _)| {
                let (tf, deps) = calc_taylor_factor(gdot);
                (ev * inv_total_lat_vol * tf, ev * inv_total_lat_vol * deps)
            })
            .reduce(|tfdeps, tfdeps_elem|
                (tfdeps.0 + tfdeps_elem.0, tfdeps.1 + tfdeps_elem.1)
            ).unwrap_or((0.0_f64, 0.0_f64));
            tay_fac_hkl[istep] += tay_fac_hkl_step;
            eps_rate_hkl[istep] += eps_rate_hkl_step;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "calc_directional_stiffness_lattice_fiber")]
fn calc_directional_stiffness_lattice_fiber<'py>(
    cauchy_stress: &Bound<'_, PyArray3<f64>>,
    sample_strain: &Bound<'_, PyArray3<f64>>,
    directional_stiffness: &Bound<'_, PyArray2<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    per_rank_update: bool
) -> anyhow::Result<()> {
    let num_elems: usize;
    let num_steps: usize;

    let s_in_fibers = in_fibers.readonly();
    {
        let in_fiber_av = s_in_fibers.as_array();
        (_, num_steps, num_elems) = in_fiber_av.dim();
    }
    let in_fiber_hkls = s_in_fibers.as_slice()?;
    let s_elem_vols = elem_vols.readonly();
    let evs = s_elem_vols.as_slice()?;

    let cauchy = cauchy_stress.readonly();
    let stress = cauchy.as_slice()?;
    let sstrain = sample_strain.readonly();
    let strain = sstrain.as_slice()?;

    let mut dir_stiffness = directional_stiffness.readwrite();
    let dir_stiff = dir_stiffness.as_slice_mut()?;

    let lat_ds_in_fiber_iter = dir_stiff.chunks_exact_mut(num_steps).zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));

    for (dir_stiff_hkl, in_fiber_hkl) in lat_ds_in_fiber_iter.into_iter() {
        let stress_strains = stress.chunks_exact(num_elems * 6).zip(strain.chunks_exact(num_elems * 6));
        let ev_sst_iter = evs.chunks_exact(num_elems).zip(stress_strains);
        let ev_sst_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_sst_iter);
        for (istep, (in_fiber_hkl_step, (evs_step, (stress_step, strain_step)))) in ev_sst_in_fiber_iter.enumerate() {
            let (_, inv_total_lat_vol) = calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);

            dir_stiff_hkl[istep] += stress_step.chunks_exact(6)
            .zip(strain_step.chunks_exact(6))
            .zip(evs_step)
            .zip(in_fiber_hkl_step)
            .filter(|ev_sst_in_fiber| *ev_sst_in_fiber.1)
            .map(|((sst, ev), _)| {
                if f64::abs(sst.1[2]) < f64::EPSILON {
                    0.0_f64
                }
                else {
                    (sst.0[2] / sst.1[2]) * ev * inv_total_lat_vol
                }
            })
            .reduce(|dir_modulus, dir_modulus_elem|
                dir_modulus + dir_modulus_elem
            ).unwrap_or(0.0_f64);
        }
    }

    Ok(())
}

#[pyfunction]
#[pyo3(name = "calc_within_fibers")]
fn calc_within_fibers<'py>(
    lattice_orientations: &Bound<'_, PyArray3<f64>>,
    sample_dir: &Bound<'_, PyArray1<f64>>,
    hkls: &Bound<'_, PyArray2<f64>>,
    lattice_param_a: f64,
    distance_tolerance_rad: f64,
    in_fibers: &Bound<'_, PyArray3<bool>>,
) -> anyhow::Result<()> {

    let num_elems: usize;
    let num_steps: usize;

    let mut s_in_fibers = in_fibers.readwrite();
    {
        let in_fiber_av = s_in_fibers.as_array();
        (_, num_steps, num_elems) = in_fiber_av.dim();
    }

    let in_fiber_hkls = s_in_fibers.as_slice_mut()?;

    let xtal_o = lattice_orientations.readonly();
    let xtal_orientations = xtal_o.as_slice()?;

    let s_hkls = hkls.readonly();
    let slice_hkls = s_hkls.as_slice()?;

    let s_sample_dir = sample_dir.readonly();
    let s_dir = s_sample_dir.as_slice()?;

    in_fiber_hkls.chunks_exact_mut(num_steps * num_elems)
    .zip(slice_hkls.chunks_exact(3))
    .for_each(|(in_fibers, hkl)| {
        calculate_in_fibers(lattice_param_a, hkl, s_dir, xtal_orientations, distance_tolerance_rad, in_fibers);
    });

    Ok(())
}

#[pymodule]
fn xtal_light_up(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(strain_lattice2sample, m)?)?;
    m.add_function(wrap_pyfunction!(calc_taylor_factors, m)?)?;
    m.add_function(wrap_pyfunction!(calc_directional_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(calc_lattice_strains, m)?)?;
    m.add_function(wrap_pyfunction!(calc_taylor_factors_lattice_fiber, m)?)?;
    m.add_function(wrap_pyfunction!(calc_directional_stiffness_lattice_fiber, m)?)?;
    m.add_function(wrap_pyfunction!(calc_within_fibers, m)?)?;
    Ok(())
}

#[inline(always)]
fn calc_volume_terms_fiber(evs_step: &[f64],
                           in_fiber_hkl_step: &[bool],
                           per_rank_update: bool
                          ) -> (f64, f64) {
    let total_lat_vol = evs_step.into_iter()
                        .zip(in_fiber_hkl_step)
                        .filter(|ev_in_fiber| *ev_in_fiber.1)
                        .map(|(ev, _)| *ev)
                        .reduce(|ev_total, ev| {
                            ev_total + ev
                        }).unwrap_or(0.0_f64);

    let inv_total_lat_vol = 
    if per_rank_update {
        1.0_f64
    } else {
        if total_lat_vol > f64::EPSILON {
            1.0_f64 / total_lat_vol
        } else {
            0.0_f64
        }
    };
    (total_lat_vol, inv_total_lat_vol)
}

#[inline(always)]
fn calculate_in_fibers(lparam_a: f64,
                       hkl: &[f64],
                       s_dir: &[f64],
                       quats: &[f64],
                       distance_tolerance: f64,
                       in_fiber: &mut [bool]
) {
    // Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
    let lat_vec_ops_b = compute_lattice_b_param_cubic(lparam_a);

    // compute crystal direction from planeData
    let c_dir = {
        let mut tmp_cdir = [0.0_f64; 3];
        mat_t_vec_mult::<3, 3, f64>(&lat_vec_ops_b, &hkl, &mut tmp_cdir);
        tmp_cdir
    };

    let symm_quat = symmetric_cubic_quaternions();

    within_fiber::<24>(&c_dir, s_dir, quats, &symm_quat, distance_tolerance, in_fiber);
}

#[inline(always)]
fn quat2rmat(quat: &[f64]) -> [[f64; 3]; 3] {
    assert!(quat.len() >= 4);
    let qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    let mut rmat = [[0.0_f64; 3]; 3];

    rmat[0][0] = qbar + 2.0_f64 * quat[1] * quat[1];
    rmat[1][0] = 2.0_f64 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0_f64 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0_f64 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0_f64 * quat[2] * quat[2];
    rmat[2][1] = 2.0_f64 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0_f64 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0_f64 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0_f64 * quat[3] * quat[3];

    rmat
}

#[inline(always)]
fn rotate_strain_to_sample(quat: &[f64], strain_vec: &mut[f64]) {
    assert!(quat.len() >= 4);
    assert!(strain_vec.len() >= 6);
    let rmat = quat2rmat(quat);
    let strain = 
    {
        let mut strain = [[0.0_f64; 3]; 3];

        strain[0][0] = strain_vec[0];
        strain[1][1] = strain_vec[1];
        strain[2][2] = strain_vec[2];
        strain[1][2] = strain_vec[3]; strain[2][1] = strain[1][2];
        strain[0][2] = strain_vec[4]; strain[2][0] = strain[0][2];
        strain[0][1] = strain_vec[5]; strain[1][0] = strain[0][1];

        strain
    };

    {
        let mut strain_samp = [[0.0_f64; 3]; 3];
        rotate_matrix::<3, false, f64>(&rmat, &strain, &mut strain_samp);

        strain_vec[0] = strain_samp[0][0];
        strain_vec[1] = strain_samp[1][1];
        strain_vec[2] = strain_samp[2][2];
        strain_vec[3] = strain_samp[1][2];
        strain_vec[4] = strain_samp[0][2];
        strain_vec[5] = strain_samp[0][1];
    }
}

#[inline(always)]
fn calc_taylor_factor(gdots: &[f64]) -> (f64, f64) {
    assert!(gdots.len() >= 12);
    let symm_schmid = calculate_fcc_symm_schmid_tensor();
    let mut plastic_def_rate_vec = [0.0_f64; 6];
    mat_t_vec_mult::<12, 6, f64>(&symm_schmid, gdots, &mut plastic_def_rate_vec);

    let eff_plastic_def_rate = { 
        let norm_vec = norm::<6, f64>(&plastic_def_rate_vec);
        norm_vec * f64::sqrt(2.0_f64 / 3.0_f64) 
    };

    if eff_plastic_def_rate <= f64::EPSILON {
        return (0.0_f64, 0.0)
    }

    let abs_sum_shear_rate = {
        let mut sum = 0.0_f64;
        for gdot in gdots.iter() {
            sum += f64::abs(*gdot);
        }
        sum
    };

    (abs_sum_shear_rate / eff_plastic_def_rate, eff_plastic_def_rate)
}

#[inline(always)]
fn calculate_fcc_symm_schmid_tensor() -> [[f64; 6]; 12] {
    let two = 2.0_f64;
    let three = 3.0_f64;
    let sqrt_3i = 1.0_f64 / f64::sqrt(three);
    let sqrt_2i = 1.0 / f64::sqrt(two);

    let slip_direction = [
        [sqrt_3i, sqrt_3i, sqrt_3i],
        [sqrt_3i, sqrt_3i, sqrt_3i],
        [sqrt_3i, sqrt_3i, sqrt_3i],
        [-sqrt_3i, sqrt_3i, sqrt_3i],
        [-sqrt_3i, sqrt_3i, sqrt_3i],
        [-sqrt_3i, sqrt_3i, sqrt_3i],
        [-sqrt_3i, -sqrt_3i, sqrt_3i],
        [-sqrt_3i, -sqrt_3i, sqrt_3i],
        [-sqrt_3i, -sqrt_3i, sqrt_3i],
        [sqrt_3i, -sqrt_3i, sqrt_3i],
        [sqrt_3i, -sqrt_3i, sqrt_3i],
        [sqrt_3i, -sqrt_3i, sqrt_3i],
    ];

    let slip_plane_normal = [
        [0.0_f64, sqrt_2i, -sqrt_2i],
        [-sqrt_2i, 0.0_f64, sqrt_2i],
        [sqrt_2i, -sqrt_2i, 0.0_f64],
        [-sqrt_2i, 0.0_f64, -sqrt_2i],
        [0.0_f64, -sqrt_2i, sqrt_2i],
        [sqrt_2i, sqrt_2i, 0.0_f64],
        [0.0_f64, -sqrt_2i, -sqrt_2i],
        [sqrt_2i, 0.0_f64, sqrt_2i],
        [-sqrt_2i, sqrt_2i, 0.0_f64],
        [sqrt_2i, 0.0_f64, -sqrt_2i],
        [0.0_f64, sqrt_2i, sqrt_2i],
        [-sqrt_2i, -sqrt_2i, 0.0_f64],
    ];

    let mut symm_schmid = [[0.0_f64; 6]; 12];
    let mut skw_schmid = [[0.0_f64; 3]; 12];

    calculate_schmid_tensor::<12>(
        &slip_plane_normal,
        &slip_direction,
        &mut symm_schmid,
        &mut skw_schmid,
    );

    symm_schmid
}

fn calculate_schmid_tensor<const NSLIP: usize>(
    slip_plane_normal: &[[f64; 3]],
    slip_direction: &[[f64; 3]],
    symm_schmid: &mut [[f64; 6]],
    skw_schmid: &mut [[f64; 3]],
)
{
    assert!(slip_plane_normal.len() >= NSLIP);
    assert!(slip_direction.len() >= NSLIP);
    assert!(symm_schmid.len() >= NSLIP);
    assert!(skw_schmid.len() >= NSLIP);

    let one_half = 0.5_f64;
    let sqrt_2i = 1.0 / f64::sqrt(2.0_f64);

    let mut schmid = [[0.0_f64; 3]; 3];

    for islip in 0..NSLIP {
        outer_prod::<3, 3, f64>(
            &slip_direction[islip],
            &slip_plane_normal[islip],
            &mut schmid,
        );

        // Replace with an inline set of functions
        // skew(schmid)[2, 1] = 1/2 * (schmid[2, 1] - schmid[1, 2])
        skw_schmid[islip][0] = one_half * (schmid[2][1] - schmid[1][2]);
        // skew(schmid)[0, 2] = 1/2 * (schmid[0, 2] - schmid[2, 0])
        skw_schmid[islip][1] = one_half * (schmid[0][2] - schmid[2][0]);
        // skew(schmid)[1, 0] = 1/2 * (schmid[1, 0] - schmid[0, 1])
        skw_schmid[islip][2] = one_half * (schmid[1][0] - schmid[0][1]);

        // Replace with an inline set of functions
        // sym(schmid)[0, 0] = 1/2 * (schmid[0, 0] + schmid[0, 0])
        symm_schmid[islip][0] = schmid[0][0];
        // sym(schmid)[1, 1] = 1/2 * (schmid[1, 1] + schmid[1, 1])
        symm_schmid[islip][1] = schmid[1][1];
        // skew(schmid)[2, 2] = 1/2 * (schmid[2, 2] + schmid[2, 2])
        symm_schmid[islip][2] = schmid[2][2];
        // sym(schmid)[1, 2] = 1/2 * (schmid[1, 2] + schmid[2, 1])
        // For consistent dot products replace with sqrt(2)/2 = 1/sqrt(2)
        symm_schmid[islip][3] = sqrt_2i * (schmid[2][1] + schmid[1][2]);
        // sym(schmid)[0, 2] = 1/2 * (schmid[0, 2] + schmid[2, 0])
        symm_schmid[islip][4] = sqrt_2i * (schmid[0][2] + schmid[2][0]);
        // sym(schmid)[0, 1] = 1/2 * (schmid[0, 1] + schmid[1, 0])
        symm_schmid[islip][5] = sqrt_2i * (schmid[1][0] + schmid[0][1]);
    }
}

/// Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
#[inline(always)]
fn compute_lattice_b_param_cubic(lparam_a: f64) -> [[f64; 3]; 3] {
    let deg90 = std::f64::consts::PI / 2.0_f64;
    let cellparms = [lparam_a, lparam_a, lparam_a, deg90, deg90, deg90];

    let alfa = cellparms[3];
    let beta = cellparms[4];
    let gamma = cellparms[5];

    let cosalfar = (f64::cos(beta) * f64::cos(gamma) - f64::cos(alfa)) / (f64::sin(beta) * f64::sin(gamma));
    let sinalfar = f64::sqrt(1.0_f64 - cosalfar * cosalfar);

    let a = [cellparms[0], 0.0_f64, 0.0_f64];
    let b = [cellparms[1] * f64::cos(gamma), cellparms[1] * f64::sin(gamma), 0.0_f64];
    let c = [cellparms[2] * f64::cos(beta), -cellparms[2] * cosalfar * f64::sin(beta), cellparms[2] * sinalfar * f64::sin(beta)];

    // Cell volume
    let inv_vol = {
        let v_temp = cross_prod(&b, &c);
        1.0 / dot_prod::<3, f64>(&a, &v_temp)
    };

    // Reciprocal lattice vectors
    let cross_prod_inv_v = |vec1: &[f64], vec2: &[f64], inv_vol: f64| -> [f64; 3] 
    {
        let mut tmp = cross_prod(vec1, vec2);
        tmp[0] *= inv_vol;
        tmp[1] *= inv_vol;
        tmp[2] *= inv_vol;
        tmp
    };

    let astar = cross_prod_inv_v(&b, &c, inv_vol);
    let bstar = cross_prod_inv_v(&c, &a, inv_vol);
    let cstar = cross_prod_inv_v(&a, &b, inv_vol);

    // B takes components in the reciprocal lattice to X
    [astar, bstar, cstar]
}

#[inline(always)]
fn symmetric_cubic_quaternions() -> [[f64; 4]; 24] {
    let angle_axis_symm = [
            [0.0_f64,       1.0_f64,    0.0_f64,    0.0_f64],  // identity
            [std::f64::consts::FRAC_PI_2,     1.0_f64,    0.0_f64,    0.0_f64],  // fourfold about   1  0  0 (x1)
            [std::f64::consts::PI,        1.0_f64,    0.0_f64,    0.0_f64],  //
            [std::f64::consts::FRAC_PI_2 * 3.0_f64,   1.0_f64,    0.0_f64,    0.0_f64],  //
            [std::f64::consts::FRAC_PI_2,     0.0_f64,    1.0_f64,    0.0_f64],  // fourfold about   0  1  0 (x2)
            [std::f64::consts::PI,        0.0_f64,    1.0_f64,    0.0_f64],  //
            [std::f64::consts::FRAC_PI_2 * 3.0_f64,   0.0_f64,    1.0_f64,    0.0_f64],  //
            [std::f64::consts::FRAC_PI_2,     0.0_f64,    0.0_f64,    1.0_f64],  // fourfold about   0  0  1 (x3)
            [std::f64::consts::PI,        0.0_f64,    0.0_f64,    1.0_f64],  //
            [std::f64::consts::FRAC_PI_2 * 3.0_f64,   0.0_f64,    0.0_f64,    1.0_f64],  //
            [std::f64::consts::FRAC_PI_3 * 2.0_f64,   1.0_f64,    1.0_f64,    1.0_f64],  // threefold about  1  1  1
            [std::f64::consts::FRAC_PI_3 * 4.0_f64,   1.0_f64,    1.0_f64,    1.0_f64],  //
            [std::f64::consts::FRAC_PI_3 * 2.0_f64,  -1.0_f64,    1.0_f64,    1.0_f64],  // threefold about -1  1  1
            [std::f64::consts::FRAC_PI_3 * 4.0_f64,  -1.0_f64,    1.0_f64,    1.0_f64],  //
            [std::f64::consts::FRAC_PI_3 * 2.0_f64,  -1.0_f64,   -1.0_f64,    1.0_f64],  // threefold about -1 -1  1
            [std::f64::consts::FRAC_PI_3 * 4.0_f64,  -1.0_f64,   -1.0_f64,    1.0_f64],  //
            [std::f64::consts::FRAC_PI_3 * 2.0_f64,   1.0_f64,   -1.0_f64,    1.0_f64],  // threefold about  1 -1  1
            [std::f64::consts::FRAC_PI_3 * 4.0_f64,   1.0_f64,   -1.0_f64,    1.0_f64],  //
            [std::f64::consts::PI,        1.0_f64,    1.0_f64,    0.0_f64],  // twofold about    1  1  0
            [std::f64::consts::PI,       -1.0_f64,    1.0_f64,    0.0_f64],  // twofold about   -1  1  0
            [std::f64::consts::PI,        1.0_f64,    0.0_f64,    1.0_f64],  // twofold about    1  0  1
            [std::f64::consts::PI,        0.0_f64,    1.0_f64,    1.0_f64],  // twofold about    0  1  1
            [std::f64::consts::PI,       -1.0_f64,    0.0_f64,    1.0_f64],  // twofold about   -1  0  1
            [std::f64::consts::PI,        0.0_f64,   -1.0_f64,    1.0_f64],  // twofold about    0 -1  1
    ];

    let inv2 = 1.0_f64 / 2.0_f64;
    let mut quat_symm = [[0.0_f64; 4]; 24];
    quat_symm.iter_mut()
    .zip(angle_axis_symm)
    .for_each(|(quat, ang_axis)| {
        let s = f64::sin(inv2 * ang_axis[0]); 
        quat[0] = f64::cos(inv2 * ang_axis[0]);
        let mut inv_norm_axis = 1.0 / norm::<3, f64>(&ang_axis[1..4]);
        quat[1] = s * ang_axis[1] * inv_norm_axis;
        quat[2] = s * ang_axis[2] * inv_norm_axis;
        quat[3] = s * ang_axis[3] * inv_norm_axis;

        inv_norm_axis = 1.0_f64;
        if quat[0] < 0.0 {
            inv_norm_axis *= -1.0_f64;
        }

        quat[0] *= inv_norm_axis;
        quat[1] *= inv_norm_axis;
        quat[2] *= inv_norm_axis;
        quat[3] *= inv_norm_axis;
    });

    quat_symm
}

/// Returns all that aren't 
// #[inline(always)]
// fn find_unique_tolerance<const SYM_LEN: usize>(rmat_fr_qsym_c_dir: &[[f64; 3]], tolerance: f64) -> Vec<[f64; 3]> {
//     rmat_fr_qsym_c_dir.to_vec()
// }

#[inline(always)]
fn within_fiber<const SYM_LEN: usize>(c_dir: &[f64],
                                      s_dir: &[f64],
                                      quats: &[f64],
                                      symm_quat: &[[f64; 4]],
                                      distance_tolerance: f64,
                                      in_fibers: &mut [bool]) {

    assert!(c_dir.len() >= 3);
    assert!(s_dir.len() >= 3);
    assert!(quats.len() >= 4);
    assert!(symm_quat.len() >= SYM_LEN);

    assert!(in_fibers.len() == (quats.len() / 4));

    let c = {
        let inv_c_norm = 1.0 / norm::<3, f64>(c_dir);
        [c_dir[0] * inv_c_norm, c_dir[1] * inv_c_norm, c_dir[2] * inv_c_norm]
    };

    let s = {
        let inv_s_norm = 1.0 / norm::<3, f64>(s_dir);
        [s_dir[0] * inv_s_norm, s_dir[1] * inv_s_norm, s_dir[2] * inv_s_norm]
    };

    // Could maybe move this over to a vec if we want this to be easily generic over a ton of symmetry conditions...
    let mut rmat_fr_qsym_c_dir = [[0.0_f64; 3]; SYM_LEN];
    rmat_fr_qsym_c_dir.iter_mut()
    .zip(symm_quat)
    .for_each(|(prod, quat)| {
        let rmat = quat2rmat(quat);
        // Might need to make this the transpose...
        mat_t_vec_mult::<3, 3, f64>(&rmat, &c, prod);
    });

    // If we really wanted to we could lower try and calculate the elements
    // that aren't unique here but that's not worth the effort at all given
    // how fast things are
    // let c_syms: Vec<[f64; 3]> = find_unique_tolerance::<SYM_LEN>(&rmat_fr_qsym_c_dir, f64::sqrt(f64::EPSILON));

    let s_rmat_csym_prod: Vec<f64> = quats.chunks_exact(4).map(|quat| {
        let sine = rmat_fr_qsym_c_dir.iter().map(|c_sym| {
            let rmat = quat2rmat(quat);
            let mut prod = [0.0_f64; 3];
            // Might need to make this the transpose...
            mat_vec_mult::<3, 3, f64>(&rmat, c_sym, &mut prod);
            dot_prod::<3, f64>(&s, &prod)
        })
        .fold(std::f64::MIN, |a, b| a.max(b));
        sine
    }).collect();

    in_fibers.iter_mut()
    .zip(s_rmat_csym_prod)
    .for_each(|(in_fiber, sine)| {
        let sine_safe = 
        {
            if f64::abs(sine) > 1.00000001 {
                sine.signum()
            }
            else {
                sine
            }
        };
        
        let distance = f64::acos(sine_safe);
        *in_fiber = distance <= distance_tolerance;
    })
}