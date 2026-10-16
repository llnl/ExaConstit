//! Python bindings for computing lattice strain / "light-up" quantities from
//! ExaConstit results, mirroring ExaConstit's in-situ light-up calculations
//! (`LightUp` in src/postprocessing/mechanics_lightup.cpp) so the two produce the
//! same results for every supported crystal system.
//!
//! The usual workflow (see light_up_py/fiber_calcs_rank.py) is:
//! 1. calc_within_fibers to flag which elements are in each HKL fiber
//! 2. calc_lattice_strains, calc_directional_stiffness_lattice_fiber and
//!    calc_taylor_factors_lattice_fiber to get the fiber (volume) averaged values
//!
//! The non-fiber versions (calc_directional_stiffness, calc_taylor_factors) give
//! the same per-element quantities without any fiber averaging.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::PyModuleMethods;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, Bound, FromPyObject, PyResult};

pub(crate) mod lattice;
pub(crate) mod math;
use lattice::{compute_lattice_b_param, LatticeType};
use math::*;

/// Rotates strains from the lattice (crystal) frame into the sample frame in place.
///
/// Arguments:
/// - lattice_orientations: (steps, elems, 4) quaternions for each element
/// - strains: (steps, elems, 6) lattice frame strains, overwritten with the sample
///   frame strains
///
/// ExaConstit v0.9+ already writes its elastic strains in the sample frame, so this
/// is only needed for older results.
#[pyfunction]
#[pyo3(name = "strain_lattice2sample")]
fn strain_lattice2sample(
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

    xtal_strains
        .as_chunks_mut::<6>()
        .0
        .iter_mut()
        .zip(xtal_orientations.as_chunks::<4>().0)
        .for_each(|(strain_vec, quat)| {
            rotate_strain_to_sample(quat, strain_vec);
        });
    Ok(())
}

/// Taylor factor for every element: sum over the slip systems of |shear rate|
/// divided by the equivalent plastic strain rate, the same way ExaConstit's in-situ
/// light-up computes it. Elements with an equivalent plastic strain rate of
/// essentially zero (<= 1e-14) get a Taylor factor of 0.
///
/// Arguments:
/// - shear_rates: (steps, elems, nslip) slip system shear rates ("Shearing Rate"
///   output); any number of slip systems works
/// - eq_pl_strain_rate: (steps, elems) ExaCMech's equivalent plastic strain rate
///   ("Equivalent Plastic Strain Rate" output)
/// - taylor_factors: (steps, elems) output, overwritten
#[pyfunction]
#[pyo3(name = "calc_taylor_factors")]
fn calc_taylor_factors(
    shear_rates: &Bound<'_, PyArray3<f64>>,
    eq_pl_strain_rate: &Bound<'_, PyArray2<f64>>,
    taylor_factors: &Bound<'_, PyArray2<f64>>,
) -> anyhow::Result<()> {
    let num_slip = shear_rates.readonly().as_array().dim().2;
    let gamma_dots = shear_rates.readonly();
    let gammadots = gamma_dots.as_slice()?;
    let eq_pl_rate = eq_pl_strain_rate.readonly();
    let dpeffs = eq_pl_rate.as_slice()?;
    let mut tay_factors = taylor_factors.readwrite();
    let tayfacs = tay_factors.as_slice_mut()?;

    check_taylor_inputs(num_slip, gammadots.len(), dpeffs.len(), tayfacs.len())?;

    tayfacs
        .iter_mut()
        .zip(gammadots.chunks_exact(num_slip))
        .zip(dpeffs)
        .for_each(|((tayfac, gdots), dpeff)| {
            *tayfac = calc_taylor_factor(gdots, *dpeff);
        });
    Ok(())
}

/// Directional stiffness for every element: the sample frame 33 (z) Cauchy stress
/// divided by the sample frame 33 elastic strain. Elements whose 33 strain is
/// essentially zero (< machine epsilon) get a value of 0.
///
/// Arguments:
/// - cauchy_stress: (steps, elems, 6) sample frame Cauchy stress
/// - sample_strain: (steps, elems, 6) sample frame elastic strain for the same steps
/// - directional_stiffness: (steps, elems) output, overwritten
#[pyfunction]
#[pyo3(name = "calc_directional_stiffness")]
fn calc_directional_stiffness(
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

    let stress_strains = stress
        .as_chunks::<6>()
        .0
        .iter()
        .zip(strain.as_chunks::<6>().0);

    dir_stiff
        .iter_mut()
        .zip(stress_strains)
        .for_each(|(dir_stiff, stress_strains)| {
            if f64::abs(stress_strains.1[2]) < f64::EPSILON {
                *dir_stiff = 0.0_f64
            } else {
                *dir_stiff = stress_strains.0[2] / stress_strains.1[2];
            }
        });
    Ok(())
}

/// Lattice strain for each HKL fiber: the sample frame elastic strain projected onto
/// the sample direction (s . strain . s), volume weighted over the elements in the
/// fiber.
///
/// Arguments:
/// - sample_strains: (steps, elems, 6) sample frame elastic strains
/// - sample_dir: (3,) sample direction, normalized internally
/// - elem_vols: (steps, elems) element volumes
/// - in_fibers: (nhkl, steps, elems) from calc_within_fibers
/// - lattice_strains: (nhkl, steps) output, added to
/// - lattice_volumes: (nhkl, steps) output, added to; the total in-fiber volume
/// - per_rank_update: if true the volume weighted sums are added to the outputs
///   without normalizing, so results from several ExaConstit ranks / data blocks can
///   be accumulated, and you divide lattice_strains by lattice_volumes once at the
///   end. If false each call adds the volume average for just its own elements.
#[pyfunction]
#[pyo3(name = "calc_lattice_strains")]
fn calc_lattice_strains(
    sample_strains: &Bound<'_, PyArray3<f64>>,
    sample_dir: &Bound<'_, PyArray1<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    lattice_strains: &Bound<'_, PyArray2<f64>>,
    lattice_volumes: &Bound<'_, PyArray2<f64>>,
    per_rank_update: bool,
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

    let project_vec = [
        s_dir[0] * s_dir[0],
        s_dir[1] * s_dir[1],
        s_dir[2] * s_dir[2],
        2.0_f64 * s_dir[1] * s_dir[2],
        2.0_f64 * s_dir[0] * s_dir[2],
        2.0_f64 * s_dir[0] * s_dir[1],
    ];

    let lat_strain_in_fiber_iter = lat_strains
        .chunks_exact_mut(num_steps)
        .zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));

    let lat_vol_strain_in_fiber_iter = lat_vols
        .chunks_exact_mut(num_steps)
        .zip(lat_strain_in_fiber_iter);

    for (lat_vols_hkl, (lat_strains_hkl, in_fiber_hkl)) in lat_vol_strain_in_fiber_iter.into_iter()
    {
        let ev_strains_iter = evs
            .chunks_exact(num_elems)
            .zip(strains.chunks_exact(num_elems * 6));
        let ev_strains_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_strains_iter);

        for (istep, (in_fiber_hkl_step, (evs_step, strains_step))) in
            ev_strains_in_fiber_iter.enumerate()
        {
            let (total_lat_vol, inv_total_lat_vol) =
                calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);

            lat_vols_hkl[istep] += total_lat_vol;

            lat_strains_hkl[istep] += strains_step
                .as_chunks::<6>()
                .0
                .iter()
                .zip(evs_step)
                .zip(in_fiber_hkl_step)
                .filter(|ev_strain_in_fiber| *ev_strain_in_fiber.1)
                .map(|((strain, ev), _)| {
                    ev * inv_total_lat_vol * dot_prod::<6, f64>(&project_vec, strain)
                })
                .reduce(|lat_strain, lat_strain_elem| lat_strain + lat_strain_elem)
                .unwrap_or(0.0_f64);
        }
    }
    Ok(())
}

/// Volume weighted Taylor factor and equivalent plastic strain rate for each HKL
/// fiber, mirroring LightUp::CalcLatticeTaylorFactorDpeff in ExaConstit. See
/// calc_taylor_factors for how the per-element Taylor factor is computed.
///
/// Arguments:
/// - shear_rates: (steps, elems, nslip) slip system shear rates; any number of slip
///   systems works
/// - eq_pl_strain_rate: (steps, elems) ExaCMech's equivalent plastic strain rate
/// - taylor_factors: (nhkl, steps) output, added to
/// - lattice_eq_pl_strain_rate: (nhkl, steps) output, added to
/// - elem_vols: (steps, elems) element volumes for the same steps
/// - in_fibers: (nhkl, steps, elems) from calc_within_fibers for the same steps
/// - per_rank_update: see calc_lattice_strains; when true divide the outputs by the
///   in-fiber volumes (lattice_volumes) once everything has been accumulated
#[pyfunction]
#[pyo3(name = "calc_taylor_factors_lattice_fiber")]
fn calc_taylor_factors_lattice_fiber(
    shear_rates: &Bound<'_, PyArray3<f64>>,
    eq_pl_strain_rate: &Bound<'_, PyArray2<f64>>,
    taylor_factors: &Bound<'_, PyArray2<f64>>,
    lattice_eq_pl_strain_rate: &Bound<'_, PyArray2<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    per_rank_update: bool,
) -> anyhow::Result<()> {
    let num_slip = shear_rates.readonly().as_array().dim().2;
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
    let mut lat_eq_pl_rate = lattice_eq_pl_strain_rate.readwrite();
    let eps_rate = lat_eq_pl_rate.as_slice_mut()?;
    let eq_pl_rate = eq_pl_strain_rate.readonly();
    let dpeffs = eq_pl_rate.as_slice()?;

    check_taylor_inputs(num_slip, gammadots.len(), dpeffs.len(), evs.len())?;

    let lat_eps_in_fiber_iter = eps_rate
        .chunks_exact_mut(num_steps)
        .zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));
    let ltf_leps_in_fiber_iter = tayfacs
        .chunks_exact_mut(num_steps)
        .zip(lat_eps_in_fiber_iter);

    for (tay_fac_hkl, (eps_rate_hkl, in_fiber_hkl)) in ltf_leps_in_fiber_iter.into_iter() {
        let ev_gdot_iter = evs
            .chunks_exact(num_elems)
            .zip(gammadots.chunks_exact(num_elems * num_slip));
        let ev_gdot_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_gdot_iter);
        for (istep, (in_fiber_hkl_step, (evs_step, gdots_step))) in
            ev_gdot_in_fiber_iter.enumerate()
        {
            let (_, inv_total_lat_vol) =
                calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);
            let dpeff_step = &dpeffs[istep * num_elems..(istep + 1) * num_elems];

            let (tay_fac_hkl_step, eps_rate_hkl_step) = gdots_step
                .chunks_exact(num_slip)
                .zip(evs_step)
                .zip(in_fiber_hkl_step)
                .enumerate()
                .filter(|(_, ev_gdot_in_fiber)| *ev_gdot_in_fiber.1)
                .map(|(ielem, ((gdot, ev), _))| {
                    let deps = dpeff_step[ielem];
                    let tf = calc_taylor_factor(gdot, deps);
                    (ev * inv_total_lat_vol * tf, ev * inv_total_lat_vol * deps)
                })
                .reduce(|tfdeps, tfdeps_elem| (tfdeps.0 + tfdeps_elem.0, tfdeps.1 + tfdeps_elem.1))
                .unwrap_or((0.0_f64, 0.0_f64));
            tay_fac_hkl[istep] += tay_fac_hkl_step;
            eps_rate_hkl[istep] += eps_rate_hkl_step;
        }
    }
    Ok(())
}

/// Volume weighted directional stiffness (sample frame 33 Cauchy stress / 33
/// elastic strain, see calc_directional_stiffness) for each HKL fiber.
///
/// Arguments:
/// - cauchy_stress: (steps, elems, 6) sample frame Cauchy stress
/// - sample_strain: (steps, elems, 6) sample frame elastic strain for the same steps
/// - directional_stiffness: (nhkl, steps) output, added to
/// - elem_vols: (steps, elems) element volumes for the same steps
/// - in_fibers: (nhkl, steps, elems) from calc_within_fibers for the same steps
/// - per_rank_update: see calc_lattice_strains
#[pyfunction]
#[pyo3(name = "calc_directional_stiffness_lattice_fiber")]
fn calc_directional_stiffness_lattice_fiber(
    cauchy_stress: &Bound<'_, PyArray3<f64>>,
    sample_strain: &Bound<'_, PyArray3<f64>>,
    directional_stiffness: &Bound<'_, PyArray2<f64>>,
    elem_vols: &Bound<'_, PyArray2<f64>>,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    per_rank_update: bool,
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

    let lat_ds_in_fiber_iter = dir_stiff
        .chunks_exact_mut(num_steps)
        .zip(in_fiber_hkls.chunks_exact(num_steps * num_elems));

    for (dir_stiff_hkl, in_fiber_hkl) in lat_ds_in_fiber_iter.into_iter() {
        let stress_strains = stress
            .chunks_exact(num_elems * 6)
            .zip(strain.chunks_exact(num_elems * 6));
        let ev_sst_iter = evs.chunks_exact(num_elems).zip(stress_strains);
        let ev_sst_in_fiber_iter = in_fiber_hkl.chunks_exact(num_elems).zip(ev_sst_iter);
        for (istep, (in_fiber_hkl_step, (evs_step, (stress_step, strain_step)))) in
            ev_sst_in_fiber_iter.enumerate()
        {
            let (_, inv_total_lat_vol) =
                calc_volume_terms_fiber(evs_step, in_fiber_hkl_step, per_rank_update);

            dir_stiff_hkl[istep] += stress_step
                .as_chunks::<6>()
                .0
                .iter()
                .zip(strain_step.as_chunks::<6>().0)
                .zip(evs_step)
                .zip(in_fiber_hkl_step)
                .filter(|ev_sst_in_fiber| *ev_sst_in_fiber.1)
                .map(|((sst, ev), _)| {
                    if f64::abs(sst.1[2]) < f64::EPSILON {
                        0.0_f64
                    } else {
                        (sst.0[2] / sst.1[2]) * ev * inv_total_lat_vol
                    }
                })
                .reduce(|dir_modulus, dir_modulus_elem| dir_modulus + dir_modulus_elem)
                .unwrap_or(0.0_f64);
        }
    }

    Ok(())
}

/// Lattice parameters can be given as a single value (e.g. the cubic lattice
/// parameter, which is how this function was originally called) or as a list.
#[derive(FromPyObject)]
enum LatticeParams {
    Scalar(f64),
    List(Vec<f64>),
}

impl LatticeParams {
    fn to_vec(&self) -> Vec<f64> {
        match self {
            LatticeParams::Scalar(a) => vec![*a],
            LatticeParams::List(params) => params.clone(),
        }
    }
}

/// Flags which elements are "in fiber" for each HKL: an element is in the fiber when
/// one of the symmetrically equivalent crystal directions of the HKL plane normal,
/// once rotated by the element's orientation, lies within distance_tolerance_rad of
/// the sample direction. Mirrors LightUp::CalculateInFibers in ExaConstit.
///
/// Arguments:
/// - lattice_orientations: (steps, elems, 4) quaternions for each element
/// - sample_dir: (3,) sample direction, normalized internally
/// - hkls: (nhkl, 3) Miller indices of the planes (3-index form, also for
///   hexagonal / trigonal)
/// - lattice_params: lattice parameter(s), a single value or a list; see below
/// - distance_tolerance_rad: angular tolerance in radians
/// - in_fibers: (nhkl, steps, elems) bool output, overwritten
/// - lattice_type: crystal system, default "cubic"
///
/// lattice_type uses the same names as ExaConstit's `laue_type` option and
/// lattice_params the same parameters as its `lattice_params` option, with any
/// angles in radians:
/// - cubic -> a
/// - hexagonal / trigonal / tetragonal -> a, c
/// - rhombohedral -> a, alpha
/// - orthorhombic -> a, b, c
/// - monoclinic -> a, b, c, beta
/// - triclinic -> a, b, c, alpha, beta, gamma
///
/// The wrong number of parameters, non-positive parameters or an unknown lattice
/// type raise an error.
#[pyfunction]
#[pyo3(
    name = "calc_within_fibers",
    signature = (lattice_orientations, sample_dir, hkls, lattice_params, distance_tolerance_rad, in_fibers, lattice_type = "cubic")
)]
fn calc_within_fibers(
    lattice_orientations: &Bound<'_, PyArray3<f64>>,
    sample_dir: &Bound<'_, PyArray1<f64>>,
    hkls: &Bound<'_, PyArray2<f64>>,
    lattice_params: LatticeParams,
    distance_tolerance_rad: f64,
    in_fibers: &Bound<'_, PyArray3<bool>>,
    lattice_type: &str,
) -> anyhow::Result<()> {
    let lat_type = LatticeType::from_name(lattice_type)?;
    let cell_params = lat_type.cell_parameters(&lattice_params.to_vec())?;
    // Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's the easiest way of doing things
    let lat_vec_ops_b = compute_lattice_b_param(&cell_params);
    let symm_quat = lat_type.symmetric_quaternions();

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

    in_fiber_hkls
        .chunks_exact_mut(num_steps * num_elems)
        .zip(slice_hkls.as_chunks::<3>().0)
        .for_each(|(in_fibers, hkl)| {
            calculate_in_fibers(
                &lat_vec_ops_b,
                &symm_quat,
                hkl,
                s_dir,
                xtal_orientations,
                distance_tolerance_rad,
                in_fibers,
            );
        });

    Ok(())
}

/// Lattice strain / light-up calculations for ExaConstit results. See each
/// function's documentation (help(xtal_light_up.xtal_light_up.<name>)) for the
/// expected array shapes.
///
/// Conventions shared by all of the functions in this module:
/// - Arrays must be C-contiguous numpy float64 arrays (bool for in_fibers); a
///   non-contiguous array raises an error rather than being silently copied.
/// - Symmetric tensors are stored as 6-vectors in the order
///   [11, 22, 33, 23, 13, 12] (tensor components, not engineering shear strains).
/// - Quaternions are [w, x, y, z] and rotate lattice (crystal) frame vectors into
///   the sample frame, the same convention ExaConstit / ExaCMech use.
#[pymodule]
fn xtal_light_up(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(strain_lattice2sample, m)?)?;
    m.add_function(wrap_pyfunction!(calc_taylor_factors, m)?)?;
    m.add_function(wrap_pyfunction!(calc_directional_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(calc_lattice_strains, m)?)?;
    m.add_function(wrap_pyfunction!(calc_taylor_factors_lattice_fiber, m)?)?;
    m.add_function(wrap_pyfunction!(
        calc_directional_stiffness_lattice_fiber,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calc_within_fibers, m)?)?;
    Ok(())
}

/// Total volume of the in-fiber elements for one HKL / step, and the factor each
/// element's volume weighted contribution gets multiplied by. With per_rank_update
/// that factor is 1 (raw volume weighted sums, normalized by the caller once all the
/// data has been accumulated); otherwise it is 1 / total volume (0 if the fiber is
/// empty) so the result is the volume average.
#[inline(always)]
fn calc_volume_terms_fiber(
    evs_step: &[f64],
    in_fiber_hkl_step: &[bool],
    per_rank_update: bool,
) -> (f64, f64) {
    let total_lat_vol = evs_step
        .iter()
        .zip(in_fiber_hkl_step)
        .filter(|ev_in_fiber| *ev_in_fiber.1)
        .map(|(ev, _)| *ev)
        .reduce(|ev_total, ev| ev_total + ev)
        .unwrap_or(0.0_f64);

    let inv_total_lat_vol = if per_rank_update {
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

/// In-fiber flags for a single HKL: converts the Miller indices to a crystal frame
/// direction with the reciprocal lattice (c = B^T hkl) and hands off to within_fiber.
#[inline(always)]
fn calculate_in_fibers(
    lat_vec_ops_b: &[[f64; 3]; 3],
    symm_quat: &[[f64; 4]],
    hkl: &[f64],
    s_dir: &[f64],
    quats: &[f64],
    distance_tolerance: f64,
    in_fiber: &mut [bool],
) {
    // compute crystal direction from planeData
    let c_dir = {
        let mut tmp_cdir = [0.0_f64; 3];
        mat_t_vec_mult::<3, 3, f64>(lat_vec_ops_b, hkl, &mut tmp_cdir);
        tmp_cdir
    };

    within_fiber(
        &c_dir,
        s_dir,
        quats,
        symm_quat,
        distance_tolerance,
        in_fiber,
    );
}

/// Rotation matrix for a unit quaternion [w, x, y, z]. Same as ExaConstit's
/// Quat2RMat: it takes lattice (crystal) frame vectors into the sample frame.
#[inline(always)]
fn quat2rmat(quat: &[f64]) -> [[f64; 3]; 3] {
    assert!(quat.len() >= 4);
    let qbar = quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

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

/// Rotates a lattice frame strain 6-vector [11, 22, 33, 23, 13, 12] into the sample
/// frame in place (R strain R^T, with R from quat2rmat).
#[inline(always)]
fn rotate_strain_to_sample(quat: &[f64], strain_vec: &mut [f64]) {
    assert!(quat.len() >= 4);
    assert!(strain_vec.len() >= 6);
    let rmat = quat2rmat(quat);
    let strain = {
        let mut strain = [[0.0_f64; 3]; 3];

        strain[0][0] = strain_vec[0];
        strain[1][1] = strain_vec[1];
        strain[2][2] = strain_vec[2];
        strain[1][2] = strain_vec[3];
        strain[2][1] = strain[1][2];
        strain[0][2] = strain_vec[4];
        strain[2][0] = strain[0][2];
        strain[0][1] = strain_vec[5];
        strain[1][0] = strain[0][1];

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

/// Makes sure the Taylor factor inputs line up: one equivalent plastic strain rate
/// and num_slip shear rates for every element and step.
fn check_taylor_inputs(
    num_slip: usize,
    num_gdots: usize,
    num_dpeffs: usize,
    num_values: usize,
) -> anyhow::Result<()> {
    if num_slip == 0 {
        anyhow::bail!("shear_rates must have at least one slip system");
    }
    if num_dpeffs != num_values {
        anyhow::bail!(
            "eq_pl_strain_rate has {} values but {} were expected (one per element and step)",
            num_dpeffs,
            num_values
        );
    }
    if num_gdots != num_values * num_slip {
        anyhow::bail!(
            "shear_rates has {} values but {} were expected ({} slip systems per element and step)",
            num_gdots,
            num_values * num_slip,
            num_slip
        );
    }
    Ok(())
}

/// Taylor factor from the slip system shear rates and ExaCMech's equivalent plastic
/// strain rate. Mirrors LightUp::CalcLatticeTaylorFactorDpeff in ExaConstit, so it
/// works for any crystal system / number of slip systems.
#[inline(always)]
fn calc_taylor_factor(gdots: &[f64], dpeff: f64) -> f64 {
    let abs_sum_shear_rate: f64 = gdots.iter().map(|gdot| f64::abs(*gdot)).sum();
    if f64::abs(dpeff) <= 1.0e-14 {
        0.0_f64
    } else {
        abs_sum_shear_rate / dpeff
    }
}

/// Core in-fiber test shared with ExaConstit (LightUp::CalculateInFibers).
///
/// For every orientation (quats holds [w, x, y, z] per element) this finds the
/// largest cosine between the sample direction and the symmetrically equivalent
/// crystal directions R_q R_sym^T c, and flags the element as in fiber when the
/// corresponding angle is <= distance_tolerance (radians). in_fibers has one entry
/// per quaternion.
#[inline(always)]
fn within_fiber(
    c_dir: &[f64],
    s_dir: &[f64],
    quats: &[f64],
    symm_quat: &[[f64; 4]],
    distance_tolerance: f64,
    in_fibers: &mut [bool],
) {
    assert!(c_dir.len() >= 3);
    assert!(s_dir.len() >= 3);
    assert!(quats.len() >= 4);
    assert!(!symm_quat.is_empty());

    assert!(in_fibers.len() == (quats.len() / 4));

    let c = {
        let inv_c_norm = 1.0 / norm::<3, f64>(c_dir);
        [
            c_dir[0] * inv_c_norm,
            c_dir[1] * inv_c_norm,
            c_dir[2] * inv_c_norm,
        ]
    };

    let s = {
        let inv_s_norm = 1.0 / norm::<3, f64>(s_dir);
        [
            s_dir[0] * inv_s_norm,
            s_dir[1] * inv_s_norm,
            s_dir[2] * inv_s_norm,
        ]
    };

    // Could maybe move this over to a vec if we want this to be easily generic over a ton of symmetry conditions...
    let mut rmat_fr_qsym_c_dir = vec![[0.0_f64; 3]; symm_quat.len()];
    rmat_fr_qsym_c_dir
        .iter_mut()
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

    let s_rmat_csym_prod: Vec<f64> = quats
        .as_chunks::<4>()
        .0
        .iter()
        .map(|quat| {
            let sine = rmat_fr_qsym_c_dir
                .iter()
                .map(|c_sym| {
                    let rmat = quat2rmat(quat);
                    let mut prod = [0.0_f64; 3];
                    // Might need to make this the transpose...
                    mat_vec_mult::<3, 3, f64>(&rmat, c_sym, &mut prod);
                    dot_prod::<3, f64>(&s, &prod)
                })
                .fold(f64::MIN, |a, b| a.max(b));
            sine
        })
        .collect();

    in_fibers
        .iter_mut()
        .zip(s_rmat_csym_prod)
        .for_each(|(in_fiber, sine)| {
            let sine_safe = {
                if f64::abs(sine) > 1.00000001 {
                    sine.signum()
                } else {
                    sine
                }
            };

            let distance = f64::acos(sine_safe);
            *in_fiber = distance <= distance_tolerance;
        })
}
