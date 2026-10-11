//! Crystal lattice description for the light-up calculations.
//!
//! This mirrors `LatticeTypeGeneral` / `GetSymmetryGroups` in ExaConstit's
//! in-situ light-up implementation (src/postprocessing/mechanics_lightup.cpp)
//! so the post-processing results line up with what ExaConstit computes in-situ
//! for every supported crystal system:
//!
//! | lattice type | symmetry ops | lattice parameters                          |
//! |--------------|--------------|---------------------------------------------|
//! | cubic        | 24           | a                                           |
//! | hexagonal    | 12           | a, c                                        |
//! | trigonal     | 6            | a, c                                        |
//! | rhombohedral | 6            | a, alpha (radians)                          |
//! | tetragonal   | 8            | a, c                                        |
//! | orthorhombic | 4            | a, b, c                                     |
//! | monoclinic   | 2            | a, b, c, beta (radians)                     |
//! | triclinic    | 1            | a, b, c, alpha, beta, gamma (radians)       |

use anyhow::{bail, Result};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

use super::math::{cross_prod, dot_prod, norm};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatticeType {
    Cubic,
    Hexagonal,
    Trigonal,
    Rhombohedral,
    Tetragonal,
    Orthorhombic,
    Monoclinic,
    Triclinic,
}

impl LatticeType {
    #[cfg(test)]
    pub const ALL: [LatticeType; 8] = [
        LatticeType::Cubic,
        LatticeType::Hexagonal,
        LatticeType::Trigonal,
        LatticeType::Rhombohedral,
        LatticeType::Tetragonal,
        LatticeType::Orthorhombic,
        LatticeType::Monoclinic,
        LatticeType::Triclinic,
    ];

    /// Parses the same names ExaConstit's `laue_type` option uses (case insensitive)
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name.trim().to_ascii_uppercase().as_str() {
            "CUBIC" => LatticeType::Cubic,
            "HEXAGONAL" => LatticeType::Hexagonal,
            "TRIGONAL" => LatticeType::Trigonal,
            "RHOMBOHEDRAL" => LatticeType::Rhombohedral,
            "TETRAGONAL" => LatticeType::Tetragonal,
            "ORTHORHOMBIC" => LatticeType::Orthorhombic,
            "MONOCLINIC" => LatticeType::Monoclinic,
            "TRICLINIC" => LatticeType::Triclinic,
            _ => bail!(
                "Unknown lattice type '{}'. Valid options are: cubic, hexagonal, trigonal, \
                 rhombohedral, tetragonal, orthorhombic, monoclinic, triclinic",
                name
            ),
        })
    }

    /// Lattice parameters expected for this crystal system, in order
    pub fn parameter_names(&self) -> &'static str {
        match self {
            LatticeType::Cubic => "a",
            LatticeType::Hexagonal | LatticeType::Trigonal | LatticeType::Tetragonal => "a, c",
            LatticeType::Rhombohedral => "a, alpha (in radians)",
            LatticeType::Orthorhombic => "a, b, c",
            LatticeType::Monoclinic => "a, b, c, beta (in radians)",
            LatticeType::Triclinic => "a, b, c, alpha, beta, gamma (in radians)",
        }
    }

    pub fn num_parameters(&self) -> usize {
        match self {
            LatticeType::Cubic => 1,
            LatticeType::Hexagonal
            | LatticeType::Trigonal
            | LatticeType::Tetragonal
            | LatticeType::Rhombohedral => 2,
            LatticeType::Orthorhombic => 3,
            LatticeType::Monoclinic => 4,
            LatticeType::Triclinic => 6,
        }
    }

    /// Full set of cell parameters [a, b, c, alpha, beta, gamma] for this system
    /// built from the user supplied lattice parameters.
    pub fn cell_parameters(&self, lparams: &[f64]) -> Result<[f64; 6]> {
        if lparams.len() != self.num_parameters() {
            bail!(
                "Lattice type {:?} needs {} lattice parameter(s) ({}) but {} were given",
                self,
                self.num_parameters(),
                self.parameter_names(),
                lparams.len()
            );
        }
        if lparams.iter().any(|lp| !lp.is_finite() || *lp <= 0.0) {
            bail!("Lattice parameters must all be positive, got {:?}", lparams);
        }
        let p = lparams;
        Ok(match self {
            LatticeType::Cubic => [p[0], p[0], p[0], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2],
            LatticeType::Hexagonal | LatticeType::Trigonal => {
                [p[0], p[0], p[1], FRAC_PI_2, FRAC_PI_2, 2.0 * FRAC_PI_3]
            }
            LatticeType::Rhombohedral => [p[0], p[0], p[0], p[1], p[1], p[1]],
            LatticeType::Tetragonal => [p[0], p[0], p[1], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2],
            LatticeType::Orthorhombic => [p[0], p[1], p[2], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2],
            LatticeType::Monoclinic => [p[0], p[1], p[2], FRAC_PI_2, p[3], FRAC_PI_2],
            LatticeType::Triclinic => [p[0], p[1], p[2], p[3], p[4], p[5]],
        })
    }

    /// Point group rotations as [angle (radians), axis_x, axis_y, axis_z].
    /// Unless noted, these are taken from HEXRD:
    /// https://github.com/HEXRD/hexrd/blob/3060f506148ee29ef561c48c3331238e41fb928e/hexrd/rotations.py#L1327-L1514
    pub fn symmetry_angle_axis(&self) -> Vec<[f64; 4]> {
        let sqrt3_2 = f64::sqrt(3.0) / 2.0;
        let isqrt6 = 1.0 / f64::sqrt(6.0);
        match self {
            LatticeType::Cubic => vec![
                [0.0, 1.0, 0.0, 0.0],               // identity
                [FRAC_PI_2, 1.0, 0.0, 0.0],         // fourfold about   1  0  0 (x1)
                [PI, 1.0, 0.0, 0.0],                //
                [FRAC_PI_2 * 3.0, 1.0, 0.0, 0.0],   //
                [FRAC_PI_2, 0.0, 1.0, 0.0],         // fourfold about   0  1  0 (x2)
                [PI, 0.0, 1.0, 0.0],                //
                [FRAC_PI_2 * 3.0, 0.0, 1.0, 0.0],   //
                [FRAC_PI_2, 0.0, 0.0, 1.0],         // fourfold about   0  0  1 (x3)
                [PI, 0.0, 0.0, 1.0],                //
                [FRAC_PI_2 * 3.0, 0.0, 0.0, 1.0],   //
                [FRAC_PI_3 * 2.0, 1.0, 1.0, 1.0],   // threefold about  1  1  1
                [FRAC_PI_3 * 4.0, 1.0, 1.0, 1.0],   //
                [FRAC_PI_3 * 2.0, -1.0, 1.0, 1.0],  // threefold about -1  1  1
                [FRAC_PI_3 * 4.0, -1.0, 1.0, 1.0],  //
                [FRAC_PI_3 * 2.0, -1.0, -1.0, 1.0], // threefold about -1 -1  1
                [FRAC_PI_3 * 4.0, -1.0, -1.0, 1.0], //
                [FRAC_PI_3 * 2.0, 1.0, -1.0, 1.0],  // threefold about  1 -1  1
                [FRAC_PI_3 * 4.0, 1.0, -1.0, 1.0],  //
                [PI, 1.0, 1.0, 0.0],                // twofold about    1  1  0
                [PI, -1.0, 1.0, 0.0],               // twofold about   -1  1  0
                [PI, 1.0, 0.0, 1.0],                // twofold about    1  0  1
                [PI, 0.0, 1.0, 1.0],                // twofold about    0  1  1
                [PI, -1.0, 0.0, 1.0],               // twofold about   -1  0  1
                [PI, 0.0, -1.0, 1.0],               // twofold about    0 -1  1
            ],
            LatticeType::Hexagonal => vec![
                [0.0, 1.0, 0.0, 0.0],       // identity
                [FRAC_PI_3, 0.0, 0.0, 1.0], // sixfold about  0  0  1 (x3,c)
                [FRAC_PI_3 * 2.0, 0.0, 0.0, 1.0],
                [PI, 0.0, 0.0, 1.0],
                [FRAC_PI_3 * 4.0, 0.0, 0.0, 1.0],
                [FRAC_PI_3 * 5.0, 0.0, 0.0, 1.0],
                [PI, 1.0, 0.0, 0.0],       // twofold about  2 -1  0 (x1,a1)
                [PI, -0.5, sqrt3_2, 0.0],  // twofold about -1  2  0 (a2)
                [PI, -0.5, -sqrt3_2, 0.0], // twofold about -1 -1  0 (a3)
                [PI, sqrt3_2, 0.5, 0.0],   // twofold about  1  0  0
                [PI, 0.0, 1.0, 0.0],       // twofold about -1  1  0 (x2)
                [PI, -sqrt3_2, 0.5, 0.0],  // twofold about  0 -1  0
            ],
            LatticeType::Trigonal => vec![
                [0.0, 1.0, 0.0, 0.0],             // identity
                [FRAC_PI_3 * 2.0, 0.0, 0.0, 1.0], // threefold about 0001 (x3,c)
                [FRAC_PI_3 * 4.0, 0.0, 0.0, 1.0],
                [PI, 1.0, 0.0, 0.0],       // twofold about  2 -1 -1  0 (x1,a1)
                [PI, -0.5, sqrt3_2, 0.0],  // twofold about -1  2 -1  0 (a2)
                [PI, -0.5, -sqrt3_2, 0.0], // twofold about -1 -1  2  0 (a3)
            ],
            LatticeType::Rhombohedral => vec![
                [0.0, 1.0, 0.0, 0.0],             // identity
                [FRAC_PI_3 * 2.0, 1.0, 1.0, 1.0], // threefold about [111]
                [FRAC_PI_3 * 4.0, 1.0, 1.0, 1.0],
                [PI, 2.0 * isqrt6, -isqrt6, -isqrt6], // twofold perpendicular to [111]
                [PI, -isqrt6, 2.0 * isqrt6, -isqrt6],
                [PI, -isqrt6, -isqrt6, 2.0 * isqrt6],
            ],
            LatticeType::Tetragonal => vec![
                [0.0, 1.0, 0.0, 0.0],       // identity
                [FRAC_PI_2, 0.0, 0.0, 1.0], // fourfold about 0  0  1 (x3)
                [PI, 0.0, 0.0, 1.0],
                [FRAC_PI_2 * 3.0, 0.0, 0.0, 1.0],
                [PI, 1.0, 0.0, 0.0],  // twofold about  1  0  0 (x1)
                [PI, 0.0, 1.0, 0.0],  // twofold about  0  1  0 (x2)
                [PI, 1.0, 1.0, 0.0],  // twofold about  1  1  0
                [PI, -1.0, 1.0, 0.0], // twofold about -1  1  0
            ],
            LatticeType::Orthorhombic => vec![
                [0.0, 1.0, 0.0, 0.0], // identity
                [PI, 1.0, 0.0, 0.0],  // twofold about  1  0  0
                [PI, 0.0, 1.0, 0.0],  // twofold about  0  1  0
                [PI, 0.0, 0.0, 1.0],  // twofold about  0  0  1
            ],
            LatticeType::Monoclinic => vec![
                [0.0, 1.0, 0.0, 0.0], // identity
                [PI, 0.0, 1.0, 0.0],  // twofold about 010 (x2)
            ],
            LatticeType::Triclinic => vec![
                [0.0, 1.0, 0.0, 0.0], // identity
            ],
        }
    }

    /// Point group rotations as unit quaternions [w, x, y, z] with w >= 0
    pub fn symmetric_quaternions(&self) -> Vec<[f64; 4]> {
        self.symmetry_angle_axis()
            .iter()
            .map(|ang_axis| {
                let half = 0.5 * ang_axis[0];
                let s = f64::sin(half);
                let inv_norm_axis = 1.0 / norm::<3, f64>(&ang_axis[1..4]);
                let mut quat = [
                    f64::cos(half),
                    s * ang_axis[1] * inv_norm_axis,
                    s * ang_axis[2] * inv_norm_axis,
                    s * ang_axis[3] * inv_norm_axis,
                ];
                if quat[0] < 0.0 {
                    quat.iter_mut().for_each(|q| *q *= -1.0);
                }
                quat
            })
            .collect()
    }
}

/// Computes the reciprocal lattice B from cell parameters [a, b, c, alpha, beta, gamma].
/// Different from HEXRD we return it as a row matrix (each row is a reciprocal lattice
/// vector) as that's the easiest way of doing things, the same as ExaConstit's version.
pub fn compute_lattice_b_param(cellparms: &[f64; 6]) -> [[f64; 3]; 3] {
    let alfa = cellparms[3];
    let beta = cellparms[4];
    let gamma = cellparms[5];

    let cosalfar =
        (f64::cos(beta) * f64::cos(gamma) - f64::cos(alfa)) / (f64::sin(beta) * f64::sin(gamma));
    let sinalfar = f64::sqrt(1.0_f64 - cosalfar * cosalfar);

    let a = [cellparms[0], 0.0_f64, 0.0_f64];
    let b = [
        cellparms[1] * f64::cos(gamma),
        cellparms[1] * f64::sin(gamma),
        0.0_f64,
    ];
    let c = [
        cellparms[2] * f64::cos(beta),
        -cellparms[2] * cosalfar * f64::sin(beta),
        cellparms[2] * sinalfar * f64::sin(beta),
    ];

    // Cell volume
    let inv_vol = {
        let v_temp = cross_prod(&b, &c);
        1.0 / dot_prod::<3, f64>(&a, &v_temp)
    };

    // Reciprocal lattice vectors
    let cross_prod_inv_v = |vec1: &[f64], vec2: &[f64]| -> [f64; 3] {
        let mut tmp = cross_prod(vec1, vec2);
        tmp.iter_mut().for_each(|t| *t *= inv_vol);
        tmp
    };

    // B takes components in the reciprocal lattice to X
    [
        cross_prod_inv_v(&b, &c),
        cross_prod_inv_v(&c, &a),
        cross_prod_inv_v(&a, &b),
    ]
}

/// Direct lattice vectors (rows) for the same cell, used to check the reciprocal lattice
#[cfg(test)]
fn direct_lattice(cellparms: &[f64; 6]) -> [[f64; 3]; 3] {
    let (alfa, beta, gamma) = (cellparms[3], cellparms[4], cellparms[5]);
    let cosalfar =
        (f64::cos(beta) * f64::cos(gamma) - f64::cos(alfa)) / (f64::sin(beta) * f64::sin(gamma));
    let sinalfar = f64::sqrt(1.0_f64 - cosalfar * cosalfar);
    [
        [cellparms[0], 0.0, 0.0],
        [
            cellparms[1] * f64::cos(gamma),
            cellparms[1] * f64::sin(gamma),
            0.0,
        ],
        [
            cellparms[2] * f64::cos(beta),
            -cellparms[2] * cosalfar * f64::sin(beta),
            cellparms[2] * sinalfar * f64::sin(beta),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quat_mult(p: &[f64; 4], q: &[f64; 4]) -> [f64; 4] {
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    }

    fn same_rotation(p: &[f64; 4], q: &[f64; 4]) -> bool {
        let diff: f64 = (0..4).map(|i| (p[i] - q[i]).abs()).sum();
        let sum: f64 = (0..4).map(|i| (p[i] + q[i]).abs()).sum();
        diff < 1e-12 || sum < 1e-12
    }

    #[test]
    fn symmetry_groups_are_closed_and_distinct() {
        let expected = [24, 12, 6, 6, 8, 4, 2, 1];
        for (lt, nsym) in LatticeType::ALL.iter().zip(expected) {
            let quats = lt.symmetric_quaternions();
            assert_eq!(quats.len(), nsym, "{:?}", lt);
            for (i, p) in quats.iter().enumerate() {
                let n = norm::<4, f64>(p);
                assert!((n - 1.0).abs() < 1e-14, "{:?} op {} not unit", lt, i);
                for q in quats.iter().skip(i + 1) {
                    assert!(!same_rotation(p, q), "{:?} has a duplicate operator", lt);
                }
                for q in quats.iter() {
                    let pq = quat_mult(p, q);
                    assert!(
                        quats.iter().any(|r| same_rotation(&pq, r)),
                        "{:?} is not closed under composition",
                        lt
                    );
                }
            }
        }
    }

    #[test]
    fn reciprocal_lattice_is_dual_to_direct_lattice() {
        let params: [(LatticeType, Vec<f64>); 8] = [
            (LatticeType::Cubic, vec![3.6]),
            (LatticeType::Hexagonal, vec![2.95, 4.68]),
            (LatticeType::Trigonal, vec![4.76, 12.99]),
            (LatticeType::Rhombohedral, vec![5.1, 0.95]),
            (LatticeType::Tetragonal, vec![4.59, 2.96]),
            (LatticeType::Orthorhombic, vec![2.85, 5.87, 4.96]),
            (LatticeType::Monoclinic, vec![5.15, 5.21, 5.31, 1.73]),
            (
                LatticeType::Triclinic,
                vec![7.1, 7.8, 8.2, 1.62, 1.75, 1.45],
            ),
        ];
        for (lt, p) in params.iter() {
            let cell = lt.cell_parameters(p).unwrap();
            let bmat = compute_lattice_b_param(&cell);
            let amat = direct_lattice(&cell);
            for (i, a_i) in amat.iter().enumerate() {
                for (j, b_j) in bmat.iter().enumerate() {
                    let d = dot_prod::<3, f64>(a_i, b_j);
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (d - expected).abs() < 1e-12,
                        "{:?} a_{} . b*_{} = {}",
                        lt,
                        i,
                        j,
                        d
                    );
                }
            }
        }
    }

    #[test]
    fn cubic_reciprocal_lattice_is_scaled_identity() {
        let bmat = compute_lattice_b_param(&LatticeType::Cubic.cell_parameters(&[4.0]).unwrap());
        for (i, row) in bmat.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                let expected = if i == j { 0.25 } else { 0.0 };
                assert!((val - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn parameter_validation() {
        assert!(LatticeType::from_name("Hexagonal").is_ok());
        assert!(LatticeType::from_name("hcp").is_err());
        assert!(LatticeType::Hexagonal.cell_parameters(&[2.95]).is_err());
        assert!(LatticeType::Cubic.cell_parameters(&[-3.6]).is_err());
        assert!(LatticeType::Cubic.cell_parameters(&[3.6]).is_ok());
    }
}
