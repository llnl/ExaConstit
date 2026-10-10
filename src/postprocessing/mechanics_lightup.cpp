#include "postprocessing/mechanics_lightup.hpp"

#include "utilities/mechanics_kernels.hpp"
#include "utilities/rotations.hpp"

#include "mfem/general/forall.hpp"

#include "ECMech_const.h"
#include "ECMech_gpu_portability.h"
#include "SNLS_linalg.h"

/**
 * @brief Type trait for detecting std::array types
 *
 * Helper template for template metaprogramming to distinguish
 * std::array types from other types in generic printing functions.
 */
namespace no_std {
template <typename T>
struct IsStdArray : std::false_type {};
template <typename T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};
} // namespace no_std

/**
 * @brief Print std::array to output stream with formatting
 *
 * @tparam T Array element type
 * @tparam N Array size
 * @param stream Output stream for writing
 * @param array Array to print
 *
 * Formats std::array output as "[ val1, val2, val3 ]" with scientific
 * notation and 6-digit precision. Used for consistent formatting of
 * HKL directions and other array data in output files.
 */
template <typename T, std::size_t N>
void printArray(std::ostream& stream, std::array<T, N>& array) {
    stream << "\"[ ";
    for (size_t i = 0; i < N - 1; i++) {
        stream << std::scientific << std::setprecision(6) << array[i] << ",";
    }
    stream << array[N - 1] << " ]\"\t";
}

/**
 * @brief Generic value printing with type-specific formatting
 *
 * @tparam T Value type
 * @param stream Output stream for writing
 * @param t Value to print
 *
 * Prints values with appropriate formatting based on type:
 * - std::array types use printArray() for structured output
 * - Other types use scientific notation with 6-digit precision
 *
 * Enables generic output formatting for different data types
 * in LightUp file output operations.
 */
template <typename T>
void printValues(std::ostream& stream, T& t) {
    if constexpr (no_std::IsStdArray<T>::value) {
        printArray(stream, t);
    } else {
        stream << std::scientific << std::setprecision(6) << t << "\t";
    }
}

/**
 * @brief Generate region-specific lattice output basename
 *
 * @param lattice_basename Base filename from configuration
 * @param region_id Region identifier
 * @return Region-specific filename prefix
 *
 * Constructs unique output file basename by appending region identifier.
 * Format: "basename_region_N_" where N is the region ID. Ensures
 * separate output files for each material region in multi-region simulations.
 */
std::string get_lattice_basename(const fs::path& lattice_basename, const int region_id) {
    return lattice_basename.string() + "region_" + std::to_string(region_id) + "_";
}

/**
 * @brief Get crystallographic point group symmetry operations
 *
 * @param lattice_type Crystal system type specifying the point group
 * @return Vector of quaternions representing symmetry operations
 *
 * Returns the complete set of point group symmetry operations for the
 * specified crystal system as quaternions in the form [angle, x, y, z].
 * Each quaternion represents a rotation operation that maps crystallographic
 * directions to their symmetrically equivalent counterparts.
 *
 * The number and type of symmetry operations depend on the crystal system:
 * - Cubic: 24 operations (identity, 3-fold, 4-fold, 2-fold rotations)
 * - Hexagonal: 12 operations (identity, 6-fold, 3-fold, 2-fold rotations)
 * - Trigonal: 6 operations (identity, 3-fold, 2-fold rotations)
 * - Rhombohedral: 6 operations (identity, 3-fold about [111], 2-fold perpendicular to [111])
 * - Tetragonal: 8 operations (identity, 4-fold, 2-fold rotations)
 * - Orthorhombic: 4 operations (identity, three 2-fold rotations)
 * - Monoclinic: 2 operations (identity, one 2-fold rotation)
 * - Triclinic: 1 operation (identity only)
 *
 * These symmetry operations are used by LatticeTypeGeneral to generate
 * symmetrically equivalent HKL directions for lattice strain calculations
 * in powder diffraction simulations.
 *
 * @note Quaternions use the convention [angle, axis_x, axis_y, axis_z]
 *       where angle is in radians and the axis is normalized.
 */
std::vector<std::array<double, 4>> GetSymmetryGroups(const LatticeType& lattice_type) {
    // If not mentioned specifically these are taken from:
    // https://github.com/HEXRD/hexrd/blob/3060f506148ee29ef561c48c3331238e41fb928e/hexrd/rotations.py#L1327-L1514
    constexpr double PI = 3.14159265358979323846264338327950288;
    constexpr double FRAC_PI_2 = 1.57079632679489661923132169163975144;
    constexpr double FRAC_PI_3 = 1.04719755119659774615421446109316763;
    const double SQRT3_2 = std::sqrt(3.0) / 2.0;
    const double ISQRT6 = 1.0 / std::sqrt(6.0);

    std::vector<std::array<double, 4>> lattice_symm;
    switch (lattice_type) {
    case LatticeType::CUBIC: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0},               // identity
            {FRAC_PI_2, 1.0, 0.0, 0.0},         // fourfold about   1  0  0 (x1)
            {PI, 1.0, 0.0, 0.0},                //
            {FRAC_PI_2 * 3.0, 1.0, 0.0, 0.0},   //
            {FRAC_PI_2, 0.0, 1.0, 0.0},         // fourfold about   0  1  0 (x2)
            {PI, 0.0, 1.0, 0.0},                //
            {FRAC_PI_2 * 3.0, 0.0, 1.0, 0.0},   //
            {FRAC_PI_2, 0.0, 0.0, 1.0},         // fourfold about   0  0  1 (x3)
            {PI, 0.0, 0.0, 1.0},                //
            {FRAC_PI_2 * 3.0, 0.0, 0.0, 1.0},   //
            {FRAC_PI_3 * 2.0, 1.0, 1.0, 1.0},   // threefold about  1  1  1
            {FRAC_PI_3 * 4.0, 1.0, 1.0, 1.0},   //
            {FRAC_PI_3 * 2.0, -1.0, 1.0, 1.0},  // threefold about -1  1  1
            {FRAC_PI_3 * 4.0, -1.0, 1.0, 1.0},  //
            {FRAC_PI_3 * 2.0, -1.0, -1.0, 1.0}, // threefold about -1 -1  1
            {FRAC_PI_3 * 4.0, -1.0, -1.0, 1.0}, //
            {FRAC_PI_3 * 2.0, 1.0, -1.0, 1.0},  // threefold about  1 -1  1
            {FRAC_PI_3 * 4.0, 1.0, -1.0, 1.0},  //
            {PI, 1.0, 1.0, 0.0},                // twofold about    1  1  0
            {PI, -1.0, 1.0, 0.0},               // twofold about   -1  1  0
            {PI, 1.0, 0.0, 1.0},                // twofold about    1  0  1
            {PI, 0.0, 1.0, 1.0},                // twofold about    0  1  1
            {PI, -1.0, 0.0, 1.0},               // twofold about   -1  0  1
            {PI, 0.0, -1.0, 1.0},               // twofold about    0 -1  1
        };
        break;
    }
    case LatticeType::HEXAGONAL: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0},       // identity
            {FRAC_PI_3, 0.0, 0.0, 1.0}, // sixfold about  0  0  1 (x3,c)
            {FRAC_PI_3 * 2.0, 0.0, 0.0, 1.0},
            {PI, 0.0, 0.0, 1.0},
            {FRAC_PI_3 * 4.0, 0.0, 0.0, 1.0},
            {FRAC_PI_3 * 5.0, 0.0, 0.0, 1.0},
            {PI, 1.0, 0.0, 0.0},       // twofold about  2 -1  0 (x1,a1)
            {PI, -0.5, SQRT3_2, 0.0},  // twofold about -1  2  0 (a2)
            {PI, -0.5, -SQRT3_2, 0.0}, // twofold about -1 -1  0 (a3)
            {PI, SQRT3_2, 0.5, 0.0},   // twofold about  1  0  0
            {PI, 0.0, 1.0, 0.0},       // twofold about -1  1  0 (x2)
            {PI, -SQRT3_2, 0.5, 0.0}   // twofold about  0 -1  0
        };
        break;
    }
    case LatticeType::TRIGONAL: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0},             // identity
            {FRAC_PI_3 * 2.0, 0.0, 0.0, 1.0}, // threefold about 0001 (x3,c)
            {FRAC_PI_3 * 4.0, 0.0, 0.0, 1.0},
            {PI, 1.0, 0.0, 0.0},      // twofold about  2 -1 -1  0 (x1,a1)
            {PI, -0.5, SQRT3_2, 0.0}, // twofold about -1  2 -1  0 (a2)
            {PI, -0.5, SQRT3_2, 0.0}  // twofold about -1 -1  2  0 (a3)
        };
        break;
    }
    case LatticeType::RHOMBOHEDRAL: {
        // Claude generated these symmetry groups
        lattice_symm = {{0.0, 1.0, 0.0, 0.0}, // identity
                        {FRAC_PI_3 * 2.0,
                         1.0,
                         1.0,
                         1.0}, // 3-fold rotations about [111] direction (2 operations)
                        {FRAC_PI_3 * 4.0, 1.0, 1.0, 1.0},
                        {PI,
                         2.0 * ISQRT6,
                         -ISQRT6,
                         -ISQRT6}, // 2-fold rotations perpendicular to [111] (3 operations)
                        {PI, -ISQRT6, 2.0 * ISQRT6, -ISQRT6},
                        {PI, -ISQRT6, -ISQRT6, 2.0 * ISQRT6}};
        break;
    }
    case LatticeType::TETRAGONAL: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0},       // identity
            {FRAC_PI_2, 0.0, 0.0, 1.0}, // fourfold about 0  0  1 (x3)
            {PI, 0.0, 0.0, 1.0},
            {FRAC_PI_2 * 3.0, 0.0, 0.0, 1.0},
            {PI, 1.0, 0.0, 0.0}, // twofold about  1  0  0 (x1)
            {PI, 0.0, 1.0, 0.0}, // twofold about  0  1  0 (x2)
            {PI, 1.0, 1.0, 0.0}, // twofold about  1  1  0
            {PI, -1.0, 1.0, 0.0} // twofold about -1  1  0
        };
        break;
    }
    case LatticeType::ORTHORHOMBIC: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0}, // identity
            {PI, 1.0, 0.0, 0.0},  // twofold about  1  0  0
            {PI, 0.0, 1.0, 0.0},  // twofold about  0  1  0
            {PI, 0.0, 0.0, 1.0}   // twofold about  0  0  1
        };
        break;
    }
    case LatticeType::MONOCLINIC: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0}, // identity
            {PI, 0.0, 1.0, 0.0}   // twofold about 010 (x2)
        };
        break;
    }
    case LatticeType::TRICLINIC: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0} // identity
        };
        break;
    }
    default: {
        lattice_symm = {
            {0.0, 1.0, 0.0, 0.0} // identity
        };
        break;
    }
    }
    return lattice_symm;
}

/**
 * @brief Get number of symmetry operations for a crystal system
 *
 * @param lattice_type Crystal system type specifying the point group
 * @return Number of symmetry operations in the point group
 *
 * Returns the total number of symmetry operations for the specified
 * crystal system's point group. This count includes the identity operation
 * and all rotational symmetries of the crystal structure.
 *
 * The count varies by crystal system:
 * - Cubic: 24 symmetry operations
 * - Hexagonal: 12 symmetry operations
 * - Trigonal: 6 symmetry operations
 * - Rhombohedral: 6 symmetry operations
 * - Tetragonal: 8 symmetry operations
 * - Orthorhombic: 4 symmetry operations
 * - Monoclinic: 2 symmetry operations
 * - Triclinic: 1 symmetry operation
 *
 * This function is used to initialize the NSYM member variable in
 * LatticeTypeGeneral and to allocate appropriate storage for
 * symmetry-related calculations.
 *
 * @see GetSymmetryGroups() for the actual symmetry operation quaternions
 */
size_t GetNumberSymmetryOperations(const LatticeType& lattice_type) {
    return GetSymmetryGroups(lattice_type).size();
}

LatticeTypeGeneral::LatticeTypeGeneral(const std::vector<double>& lattice_param_a,
                                       const LatticeType& lattice_type)
    : NSYM(GetNumberSymmetryOperations(lattice_type)) {
    SymmetricQuaternions(lattice_type);
    ComputeLatticeBParam(lattice_param_a, lattice_type);
}

void LatticeTypeGeneral::ComputeLatticeBParam(const std::vector<double>& lparam_a,
                                              const LatticeType& lattice_type) {
    constexpr double FRAC_PI_2 = 1.57079632679489661923132169163975144;
    constexpr double FRAC_PI_4_3 = FRAC_PI_2 * 4.0 / 3.0;
    std::vector<double> cellparms(6, 0.0);

    switch (lattice_type) {
    case LatticeType::CUBIC: {
        cellparms = {lparam_a[0], lparam_a[0], lparam_a[0], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2};
        break;
    }
    case LatticeType::HEXAGONAL:
    case LatticeType::TRIGONAL: {
        cellparms = {lparam_a[0], lparam_a[0], lparam_a[1], FRAC_PI_2, FRAC_PI_2, FRAC_PI_4_3};
        break;
    }
    case LatticeType::RHOMBOHEDRAL: {
        cellparms = {lparam_a[0], lparam_a[0], lparam_a[0], lparam_a[1], lparam_a[1], lparam_a[1]};
        break;
    }
    case LatticeType::TETRAGONAL: {
        cellparms = {lparam_a[0], lparam_a[0], lparam_a[1], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2};
        break;
    }
    case LatticeType::ORTHORHOMBIC: {
        cellparms = {lparam_a[0], lparam_a[1], lparam_a[2], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2};
        break;
    }
    case LatticeType::MONOCLINIC: {
        cellparms = {lparam_a[0], lparam_a[1], lparam_a[2], FRAC_PI_2, lparam_a[3], FRAC_PI_2};
        break;
    }
    case LatticeType::TRICLINIC: {
        cellparms = {lparam_a[0], lparam_a[1], lparam_a[2], lparam_a[3], lparam_a[4], lparam_a[5]};
        break;
    }
    default: {
        cellparms = {lparam_a[0], lparam_a[0], lparam_a[0], FRAC_PI_2, FRAC_PI_2, FRAC_PI_2};
        break;
    }
    }

    const double alfa = cellparms[3];
    const double beta = cellparms[4];
    const double gamma = cellparms[5];

    const double cosalfar = (cos(beta) * cos(gamma) - cos(alfa)) / (sin(beta) * sin(gamma));
    const double sinalfar = sqrt(1.0 - cosalfar * cosalfar);

    const double a[3] = {cellparms[0], 0.0, 0.0};
    const double b[3] = {cellparms[1] * cos(gamma), cellparms[1] * sin(gamma), 0.0};
    const double c[3] = {cellparms[2] * cos(beta),
                         -cellparms[2] * cosalfar * sin(beta),
                         cellparms[2] * sinalfar * sin(beta)};

    // Cell volume
    double vol[3] = {};
    auto cross_prod = [&](const double* const vec1, const double* const vec2, double* const prod) {
        prod[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        prod[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        prod[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    };

    cross_prod(b, c, vol);
    const double inv_vol = 1.0 / snls::linalg::dotProd<3>(a, vol);

    // Reciprocal lattice vectors
    auto cross_prod_inv_v =
        [&](const double* const vec1, const double* const vec2, double* cross_prod_v) {
            cross_prod(vec1, vec2, cross_prod_v);
            cross_prod_v[0] *= inv_vol;
            cross_prod_v[1] *= inv_vol;
            cross_prod_v[2] *= inv_vol;
        };

    double* latb[3] = {&lattice_b[0], &lattice_b[3], &lattice_b[6]};
    // B takes components in the reciprocal lattice to X
    cross_prod_inv_v(b, c, latb[0]);
    cross_prod_inv_v(c, a, latb[1]);
    cross_prod_inv_v(a, b, latb[2]);
}

void LatticeTypeGeneral::SymmetricQuaternions(const LatticeType& lattice_type) {
    constexpr double inv2 = 1.0 / 2.0;
    auto angle_axis_symm = GetSymmetryGroups(lattice_type);

    for (size_t isym = 0; isym < NSYM * 4; isym++) {
        quat_symm.push_back(0.0);
    }

    for (size_t isym = 0; isym < NSYM; isym++) {
        double* symm_quat = &quat_symm[isym * 4];
        const double s = sin(inv2 * angle_axis_symm[isym][0]);
        symm_quat[0] = cos(inv2 * angle_axis_symm[isym][0]);
        double inv_norm_axis = 1.0 / snls::linalg::norm<3>(&angle_axis_symm[isym][1]);
        symm_quat[1] = s * angle_axis_symm[isym][1] * inv_norm_axis;
        symm_quat[2] = s * angle_axis_symm[isym][2] * inv_norm_axis;
        symm_quat[3] = s * angle_axis_symm[isym][3] * inv_norm_axis;

        inv_norm_axis = 1.0;
        if (symm_quat[0] < 0.0) {
            inv_norm_axis *= -1.0;
        }

        symm_quat[0] *= inv_norm_axis;
        symm_quat[1] *= inv_norm_axis;
        symm_quat[2] *= inv_norm_axis;
        symm_quat[3] *= inv_norm_axis;
    }
}

LightUp::LightUp(const std::vector<std::array<double, 3>>& hkls,
                 const double distance_tolerance,
                 const std::array<double, 3> s_dir,
                 std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                 const std::shared_ptr<SimulationState> sim_state,
                 const int region,
                 const RTModel& rtmodel,
                 const fs::path& lattice_basename,
                 const std::vector<double>& lattice_params,
                 const LatticeType& lattice_type)
    : m_hkls(hkls), m_distance_tolerance(distance_tolerance),
      m_npts(static_cast<size_t>(qspace->GetSize())), m_class_device(rtmodel),
      m_sim_state(sim_state), m_region(region),
      m_lattice_basename(get_lattice_basename(lattice_basename, region)),
      m_lattice(lattice_params, lattice_type), m_workspace(qspace, 3) {
    m_s_dir[0] = s_dir[0];
    m_s_dir[1] = s_dir[1];
    m_s_dir[2] = s_dir[2];

    const double inv_s_norm = 1.0 / snls::linalg::norm<3>(m_s_dir);
    m_s_dir[0] *= inv_s_norm;
    m_s_dir[1] *= inv_s_norm;
    m_s_dir[2] *= inv_s_norm;

    auto lat_vec_ops_b = m_lattice.lattice_b;
    // First one we'll always set to be all the values
    m_in_fibers.push_back(mfem::Array<bool>(static_cast<int>(m_npts)));
    for (auto& hkl : hkls) {
        m_in_fibers.push_back(mfem::Array<bool>(static_cast<int>(m_npts)));
        // Computes reciprocal lattice B but different from HEXRD we return as row matrix as that's
        // the easiest way of doing things
        double c_dir[3];
        // compute crystal direction from planeData
        snls::linalg::matTVecMult<3, 3>(lat_vec_ops_b, hkl.data(), c_dir);

        const double inv_c_norm = 1.0 / snls::linalg::norm<3>(c_dir);
        c_dir[0] *= inv_c_norm;
        c_dir[1] *= inv_c_norm;
        c_dir[2] *= inv_c_norm;

        // Could maybe move this over to a vec if we want this to be easily generic over a ton of
        // symmetry conditions...
        std::vector<std::array<double, 3>> rmat_fr_qsym_c_dir;
        mfem::Vector tmp(static_cast<int>(m_lattice.NSYM) * 3);
        for (size_t isym = 0; isym < m_lattice.NSYM; isym++) {
            rmat_fr_qsym_c_dir.push_back({0.0, 0.0, 0.0});
            double rmat[3 * 3] = {};
            Quat2RMat(&m_lattice.quat_symm[isym * 4], rmat);
            snls::linalg::matTVecMult<3, 3>(rmat, c_dir, rmat_fr_qsym_c_dir[isym].data());
            const int offset = static_cast<int>(isym * 3);
            tmp(offset + 0) = rmat_fr_qsym_c_dir[isym][0];
            tmp(offset + 1) = rmat_fr_qsym_c_dir[isym][1];
            tmp(offset + 2) = rmat_fr_qsym_c_dir[isym][2];
        }
        tmp.UseDevice(true);
        m_rmat_fr_qsym_c_dir.push_back(tmp);
    }

    m_hkls.insert(m_hkls.begin(), {0.0, 0.0, 0.0});
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    // Now we're going to save off the lattice values to a file
    if (my_id == 0) {
        auto file_line_print = [&](auto& basename, auto& name, auto& hkls) {
            fs::path filename = basename;
            filename += name;
            std::ofstream file;
            file.open(filename, std::ios_base::out);

            file << "#" << "\t";

            for (auto& item : hkls) {
                file << std::setprecision(1) << "\"[ " << item[0] << ", " << item[1] << ", "
                     << item[2] << " ]\"" << "\t";
            }
            file << std::endl;

            file.close();
        };

        file_line_print(m_lattice_basename, "strains.txt", m_hkls);
        file_line_print(m_lattice_basename, "volumes.txt", m_hkls);
        file_line_print(m_lattice_basename, "dpeff.txt", m_hkls);
        file_line_print(m_lattice_basename, "taylor_factor.txt", m_hkls);
        file_line_print(m_lattice_basename, "directional_stiffness.txt", m_hkls);
    }

    /* add a working array for the QF and in_fiber arrays */
    // If we really wanted to we could lower try and calculate the elements
    // that aren't unique here but that's not worth the effort at all given
    // how fast things are
    // let c_syms: Vec<[f64; 3]> = find_unique_tolerance::<SYM_LEN>(&rmat_fr_qsym_c_dir,
    // f64::sqrt(f64::EPSILON));

    // Move all of the above to the object constructor
    // rmat_fr_qsym_c_dir move to an mfem vector and then use it's data down here
    // same with s_dir and c_dir
    // Here iterate on which HKL we're using maybe have a map for these rmat_fr_qsym_c_dir and c_dir
}

void LightUp::CalculateLightUpData(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress) {
    std::string s_estrain = "elastic_strain";
    std::string s_rvol = "relative_volume";
    std::string s_quats = "quats";
    std::string s_gdot = "shear_rate";
    std::string s_shrateEff = "eq_pl_strain_rate";

    const size_t quats_offset = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_quats, m_region).first);
    const size_t strain_offset = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_estrain, m_region).first);
    const size_t rel_vol_offset = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_rvol, m_region).first);
    const size_t dpeff_offset = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_shrateEff, m_region).first);
    const size_t gdot_offset = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_gdot, m_region).first);
    const size_t gdot_length = static_cast<size_t>(
        m_sim_state->GetQuadratureFunctionStatePair(s_gdot, m_region).second);

    m_in_fibers[0] = true;
    for (size_t ihkl = 0; ihkl < m_rmat_fr_qsym_c_dir.size(); ihkl++) {
        CalculateInFibers(history, quats_offset, ihkl);
    }

    std::vector<double> lattice_strains_output;
    std::vector<double> lattice_volumes_output;

    CalcLatticeStrains(history,
                       strain_offset,
                       quats_offset,
                       rel_vol_offset,
                       lattice_strains_output,
                       lattice_volumes_output);

    std::vector<double> lattice_dpeff_output;
    std::vector<double> lattice_tayfac_output;

    CalcLatticeTaylorFactorDpeff(history,
                                 dpeff_offset,
                                 gdot_offset,
                                 gdot_length,
                                 lattice_tayfac_output,
                                 lattice_dpeff_output);

    std::vector<std::array<double, 3>> lattice_dir_stiff_output;

    CalcLatticeDirectionalStiffness(
        history, stress, strain_offset, quats_offset, rel_vol_offset, lattice_dir_stiff_output);

    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    // Now we're going to save off the lattice values to a file
    if (my_id == 0) {
        auto file_line_print = [&](auto& basename, auto& name, auto& vec) {
            fs::path filename = basename;
            filename += name;
            std::ofstream file;
            file.open(filename, std::ios_base::app);

            for (auto& item : vec) {
                printValues(file, item);
            }
            file << std::endl;

            file.close();
        };

        file_line_print(m_lattice_basename, "strains.txt", lattice_strains_output);
        file_line_print(m_lattice_basename, "volumes.txt", lattice_volumes_output);
        file_line_print(m_lattice_basename, "dpeff.txt", lattice_dpeff_output);
        file_line_print(m_lattice_basename, "taylor_factor.txt", lattice_tayfac_output);
        file_line_print(m_lattice_basename, "directional_stiffness.txt", lattice_dir_stiff_output);
    }
}

void LightUp::CalculateInFibers(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
    const size_t quats_offset,
    const size_t hkl_index) {
    // Same could be said for in_fiber down here
    // that way we just need to know which hkl and quats we're running with
    const size_t vdim = static_cast<size_t>(history->GetVDim());
    const auto history_data = history->Read();

    // First hkl_index is always completely true so we can easily
    // compute the total volume average values
    auto in_fiber_view = m_in_fibers[hkl_index + 1].Write();
    auto rmat_fr_qsym_c_dir = m_rmat_fr_qsym_c_dir[hkl_index].Read();

    mfem::Vector s_dir(3);
    s_dir[0] = m_s_dir[0];
    s_dir[1] = m_s_dir[1];
    s_dir[2] = m_s_dir[2];
    auto s_dir_data = s_dir.Read();
    auto distance_tolerance = m_distance_tolerance;

    const size_t NSYM = m_lattice.NSYM;

    mfem::forall(static_cast<int>(m_npts), [=] MFEM_HOST_DEVICE(int i) {
        const size_t iquats = static_cast<size_t>(i);

        const auto quats = &history_data[iquats * vdim + quats_offset];
        double rmat[3 * 3] = {};
        Quat2RMat(quats, rmat);

        double sine = -10;
        for (size_t isym = 0; isym < NSYM; isym++) {
            double prod[3] = {};
            snls::linalg::matVecMult<3, 3>(rmat, &rmat_fr_qsym_c_dir[isym * 3], prod);
            double tmp = snls::linalg::dotProd<3>(s_dir_data, prod);
            sine = (tmp > sine) ? tmp : sine;
        }
        if (fabs(sine) > 1.00000001) {
            sine = (sine >= 0) ? 1.0 : -1.0;
        }
        in_fiber_view[iquats] = acos(sine) <= distance_tolerance;
    });
}

void LightUp::CalcLatticeStrains(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
    const size_t strain_offset,
    const size_t quats_offset,
    const size_t rel_vol_offset,
    std::vector<double>& lattice_strains_output,
    std::vector<double>& lattice_volumes_output) {
    const double project_vec[6] = {m_s_dir[0] * m_s_dir[0],
                                   m_s_dir[1] * m_s_dir[1],
                                   m_s_dir[2] * m_s_dir[2],
                                   2.0 * m_s_dir[1] * m_s_dir[2],
                                   2.0 * m_s_dir[0] * m_s_dir[2],
                                   2.0 * m_s_dir[0] * m_s_dir[1]};

    const size_t vdim = static_cast<size_t>(history->GetVDim());
    const auto history_data = history->Read();
    m_workspace = 0.0;
    auto lattice_strains = m_workspace.Write();

    // Only need to compute this once
    mfem::forall(static_cast<int>(m_npts), [=] MFEM_HOST_DEVICE(int i) {
        const size_t iqpts = static_cast<size_t>(i);

        const auto strain_lat = &history_data[iqpts * vdim + strain_offset];
        const auto quats = &history_data[iqpts * vdim + quats_offset];
        const auto rel_vol = history_data[iqpts * vdim + rel_vol_offset];

        double strain[6] = {};
        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain;                      // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain;                     // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4];                    // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3];                    // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2];                    // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};

            Quat2RMat(quats, rmat);
            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];
            strain[0] = strain_m[0][0];
            strain[1] = strain_m[1][1];
            strain[2] = strain_m[2][2];
            strain[3] = strain_m[1][2];
            strain[4] = strain_m[0][2];
            strain[5] = strain_m[0][1];
        }
        const double proj_strain = snls::linalg::dotProd<6>(project_vec, strain);
        lattice_strains[iqpts] = proj_strain;
    });

    for (const auto& in_fiber_hkl : m_in_fibers) {
        mfem::Vector lattice_strain_hkl(1);
        auto region_comm = m_sim_state->GetRegionCommunicator(m_region);
        const double lat_vol = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(
            &m_workspace, &in_fiber_hkl, lattice_strain_hkl, 1, m_class_device, region_comm);

        lattice_volumes_output.push_back(lat_vol);
        lattice_strains_output.push_back(lattice_strain_hkl(0));
    }
}

void LightUp::CalcLatticeTaylorFactorDpeff(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
    const size_t dpeff_offset,
    const size_t gdot_offset,
    const size_t gdot_length,
    std::vector<double>& lattice_tay_facs,
    std::vector<double>& lattice_dpeff) {
    const size_t vdim = static_cast<size_t>(history->GetVDim());
    const auto history_data = history->Read();
    m_workspace = 0.0;
    auto lattice_tayfac_dpeffs = m_workspace.Write();

    // Only need to compute this once
    mfem::forall(static_cast<int>(m_npts), [=] MFEM_HOST_DEVICE(int i) {
        const size_t iqpts = static_cast<size_t>(i);

        const auto dpeff = &history_data[iqpts * vdim + dpeff_offset];
        const auto gdots = &history_data[iqpts * vdim + gdot_offset];
        auto lattice_tayfac_dpeff = &lattice_tayfac_dpeffs[iqpts * 2];
        double abs_gdot = 0.0;
        for (size_t islip = 0; islip < gdot_length; islip++) {
            abs_gdot += fabs(gdots[islip]);
        }
        lattice_tayfac_dpeff[0] = (fabs(*dpeff) <= 1.0e-14) ? 0.0 : (abs_gdot / *dpeff);
        lattice_tayfac_dpeff[1] = *dpeff;
    });

    for (const auto& in_fiber_hkl : m_in_fibers) {
        mfem::Vector lattice_tayfac_dpeff_hkl(2);
        auto region_comm = m_sim_state->GetRegionCommunicator(m_region);
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(
            &m_workspace, &in_fiber_hkl, lattice_tayfac_dpeff_hkl, 2, m_class_device, region_comm);
        lattice_tay_facs.push_back(lattice_tayfac_dpeff_hkl(0));
        lattice_dpeff.push_back(lattice_tayfac_dpeff_hkl(1));
    }
}

void LightUp::CalcLatticeDirectionalStiffness(
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> history,
    const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress,
    const size_t strain_offset,
    const size_t quats_offset,
    const size_t rel_vol_offset,
    std::vector<std::array<double, 3>>& lattice_dir_stiff) {
    const size_t vdim = static_cast<size_t>(history->GetVDim());
    const auto history_data = history->Read();
    const auto stress_data = stress->Read();
    m_workspace = 0.0;
    auto lattice_directional_stiffness = m_workspace.Write();

    // Only need to compute this once
    mfem::forall(static_cast<int>(m_npts), [=] MFEM_HOST_DEVICE(int i) {
        const size_t iqpts = static_cast<size_t>(i);

        const auto strain_lat = &history_data[iqpts * vdim + strain_offset];
        const auto quats = &history_data[iqpts * vdim + quats_offset];
        const auto rel_vol = history_data[iqpts * vdim + rel_vol_offset];
        const auto stress_l = &stress_data[iqpts * 6];
        auto lds = &lattice_directional_stiffness[iqpts * 3];

        double strain[6] = {};
        {
            double strainm[3 * 3] = {};
            double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
            const double t1 = ecmech::sqr2i * strain_lat[0];
            const double t2 = ecmech::sqr6i * strain_lat[1];
            //
            // Volume strain is ln(V^e_mean) term aka ln(relative volume)
            // Our plastic deformation has a det(1) aka no change in volume change
            const double elas_vol_strain = log(rel_vol);
            // We output elastic strain formulation such that the relationship
            // between V^e and \varepsilon is just V^e = I + \varepsilon
            strain_m[0][0] = (t1 - t2) + elas_vol_strain;                      // 11
            strain_m[1][1] = (-t1 - t2) + elas_vol_strain;                     // 22
            strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
            strain_m[1][2] = ecmech::sqr2i * strain_lat[4];                    // 23
            strain_m[2][0] = ecmech::sqr2i * strain_lat[3];                    // 31
            strain_m[0][1] = ecmech::sqr2i * strain_lat[2];                    // 12

            strain_m[2][1] = strain_m[1][2];
            strain_m[0][2] = strain_m[2][0];
            strain_m[1][0] = strain_m[0][1];

            double rmat[3 * 3] = {};
            double strain_samp[3 * 3] = {};

            Quat2RMat(quats, rmat);

            snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

            strain_m[0] = &strain_samp[0];
            strain_m[1] = &strain_samp[3];
            strain_m[2] = &strain_samp[6];

            strain[0] = strain_m[0][0];
            strain[1] = strain_m[1][1];
            strain[2] = strain_m[2][2];
            strain[3] = strain_m[1][2];
            strain[4] = strain_m[0][2];
            strain[5] = strain_m[0][1];
        }

        for (size_t ipt = 0; ipt < 3; ipt++) {
            lds[ipt] = (fabs(strain[ipt]) < 1e-12) ? 0.0 : (stress_l[ipt] / strain[ipt]);
        }
    });

    for (const auto& in_fiber_hkl : m_in_fibers) {
        mfem::Vector lattice_direct_stiff(3);
        auto region_comm = m_sim_state->GetRegionCommunicator(m_region);
        [[maybe_unused]] double _ = exaconstit::kernel::ComputeVolAvgTensorFilterFromPartial<true>(
            &m_workspace, &in_fiber_hkl, lattice_direct_stiff, 3, m_class_device, region_comm);
        std::array<double, 3> stiff_tmp;
        for (size_t ipt = 0; ipt < 3; ipt++) {
            stiff_tmp[ipt] = lattice_direct_stiff(static_cast<int>(ipt));
        }
        lattice_dir_stiff.push_back(stiff_tmp);
    }
}