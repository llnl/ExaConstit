#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

/**
 * @brief Convert string to MeshType enum
 * @param str String representation of mesh type ("file", "auto")
 * @return Corresponding MeshType enum value
 */
MeshType string_to_mesh_type(const std::string& str) {
    static const std::map<std::string, MeshType> mapping = {
        {"file", MeshType::FILE},
        {"auto", MeshType::AUTO},
    };

    return string_to_enum(str, mapping, MeshType::NOTYPE, "mesh");
}

/**
 * @brief Convert string to TimeStepType enum
 * @param str String representation of time step type ("fixed", "auto", "custom")
 * @return Corresponding TimeStepType enum value
 */
TimeStepType string_to_time_step_type(const std::string& str) {
    static const std::map<std::string, TimeStepType> mapping = {{"fixed", TimeStepType::FIXED},
                                                                {"auto", TimeStepType::AUTO},
                                                                {"custom", TimeStepType::CUSTOM}};

    return string_to_enum(str, mapping, TimeStepType::NOTYPE, "time step");
}

/**
 * @brief Convert string to OriType enum
 * @param str String representation of orientation type ("quat", "custom", "euler")
 * @return Corresponding OriType enum value
 */
OriType string_to_ori_type(const std::string& str) {
    static const std::map<std::string, OriType> mapping = {
        {"quat", OriType::QUAT}, {"custom", OriType::CUSTOM}, {"euler", OriType::EULER}};

    return string_to_enum(str, mapping, OriType::NOTYPE, "orientation type");
}

/**
 * @brief Convert string to MechType enum
 * @param str String representation of mechanics type ("umat", "exacmech")
 * @return Corresponding MechType enum value
 */
MechType string_to_mech_type(const std::string& str) {
    static const std::map<std::string, MechType> mapping = {{"umat", MechType::UMAT},
                                                            {"exacmech", MechType::EXACMECH}};

    return string_to_enum(str, mapping, MechType::NOTYPE, "material model");
}

/**
 * @brief Convert string to RTModel enum
 * @param str String representation of runtime model ("CPU", "OPENMP", "GPU")
 * @return Corresponding RTModel enum value
 */
RTModel string_to_rt_model(const std::string& str) {
    static const std::map<std::string, RTModel> mapping = {
        {"CPU", RTModel::CPU}, {"OPENMP", RTModel::OPENMP}, {"GPU", RTModel::GPU}};

    return string_to_enum(str, mapping, RTModel::NOTYPE, "runtime model");
}

/**
 * @brief Convert string to AssemblyType enum
 * @param str String representation of assembly type ("FULL", "PA", "EA")
 * @return Corresponding AssemblyType enum value
 */
AssemblyType string_to_assembly_type(const std::string& str) {
    static const std::map<std::string, AssemblyType> mapping = {
        {"FULL", AssemblyType::FULL}, {"PA", AssemblyType::PA}, {"EA", AssemblyType::EA}};

    return string_to_enum(str, mapping, AssemblyType::NOTYPE, "assembly");
}

/**
 * @brief Convert string to IntegrationModel enum
 * @param str String representation of integration model ("FULL", "BBAR")
 * @return Corresponding IntegrationModel enum value
 */
IntegrationModel string_to_integration_model(const std::string& str) {
    static const std::map<std::string, IntegrationModel> mapping = {
        {"FULL", IntegrationModel::DEFAULT}, {"BBAR", IntegrationModel::BBAR}};

    return string_to_enum(str, mapping, IntegrationModel::NOTYPE, "integration model");
}

/**
 * @brief Convert string to LinearSolverType enum
 * @param str String representation of linear solver type ("CG", "GMRES", "MINRES", "BICGSTAB")
 * @return Corresponding LinearSolverType enum value
 */
LinearSolverType string_to_linear_solver_type(const std::string& str) {
    static const std::map<std::string, LinearSolverType> mapping = {
        {"CG", LinearSolverType::CG},
        {"PCG", LinearSolverType::CG},
        {"GMRES", LinearSolverType::GMRES},
        {"MINRES", LinearSolverType::MINRES},
        {"BICGSTAB", LinearSolverType::BICGSTAB}};

    return string_to_enum(str, mapping, LinearSolverType::NOTYPE, "linear solver");
}

/**
 * @brief Convert string to NonlinearSolverType enum
 * @param str String representation of nonlinear solver type ("NR", "NRLS", "TRDOG")
 * @return Corresponding NonlinearSolverType enum value
 */
NonlinearSolverType string_to_nonlinear_solver_type(const std::string& str) {
    static const std::map<std::string, NonlinearSolverType> mapping = {
        {"NR", NonlinearSolverType::NR},
        {"NRLS", NonlinearSolverType::NRLS},
        {"TRDOG", NonlinearSolverType::TRDOG}};

    return string_to_enum(str, mapping, NonlinearSolverType::NOTYPE, "nonlinear solver");
}

/**
 * @brief Convert string to PreconditionerType enum
 * @param str String representation of preconditioner type ("JACOBI", "AMG", "ILU", "L1GS",
 * "CHEBYSHEV")
 * @return Corresponding PreconditionerType enum value
 */
PreconditionerType string_to_preconditioner_type(const std::string& str) {
    static const std::map<std::string, PreconditionerType> mapping = {
        {"JACOBI", PreconditionerType::JACOBI},
        {"AMG", PreconditionerType::AMG},
        {"ILU", PreconditionerType::ILU},
        {"L1GS", PreconditionerType::L1GS},
        {"CHEBYSHEV", PreconditionerType::CHEBYSHEV},
    };

    return string_to_enum(str, mapping, PreconditionerType::NOTYPE, "preconditioner");
}

/**
 * @brief Convert string to LatticeType enum
 * @param str String representation of lattice type ("CUBIC", "HEXAGONAL", "TRIGONAL",
 *             "RHOMBOHEDRAL", "TETRAGONAL", "ORTHORHOMBIC", "MONOCLINIC", "TRICLINIC")
 * @return Corresponding LatticeType enum value
 */
LatticeType string_to_lattice_type(const std::string& str) {
    static const std::map<std::string, LatticeType> mapping = {
        {"CUBIC", LatticeType::CUBIC},
        {"HEXAGONAL", LatticeType::HEXAGONAL},
        {"TRIGONAL", LatticeType::TRIGONAL},
        {"RHOMBOHEDRAL", LatticeType::RHOMBOHEDRAL},
        {"TETRAGONAL", LatticeType::TETRAGONAL},
        {"ORTHORHOMBIC", LatticeType::ORTHORHOMBIC},
        {"MONOCLINIC", LatticeType::MONOCLINIC},
        {"TRICLINIC", LatticeType::TRICLINIC}};

    return string_to_enum(str, mapping, LatticeType::CUBIC, "lattice type");
}