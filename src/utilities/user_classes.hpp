#pragma once

/**
 * @file user_classes.hpp
 * @brief ExaConstit's side of the user classes of exaconstit_user.hpp
 *
 * A user function key of the options file (register_function, initialize_function,
 * function_name) can name a factory instead of a function:
 *     key = { factory = "MakeMyClasses", variant = "my_class" }
 * The factory, a C++ function of the user library, returns an object derived from one of the
 * base classes of exaconstit_user.hpp. This file creates those objects (once, on every rank),
 * builds the inputs they ask for by name as views of ExaConstit's data, and wraps them in the
 * calls ExaConstit already makes for the function form, so the rest of ExaConstit treats both
 * forms alike.
 */

#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "exaconstit_user.hpp"

class SimulationState;

namespace exaconstit {

/**
 * @brief Which values a user class sees
 *
 * INITIALIZE: beginning-of-step data before the first step. POST_PROCESSING: the last converged
 * step (after UpdateModel(), so beginning and end of step hold the same values).
 */
enum class UserPhase { INITIALIZE, POST_PROCESSING };

/**
 * @brief Create and check every user object listed in the options
 *
 * Called by CheckUserFunctions() right after the options are read, on every rank.
 */
void CheckUserObjects(const ExaOptions& options);

/**
 * @brief Inputs a material offers to user classes, under the names of exaconstit_user.hpp
 *
 * Built once per material and phase on a rank that owns elements of the material. Quadrature
 * data is handed over as views of ExaConstit's own memory (HostRead() pointers); only values
 * that do not exist at the quadrature points (grid functions evaluated there, and def_grad of a
 * material that does not cover the whole mesh) are computed into buffers kept here and reused.
 */
class UserInputCatalog {
public:
    UserInputCatalog(std::shared_ptr<SimulationState> sim_state, int region, UserPhase phase);
    // The views refer to this object
    UserInputCatalog(const UserInputCatalog&) = delete;
    UserInputCatalog& operator=(const UserInputCatalog&) = delete;

    /// Names and shapes of every input (Context::available)
    const std::vector<user::FieldInfo>& Available() const {
        return m_available;
    }

    /// Stops the run if a name is unknown or has two meanings, listing every input
    void Check(const std::vector<std::string>& names, const std::string& purpose) const;

    /// Current views of the named inputs (Check() first)
    user::Inputs Gather(const std::vector<std::string>& names) const;

    int NumElements() const {
        return m_num_elements;
    }
    int QptsPerElement() const {
        return m_qpts_per_element;
    }

private:
    /// One input: its name and shape, and how to get a view of its current values
    struct Source {
        user::FieldInfo info;
        std::function<user::Field()> view;
    };

    void Add(user::FieldInfo info, std::function<user::Field()> view);
    void AddQuadratureFunction(const std::string& name,
                               user::Layout layout,
                               std::vector<int> shape,
                               std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf);
    void AddGridFunction(const std::string& name,
                         user::Layout layout,
                         std::shared_ptr<mfem::ParGridFunction> gf);
    user::Field MakeField(const user::FieldInfo& info, const double* data, int stride) const;

    std::shared_ptr<SimulationState> m_sim_state;
    int m_region;
    int m_num_elements = 0;
    int m_qpts_per_element = 0;
    std::vector<Source> m_sources;
    std::vector<user::FieldInfo> m_available;
};

/**
 * @brief The register_function of a material given as a factory table
 *
 * Calls names() and registers every slice with AddQuadratureFunctionStatePair. Run on every
 * rank for every material, also where the rank owns none of its elements.
 */
void RunUserRegisterObject(std::shared_ptr<SimulationState> sim_state,
                           int region,
                           const StateVariables& state_vars,
                           const std::string& material_label);

/**
 * @brief The initialize_function of a material given as a factory table
 *
 * Calls initialize() with the material's own beginning-of-step state variables as output. Run
 * on ranks owning elements of the material, not on restarts.
 */
void RunUserInitializeObject(std::shared_ptr<SimulationState> sim_state,
                             int region,
                             const StateVariables& state_vars,
                             const std::string& material_label);

/**
 * @brief Signature of the post-processing call (the function form's signature)
 */
using PostProcessingCall = std::function<void(
    std::shared_ptr<SimulationState>, int, std::shared_ptr<mfem::expt::PartialQuadratureFunction>)>;

/**
 * @brief Wrap the post-processing object of an entry given as a factory table
 *
 * @param sim_state Simulation state
 * @param index Entry index in [[PostProcessing.user_functions]]
 * @param options The entry
 * @param regions Selected regions (0-based)
 * @return A call that fills one region's values like a post-processing user function
 *
 * Calls inputs() for every selected region this rank owns elements of now, so unknown names stop
 * the run at startup.
 */
PostProcessingCall MakeUserPostProcessingCall(std::shared_ptr<SimulationState> sim_state,
                                              size_t index,
                                              const UserFunctionOptions& options,
                                              const std::vector<int>& regions);

} // namespace exaconstit
