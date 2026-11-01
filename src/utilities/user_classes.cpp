#include "utilities/user_classes.hpp"

#include "sim_state/simulation_state.hpp"
#include "utilities/user_function_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <typeinfo>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace exaconstit {

namespace {

/// Readable C++ class name of an object, e.g. "MyStartingDamage"
std::string ClassName(const user::Function& object) {
    const char* mangled = typeid(object).name();
#if defined(__GNUG__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string name(demangled);
        std::free(demangled);
        return name;
    }
#endif
    return mangled;
}

/// Every user object, by options entry; they live until the end of the run
std::map<std::string, std::shared_ptr<user::Function>>& UserObjects() {
    static std::map<std::string, std::shared_ptr<user::Function>> objects;
    return objects;
}

/**
 * @brief Run a call into user code, stopping the run with a clear message if it throws
 *
 * Only some ranks may throw (their data differs), so this aborts on the throwing rank.
 */
template <typename Call>
auto CallUser(const std::string& what, Call&& call) {
    try {
        return call();
    } catch (const std::exception& e) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MFEM_ABORT("Rank " << rank << ": the " << what << " stopped with an error:\n    "
                           << e.what());
    }
    return decltype(call())();
}

/// The context of a call on a region (everything but time)
user::Context MakeContext(SimulationState& sim_state,
                          int region,
                          const std::string& variant,
                          const UserInputCatalog* catalog) {
    const auto& material = sim_state.GetOptions().materials[static_cast<size_t>(region)];
    user::Context context;
    context.region = region;
    context.material_name = material.material_name;
    context.variant = variant;
    context.num_state_vars = material.state_vars.num_vars;
    for (const auto& [name, slice] : sim_state.GetQuadratureFunctionStatePairs(region)) {
        context.state_variables.push_back({name,
                                           slice.first,
                                           slice.second,
                                           sim_state.GetQuadratureFunctionStatePlot(name, region)});
    }
    if (catalog) {
        context.num_elements = catalog->NumElements();
        context.qpts_per_element = catalog->QptsPerElement();
        context.available = catalog->Available();
    }
    return context;
}

/// Readable shape, e.g. "[3,3]"
std::string ShapeString(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        out += (i ? "," : "") + std::to_string(shape[i]);
    }
    return out + "]";
}

const char* LayoutName(user::Layout layout) {
    switch (layout) {
    case user::Layout::scalar:
        return "scalar";
    case user::Layout::vector:
        return "vector";
    case user::Layout::voigt:
        return "voigt";
    case user::Layout::matrix:
        return "matrix";
    case user::Layout::quaternion:
        return "quaternion";
    case user::Layout::list:
        return "list";
    }
    return "list";
}

/// Readable form of the factory signature, for error messages
std::string UserFactorySignature(const std::string& factory) {
    return "std::unique_ptr<exaconstit::user::Function> " + factory +
           "(const exaconstit::user::FactoryArgs& args)";
}

/**
 * @brief The user object of one options entry, created on first use and then reused
 *
 * @param key Unique key of the options entry (e.g. "register_function:2")
 * @param library_path Library holding the factory
 * @param factory Factory name and variant
 * @param purpose Where the object is used, for error messages
 * @param base_name Name of Base, for error messages
 * @return The object; the run stops (on every rank) if the factory is missing, does not know
 *         the variant, throws, or returns an object not derived from Base
 */
template <typename Base>
std::shared_ptr<Base> GetUserObject(const std::string& key,
                                    const std::filesystem::path& library_path,
                                    const UserFactoryOptions& factory,
                                    const std::string& purpose,
                                    const char* base_name) {
    auto& objects = UserObjects();
    auto found = objects.find(key);
    if (found == objects.end()) {
        // The return type is not part of a C++ function's symbol, so the factory is found
        // through a function type with the same arguments, then called with its real type
        using LookupType = void (*)(const user::FactoryArgs&);
        const auto symbol = LoadUserFunction<LookupType>(
            library_path, factory.factory, UserFactorySignature(factory.factory), purpose);
        const auto make = reinterpret_cast<user::Factory>(symbol);

        user::FactoryArgs args;
        args.variant = factory.variant;
        std::unique_ptr<user::Function> object;
        try {
            object = make(args);
        } catch (const std::exception& e) {
            ABORT_0_OPT("The factory '" << factory.factory << "' of the " << purpose
                                        << " stopped with an error for variant '" << factory.variant
                                        << "':\n    " << e.what());
        }
        MFEM_VERIFY_0(object != nullptr,
                      "The factory '" << factory.factory << "' of the " << purpose
                                      << " does not know the variant '" << factory.variant
                                      << "' (it returned nullptr)");
        found = objects.emplace(key, std::shared_ptr<user::Function>(std::move(object))).first;
    }

    auto typed = std::dynamic_pointer_cast<Base>(found->second);
    MFEM_VERIFY_0(typed != nullptr,
                  "The " << purpose << " (" << factory.label() << ") is a "
                         << ClassName(*found->second) << ", which is not derived from "
                         << base_name);
    return typed;
}

std::string RegisterPurpose(const std::string& material_label) {
    return "register_function of material " + material_label;
}
std::string InitializePurpose(const std::string& material_label) {
    return "initialize_function of material " + material_label;
}
std::string PostProcessingPurpose(const UserFunctionOptions& options) {
    return "post-processing user function of entry '" + options.name + "'";
}

/// The object of each key (region: 0-based material index; index: entry index)
std::shared_ptr<user::RegisterFunction>
RegisterObject(const StateVariables& vars, size_t region, const std::string& material_label) {
    return GetUserObject<user::RegisterFunction>("register_function:" + std::to_string(region),
                                                 vars.library_path,
                                                 vars.register_factory,
                                                 RegisterPurpose(material_label),
                                                 "exaconstit::user::RegisterFunction");
}
std::shared_ptr<user::InitializeFunction>
InitializeObject(const StateVariables& vars, size_t region, const std::string& material_label) {
    return GetUserObject<user::InitializeFunction>("initialize_function:" + std::to_string(region),
                                                   vars.library_path,
                                                   vars.initialize_factory,
                                                   InitializePurpose(material_label),
                                                   "exaconstit::user::InitializeFunction");
}
std::shared_ptr<user::PostProcessingFunction> PostProcessingObject(const UserFunctionOptions& entry,
                                                                   size_t index) {
    return GetUserObject<user::PostProcessingFunction>("user_functions:" + std::to_string(index),
                                                       entry.library_path,
                                                       entry.factory,
                                                       PostProcessingPurpose(entry),
                                                       "exaconstit::user::PostProcessingFunction");
}

} // namespace

void CheckUserObjects(const ExaOptions& options) {
    for (size_t region = 0; region < options.materials.size(); region++) {
        const auto& material = options.materials[region];
        if (!material.state_vars.register_factory.empty()) {
            RegisterObject(material.state_vars, region, MaterialLabel(material));
        }
        if (!material.state_vars.initialize_factory.empty()) {
            InitializeObject(material.state_vars, region, MaterialLabel(material));
        }
    }
    const auto& entries = options.post_processing.user_functions;
    for (size_t index = 0; index < entries.size(); index++) {
        if (!entries[index].factory.empty()) {
            PostProcessingObject(entries[index], index);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------------

UserInputCatalog::UserInputCatalog(std::shared_ptr<SimulationState> sim_state,
                                   int region,
                                   UserPhase phase)
    : m_sim_state(std::move(sim_state)), m_region(region) {
    const bool initialize = (phase == UserPhase::INITIALIZE);
    const char* step = initialize ? "_beg" : "_end";
    auto state_vars = m_sim_state->GetQuadratureFunction(std::string("state_var") + step, region);
    MFEM_VERIFY(state_vars, "Material " << region + 1 << " has no elements on this rank");
    const auto space = state_vars->GetPartialSpaceShared();
    m_num_elements = space->GetNumLocalElements();
    m_qpts_per_element = (m_num_elements > 0) ? space->GetIntRule(0).GetNPoints() : 0;
    MFEM_VERIFY(space->GetSize() == m_num_elements * m_qpts_per_element,
                "User classes need the same number of quadrature points in every element");

    // The curated names
    AddQuadratureFunction(
        "cauchy_stress",
        user::Layout::voigt,
        {6},
        m_sim_state->GetQuadratureFunction(std::string("cauchy_stress") + step, region));
    {
        // The global deformation gradient is current at every step (the per-material copy
        // exists only with some volume averages). A material covering the whole mesh on this
        // rank shares its layout, so it is handed over directly; otherwise its points are
        // gathered into a buffer at every call.
        auto global = m_sim_state->GetQuadratureFunction("kinetic_grads");
        if (space->IsFullSpace()) {
            AddQuadratureFunction("def_grad", user::Layout::matrix, {3, 3}, global);
        } else {
            auto buffer = std::make_shared<mfem::expt::PartialQuadratureFunction>(space, 9, 0.0);
            user::FieldInfo info{"def_grad", user::Layout::matrix, {3, 3}, 9};
            Add(info, [this, info, global, buffer]() {
                *buffer = *static_cast<const mfem::QuadratureFunction*>(global.get());
                return MakeField(info, buffer->HostRead(), 9);
            });
        }
    }
    const int num_state_vars = state_vars->GetVDim();
    AddQuadratureFunction("state_vars", user::Layout::list, {num_state_vars}, state_vars);
    for (const auto& [name, slice] : m_sim_state->GetQuadratureFunctionStatePairs(region)) {
        const int offset = slice.first;
        const int length = slice.second;
        const user::Layout layout = (length == 1)                      ? user::Layout::scalar
                                    : (name == "quats" && length == 4) ? user::Layout::quaternion
                                                                       : user::Layout::list;
        user::FieldInfo info{name, layout, {length}, length};
        Add(info, [this, info, state_vars, offset, num_state_vars]() {
            return MakeField(info, state_vars->HostRead() + offset, num_state_vars);
        });
    }
    AddGridFunction("reference_coords", user::Layout::vector, m_sim_state->GetRefCoords());
    AddGridFunction("coords", user::Layout::vector, m_sim_state->GetCurrentCoords());
    AddGridFunction("displacement", user::Layout::vector, m_sim_state->GetDisplacement());
    AddGridFunction("velocity", user::Layout::vector, m_sim_state->GetVelocity());
    AddGridFunction("grains", user::Layout::scalar, m_sim_state->GetGrains());

    // Any other quadrature data of the material on its quadrature points, under its own name
    // (the element averages "*_avg" live on another space and are left out by the space check)
    static const std::vector<std::string> curated = {"cauchy_stress_beg",
                                                     "cauchy_stress_end",
                                                     "state_var_beg",
                                                     "state_var_end",
                                                     "kinetic_grads",
                                                     "tangent_stiffness"};
    for (const auto& name : m_sim_state->GetQuadratureFunctionNames(region)) {
        if (std::find(curated.begin(), curated.end(), name) != curated.end()) {
            continue;
        }
        auto qf = m_sim_state->GetQuadratureFunction(name, region);
        if (!qf || qf->GetPartialSpaceShared() != space) {
            continue;
        }
        AddQuadratureFunction(name, user::Layout::list, {qf->GetVDim()}, qf);
    }
}

void UserInputCatalog::Add(user::FieldInfo info, std::function<user::Field()> view) {
    m_available.push_back(info);
    m_sources.push_back({std::move(info), std::move(view)});
}

user::Field
UserInputCatalog::MakeField(const user::FieldInfo& info, const double* data, int stride) const {
    user::Field field;
    static_cast<user::FieldInfo&>(field) = info;
    field.num_elements = m_num_elements;
    field.qpts_per_element = m_qpts_per_element;
    field.data = data;
    field.stride = stride;
    return field;
}

void UserInputCatalog::AddQuadratureFunction(
    const std::string& name,
    user::Layout layout,
    std::vector<int> shape,
    std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf) {
    if (!qf) {
        return;
    }
    const int vdim = qf->GetVDim();
    user::FieldInfo info{name, layout, std::move(shape), vdim};
    // HostRead() at every call: the pointer moves with UpdateModel()'s swaps, and on GPU runs
    // it brings the current values to the host
    Add(info, [this, info, qf, vdim]() {
        return MakeField(info, qf->HostRead(), vdim);
    });
}

void UserInputCatalog::AddGridFunction(const std::string& name,
                                       user::Layout layout,
                                       std::shared_ptr<mfem::ParGridFunction> gf) {
    if (!gf) {
        return;
    }
    const int vdim = gf->ParFESpace()->GetVDim();
    user::FieldInfo info{name, layout, {vdim}, vdim};
    auto space =
        m_sim_state->GetQuadratureFunction("state_var_beg", m_region)->GetPartialSpaceShared();
    auto mesh = m_sim_state->GetMesh();
    auto buffer = std::make_shared<std::vector<double>>(static_cast<size_t>(m_num_elements) *
                                                        m_qpts_per_element * vdim);
    // The grid function evaluated at the material's quadrature points, element by element
    Add(info, [this, info, gf, space, mesh, buffer, vdim]() {
        gf->HostRead();
        const int* local2global = space->GetLocal2Global().HostRead();
        mfem::DenseMatrix values;
        for (int e = 0; e < m_num_elements; e++) {
            auto* transformation = mesh->GetElementTransformation(local2global[e]);
            const auto& rule = space->GetIntRule(e);
            gf->GetVectorValues(*transformation, rule, values);
            for (int q = 0; q < m_qpts_per_element; q++) {
                for (int c = 0; c < vdim; c++) {
                    (*buffer)[(static_cast<size_t>(e) * m_qpts_per_element + q) * vdim + c] =
                        values(c, q);
                }
            }
        }
        return MakeField(info, buffer->data(), vdim);
    });
}

void UserInputCatalog::Check(const std::vector<std::string>& names,
                             const std::string& purpose) const {
    for (const auto& name : names) {
        const auto count = std::count_if(
            m_sources.begin(), m_sources.end(), [&name](const Source& s) {
                return s.info.name == name;
            });
        if (count == 1) {
            continue;
        }
        std::ostringstream list;
        for (const auto& info : m_available) {
            list << "\n    " << info.name << "  " << LayoutName(info.layout) << " "
                 << ShapeString(info.shape);
        }
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (count == 0) {
            MFEM_ABORT("Rank " << rank << ": the " << purpose << " asks for the input '" << name
                               << "', which material " << m_sim_state->GetRegionName(m_region)
                               << " does not have. Its inputs are:" << list.str());
        }
        MFEM_ABORT("Rank " << rank << ": the " << purpose << " asks for the input '" << name
                           << "', which names two different data sets of material "
                           << m_sim_state->GetRegionName(m_region)
                           << " (a state variable and quadrature data); rename the state "
                              "variable. Its inputs are:"
                           << list.str());
    }
}

user::Inputs UserInputCatalog::Gather(const std::vector<std::string>& names) const {
    user::Inputs inputs;
    for (const auto& name : names) {
        for (const auto& source : m_sources) {
            if (source.info.name == name) {
                inputs.fields.emplace(name, source.view());
                break;
            }
        }
    }
    return inputs;
}

// ---------------------------------------------------------------------------------------------
// The three keys
// ---------------------------------------------------------------------------------------------

void RunUserRegisterObject(std::shared_ptr<SimulationState> sim_state,
                           int region,
                           const StateVariables& state_vars,
                           const std::string& material_label) {
    const std::string purpose = RegisterPurpose(material_label);
    auto object = RegisterObject(state_vars, static_cast<size_t>(region), material_label);
    const auto context = MakeContext(
        *sim_state, region, state_vars.register_factory.variant, nullptr);
    const auto names = CallUser(purpose + " (names)", [&]() {
        return object->names(context);
    });
    for (const auto& name : names) {
        // The slice itself (inside the state variables, no overlap) is checked there
        const bool added = sim_state->AddQuadratureFunctionStatePair(
            name.name, {name.offset, name.length}, region, name.plot);
        MFEM_VERIFY_0(added,
                      "The " << purpose << " names the state variable '" << name.name
                             << "', which material " << material_label
                             << " already has; choose another name");
    }
}

void RunUserInitializeObject(std::shared_ptr<SimulationState> sim_state,
                             int region,
                             const StateVariables& state_vars,
                             const std::string& material_label) {
    const std::string purpose = InitializePurpose(material_label);
    auto object = InitializeObject(state_vars, static_cast<size_t>(region), material_label);
    const UserInputCatalog catalog(sim_state, region, UserPhase::INITIALIZE);
    auto context = MakeContext(*sim_state, region, state_vars.initialize_factory.variant, &catalog);
    context.time = sim_state->GetTime();
    context.delta_time = 0.0;

    const auto names = CallUser(purpose + " (inputs)", [&]() {
        return object->inputs(context);
    });
    catalog.Check(names, purpose);
    const auto inputs = catalog.Gather(names);

    // The output is the material's own state variable array: no copy in or out
    auto state_var_beg = sim_state->GetQuadratureFunction("state_var_beg", region);
    user::OutputField output;
    static_cast<user::FieldInfo&>(output) = user::FieldInfo{
        "state_vars", user::Layout::list, {state_var_beg->GetVDim()}, state_var_beg->GetVDim()};
    output.num_elements = catalog.NumElements();
    output.qpts_per_element = catalog.QptsPerElement();
    output.data = state_var_beg->HostReadWrite();
    output.stride = state_var_beg->GetVDim();
    CallUser(purpose, [&]() {
        object->initialize(context, inputs, output);
    });
}

PostProcessingCall MakeUserPostProcessingCall(std::shared_ptr<SimulationState> sim_state,
                                              size_t index,
                                              const UserFunctionOptions& options,
                                              const std::vector<int>& regions) {
    const std::string purpose = PostProcessingPurpose(options);
    auto object = PostProcessingObject(options, index);

    // Per region this rank owns elements of: the inputs offered, the context and the names asked
    // for (asked once, here, so an unknown name stops the run at startup)
    struct RegionData {
        std::shared_ptr<UserInputCatalog> catalog;
        user::Context context;
        std::vector<std::string> names;
    };
    auto data = std::make_shared<std::map<int, RegionData>>();
    for (const int region : regions) {
        if (!sim_state->IsRegionActive(region)) {
            continue;
        }
        RegionData region_data;
        region_data.catalog = std::make_shared<UserInputCatalog>(
            sim_state, region, UserPhase::POST_PROCESSING);
        region_data.context = MakeContext(
            *sim_state, region, options.factory.variant, region_data.catalog.get());
        const std::string where = purpose + ", material " + sim_state->GetRegionName(region);
        region_data.names = CallUser(where + " (inputs)", [&]() {
            return object->inputs(region_data.context);
        });
        region_data.catalog->Check(region_data.names, where);
        data->emplace(region, std::move(region_data));
    }

    const int num_components = options.num_components;
    return [object, data, purpose, num_components](
               std::shared_ptr<SimulationState> state,
               int region,
               std::shared_ptr<mfem::expt::PartialQuadratureFunction> values) {
        auto& region_data = data->at(region);
        auto& context = region_data.context;
        context.time = state->GetTime();
        context.delta_time = state->GetDeltaTime();
        const auto inputs = region_data.catalog->Gather(region_data.names);

        // The output is ExaConstit's own array, zero-filled in place
        user::OutputField output;
        static_cast<user::FieldInfo&>(output) = user::FieldInfo{
            "output", user::Layout::list, {num_components}, num_components};
        output.num_elements = region_data.catalog->NumElements();
        output.qpts_per_element = region_data.catalog->QptsPerElement();
        output.data = values->HostWrite();
        output.stride = num_components;
        std::fill(output.data, output.data + values->Size(), 0.0);
        const std::string where = purpose + ", material " + state->GetRegionName(region);
        CallUser(where, [&]() {
            object->compute(context, inputs, output);
        });
    };
}

} // namespace exaconstit
