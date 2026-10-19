#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

// Time options nested classes implementation
TimeOptions::AutoTimeOptions
TimeOptions::AutoTimeOptions::from_toml(const toml::value& toml_input) {
    AutoTimeOptions options;

    if (toml_input.contains("dt_start")) {
        options.dt_start = toml::find<double>(toml_input, "dt_start");
    }

    if (toml_input.contains("dt_min")) {
        options.dt_min = toml::find<double>(toml_input, "dt_min");
    }

    if (toml_input.contains("dt_max")) {
        options.dt_max = toml::find<double>(toml_input, "dt_max");
    }

    if (toml_input.contains("dt_scale")) {
        options.dt_scale = toml::find<double>(toml_input, "dt_scale");
    }

    if (toml_input.contains("t_final")) {
        options.t_final = toml::find<double>(toml_input, "t_final");
    }

    return options;
}

TimeOptions::FixedTimeOptions
TimeOptions::FixedTimeOptions::from_toml(const toml::value& toml_input) {
    FixedTimeOptions options;

    if (toml_input.contains("dt")) {
        options.dt = toml::find<double>(toml_input, "dt");
    }

    if (toml_input.contains("t_final")) {
        options.t_final = toml::find<double>(toml_input, "t_final");
    }

    return options;
}

TimeOptions::CustomTimeOptions
TimeOptions::CustomTimeOptions::from_toml(const toml::value& toml_input) {
    CustomTimeOptions options;

    if (toml_input.contains("nsteps")) {
        options.nsteps = toml::find<int>(toml_input, "nsteps");
    }

    if (toml_input.contains("floc")) {
        options.floc = toml::find<std::string>(toml_input, "floc");
    }

    return options;
}

bool TimeOptions::CustomTimeOptions::load_custom_dt_values() {
    try {
        std::ifstream file(floc);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + floc.string());
        }

        dt_values.clear();
        double value;
        while (file >> value) {
            if (value <= 0) {
                WARNING_0_OPT("Error: `Time.Custom` file had value less than 0");
                return false;
            }
            dt_values.push_back(value);
        }
        if (dt_values.size() >= static_cast<size_t>(nsteps)) {
            dt_values.resize(static_cast<size_t>(nsteps));
            return true;
        } else {
            std::ostringstream err;
            err << "Error: `Time.Custom` floc: " << floc << " provided does not contain "
                << std::to_string(nsteps)
                << " steps but rather: " << std::to_string(dt_values.size());
            WARNING_0_OPT(err.str());
            return false;
        }
    } catch (...) {
        return false;
    }
}

void TimeOptions::determine_time_type(const std::optional<TimeStepType>& requested_type) {
    // An explicit time_type takes precedence over which sections are present. An unknown
    // string has already been warned about and comes in as NOTYPE: fall back to the priority.
    if (requested_type.has_value() && requested_type.value() != TimeStepType::NOTYPE) {
        time_type = requested_type.value();
        time_type_explicit = true;
        if (time_type == TimeStepType::FIXED && !fixed_time.has_value()) {
            fixed_time = FixedTimeOptions{};
        } else if ((time_type == TimeStepType::AUTO && !auto_time.has_value()) ||
                   (time_type == TimeStepType::CUSTOM && !custom_time.has_value())) {
            const std::string section = (time_type == TimeStepType::AUTO) ? "Auto" : "Custom";
            WARNING_0_OPT("Error: Time.time_type requests " + section +
                          " time stepping but no [Time." + section + "] section was provided");
        }
        return;
    }

    time_type_explicit = false;
    if (custom_time.has_value()) {
        time_type = TimeStepType::CUSTOM;
    } else if (auto_time.has_value()) {
        time_type = TimeStepType::AUTO;
    } else if (fixed_time.has_value()) {
        time_type = TimeStepType::FIXED;
    } else {
        // Default to fixed with defaults
        fixed_time = FixedTimeOptions{};
        time_type = TimeStepType::FIXED;
    }
}

TimeOptions TimeOptions::from_toml(const toml::value& toml_input) {
    TimeOptions options;

    // Check for nested time stepping sections
    if (toml_input.contains("Auto")) {
        options.auto_time = AutoTimeOptions::from_toml(toml::find(toml_input, "Auto"));
    }

    if (toml_input.contains("Fixed")) {
        options.fixed_time = FixedTimeOptions::from_toml(toml::find(toml_input, "Fixed"));
    }

    if (toml_input.contains("Custom")) {
        options.custom_time = CustomTimeOptions::from_toml(toml::find(toml_input, "Custom"));
    }

    // Determine which time stepping mode to use
    std::optional<TimeStepType> requested_type;
    if (toml_input.contains("time_type")) {
        requested_type = string_to_time_step_type(toml::find<std::string>(toml_input, "time_type"));
    }
    options.determine_time_type(requested_type);

    return options;
}

bool TimeOptions::validate() {
    switch (time_type) {
    case TimeStepType::CUSTOM: {
        if (!custom_time.has_value()) {
            return false;
        }
        return custom_time->load_custom_dt_values();
    }
    case TimeStepType::AUTO: {
        if (!auto_time.has_value()) {
            return false;
        }
        const bool auto_time_good = auto_time->dt_min > 0.0 && auto_time->dt_start > 0.0 &&
                                    auto_time->dt_max > auto_time->dt_min &&
                                    auto_time->dt_scale > 0.0 && auto_time->dt_scale < 1.0 &&
                                    auto_time->t_final >= auto_time->dt_start;
        if (!auto_time_good) {
            std::ostringstream err;
            err << "Error: Time.Auto had issues make sure it satisfies the following conditions:"
                << std::endl;
            err << "       dt_min > 0.0; dt_start > 0.0; dt_max > dt_min;" << std::endl;
            err << "       dt_scale > 0.0; dt_scale < 1.0; t_final >= dt_start";
            WARNING_0_OPT(err.str());
        }
        return auto_time_good;
    }
    case TimeStepType::FIXED: {
        if (!fixed_time.has_value()) {
            return false;
        }
        const bool fixed_time_good = fixed_time->dt > 0.0 && fixed_time->t_final >= fixed_time->dt;
        if (!fixed_time_good) {
            std::ostringstream err;
            err << "Error: Time.Fixed had issues make sure it satisfies the following conditions:"
                << std::endl;
            err << "       dt > 0.0; t_final > dt" << std::endl;
            WARNING_0_OPT(err.str());
        }
        return fixed_time_good;
    }
    default:
        return false;
    }
    return true;
}
