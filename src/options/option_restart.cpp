#include "options/option_parser_v2.hpp"
#include "options/option_util.hpp"

#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

RestartOptions RestartOptions::from_toml(const toml::value& toml_input) {
    RestartOptions options;

    if (toml_input.contains("write")) {
        options.write = toml::find<bool>(toml_input, "write");
    }

    if (toml_input.contains("interval")) {
        options.interval = toml::find<int>(toml_input, "interval");
    }

    if (toml_input.contains("write_final")) {
        options.write_final = toml::find<bool>(toml_input, "write_final");
    }

    if (toml_input.contains("keep_last")) {
        options.keep_last = toml::find<int>(toml_input, "keep_last");
    }

    if (toml_input.contains("read")) {
        options.read = toml::find<bool>(toml_input, "read");
    }

    if (toml_input.contains("read_cycle")) {
        options.read_cycle = toml::find<size_t>(toml_input, "read_cycle");
    }

    if (toml_input.contains("directory")) {
        options.directory = toml::find<std::string>(toml_input, "directory");
    }

    if (toml_input.contains("read_directory")) {
        options.read_directory = toml::find<std::string>(toml_input, "read_directory");
    }

    return options;
}

bool RestartOptions::validate() const {
    // Checkpoints are Axom Sidre files, so a build without Axom can neither write nor read
    // them. Without this check the request would be silently ignored.
#if !defined(EXACONSTIT_USE_AXOM)
    if (write || read) {
        WARNING_0_OPT("Error: The Restart table asks to write or read checkpoints, but ExaConstit "
                      "was built without Axom (configure with AXOM_DIR pointing at an Axom "
                      "built with HDF5).");
        return false;
    }
#endif

    if (interval < 0) {
        WARNING_0_OPT("Error: Restart table has `interval` set to a value less than 0 (use 0 to "
                      "disable interval checkpoints).");
        return false;
    }

    if (keep_last < 0) {
        WARNING_0_OPT("Error: Restart table has `keep_last` set to a value less than 0 (use 0 to "
                      "keep every checkpoint).");
        return false;
    }

    if ((write || read) && directory.empty()) {
        WARNING_0_OPT("Error: Restart table has an empty `directory`.");
        return false;
    }

    if (read_directory.has_value() && read_directory->empty()) {
        WARNING_0_OPT("Error: Restart table has an empty `read_directory`.");
        return false;
    }

    // Checking the directory here gives a clear message before any setup work is done. Which
    // checkpoint inside it is used, and whether it is complete, is decided when the
    // simulation starts.
    if (read && !fs::is_directory(GetReadDirectory())) {
        std::ostringstream err;
        err << "Error: Restart table has `read` = true, but the checkpoint directory '"
            << GetReadDirectory().string() << "' does not exist.";
        WARNING_0_OPT(err.str());
        return false;
    }

    // The remaining cases are allowed but probably not what was intended.
    if (read_directory.has_value() && !read) {
        WARNING_0_OPT("Restart table sets `read_directory` but `read` = false, so it is ignored.");
    }

    if (read_cycle.has_value() && !read) {
        WARNING_0_OPT("Restart table sets `read_cycle` but `read` = false, so it is ignored.");
    }

    if (write && interval == 0 && !write_final) {
        WARNING_0_OPT("Restart table has `write` = true, but `interval` = 0 and `write_final` = "
                      "false, so no checkpoints will be written.");
    }

    return true;
}
