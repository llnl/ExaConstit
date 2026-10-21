// Placeholder for the microstructure_generator tool, so the executable target exists and links.
// The real command-line tool (options file and -m -n -s -o --parts flags, see
// microstructure_options.hpp) is not implemented yet.
#include "microstructure/microstructure_driver.hpp"

#include <iostream>

// Prints a "not implemented" message and returns a nonzero exit code, so scripts that call the
// tool fail loudly instead of silently producing no output.
int main() {
    std::cerr << "microstructure_generator is not implemented yet" << std::endl;
    return 1;
}
