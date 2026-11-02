#include <cstdlib>
#include <iostream>
#include <string>

#include "axom/sidre.hpp"

// Diagnostic tool: lists the top-level groups of an ExaConstit checkpoint rank file and how
// each quadrature function is stored in it, either as values (with their count) or as
// `copy_of` another function (see RestartCheckpoint::WriteQuadratureFunctions()).
//
// Usage: inspect_restart_rank_file <checkpoint directory>/cycle_<N>/rank_<R>.hdf5
//
// Only the file's structure is read; the stored arrays themselves are not loaded.

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <rank_file.hdf5>" << std::endl;
        return EXIT_FAILURE;
    }

    axom::sidre::DataStore store;
    axom::sidre::Group* root = store.getRoot();
    if (!root->load(argv[1], "sidre_hdf5")) {
        std::cerr << "Unable to load " << argv[1] << " as a Sidre HDF5 file" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Top-level groups:";
    for (axom::sidre::Group& group : root->groups()) {
        std::cout << " " << group.getName();
    }
    std::cout << std::endl;

    if (!root->hasGroup("qfuncs")) {
        std::cout << "No qfuncs/ group; this is not an ExaConstit checkpoint rank file"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Quadrature functions:" << std::endl;
    for (axom::sidre::Group& entry : root->getGroup("qfuncs")->groups()) {
        std::string storage;
        if (entry.hasView("copy_of")) {
            storage = std::string("copy_of ") + entry.getView("copy_of")->getString();
        } else if (entry.hasView("values")) {
            storage = "values  " + std::to_string(entry.getView("values")->getNumElements()) +
                      " doubles";
        } else {
            storage = "(no values)";
        }
        std::cout << "  " << entry.getName() << ": " << storage << std::endl;
    }
    return EXIT_SUCCESS;
}
