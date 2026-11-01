/**
 * @file write_statistics_mesh.cpp
 * @brief Writes the parallel mesh decomposition of the volume statistics test
 *
 * Usage: write_statistics_mesh <grain file> <nx> <ny> <nz> <prefix> <number of parts>
 *
 * The statistics test needs ranks, rank 0 included, that own no element of material 2, so that
 * the paths for ranks without a region's elements and the gather of the region's results to
 * rank 0 run. Left to the partitioner (METIS), which elements rank 0 gets depends on the
 * machine, so the test reads this decomposition ([Mesh] type = "parallel") instead.
 *
 * The mesh is the auto mesh of the deck (exaconstit::MakeAutoHexMesh on the unit cube, with
 * ExaConstit's boundary attributes), with the grain IDs of the grain file (one per element, in
 * the element order of the auto mesh) as element attributes, as an auto mesh with grain_file
 * gets them. Part 0 holds the elements numbered before the first element of grain 2, so it
 * owns no element of material 2; the other parts split the remaining elements into contiguous
 * blocks of nearly equal size. Each part is written with mfem::MeshPart::Print to
 * mfem::MakeParFilename(prefix, part).
 */

#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "usage: write_statistics_mesh <grain file> <nx> <ny> <nz> <prefix> "
                     "<number of parts>"
                  << std::endl;
        return 1;
    }
    const std::string grain_path = argv[1];
    const std::array<int, 3> nxyz = {std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4])};
    const std::string prefix = argv[5];
    const int num_parts = std::stoi(argv[6]);

    mfem::Mesh mesh = exaconstit::MakeAutoHexMesh(nxyz, {1.0, 1.0, 1.0});
    const int num_elements = mesh.GetNE();

    std::ifstream grain_file(grain_path);
    std::vector<int> grains(num_elements);
    for (int& grain : grains) {
        grain_file >> grain;
    }
    if (!grain_file) {
        std::cerr << grain_path << " has fewer than " << num_elements << " grain IDs" << std::endl;
        return 1;
    }
    for (int e = 0; e < num_elements; e++) {
        mesh.SetAttribute(e, grains[e]);
    }
    mesh.SetAttributes();
    exaconstit::ApplyAutoMeshBoundaryAttributes(mesh);

    // Part 0: the elements before the first one of grain 2; parts 1..num_parts-1: the rest
    const int first_grain2 = static_cast<int>(std::find(grains.begin(), grains.end(), 2) -
                                              grains.begin());
    if (num_parts < 2 || first_grain2 == 0 || first_grain2 == num_elements ||
        num_elements - first_grain2 < num_parts - 1) {
        std::cerr << "need at least 2 parts, elements of grain 2 that are not the first element, "
                     "and an element after it for every part but part 0"
                  << std::endl;
        return 1;
    }
    mfem::Array<int> partitioning(num_elements);
    const int num_rest = num_elements - first_grain2;
    for (int e = 0; e < num_elements; e++) {
        partitioning[e] = (e < first_grain2)
                              ? 0
                              : 1 + static_cast<int>(static_cast<long>(e - first_grain2) *
                                                     (num_parts - 1) / num_rest);
    }

    mfem::MeshPartitioner partitioner(mesh, num_parts, partitioning.GetData());
    mfem::MeshPart mesh_part;
    for (int part = 0; part < num_parts; part++) {
        partitioner.ExtractPart(part, mesh_part);
        std::ofstream part_stream(mfem::MakeParFilename(prefix, part));
        part_stream.precision(17);
        mesh_part.Print(part_stream);
    }
    return 0;
}
