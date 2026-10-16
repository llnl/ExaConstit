
#ifndef BCDATA
#define BCDATA

#include "mfem.hpp"
#include "mfem/linalg/vector.hpp"

#include <array>
#include <fstream>

/**
 * @brief Individual boundary condition data container and processor
 *
 * @details This class stores and processes data for a single boundary condition instance.
 * It handles the application of Dirichlet boundary conditions for velocity-based formulations
 * and manages component-wise scaling for different constraint types.
 *
 * The class supports component-wise boundary conditions where different velocity components
 * can be constrained independently using a component ID system:
 * - 0: No constraints
 * - 1: X-component only
 * - 2: Y-component only
 * - 3: Z-component only
 * - 4: X and Y components
 * - 5: Y and Z components
 * - 6: X and Z components
 * - 7: All components (X, Y, Z)
 */
class BCData {
public:
    /**
     * @brief Default constructor
     *
     * @details Initializes a BCData object with default values. Currently a stub
     * implementation that should be expanded based on initialization requirements.
     */
    BCData();

    /**
     * @brief Destructor
     *
     * @details Cleans up BCData resources. Currently a stub implementation.
     */
    ~BCData();

    /** @brief Essential velocity values for each component [x, y, z] */
    double ess_vel[3];

    /** @brief Scaling factors for each velocity component [x, y, z] */
    double scale[3];

    /** @brief Component ID indicating which velocity components are constrained */
    int comp_id;

    /**
     * @brief Apply Dirichlet boundary conditions to a velocity vector
     *
     * @param y Output velocity vector where boundary conditions will be applied
     *
     * @details Sets the velocity vector components based on the essential velocity values
     * and their corresponding scaling factors. For velocity-based methods, this function:
     * - Initializes the output vector to zero
     * - Applies scaled essential velocities: y[i] = ess_vel[i] * scale[i]
     *
     * This is used during the assembly process to enforce velocity boundary conditions.
     */
    void SetDirBCs(mfem::Vector& y);

    /**
     * @brief Set scaling factors based on component ID
     *
     * @details Configures the scale array based on the comp_id value to determine which
     * velocity components should be constrained. The scaling pattern is:
     * - comp_id = 0: No scaling (all zeros)
     * - comp_id = 1: X-component only (1,0,0)
     * - comp_id = 2: Y-component only (0,1,0)
     * - comp_id = 3: Z-component only (0,0,1)
     * - comp_id = 4: X,Y components (1,1,0)
     * - comp_id = 5: Y,Z components (0,1,1)
     * - comp_id = 6: X,Z components (1,0,1)
     * - comp_id = 7: All components (1,1,1)
     */
    void SetScales();

    /**
     * @brief Static utility to decode component ID into boolean flags
     *
     * @param id Component ID to decode
     * @param component Output array of boolean flags for each component [x, y, z]
     *
     * @details Converts a component ID integer into a boolean array indicating which
     * velocity components are active. This is used throughout the boundary condition
     * system to determine which degrees of freedom should be constrained.
     *
     * The mapping follows the same pattern as SetScales():
     * - id = 0: (false, false, false)
     * - id = 1: (true, false, false)
     * - id = 2: (false, true, false)
     * - id = 3: (false, false, true)
     * - id = 4: (true, true, false)
     * - id = 5: (false, true, true)
     * - id = 6: (true, false, true)
     * - id = 7: (true, true, true)
     */
    static void GetComponents(int id, std::array<bool, 3>& component);
};
#endif
