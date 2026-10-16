#pragma once

#include "mfem/config/config.hpp"
#include "mfem/fem/qspace.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace mfem::expt {

/**
 * @brief Class representing a subset of a QuadratureSpace for efficient operations on subdomains.
 *
 * PartialQuadratureSpace extends MFEM's QuadratureSpaceBase to provide efficient finite element
 * operations on subsets of mesh elements. This is particularly useful in ExaConstit for handling
 * multi-material simulations where different constitutive models apply to different regions of
 * the mesh (e.g., different crystal orientations in polycrystalline materials).
 *
 * The class maintains bidirectional mappings between local element indices (within the partial set)
 * and global element indices (in the full mesh), enabling efficient data access and computation
 * while maintaining compatibility with MFEM's finite element framework.
 *
 * Key features:
 * - Efficient memory usage by storing data only for active elements
 * - Optimized performance through local-to-global index mapping
 * - Full compatibility with MFEM's QuadratureSpaceBase interface
 * - Support for both partial and full mesh coverage with automatic optimization
 *
 * @ingroup ExaConstit_mfem_expt
 */
class PartialQuadratureSpace : public mfem::QuadratureSpaceBase {
protected:
    friend class PartialQuadratureFunction; // Uses the offsets.
    /**
     * @brief Maps local element indices to global mesh element indices.
     *
     * This array provides the forward mapping from the local element indexing scheme
     * used within the PartialQuadratureSpace to the global element indexing scheme
     * of the underlying mesh. Size equals the number of elements in the partial set.
     *
     * local2global[local_idx] = global_idx
     */
    mfem::Array<int> local2global;
    /**
     * @brief Maps global mesh element indices to local element indices.
     *
     * This array provides the reverse mapping from global mesh element indices to
     * local partial space indices. Contains -1 for elements not in the partial set.
     * Size equals the total number of elements in the mesh, or 1 if optimization
     * for full-space coverage is enabled.
     *
     * global2local[global_idx] = local_idx (or -1 if not in partial set)
     */
    mfem::Array<int> global2local;
    /**
     * @brief Maps global mesh element indices to quadrature point offsets.
     *
     * This array provides offset information for all elements in the global mesh,
     * facilitating efficient data transfer between partial and full quadrature spaces.
     * Used internally for mapping operations when the partial space doesn't cover
     * the entire mesh.
     *
     * global_offsets[global_idx] = starting offset for element's quadrature points
     */
    mfem::Array<int> global_offsets;

protected:
    /**
     * @brief Implementation of GetGeometricFactorWeights required by the base class.
     *
     * @return Const reference to the geometric factor weights vector
     *
     * This method computes and returns the geometric factor weights (determinants of
     * element transformations) for elements in the partial quadrature space. It extracts
     * the relevant weights from the full mesh's geometric factors and constructs a
     * partial weights vector containing only the data for elements in the partial set.
     *
     * The weights are essential for proper numerical integration over the partial domain.
     * Currently assumes a single integration rule type across all elements.
     */
    virtual const mfem::Vector& GetGeometricFactorWeights() const override;

    /**
     * @brief Constructs the offset arrays for quadrature points in partial elements.
     *
     * This method builds the offsets array that maps local element indices to their
     * corresponding quadrature point ranges in the flattened data structure. It iterates
     * through all elements in the partial set and accumulates the number of quadrature
     * points based on the integration rules for each element's geometry type.
     *
     * The offsets array has size (num_partial_elements + 1), where offsets[i] gives
     * the starting index for element i's quadrature points, and offsets[i+1] - offsets[i]
     * gives the number of quadrature points for element i.
     */
    void ConstructOffsets();

    /**
     * @brief Constructs global offset arrays for full mesh compatibility.
     *
     * This method builds the global_offsets array that provides offset information
     * for all elements in the global mesh, facilitating data mapping between partial
     * and full quadrature spaces. When the partial space covers all elements,
     * this creates a minimal offset array. Otherwise, it creates a complete mapping
     * for all global elements.
     *
     * Used internally for efficient data transfer operations between partial and
     * full quadrature functions.
     */
    void ConstructGlobalOffsets();

    /**
     * @brief Main construction method that builds all internal data structures.
     *
     * This method orchestrates the construction of the PartialQuadratureSpace by
     * calling the appropriate sub-construction methods in the correct order:
     * 1. ConstructIntRules() - sets up integration rules for the mesh dimension
     * 2. ConstructOffsets() - builds local element offset arrays
     * 3. ConstructGlobalOffsets() - builds global element offset arrays
     *
     * This method should be called after all mapping arrays have been initialized.
     */
    void Construct();

    /**
     * @brief Constructs local-to-global and global-to-local element mappings.
     *
     * @param mesh_ Shared pointer to the mesh object
     * @param partial_index Boolean array indicating which elements are included in the partial set
     *
     * This method builds the bidirectional mapping between local element indices (in the partial
     * space) and global element indices (in the full mesh). It handles two cases:
     * 1. Partial coverage: Creates both local2global and global2local mapping arrays
     * 2. Full coverage: Optimizes for the case where all elements are included
     *
     * The local2global array maps from local element index to global element index.
     * The global2local array maps from global element index to local element index (-1 if not
     * included).
     */
    void ConstructMappings(std::shared_ptr<mfem::Mesh> mesh, mfem::Array<bool>& partial_index);

public:
    /**
     * @brief Create a PartialQuadratureSpace based on the global rules from IntRules.
     *
     * @param mesh_ Shared pointer to the mesh object
     * @param order_ Integration order for automatic quadrature rule selection
     * @param partial_index Boolean array indicating which elements to include in the partial set
     *
     * This constructor creates a PartialQuadratureSpace using automatically selected
     * integration rules based on the specified order. The partial_index array determines
     * which mesh elements are included in the partial quadrature space. If partial_index
     * is empty or includes all elements, optimizations for full-space coverage are applied.
     */
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_,
                           int order_,
                           mfem::Array<bool>& partial_index);

    /**
     * @brief Create a PartialQuadratureSpace based on the global rules from IntRules (deprecated).
     *
     * @param mesh_ Raw pointer to the mesh object
     * @param order_ Integration order for automatic quadrature rule selection
     * @param partial_index Boolean array indicating which elements to include in the partial set
     *
     * @deprecated Use constructor with std::shared_ptr<mfem::Mesh> instead for better memory
     * management
     */
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_, int order_, mfem::Array<bool>& partial_index)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), order_, partial_index) {}

    /**
     * @brief Constructor with explicit IntegrationRule for single-element-type meshes.
     *
     * @param mesh_ Shared pointer to the mesh object
     * @param ir Integration rule to use for all elements
     * @param partial_index Boolean array indicating which elements to include
     *
     * This constructor creates a PartialQuadratureSpace using a specific integration
     * rule rather than deriving rules from an order. It's only valid for meshes
     * that have a single element type (e.g., all tetrahedra or all hexahedra).
     *
     * The constructor verifies that the mesh has at most one geometry type before
     * proceeding with construction. It then builds the element mappings and
     * constructs the offset arrays for the specified partial element set.
     */
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_,
                           const mfem::IntegrationRule& ir,
                           mfem::Array<bool>& partial_index);

    /**
     * @brief Create a PartialQuadratureSpace with an IntegrationRule (deprecated).
     *
     * @param mesh_ Raw pointer to the mesh object
     * @param ir Integration rule to use for all elements
     * @param partial_index Boolean array indicating which elements to include in the partial set
     *
     * @deprecated Use constructor with std::shared_ptr<mfem::Mesh> instead for better memory
     * management
     */
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_,
                           const mfem::IntegrationRule& ir,
                           mfem::Array<bool>& partial_index)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), ir, partial_index) {}

    /**
     * @brief Constructor that reads PartialQuadratureSpace from an input stream.
     *
     * @param mesh_ Shared pointer to the mesh object
     * @param in Input stream containing serialized PartialQuadratureSpace data
     *
     * This constructor deserializes a PartialQuadratureSpace from a stream that was
     * previously written using the Save() method. It reads the quadrature order
     * and element mapping information, then reconstructs all internal data structures.
     *
     * The expected stream format includes:
     * - Header: "PartialQuadratureSpace"
     * - Type: "default_quadrature"
     * - Order: Integration order
     * - PartialIndices: Number of elements followed by local2global mapping
     *
     * After reading the mapping data, it calls Construct() to build the complete
     * quadrature space data structures.
     */
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_, std::istream& in);

    /**
     * @brief Read a PartialQuadratureSpace from the stream (deprecated).
     *
     * @param mesh_ Raw pointer to the mesh object
     * @param in Input stream containing serialized PartialQuadratureSpace data
     *
     * @deprecated Use constructor with std::shared_ptr<mfem::Mesh> instead for better memory
     * management
     */
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_, std::istream& in)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), in) {}

    /**
     * @brief Converts a local element index to the corresponding global element index.
     *
     * @param local_idx Local element index in the partial quadrature space
     * @return Global element index in the full mesh, or -1 if invalid local index
     *
     * This method provides the mapping from the local element indexing scheme used
     * within the PartialQuadratureSpace to the global element indexing scheme of
     * the underlying mesh. Essential for accessing mesh-level element data.
     */
    [[nodiscard]]
    int LocalToGlobal(int local_idx) const {
        if (local_idx >= 0 && local_idx < local2global.Size()) {
            return local2global[local_idx];
        }
        return -1;
    }

    /**
     * @brief Converts a global element index to the corresponding local element index.
     *
     * @param global_idx Global element index in the full mesh
     * @return Local element index in the partial space, or -1 if element not in partial set
     *
     * This method provides the reverse mapping from global mesh element indices to
     * local partial space indices. Returns -1 for elements not included in the partial set.
     * Handles the special case where the partial space covers the entire mesh.
     */
    [[nodiscard]]
    int GlobalToLocal(int global_idx) const {
        if (global_idx >= 0 && global_idx < global2local.Size()) {
            return global2local[global_idx];
        } else if (global_idx >= 0 && global2local.Size() == 1) {
            return global_idx;
        }
        return -1;
    }

    /**
     * @brief Get read-only access to the global-to-local mapping array.
     *
     * @return Const reference to the global2local array
     *
     * The returned array maps global element indices to local element indices,
     * with -1 indicating elements not in the partial set. For optimization,
     * when the partial space covers all elements, this array has size 1.
     */
    const mfem::Array<int>& GetGlobal2Local() const {
        return global2local;
    }

    /**
     * @brief Get read-only access to the local-to-global mapping array.
     *
     * @return Const reference to the local2global array
     *
     * The returned array provides the mapping from local element indices
     * (within the partial space) to global element indices (in the full mesh).
     */
    const mfem::Array<int>& GetLocal2Global() const {
        return local2global;
    }

    /**
     * @brief Get read-only access to the global offset array.
     *
     * @return Const reference to the global_offsets array
     *
     * The global offset array provides quadrature point offset information
     * for all elements in the global mesh, facilitating efficient data
     * transfer between partial and full quadrature spaces.
     */
    const mfem::Array<int>& GetGlobalOffset() const {
        return global_offsets;
    }

    /**
     * @brief Get the number of elements in the local partial space.
     *
     * @return Number of elements included in this partial quadrature space
     *
     * This count represents the subset of mesh elements that are active
     * in this PartialQuadratureSpace, which may be less than the total
     * number of elements in the underlying mesh.
     */
    int GetNumLocalElements() const {
        return local2global.Size();
    }

    /**
     * @brief Check if this partial space covers the entire mesh.
     *
     * @return True if all mesh elements are included in this partial space
     *
     * This method returns true when the partial space is actually equivalent
     * to a full quadrature space, enabling certain optimizations in data
     * handling and memory management.
     */
    bool IsFullSpace() const {
        return (global2local.Size() == 1);
    }

    /**
     * @brief Get the element transformation for a local entity index.
     *
     * @param idx Local element index in the partial space
     * @return Pointer to the ElementTransformation for the corresponding global element
     *
     * This method converts the local element index to a global index and retrieves
     * the element transformation from the underlying mesh. The transformation
     * contains geometric information needed for integration and finite element assembly.
     */
    [[nodiscard]]
    virtual mfem::ElementTransformation* GetTransformation(int idx) override {
        int global_idx = LocalToGlobal(idx);
        return mesh->GetElementTransformation(global_idx);
    }

    /**
     * @brief Return the geometry type of the entity with local index idx.
     *
     * @param idx Local element index in the partial space
     * @return Geometry type (e.g., Triangle, Quadrilateral, Tetrahedron, Hexahedron)
     *
     * This method maps the local element index to the global mesh and returns
     * the geometric type of that element, which determines the appropriate
     * integration rules and basis functions.
     */
    [[nodiscard]]
    virtual mfem::Geometry::Type GetGeometry(int idx) const override {
        // QuadratureSpaceBase::GetIntRule(0) asks for a geometry even when this
        // rank's partial space is empty. There is no local-to-global element in
        // that case, so use MFEM's globally representative geometry instead of
        // passing -1 to Mesh::GetElementGeometry().
        if (local2global.Size() == 0) {
            return mesh->GetTypicalElementGeometry();
        }
        int global_idx = LocalToGlobal(idx);
        return mesh->GetElementGeometry(global_idx);
    }

    /**
     * @brief Get the permuted quadrature point index (trivial for element spaces).
     *
     * @param idx Element index (unused for element quadrature spaces)
     * @param iq Quadrature point index
     * @return The same quadrature point index (no permutation for elements)
     *
     * For element quadrature spaces, quadrature point permutation is trivial,
     * so this method simply returns the input quadrature point index unchanged.
     */
    [[nodiscard]]
    virtual int GetPermutedIndex([[maybe_unused]] int idx, int iq) const override {
        // For element quadrature spaces, the permutation is trivial
        return iq;
    }

    /**
     * @brief Save the PartialQuadratureSpace to a stream.
     *
     * @param out Output stream to write the PartialQuadratureSpace data
     *
     * This method serializes the PartialQuadratureSpace configuration to a stream,
     * including the quadrature order and the mapping of partial elements. The output
     * format can be read back using the stream constructor.
     */
    virtual void Save(std::ostream& out) const override;

    /**
     * @brief Returns the element index for the given ElementTransformation.
     *
     * @param T Reference to an ElementTransformation object
     * @return Element index from the transformation (T.ElementNo)
     *
     * This method extracts the element index directly from the ElementTransformation
     * object, providing the interface required by the QuadratureSpaceBase class.
     */
    [[nodiscard]]
    virtual int GetEntityIndex(const mfem::ElementTransformation& T) const override {
        return T.ElementNo;
    }

    // Factory methods

    /**
     * @brief Factory method to create a shared_ptr PartialQuadratureSpace.
     *
     * @param mesh Shared pointer to the mesh object
     * @param order Integration order for quadrature rules
     * @param partial_index Boolean array indicating which elements to include
     * @return Shared pointer to the created PartialQuadratureSpace
     *
     * This factory method provides the recommended way to create PartialQuadratureSpace
     * objects with proper memory management using shared_ptr. It handles the construction
     * of all internal data structures and mappings.
     */
    static std::shared_ptr<PartialQuadratureSpace>
    Create(std::shared_ptr<mfem::Mesh> mesh, int order, mfem::Array<bool> partial_index) {
        return std::make_shared<PartialQuadratureSpace>(std::move(mesh), order, partial_index);
    }

    /**
     * @brief Factory method to create a shared_ptr PartialQuadratureSpace.
     *
     * @param mesh Raw pointer to the mesh object
     * @param order Integration order for quadrature rules
     * @param partial_index Boolean array indicating which elements to include
     * @return Shared pointer to the created PartialQuadratureSpace
     *
     * This factory method provides the recommended way to create PartialQuadratureSpace
     * objects with proper memory management using shared_ptr. It handles the construction
     * of all internal data structures and mappings.
     */
    [[deprecated("Use Create() with std::shared_ptr<mfem::Mesh> instead")]]
    static std::shared_ptr<PartialQuadratureSpace>
    Create(mfem::Mesh* mesh, int order, mfem::Array<bool> partial_index) {
        return std::make_shared<PartialQuadratureSpace>(
            ptr_utils::borrow_ptr(mesh), order, partial_index);
    }
};

} // namespace mfem::expt
