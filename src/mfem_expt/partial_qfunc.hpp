#pragma once

#include "mfem_expt/partial_qspace.hpp"

#include "mfem/config/config.hpp"
#include "mfem/fem/qfunction.hpp"
#include "mfem/fem/qspace.hpp"
#include "mfem/general/forall.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace mfem::expt {

/**
 * @brief Class for representing quadrature functions on a subset of mesh elements.
 *
 * PartialQuadratureFunction extends MFEM's QuadratureFunction to efficiently store and
 * manipulate quadrature point data for only a subset of mesh elements. This is essential
 * in ExaConstit for multi-material simulations where different constitutive models and
 * state variables apply to different regions of the mesh.
 *
 * The class maintains compatibility with MFEM's QuadratureFunction interface while
 * providing optimized memory usage and performance for partial element sets. It handles
 * the mapping between partial and full quadrature spaces automatically and provides
 * default values for elements not in the partial set.
 *
 * Key features:
 * - Memory-efficient storage for sparse element data
 * - Automatic handling of default values for non-partial elements
 * - Full compatibility with MFEM's QuadratureFunction operations
 * - Efficient data transfer between partial and full quadrature spaces
 * - Support for multi-component vector fields at quadrature points
 *
 * @ingroup ExaConstit_mfem_expt
 */
class PartialQuadratureFunction : public QuadratureFunction {
private:
    /**
     * @brief Reference to the specialized PartialQuadratureSpace.
     *
     * This shared pointer maintains a reference to the PartialQuadratureSpace that
     * defines the element subset and quadrature point layout for this function.
     * The space provides the mapping between local and global element indices
     * needed for efficient data access and manipulation.
     */
    std::shared_ptr<PartialQuadratureSpace> part_quad_space;

    /**
     * @brief Default value for elements not in the partial set.
     *
     * This value is returned when accessing data for elements that are not
     * included in the partial quadrature space. It allows the function to
     * appear as if it has values defined over the entire mesh while only
     * storing data for the relevant subset of elements.
     */
    double default_value;

public:
    /**
     * @brief Constructor with shared_ptr to PartialQuadratureSpace.
     *
     * @param qspace_ Shared pointer to the PartialQuadratureSpace defining the element subset
     * @param vdim_ Vector dimension of the function (number of components per quadrature point)
     * @param default_val Default value for elements not in the partial set
     *
     * This is the recommended constructor that creates a PartialQuadratureFunction with
     * proper memory management using shared_ptr. The vector dimension determines how many
     * scalar values are stored at each quadrature point (e.g., vdim=1 for scalar fields,
     * vdim=3 for vector fields, vdim=9 for tensor fields).
     */
    PartialQuadratureFunction(std::shared_ptr<PartialQuadratureSpace> qspace_,
                              int vdim_ = 1,
                              double default_val = -1.0)
        : QuadratureFunction(std::static_pointer_cast<QuadratureSpaceBase>(qspace_), vdim_),
          part_quad_space(std::move(qspace_)), default_value(default_val) {}

    /**
     * @brief Constructor with raw pointer to PartialQuadratureSpace (deprecated).
     *
     * @param qspace_ Raw pointer to the PartialQuadratureSpace defining the element subset
     * @param vdim_ Vector dimension of the function (number of components per quadrature point)
     * @param default_val Default value for elements not in the partial set
     *
     * @deprecated Use constructor with std::shared_ptr<PartialQuadratureSpace> instead for better
     * memory management
     */
    [[deprecated("Use constructor with std::shared_ptr<PartialQuadratureSpace> instead")]]
    PartialQuadratureFunction(PartialQuadratureSpace* qspace_,
                              int vdim_ = 1,
                              double default_val = -1.0)
        : PartialQuadratureFunction(ptr_utils::borrow_ptr(qspace_), vdim_, default_val) {}

    /**
     * @brief Get the specialized PartialQuadratureSpace as shared_ptr.
     *
     * @return Shared pointer to the underlying PartialQuadratureSpace
     *
     * This method provides access to the PartialQuadratureSpace that defines the
     * element subset and quadrature point layout for this function. Useful for
     * accessing mapping information and space properties.
     */
    [[nodiscard]]
    std::shared_ptr<PartialQuadratureSpace> GetPartialSpaceShared() const {
        return part_quad_space;
    }

    /**
     * @brief Get the specialized PartialQuadratureSpace as raw pointer (deprecated).
     *
     * @return Raw pointer to the underlying PartialQuadratureSpace
     *
     * @deprecated Use GetPartialSpaceShared() instead for better memory management
     */
    [[deprecated("Use GetPartialSpaceShared() instead")]] [[nodiscard]]
    PartialQuadratureSpace* GetPartialSpace() const {
        return part_quad_space.get();
    }

    /**
     * @brief Set this equal to a constant value.
     *
     * @param value Constant value to assign to all quadrature points in the partial set
     * @return Reference to this PartialQuadratureFunction for method chaining
     *
     * This operator assigns the specified constant value to all quadrature points
     * within the partial element set. Elements outside the partial set are not
     * affected and will continue to return the default value.
     */
    PartialQuadratureFunction& operator=(double value) override {
        QuadratureFunction::operator=(value);
        return *this;
    }

    /**
     * @brief Copy the data from a Vector.
     *
     * @param vec Vector containing the data to copy (must match the size of this function)
     * @return Reference to this PartialQuadratureFunction for method chaining
     *
     * This operator copies data from a Vector into the PartialQuadratureFunction.
     * The vector size must exactly match the size of the partial quadrature space.
     * The data is interpreted as [comp0_qp0, comp1_qp0, ..., comp0_qp1, comp1_qp1, ...].
     */
    PartialQuadratureFunction& operator=(const Vector& vec) override {
        MFEM_ASSERT(part_quad_space && vec.Size() == this->Size(), "");
        QuadratureFunction::operator=(vec);
        return *this;
    }

    /**
     * @brief Copy the data from another QuadratureFunction.
     *
     * @param qf Source QuadratureFunction to copy data from
     * @return Reference to this PartialQuadratureFunction for method chaining
     *
     * This operator intelligently copies data from a QuadratureFunction, handling
     * both cases where the source function has the same size (direct copy) or
     * different size (element-by-element mapping). For different sizes, it validates
     * mesh compatibility and integration rule consistency before performing the
     * element-wise data transfer using the local-to-global mapping.
     */
    PartialQuadratureFunction& operator=(const QuadratureFunction& qf);

    /**
     * @brief Copy constructor: a new function on the same partial space with a copy of the data.
     *
     * @details Declared (defaulted) explicitly because the copy assignment below is
     * user-provided; the behaviour is that of the former implicit copy constructor.
     */
    PartialQuadratureFunction(const PartialQuadratureFunction&) = default;

    /**
     * @brief Copy assignment with value semantics: copies the DATA only.
     * @param rhs Function on an identically laid-out partial space (same vdim and size)
     * @return Reference to this PartialQuadratureFunction
     *
     * @details The implicitly-declared version relied on mfem::QuadratureFunction's
     * implicitly-declared copy assignment, which is deprecated because QuadratureFunction has a
     * user-provided copy constructor (-Wdeprecated-copy), and it also re-pointed this function's
     * space members at the source's. It is reached from SimulationState::SetupModelVariables
     * (every registered begin/end pair) and ExaCMechModel::InitStateVars; both copy between
     * functions on the same space, for which data-only copying is exactly the intended operation.
     */
    PartialQuadratureFunction& operator=(const PartialQuadratureFunction& rhs) {
        MFEM_ASSERT(rhs.GetVDim() == vdim && rhs.Size() == this->Size(),
                    "PartialQuadratureFunction copy assignment requires identical layouts");
        if (this != &rhs) {
            QuadratureFunction::operator=(static_cast<const Vector&>(rhs));
        }
        return *this;
    }

    /**
     * @brief Fill a global QuadratureFunction with data from this partial function.
     *
     * @param qf Reference to the global QuadratureFunction to fill
     * @param fill Whether to initialize non-partial elements with default value
     *
     * This method transfers data from the PartialQuadratureFunction to a global
     * QuadratureFunction that spans the entire mesh. For elements in the partial set,
     * it copies the stored values. For elements not in the partial set, it optionally
     * fills with the default value if fill=true.
     *
     * The method handles two cases:
     * 1. Same size spaces: Direct copy operation
     * 2. Different size spaces: Element-by-element mapping with validation
     *
     * Validation checks ensure compatible vector dimensions, mesh compatibility,
     * and matching integration orders before performing the data transfer.
     */
    void FillQuadratureFunction(QuadratureFunction& qf, const bool fill = false);

    /**
     * @brief Override ProjectGridFunction to project only onto the partial space.
     *
     * @param gf GridFunction to project (parameter currently unused)
     *
     * This method is currently unsupported and will abort if called. It's included
     * for interface completeness and may be implemented in future versions to
     * project GridFunction data onto the partial quadrature space.
     */
    void ProjectGridFunction([[maybe_unused]] const GridFunction& gf) override {
        MFEM_ABORT("Unsupported case.");
    }

    /**
     * @brief Return all values associated with mesh element as a reference Vector.
     *
     * @param idx Global element index
     * @param values Output vector that will reference the internal data or be filled with defaults
     *
     * This method provides access to all quadrature point values for the specified element.
     * For elements in the partial set, it creates a reference to the internal data for
     * efficient access. For elements not in the partial set, it creates a new vector
     * filled with default values.
     *
     * The values vector is organized as [comp0_qp0, comp1_qp0, ..., comp0_qp1, comp1_qp1, ...].
     */
    virtual void GetValues(int idx, Vector& values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.MakeRef(*this, vdim * s_offset, vdim * sl_size);
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Destroy();
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++) {
                values(i) = default_value;
            }
        }
    }

    /**
     * @brief Return all values associated with mesh element as a copy Vector.
     *
     * @param idx Global element index
     * @param values Output vector to store the copied values
     *
     * This method retrieves all quadrature point values for the specified element as
     * a copy rather than a reference. For elements in the partial set, it copies the
     * stored values. For elements not in the partial set, it fills the output vector
     * with default values.
     *
     * The values vector is organized as [comp0_qp0, comp1_qp0, ..., comp0_qp1, comp1_qp1, ...].
     */
    virtual void GetValues(int idx, Vector& values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            const real_t* q = HostRead() + vdim * s_offset;
            for (int i = 0; i < values.Size(); i++) {
                values(i) = *(q++);
            }
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++) {
                values(i) = default_value;
            }
        }
    }

    /**
     * @brief Return quadrature function values at a specific integration point as reference.
     *
     * @param idx Global element index
     * @param ip_num Quadrature point number within the element
     * @param values Output vector that will reference the internal data or be filled with defaults
     *
     * This method provides access to the values at a single quadrature point within an element.
     * For elements in the partial set, it creates a reference to the internal data.
     * For elements not in the partial set, it creates a new vector filled with default values.
     */
    virtual void GetValues(int idx, const int ip_num, Vector& values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
            values.MakeRef(*this, s_offset, vdim);
        } else {
            values.Destroy();
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++) {
                values(i) = default_value;
            }
        }
    }

    /**
     * @brief Return quadrature function values at a specific integration point as copy.
     *
     * @param idx Global element index
     * @param ip_num Quadrature point number within the element
     * @param values Output vector to store the copied values
     *
     * This method retrieves the values at a single quadrature point within an element
     * as a copy rather than a reference. For elements in the partial set, it copies the
     * stored values. For elements not in the partial set, it fills the output vector
     * with default values.
     */
    virtual void GetValues(int idx, const int ip_num, Vector& values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
            const real_t* q = HostRead() + s_offset;
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++) {
                values(i) = *(q++);
            }
        } else {
            values.Destroy();
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++) {
                values(i) = default_value;
            }
        }
    }

    /**
     * @brief Return all values associated with mesh element as a reference DenseMatrix.
     *
     * @param idx Global element index
     * @param values Output matrix that will reference the internal data or be filled with defaults
     *
     * This method provides access to all quadrature point values for the specified element
     * in matrix form. For elements in the partial set, it creates a memory alias to the
     * internal data for efficient access. For elements not in the partial set, it creates
     * a new matrix filled with default values.
     *
     * The matrix entry (i,j) corresponds to the i-th vector component at the j-th quadrature point.
     */
    virtual void GetValues(int idx, DenseMatrix& values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            // Make the values matrix memory an alias of the quadrature function memory
            Memory<real_t>& values_mem = values.GetMemory();
            values_mem.Delete();
            values_mem.MakeAlias(GetMemory(), vdim * s_offset, vdim * sl_size);
            values.SetSize(vdim, sl_size);
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Clear();
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            for (int j = 0; j < sl_size; j++) {
                for (int i = 0; i < vdim; i++) {
                    values(i, j) = default_value;
                }
            }
        }
    }

    /**
     * @brief Return all values associated with mesh element as a copy DenseMatrix.
     *
     * @param idx Global element index
     * @param values Output matrix to store the copied values
     *
     * This method retrieves all quadrature point values for the specified element as
     * a copy in matrix form. For elements in the partial set, it copies the stored values.
     * For elements not in the partial set, it fills the output matrix with default values.
     *
     * The matrix entry (i,j) corresponds to the i-th vector component at the j-th quadrature point.
     */
    virtual void GetValues(int idx, DenseMatrix& values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            const real_t* q = HostRead() + vdim * s_offset;
            for (int j = 0; j < sl_size; j++) {
                for (int i = 0; i < vdim; i++) {
                    values(i, j) = *(q++);
                }
            }
        } else {
            // Make the values matrix memory an alias of the quadrature function memory
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Clear();
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            for (int j = 0; j < sl_size; j++) {
                for (int i = 0; i < vdim; i++) {
                    values(i, j) = default_value;
                }
            }
        }
    }

    /**
     * @brief Get the IntegrationRule associated with entity (element or face).
     *
     * This uses the base class implementation from QuadratureFunction to provide
     * access to the integration rules associated with mesh entities.
     */
    using QuadratureFunction::GetIntRule;

    /**
     * @brief Write the PartialQuadratureFunction to a stream.
     *
     * @param out Output stream to write the function data
     *
     * This method serializes the PartialQuadratureFunction to a stream. Currently,
     * it only supports partial spaces that cover the full mesh (optimization case).
     * For true partial spaces, an error is thrown indicating the feature is not
     * yet implemented.
     */
    virtual void Save(std::ostream& out) const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::Save(out);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /**
     * @brief Write the PartialQuadratureFunction to an output stream in VTU format.
     *
     * @param out Output stream for VTU data
     * @param format VTK format (ASCII or BINARY)
     * @param compression_level Compression level for binary output
     * @param field_name Name of the field in the VTU file
     *
     * This method saves the quadrature function data to ParaView's VTU format for
     * visualization. Currently only supported for partial spaces that cover the full
     * mesh. For true partial spaces, an error is thrown indicating the feature is
     * not yet implemented.
     */
    virtual void SaveVTU(std::ostream& out,
                         VTKFormat format = VTKFormat::ASCII,
                         int compression_level = 0,
                         const std::string& field_name = "u") const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::SaveVTU(out, format, compression_level, field_name);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /**
     * @brief Save the PartialQuadratureFunction to a VTU (ParaView) file.
     *
     * @param filename Output filename (extension ".vtu" will be appended)
     * @param format VTK format (ASCII or BINARY)
     * @param compression_level Compression level for binary output
     * @param field_name Name of the field in the VTU file
     *
     * This method saves the quadrature function data to a ParaView VTU file for
     * visualization. Currently only supported for partial spaces that cover the full
     * mesh. For true partial spaces, an error is thrown indicating the feature is
     * not yet implemented.
     */
    virtual void SaveVTU(const std::string& filename,
                         VTKFormat format = VTKFormat::ASCII,
                         int compression_level = 0,
                         const std::string& field_name = "u") const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::SaveVTU(filename, format, compression_level, field_name);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /**
     * @brief Return the integral of the quadrature function (vdim = 1 only).
     *
     * @return Integral value over the partial domain
     *
     * This method computes the integral of the quadrature function over the elements
     * in the partial space. Currently only supported for partial spaces that cover
     * the full mesh. For true partial spaces, an error is thrown indicating the
     * feature is not yet implemented.
     */
    [[nodiscard]] virtual real_t Integrate() const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            return QuadratureFunction::Integrate();
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
        return default_value;
    }

    /**
     * @brief Integrate the vector-valued quadrature function.
     *
     * @param integrals Output vector to store integration results (one per vector component)
     *
     * This method computes the integral of each component of a vector-valued quadrature
     * function over the partial domain. Currently only supported for partial spaces that
     * cover the full mesh. For true partial spaces, an error is thrown indicating the
     * feature is not yet implemented.
     */
    virtual void Integrate(Vector& integrals) const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::Integrate(integrals);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /**
     * @brief Factory method to create a shared_ptr PartialQuadratureFunction.
     *
     * @param qspace Shared pointer to the PartialQuadratureSpace
     * @param vdim Vector dimension of the function (default: 1)
     * @param default_val Default value for elements not in partial set (default: -1.0)
     * @return Shared pointer to the created PartialQuadratureFunction
     *
     * This factory method provides the recommended way to create PartialQuadratureFunction
     * objects with proper memory management using shared_ptr. The vector dimension
     * determines how many components the function has at each quadrature point.
     */
    static std::shared_ptr<PartialQuadratureFunction> Create(
        std::shared_ptr<PartialQuadratureSpace> qspace, int vdim = 1, double default_val = -1.0) {
        return std::make_shared<PartialQuadratureFunction>(std::move(qspace), vdim, default_val);
    }

    /**
     * @brief Factory method to create a shared_ptr PartialQuadratureFunction (deprecated).
     *
     * @param qspace Raw pointer to the PartialQuadratureSpace
     * @param vdim Vector dimension of the function (default: 1)
     * @param default_val Default value for elements not in partial set (default: -1.0)
     * @return Shared pointer to the created PartialQuadratureFunction
     *
     * @deprecated Use Create() with std::shared_ptr<PartialQuadratureSpace> instead for better
     * memory management
     */
    [[deprecated("Use Create() with std::shared_ptr<PartialQuadratureSpace> instead")]]
    static std::shared_ptr<PartialQuadratureFunction>
    Create(PartialQuadratureSpace* qspace, int vdim = 1, double default_val = -1.0) {
        return std::make_shared<PartialQuadratureFunction>(
            ptr_utils::borrow_ptr(qspace), vdim, default_val);
    }
};

} // namespace mfem::expt