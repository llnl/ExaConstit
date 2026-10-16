

#include "boundary_conditions/BCManager.hpp"

#include "mfem.hpp"

#include <fstream>

void BCManager::UpdateBCData(std::unordered_map<std::string, mfem::Array<int>>& ess_bdr,
                             mfem::Array2D<double>& scale,
                             mfem::Vector& vgrad,
                             std::unordered_map<std::string, mfem::Array2D<bool>>& component) {
    ess_bdr["total"] = 0;
    scale = 0.0;

    const auto& ess_comp = map_ess_comp["total"].find(step)->second;
    const auto& ess_id = map_ess_id["total"].find(step)->second;

    std::array<bool, 3> cmp_row;

    component["total"] = false;

    for (size_t i = 0; i < ess_id.size(); ++i) {
        // set the active boundary attributes
        if (ess_comp[i] != 0) {
            const int bcID = ess_id[i] - 1;
            ess_bdr["total"][bcID] = 1;
            BCData::GetComponents(std::abs(ess_comp[i]), cmp_row);

            component["total"](bcID, 0) = cmp_row[0];
            component["total"](bcID, 1) = cmp_row[1];
            component["total"](bcID, 2) = cmp_row[2];
        }
    }

    UpdateBCData(ess_bdr["ess_vel"], scale, component["ess_vel"]);
    UpdateBCData(ess_bdr["ess_vgrad"], vgrad, component["ess_vgrad"]);
}

void BCManager::UpdateBCData(mfem::Array<int>& ess_bdr,
                             mfem::Array2D<double>& scale,
                             mfem::Array2D<bool>& component) {
    m_bc_instances.clear();
    ess_bdr = 0;
    scale = 0.0;

    // The size here is set explicitly
    component.SetSize(ess_bdr.Size(), 3);
    std::array<bool, 3> cmp_row;

    component = false;

    if (map_ess_vel.find(step) == map_ess_vel.end()) {
        return;
    }

    const auto& ess_vel = map_ess_vel.find(step)->second;
    const auto& ess_comp = map_ess_comp["ess_vel"].find(step)->second;
    const auto& ess_id = map_ess_id["ess_vel"].find(step)->second;

    for (size_t i = 0; i < ess_id.size(); ++i) {
        // set the active boundary attributes
        if (ess_comp[i] != 0) {
            // set the boundary condition id based on the attribute id
            int bcID = ess_id[i];

            // instantiate a boundary condition manager instance and
            // create a BCData object
            BCData& bc = this->CreateBCs(bcID);

            // set the velocity component values
            bc.ess_vel[0] = ess_vel[3 * i];
            bc.ess_vel[1] = ess_vel[3 * i + 1];
            bc.ess_vel[2] = ess_vel[3 * i + 2];
            bc.comp_id = ess_comp[i];

            // set the boundary condition scales
            bc.SetScales();

            scale(bcID - 1, 0) = bc.scale[0];
            scale(bcID - 1, 1) = bc.scale[1];
            scale(bcID - 1, 2) = bc.scale[2];
            ess_bdr[bcID - 1] = 1;
        }
    }

    for (size_t i = 0; i < ess_id.size(); ++i) {
        // set the active boundary attributes
        if (ess_comp[i] != 0) {
            const int bcID = ess_id[i] - 1;
            ess_bdr[bcID] = 1;
            BCData::GetComponents(ess_comp[i], cmp_row);
            component(bcID, 0) = cmp_row[0];
            component(bcID, 1) = cmp_row[1];
            component(bcID, 2) = cmp_row[2];
        }
    }
}

void BCManager::UpdateBCData(mfem::Array<int>& ess_bdr,
                             mfem::Vector& vgrad,
                             mfem::Array2D<bool>& component) {
    ess_bdr = 0;
    vgrad.HostReadWrite();
    vgrad = 0.0;
    auto data = vgrad.HostReadWrite();

    // The size here is set explicitly
    component.SetSize(ess_bdr.Size(), 3);
    std::array<bool, 3> cmp_row;

    component = false;

    if (map_ess_vgrad.find(step) == map_ess_vgrad.end()) {
        return;
    }

    const auto& ess_vgrad = map_ess_vgrad.find(step)->second;
    const auto& ess_comp = map_ess_comp["ess_vgrad"].find(step)->second;
    const auto& ess_id = map_ess_id["ess_vgrad"].find(step)->second;

    for (size_t i = 0; i < ess_vgrad.size(); ++i) {
        data[i] = ess_vgrad.at(i);
    }

    for (size_t i = 0; i < ess_id.size(); ++i) {
        // set the active boundary attributes
        if (ess_comp[i] != 0) {
            const int bcID = ess_id[i] - 1;
            ess_bdr[bcID] = 1;
            BCData::GetComponents(ess_comp[i], cmp_row);
            component(bcID, 0) = cmp_row[0];
            component(bcID, 1) = cmp_row[1];
            component(bcID, 2) = cmp_row[2];
        }
    }
}
