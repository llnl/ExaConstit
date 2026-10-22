#include "microstructure/mesh/attribute_domain.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <mpi.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exaconstit::microstructure {

/*
 * SubMeshHolder
 *
 * `mfem::ParSubMesh::CreateFromDomain` returns the sub-mesh by value. Initializing a member
 * directly from that returned temporary constructs it in place (C++17 guaranteed copy elision),
 * so no copy or move of the ParSubMesh ever happens. This matters because a mesh object owns
 * internal tables and a Nodes grid function that point back at the mesh; relying on MFEM's move
 * constructor for a derived ParSubMesh is avoided altogether. The holder is then kept behind a
 * unique_ptr (std::unique_ptr: a pointer that owns the object it points to and deletes it
 * automatically when the owner is destroyed), so its address never changes.
 */
struct AttributeDomain::SubMeshHolder {
    mfem::ParSubMesh mesh;
    SubMeshHolder(const mfem::ParMesh& parent, const mfem::Array<int>& attributes)
        : mesh(mfem::ParSubMesh::CreateFromDomain(parent, attributes)) {}
};

namespace {

/*
 * JoinInts: "a, b, c" for an error message.
 */
template <typename Container>
std::string JoinInts(const Container& values) {
    std::ostringstream out;
    bool first = true;
    for (const auto v : values) {
        out << (first ? "" : ", ") << v;
        first = false;
    }
    return out.str();
}

/*
 * GlobalAttributeCounts
 *
 * Global number of elements of every element attribute present on any rank.
 *
 * Steps:
 *  1. Count the local elements per attribute in an ordered map (std::map, sorted by attribute).
 *  2. Every rank learns how many (attribute, count) pairs each rank has (MPI_Allgather of one
 *     int), then receives all pairs (MPI_Allgatherv: each rank contributes a block of a different
 *     length and every rank receives all blocks one after another).
 *  3. Sum the counts per attribute. Integer sums, so the result is exact and identical on every
 *     rank.
 *
 * Gathering only the attributes present (instead of one counter per possible attribute value)
 * keeps memory proportional to the number of distinct attributes even when attribute values are
 * large. Collective over comm. Host only.
 */
std::map<int, std::int64_t> GlobalAttributeCounts(const mfem::ParMesh& mesh) {
    MPI_Comm comm = mesh.GetComm();
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);

    // Step 1: local counts per attribute
    std::map<int, std::int64_t> local;
    for (int e = 0; e < mesh.GetNE(); e++) {
        local[mesh.GetAttribute(e)]++;
    }
    std::vector<int> local_attr;
    std::vector<std::int64_t> local_count;
    for (const auto& entry : local) {
        local_attr.push_back(entry.first);
        local_count.push_back(entry.second);
    }

    // Step 2: block sizes, displacements, then all pairs on every rank
    int my_size = static_cast<int>(local_attr.size());
    std::vector<int> sizes(static_cast<std::size_t>(nranks), 0);
    MPI_Allgather(&my_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    int total = 0;
    for (int r = 0; r < nranks; r++) {
        displs[static_cast<std::size_t>(r)] = total;
        total += sizes[static_cast<std::size_t>(r)];
    }
    std::vector<int> all_attr(static_cast<std::size_t>(std::max(total, 1)));
    std::vector<std::int64_t> all_count(static_cast<std::size_t>(std::max(total, 1)));
    MPI_Allgatherv(local_attr.data(), my_size, MPI_INT, all_attr.data(), sizes.data(),
                   displs.data(), MPI_INT, comm);
    MPI_Allgatherv(local_count.data(), my_size, MPI_INT64_T, all_count.data(), sizes.data(),
                   displs.data(), MPI_INT64_T, comm);

    // Step 3: global sums per attribute
    std::map<int, std::int64_t> global;
    for (int i = 0; i < total; i++) {
        global[all_attr[static_cast<std::size_t>(i)]] += all_count[static_cast<std::size_t>(i)];
    }
    return global;
}

} // namespace

/*
 * AttributeDomain constructor
 *
 * Steps:
 *  1. Global element count of every attribute (GlobalAttributeCounts, collective).
 *  2. Validate the user lists against those counts (see the header for the rules). Every check
 *     uses only the user lists and globally gathered data, so all ranks reach the same verdict
 *     and throw together; no rank is left waiting in a later collective call.
 *  3. Before building a sub-mesh, two collective checks for conditions under which MFEM's
 *     ParSubMesh would abort or read an empty array (see the header): the parent must have
 *     boundary elements (ParMesh::HasBoundaryElements, an MPI_Allreduce MAX of the local counts,
 *     pmesh.cpp:1617-1624), and no communication group may have more than 32 ranks (largest
 *     GroupTopology::GetGroupSize over the parent's groups, reduced with MPI_Allreduce MAX). A
 *     "communication group" is the set of ranks that share one vertex, edge or face.
 *  4. Build the ParSubMesh of the active attributes when some attribute is passthrough
 *     (collective; see the header for the verified ParSubMesh facts), and the local
 *     active -> parent element map.
 */
AttributeDomain::AttributeDomain(mfem::ParMesh& parent,
                                 const std::vector<int>& active_attributes,
                                 const std::vector<int>& passthrough_attributes)
    : m_parent(parent) {
    // Step 1: attributes present anywhere, with their global element counts
    const std::map<int, std::int64_t> counts = GlobalAttributeCounts(parent);

    // Step 2a: listed attributes must be positive; sort and de-duplicate both lists
    const std::set<int> listed_active(active_attributes.begin(), active_attributes.end());
    const std::set<int> passthrough(passthrough_attributes.begin(),
                                    passthrough_attributes.end());
    std::set<int> non_positive;
    for (const int a : listed_active) {
        if (a <= 0) {
            non_positive.insert(a);
        }
    }
    for (const int a : passthrough) {
        if (a <= 0) {
            non_positive.insert(a);
        }
    }
    if (!non_positive.empty()) {
        throw std::invalid_argument("AttributeDomain: element attributes must be > 0; listed: " +
                                    JoinInts(non_positive));
    }

    // Step 2b: no attribute may be both active and passthrough
    std::vector<int> both;
    std::set_intersection(listed_active.begin(), listed_active.end(), passthrough.begin(),
                          passthrough.end(), std::back_inserter(both));
    if (!both.empty()) {
        throw std::invalid_argument(
            "AttributeDomain: element attribute(s) " + JoinInts(both) +
            " are listed both as active and as passthrough; each attribute must be exactly one");
    }

    // Step 2c: the effective active set ("empty" = every attribute that is not passthrough),
    // and every mesh attribute must be classified
    std::set<int> active = listed_active;
    if (active.empty()) {
        for (const auto& entry : counts) {
            if (passthrough.count(entry.first) == 0) {
                active.insert(entry.first);
            }
        }
    }
    std::vector<int> unclassified;
    for (const auto& entry : counts) {
        if (active.count(entry.first) == 0 && passthrough.count(entry.first) == 0) {
            unclassified.push_back(entry.first);
        }
    }
    if (!unclassified.empty()) {
        throw std::invalid_argument(
            "AttributeDomain: mesh element attribute(s) " + JoinInts(unclassified) +
            " are neither active nor passthrough; list each of them in active_attributes or "
            "passthrough_attributes");
    }

    // Step 2d: every passthrough attribute needs elements
    std::vector<int> empty_passthrough;
    for (const int a : passthrough) {
        const auto it = counts.find(a);
        if (it == counts.end() || it->second == 0) {
            empty_passthrough.push_back(a);
        } else {
            m_passthrough_counts.push_back(it->second);
        }
    }
    if (!empty_passthrough.empty()) {
        throw std::invalid_argument("AttributeDomain: passthrough attribute(s) " +
                                    JoinInts(empty_passthrough) + " have no elements in the mesh");
    }

    // Step 2e: the active set must contain elements somewhere
    for (const int a : active) {
        const auto it = counts.find(a);
        if (it != counts.end()) {
            m_global_active_elements += it->second;
        }
    }
    if (m_global_active_elements == 0) {
        throw std::invalid_argument("AttributeDomain: the active attributes (" +
                                    JoinInts(active) + ") have no elements in the mesh");
    }
    m_active_attributes.assign(active.begin(), active.end());
    m_passthrough_attributes.assign(passthrough.begin(), passthrough.end());

    // The active mesh. With no passthrough attribute every element is active (step 2c
    // guarantees it), so the parent itself is used and the element map is the identity.
    if (m_passthrough_attributes.empty()) {
        m_active_to_parent.SetSize(parent.GetNE());
        for (int e = 0; e < parent.GetNE(); e++) {
            m_active_to_parent[e] = e;
        }
        return;
    }

    // Step 3: guards against MFEM aborts inside the ParSubMesh construction
    if (!parent.HasBoundaryElements()) {
        throw std::invalid_argument(
            "AttributeDomain: the mesh has no boundary elements; mfem::ParSubMesh needs them "
            "(it numbers the new interface faces after the largest boundary attribute)");
    }
    int largest_group = 1;
    for (int g = 0; g < parent.GetNGroups(); g++) {
        largest_group = std::max(largest_group, parent.gtopo.GetGroupSize(g));
    }
    int global_largest_group = 1;
    MPI_Allreduce(&largest_group, &global_largest_group, 1, MPI_INT, MPI_MAX, parent.GetComm());
    constexpr int MAX_GROUP_SIZE = 32; // psubmesh.cpp:332: group_sz <= 8 * sizeof(int)
    if (global_largest_group > MAX_GROUP_SIZE) {
        throw std::invalid_argument(
            "AttributeDomain: " + std::to_string(global_largest_group) +
            " ranks share one vertex, edge or face of the mesh; mfem::ParSubMesh supports at "
            "most " +
            std::to_string(MAX_GROUP_SIZE) +
            ". Use fewer ranks for the generator, or no passthrough attributes");
    }

    // Step 4: the sub-mesh of the active attributes
    mfem::Array<int> attribute_list(static_cast<int>(m_active_attributes.size()));
    for (int i = 0; i < attribute_list.Size(); i++) {
        attribute_list[i] = m_active_attributes[static_cast<std::size_t>(i)];
    }
    m_submesh = std::make_unique<SubMeshHolder>(parent, attribute_list);
    m_active_to_parent = m_submesh->mesh.GetParentElementIDMap();
}

AttributeDomain::~AttributeDomain() = default;

mfem::ParMesh& AttributeDomain::ActiveMesh() {
    if (m_submesh) {
        return m_submesh->mesh;
    }
    return m_parent;
}

bool AttributeDomain::UsesSubMesh() const {
    return static_cast<bool>(m_submesh);
}

/*
 * ToParent
 *
 * Steps:
 *  1. Check the input on this rank (size, grain id range) and reduce the verdict over the
 *     communicator (MPI_Allreduce with MPI_MAX of a 0/1 flag), so that all ranks throw together.
 *  2. Passthrough elements: look up the position k of the element's attribute in the sorted
 *     passthrough list by binary search (std::lower_bound: repeatedly halving the sorted list
 *     until the entry is found, log2(size) comparisons) and store num_grains + 1 + k.
 *  3. Active elements: copy the grain of active element i to parent element
 *     ActiveToParentElement()[i]. Every parent element is either active or passthrough (checked
 *     in the constructor), so every entry is written exactly once.
 *
 * Host loops: the attributes live on the host and the work is one pass over the elements.
 */
mfem::Array<int> AttributeDomain::ToParent(const mfem::Array<int>& active_element_grain,
                                           int num_grains) const {
    // Step 1: local check, then a global verdict
    const int num_active = m_active_to_parent.Size();
    int bad = 0;
    if (active_element_grain.Size() != num_active || num_grains < 0) {
        bad = 1;
    } else {
        const int* grain = active_element_grain.HostRead();
        for (int i = 0; i < num_active; i++) {
            if (grain[i] < 1 || grain[i] > num_grains) {
                bad = 1;
                break;
            }
        }
    }
    int any_bad = 0;
    MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, m_parent.GetComm());
    if (any_bad) {
        throw std::invalid_argument(
            "AttributeDomain::ToParent: on some rank the grain array does not have one entry per "
            "active element, or a grain id lies outside [1, num_grains]");
    }

    // Step 2: passthrough elements get num_grains + 1 + (index of their attribute)
    const int parent_ne = m_parent.GetNE();
    mfem::Array<int> parent_grain(parent_ne);
    parent_grain = 0;
    for (int e = 0; e < parent_ne; e++) {
        const int attribute = m_parent.GetAttribute(e);
        const auto it = std::lower_bound(m_passthrough_attributes.begin(),
                                         m_passthrough_attributes.end(), attribute);
        if (it != m_passthrough_attributes.end() && *it == attribute) {
            parent_grain[e] =
                num_grains + 1 + static_cast<int>(it - m_passthrough_attributes.begin());
        }
    }

    // Step 3: active elements keep their grain
    const int* grain = active_element_grain.HostRead();
    for (int i = 0; i < num_active; i++) {
        parent_grain[m_active_to_parent[i]] = grain[i];
    }
    return parent_grain;
}

} // namespace exaconstit::microstructure
