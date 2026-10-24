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
 *  3. With passthrough attributes, the element mask: 1 for every local element whose attribute
 *     is active, 0 otherwise (local, no communication). Without passthrough attributes every
 *     element is active (step 2c makes sure every attribute is classified) and the mask stays
 *     empty, which every stage reads as "all active".
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

    // Step 3: the element mask. With no passthrough attribute every element is active (step 2c
    // guarantees it) and the mask stays empty ("all active").
    if (m_passthrough_attributes.empty()) {
        return;
    }
    m_mask.active.SetSize(parent.GetNE());
    for (int e = 0; e < parent.GetNE(); e++) {
        m_mask.active[e] = active.count(parent.GetAttribute(e)) != 0 ? 1 : 0;
    }
}

const ElementMask* AttributeDomain::Mask() const {
    return m_mask.active.Size() > 0 ? &m_mask : nullptr;
}

/*
 * ToParent
 *
 * Steps:
 *  1. Check the input on this rank (size, grain id range of the active elements) and reduce the
 *     verdict over the communicator (MPI_Allreduce with MPI_MAX of a 0/1 flag), so that all ranks
 *     throw together.
 *  2. Passthrough elements: look up the position k of the element's attribute in the sorted
 *     passthrough list by binary search (std::lower_bound: repeatedly halving the sorted list
 *     until the entry is found, log2(size) comparisons) and store num_grains + 1 + k.
 *  3. Active elements keep their grain. Every element is either active or passthrough (checked
 *     in the constructor), so every entry is written exactly once.
 *
 * Host loops: the attributes live on the host and the work is one pass over the elements.
 */
mfem::Array<int> AttributeDomain::ToParent(const mfem::Array<int>& element_grain,
                                           int num_grains) const {
    // Step 1: local check of the active entries, then a global verdict
    const int parent_ne = m_parent.GetNE();
    const ElementMask* mask = Mask();
    int bad = 0;
    if (element_grain.Size() != parent_ne || num_grains < 0) {
        bad = 1;
    } else {
        const int* grain = element_grain.HostRead();
        for (int e = 0; e < parent_ne; e++) {
            if (ElementIsActive(mask, e) && (grain[e] < 1 || grain[e] > num_grains)) {
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
            "element, or an active element's grain id lies outside [1, num_grains]");
    }

    // Steps 2-3: passthrough elements get num_grains + 1 + (index of their attribute), active
    // elements keep their grain
    const int* grain = element_grain.HostRead();
    mfem::Array<int> parent_grain(parent_ne);
    for (int e = 0; e < parent_ne; e++) {
        if (ElementIsActive(mask, e)) {
            parent_grain[e] = grain[e];
            continue;
        }
        const int attribute = m_parent.GetAttribute(e);
        const auto it = std::lower_bound(m_passthrough_attributes.begin(),
                                         m_passthrough_attributes.end(), attribute);
        parent_grain[e] = num_grains + 1 + static_cast<int>(it - m_passthrough_attributes.begin());
    }
    return parent_grain;
}

} // namespace exaconstit::microstructure
