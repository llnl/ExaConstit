/*
 * Implementation of LabelSamples (sample_labels.hpp).
 *
 * Terms used in this file:
 * - BVH (bounding volume hierarchy): a tree of nested bounding boxes that lets us skip seeds that
 *   are provably too far away; SeedIndex (geometry/seed_index.hpp) builds one and searches it.
 * - Execution space: the Axom tag type (axom::SEQ_EXEC, axom::CUDA_EXEC<256>, ...) that says
 *   where the BVH kernels run; DispatchAxomExec picks the one matching the mfem::Device.
 */
#include "microstructure/morphology/sample_labels.hpp"

#include "microstructure/exec_dispatch.hpp"
#include "microstructure/geometry/seed_index.hpp"

#include "mfem.hpp"

namespace exaconstit::microstructure {

/*
 * LabelSamples
 *
 * Algorithm (the same pattern as LabelElementsWithSeeds in labeling/tessellation_labeler.cpp):
 *  1. DispatchMetric and DispatchAxomExec call a lambda with a value whose *type* selects the
 *     SeedIndex template instantiation (metric functor x execution space).
 *  2. Build the index from the replicated seeds and run FindBest on the sample coordinates;
 *     only the best seed is needed, so the exact second-best search is off.
 *  3. Copy id1 (and f1) out of the BestPair array with an mfem::forall kernel. This kernel sits
 *     outside the dispatch lambdas on purpose: CUDA does not accept a device lambda defined
 *     inside a generic (auto-parameter) lambda.
 */
void LabelSamples(const WeightedSamples& samples, const SeedSet& seeds, MetricType metric,
                  MetricSetup setup, int num_bins, mfem::Array<int>& sample_seed,
                  mfem::Vector* sample_f) {
    setup.per_seed_shape = !seeds.shape.empty();
    const int n = samples.Size();

    // Steps 1-2: exact nearest-seed search
    mfem::Array<BestPair> best;
    DispatchMetric(metric, [&](auto metric_tag) {
        using Metric = decltype(metric_tag);
        DispatchAxomExec([&](auto exec_tag) {
            using Exec = decltype(exec_tag);
            SeedIndex<Exec, Metric> index;
            index.Build(seeds, setup, num_bins, AxomAllocatorIdForCurrentDevice());
            index.FindBest(samples.x, samples.y, samples.z, best, false);
        });
    });

    // Step 3: unpack the winning seed index and its metric value
    sample_seed.SetSize(n);
    if (sample_f != nullptr) {
        sample_f->SetSize(n);
        sample_f->UseDevice(true);
    }
    if (n == 0) {
        return;
    }
    const BestPair* best_data = best.Read();
    int* seed_out = sample_seed.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) { seed_out[q] = best_data[q].id1; });
    if (sample_f != nullptr) {
        double* f_out = sample_f->Write();
        mfem::forall(n, [=] MFEM_HOST_DEVICE(int q) { f_out[q] = best_data[q].f1; });
    }
}

} // namespace exaconstit::microstructure
