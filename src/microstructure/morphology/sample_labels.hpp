#pragma once

#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

/**
 * @file sample_labels.hpp
 * @brief Gives every sample point to the seed with the smallest metric value f_i(x) and returns
 * that value: the per-iteration labeling step shared by weight fitting and Lloyd relaxation.
 *
 * The labeling is the exact seed search of geometry/seed_index.hpp (SeedIndex::FindBest: ties go
 * to the lowest seed index, the result equals a loop over all seeds), with the metric functor
 * chosen at run time by DispatchMetric and the Axom execution space matching the mfem::Device by
 * DispatchAxomExec (exec_dispatch.hpp). A sample's label depends only on its coordinates and on
 * the replicated seeds, so labels are identical for every mesh partition and rank count.
 */

namespace exaconstit::microstructure {

/**
 * @brief Nearest seed of every sample under one metric, and the metric value there.
 *
 * Builds a SeedIndex over `seeds` (positions, `seeds.weight` for weighted metrics, `seeds.shape`
 * for per-seed quadric tensors) and queries it with the sample coordinates. Not collective (each
 * rank labels its own samples). Host function launching kernels on the mfem::Device backend.
 *
 * @param samples     this rank's samples (only the coordinates are used).
 * @param seeds       replicated seed set, N >= 1 seeds.
 * @param metric      metric family (VORONOI ignores weights).
 * @param setup       metric parameters not stored per seed (Quadric global tensor); its
 *                    per_seed_shape flag is overwritten from `seeds.shape` being non-empty.
 * @param num_bins    parameter bins of the seed search (performance only, >= 1).
 * @param sample_seed resized to the local sample count; 0-based index of the winning seed, or -1
 *                    for a non-finite sample coordinate. Written where kernels run.
 * @param sample_f    if not null, resized to the local sample count; f of the winning seed
 *                    (metric units: length^2 for Voronoi/Laguerre/Quadric, length for Additive).
 * @throws std::invalid_argument for an empty or invalid seed set (SeedIndex::Build).
 */
void LabelSamples(const WeightedSamples& samples, const SeedSet& seeds, MetricType metric,
                  MetricSetup setup, int num_bins, mfem::Array<int>& sample_seed,
                  mfem::Vector* sample_f);

} // namespace exaconstit::microstructure
