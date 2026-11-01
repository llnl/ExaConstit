#pragma once

/**
 * @file volume_statistics.hpp
 * @brief Volume-weighted statistics of a quantity over a material region and the whole sample
 *
 * The volume averages give one number per component; these functions describe the spread of
 * the quantity over the volume: mean, variance, standard deviation, skewness, excess kurtosis,
 * minimum, maximum and a fixed-bin histogram of volume fractions. Every integration point is
 * weighted by its volume (integration weight times detJ), so the mean equals the volume average.
 *
 * Per region, ComputeRegionStatistics() works on the region's per-point values in two passes
 * (the mean first, then the central moments, which avoids the loss of accuracy of raw moments)
 * and returns a packed result. CombineStatistics() merges the packed results of several regions
 * exactly (pairwise update formulas for central moments), which gives the whole-sample
 * statistics without another pass over the points. StatisticRow() / HistogramRow() turn a
 * packed result into the values written to the files (one file per statistic, one histogram
 * file per quantity).
 *
 * Packed result layout, for n components and b histogram bins (b = 0: no histogram):
 *   [mean (n) | M2 (n) | M3 (n) | M4 (n) | min (n) | max (n) | histogram volumes (n * (b + 2))]
 * with Mk = integral of (x - mean)^k dV, and per component the volumes below the range, in each
 * bin, and above the range.
 */

#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"

#include "mfem.hpp"

#include <string>
#include <vector>

namespace exaconstit {

/**
 * @brief Number of values in a packed statistics result
 * @param num_components Components per point
 * @param num_bins Histogram bins (0 for no histogram)
 */
int StatisticsDataSize(const int num_components, const int num_bins);

/**
 * @brief Statistics of a region's per-point values (collective on region_comm)
 *
 * @param values Per-point values of the region, num_components = vector dimension
 * @param histogram Histogram bins and range (bins = 0: no histogram)
 * @param class_device Execution model, as for the volume averages
 * @param region_comm Communicator of the ranks that own elements of the region
 * @param packed Output: packed result (see the file description)
 * @return Region volume
 */
double ComputeRegionStatistics(const mfem::expt::PartialQuadratureFunction& values,
                               const HistogramOptions& histogram,
                               const RTModel& class_device,
                               MPI_Comm region_comm,
                               mfem::Vector& packed);

/**
 * @brief Merge the packed results of several regions into one (exact)
 *
 * @param volumes Volume of each region (regions with zero volume are skipped)
 * @param packed Packed result of each region
 * @param num_components Components per point
 * @param num_bins Histogram bins (0 for no histogram)
 * @param combined Output: packed result of all regions together
 * @return Total volume (0 if no region had any volume)
 */
double CombineStatistics(const std::vector<double>& volumes,
                         const std::vector<mfem::Vector>& packed,
                         const int num_components,
                         const int num_bins,
                         mfem::Vector& combined);

/**
 * @brief One statistic of a packed result, for every component
 *
 * @param packed Packed result
 * @param volume Its volume
 * @param num_components Components per point
 * @param statistic "mean", "variance", "std", "skewness", "kurtosis" (excess, 0 for a normal
 *                  distribution), "min" or "max"
 * @return The statistic of each component (one value per component, in component order). A
 *         standard deviation below 1e-10 of the largest magnitude of the values (round-off of a
 *         constant field) is taken as 0, and skewness and kurtosis are 0 where it is 0.
 */
mfem::Vector StatisticRow(const mfem::Vector& packed,
                          const double volume,
                          const int num_components,
                          const std::string& statistic);

/**
 * @brief Histogram volume fractions of a packed result, component by component
 * @return For each component: fraction below the range, in each bin, above the range
 */
mfem::Vector HistogramRow(const mfem::Vector& packed,
                          const double volume,
                          const int num_components,
                          const int num_bins);

/// Column names of HistogramRow(): "<component>_below", "<component>_bin_1", ..., "_above"
std::vector<std::string> HistogramColumnNames(const std::vector<std::string>& component_names,
                                              const int num_bins);

/// Header comment line listing the histogram bin edges: "# Bin edges: e0 e1 ... eN"
std::string HistogramEdgesComment(const HistogramOptions& histogram);

} // namespace exaconstit
