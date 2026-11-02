#include "postprocessing/volume_statistics.hpp"

#include "utilities/mechanics_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace exaconstit {

namespace {
/// Blocks of the packed result (each num_components long), then the histogram volumes
enum PackedBlock { MEAN = 0, M2 = 1, M3 = 2, M4 = 3, MIN = 4, MAX = 5, NUM_BLOCKS = 6 };
} // namespace

int StatisticsDataSize(const int num_components, const int num_bins) {
    const int histogram_size = (num_bins > 0) ? num_components * (num_bins + 2) : 0;
    return NUM_BLOCKS * num_components + histogram_size;
}

double ComputeRegionStatistics(const mfem::expt::PartialQuadratureFunction& values,
                               const HistogramOptions& histogram,
                               const RTModel& class_device,
                               MPI_Comm region_comm,
                               mfem::Vector& packed) {
    const int num_components = values.GetVDim();
    const int num_points = values.Size() / num_components;
    const int num_bins = histogram.bins;
    packed.SetSize(StatisticsDataSize(num_components, num_bins));
    packed = 0.0;
    packed.HostReadWrite();

    // Pass 1: the mean and the region volume. The integral is divided here rather than in the
    // kernel, whose averaging uses an absolute volume cutoff that would zero the mean of a very
    // small region (and with it every central moment).
    mfem::Vector mean(num_components);
    const double volume = kernel::ComputeVolAvgTensorFromPartial<false>(
        &values, mean, num_components, class_device, region_comm);
    if (volume > 0.0) {
        mean /= volume;
    }

    // Pass 2: integrals of (x - mean)^2, ^3, ^4. powers(c, k, ip) holds the (k + 2)-th power of
    // component c at point ip.
    mfem::expt::PartialQuadratureFunction powers(
        values.GetPartialSpaceShared(), 3 * num_components, 0.0);
    {
        const auto x = mfem::Reshape(values.Read(), num_components, num_points);
        const auto x_mean = mfem::Reshape(mean.Read(), num_components);
        auto p = mfem::Reshape(powers.Write(), num_components, 3, num_points);
        mfem::forall(num_points, [=] MFEM_HOST_DEVICE(int ip) {
            for (int c = 0; c < num_components; c++) {
                const double d = x(c, ip) - x_mean(c);
                p(c, 0, ip) = d * d;
                p(c, 1, ip) = d * d * d;
                p(c, 2, ip) = d * d * d * d;
            }
        });
    }
    mfem::Vector central(3 * num_components);
    kernel::ComputeVolAvgTensorFromPartial<false>(
        &powers, central, 3 * num_components, class_device, region_comm);

    // Smallest and largest values
    mfem::Vector min_values, max_values;
    kernel::ComputeMinMaxFromPartial(
        &values, min_values, max_values, num_components, class_device, region_comm);

    for (int c = 0; c < num_components; c++) {
        packed[MEAN * num_components + c] = mean[c];
        packed[M2 * num_components + c] = central[c];
        packed[M3 * num_components + c] = central[num_components + c];
        packed[M4 * num_components + c] = central[2 * num_components + c];
        packed[MIN * num_components + c] = min_values[c];
        packed[MAX * num_components + c] = max_values[c];
    }

    // Histogram: the volume of the points below the range, in each bin and above the range
    if (num_bins > 0) {
        mfem::Vector bin_volumes;
        kernel::ComputeHistogramFromPartial(
            &values, num_bins, histogram.range_min, histogram.range_max, bin_volumes, region_comm);
        bin_volumes.HostRead();
        for (int i = 0; i < num_components * (num_bins + 2); i++) {
            packed[NUM_BLOCKS * num_components + i] = bin_volumes[i];
        }
    }
    return volume;
}

double CombineStatistics(const std::vector<double>& volumes,
                         const std::vector<mfem::Vector>& packed,
                         const int num_components,
                         const int num_bins,
                         mfem::Vector& combined) {
    const int size = StatisticsDataSize(num_components, num_bins);
    combined.SetSize(size);
    combined = 0.0;
    combined.HostReadWrite();

    double total_volume = 0.0;
    for (size_t r = 0; r < volumes.size(); r++) {
        const double vb = volumes[r];
        if (vb <= 0.0) {
            continue;
        }
        const mfem::Vector& b = packed[r];
        b.HostRead();
        if (total_volume <= 0.0) {
            combined = b;
            combined.HostReadWrite();
            total_volume = vb;
            continue;
        }

        // Pairwise merge of (volume, mean, M2, M3, M4) of the regions so far (a) with region b
        // (Chan, Golub and LeVeque; Pebay 2008), component by component
        const double va = total_volume;
        const double v = va + vb;
        for (int c = 0; c < num_components; c++) {
            const double mean_a = combined[MEAN * num_components + c];
            const double m2_a = combined[M2 * num_components + c];
            const double m3_a = combined[M3 * num_components + c];
            const double m4_a = combined[M4 * num_components + c];
            const double mean_b = b[MEAN * num_components + c];
            const double m2_b = b[M2 * num_components + c];
            const double m3_b = b[M3 * num_components + c];
            const double m4_b = b[M4 * num_components + c];

            const double delta = mean_b - mean_a;
            const double delta2 = delta * delta;
            const double vab = va * vb;

            combined[MEAN * num_components + c] = mean_a + delta * vb / v;
            combined[M2 * num_components + c] = m2_a + m2_b + delta2 * vab / v;
            combined[M3 * num_components + c] = m3_a + m3_b +
                                                delta * delta2 * vab * (va - vb) / (v * v) +
                                                3.0 * delta * (va * m2_b - vb * m2_a) / v;
            combined[M4 * num_components + c] = m4_a + m4_b +
                                                delta2 * delta2 * vab * (va * va - vab + vb * vb) /
                                                    (v * v * v) +
                                                6.0 * delta2 * (va * va * m2_b + vb * vb * m2_a) /
                                                    (v * v) +
                                                4.0 * delta * (va * m3_b - vb * m3_a) / v;
            combined[MIN * num_components + c] = std::min(combined[MIN * num_components + c],
                                                          b[MIN * num_components + c]);
            combined[MAX * num_components + c] = std::max(combined[MAX * num_components + c],
                                                          b[MAX * num_components + c]);
        }
        // Histogram volumes add up
        for (int i = NUM_BLOCKS * num_components; i < size; i++) {
            combined[i] += b[i];
        }
        total_volume = v;
    }
    return total_volume;
}

mfem::Vector StatisticRow(const mfem::Vector& packed,
                          const double volume,
                          const int num_components,
                          const std::string& statistic) {
    mfem::Vector row(num_components);
    packed.HostRead();
    for (int c = 0; c < num_components; c++) {
        double variance = packed[M2 * num_components + c] / volume;
        double std_dev = std::sqrt(std::max(variance, 0.0));
        // A constant field still gets a round-off spread (about 1e-16 of its size); treat a
        // spread below 1e-10 of the values' magnitude as zero, so the shape statistics of a
        // constant are 0 rather than round-off divided by round-off
        const double magnitude = std::max(std::abs(packed[MIN * num_components + c]),
                                          std::abs(packed[MAX * num_components + c]));
        if (std_dev <= 1.0e-10 * magnitude) {
            variance = 0.0;
            std_dev = 0.0;
        }
        double value = 0.0;
        if (statistic == "mean") {
            value = packed[MEAN * num_components + c];
        } else if (statistic == "variance") {
            value = variance;
        } else if (statistic == "std") {
            value = std_dev;
        } else if (statistic == "skewness") {
            value = (std_dev > 0.0)
                        ? packed[M3 * num_components + c] / volume / (std_dev * variance)
                        : 0.0;
        } else if (statistic == "kurtosis") {
            value = (variance > 0.0)
                        ? packed[M4 * num_components + c] / volume / (variance * variance) - 3.0
                        : 0.0;
        } else if (statistic == "min") {
            value = packed[MIN * num_components + c];
        } else if (statistic == "max") {
            value = packed[MAX * num_components + c];
        }
        row[c] = value;
    }
    return row;
}

mfem::Vector HistogramRow(const mfem::Vector& packed,
                          const double volume,
                          const int num_components,
                          const int num_bins) {
    // Histogram volumes are stored per component: (slot, component), slot fastest
    const int size = num_components * (num_bins + 2);
    mfem::Vector row(size);
    packed.HostRead();
    for (int i = 0; i < size; i++) {
        row[i] = packed[NUM_BLOCKS * num_components + i] / volume;
    }
    return row;
}

std::vector<std::string> HistogramColumnNames(const std::vector<std::string>& component_names,
                                              const int num_bins) {
    std::vector<std::string> names;
    for (const auto& component : component_names) {
        names.push_back(component + "_below");
        for (int b = 1; b <= num_bins; b++) {
            names.push_back(component + "_bin_" + std::to_string(b));
        }
        names.push_back(component + "_above");
    }
    return names;
}

std::vector<double> HistogramEdges(const HistogramOptions& histogram) {
    // range_min + range * b / bins rather than range_min + b * width, so edges that should be
    // round numbers (e.g. 0 in [-0.25, 0.25]) come out exact
    std::vector<double> edges;
    const double range = histogram.range_max - histogram.range_min;
    for (int b = 0; b <= histogram.bins; b++) {
        edges.push_back(histogram.range_min + range * b / histogram.bins);
    }
    return edges;
}

std::string HistogramEdgesComment(const HistogramOptions& histogram) {
    std::ostringstream comment;
    comment << "# Bin edges:";
    comment << std::scientific << std::setprecision(8);
    for (const double edge : HistogramEdges(histogram)) {
        comment << " " << edge;
    }
    return comment.str();
}

} // namespace exaconstit
