/**
 * @file test_volume_statistics.cpp
 * @brief Checks the exact combination of volume statistics across regions
 *
 * CombineStatistics() (src/postprocessing/volume_statistics.hpp) merges the packed statistics
 * of several regions (volume, mean, central moment integrals M2..M4, min, max, histogram
 * volumes) into those of all regions together, which gives the whole-sample statistics without
 * another pass over the points. Here random weighted samples (weights = point volumes, two
 * components) are split into three groups of different sizes and spreads; each group's packed
 * result is computed directly, the groups are combined, and the result must equal the direct
 * computation over all samples. The groups have nonzero spread, so every term of the pairwise
 * update formulas (including the M2 / M3 cross terms) contributes. StatisticRow() and
 * HistogramRow() are checked against the textbook definitions on the same data.
 *
 * Part of the extensive suite (EXACONSTIT_ENABLE_EXTENSIVE_TESTS).
 */
#include "postprocessing/volume_statistics.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr int num_components = 2;
constexpr int num_bins = 5;
constexpr double lower = -2.0;
constexpr double upper = 2.0;

/// Weighted samples of a quantity: weights[i] and values[i * num_components + c]
struct Samples {
    std::vector<double> weights;
    std::vector<double> values;
};

/// Packed statistics of samples, computed directly (layout of volume_statistics.hpp)
double DirectPacked(const Samples& s, mfem::Vector& packed) {
    const int n = num_components;
    packed.SetSize(exaconstit::StatisticsDataSize(n, num_bins));
    packed = 0.0;
    double volume = 0.0;
    for (const double w : s.weights) {
        volume += w;
    }
    for (int c = 0; c < n; c++) {
        double mean = 0.0;
        double min_value = 1e300;
        double max_value = -1e300;
        for (size_t i = 0; i < s.weights.size(); i++) {
            const double x = s.values[i * n + c];
            mean += s.weights[i] * x;
            min_value = std::min(min_value, x);
            max_value = std::max(max_value, x);
        }
        mean /= volume;
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (size_t i = 0; i < s.weights.size(); i++) {
            const double d = s.values[i * n + c] - mean;
            m2 += s.weights[i] * d * d;
            m3 += s.weights[i] * d * d * d;
            m4 += s.weights[i] * d * d * d * d;
            // Histogram slot: below, bins (upper edge in the last bin), above
            const double x = s.values[i * n + c];
            int slot = 0;
            if (x > upper) {
                slot = num_bins + 1;
            } else if (x >= lower) {
                slot = 1 + std::min(static_cast<int>((x - lower) / ((upper - lower) / num_bins)),
                                    num_bins - 1);
            }
            packed[6 * n + c * (num_bins + 2) + slot] += s.weights[i];
        }
        packed[0 * n + c] = mean;
        packed[1 * n + c] = m2;
        packed[2 * n + c] = m3;
        packed[3 * n + c] = m4;
        packed[4 * n + c] = min_value;
        packed[5 * n + c] = max_value;
    }
    return volume;
}

/// Group g: count samples with means / spreads that differ per group and component
Samples MakeGroup(std::mt19937& gen, const int count, const int g) {
    std::uniform_real_distribution<double> weight(0.1, 2.0);
    std::normal_distribution<double> noise(0.0, 1.0);
    Samples s;
    for (int i = 0; i < count; i++) {
        s.weights.push_back(weight(gen));
        for (int c = 0; c < num_components; c++) {
            // Skewed values (squares) shifted per group, so groups have different means,
            // spreads and shapes
            const double z = noise(gen);
            s.values.push_back(0.5 * g - 0.3 * c + (0.4 + 0.2 * g) * (z + 0.3 * z * z));
        }
    }
    return s;
}

TEST(VolumeStatistics, CombinationEqualsDirect) {
    std::mt19937 gen(12345);
    const std::vector<Samples> groups = {
        MakeGroup(gen, 50, 0), MakeGroup(gen, 13, 1), MakeGroup(gen, 200, 2)};

    // All samples together
    Samples all;
    for (const auto& g : groups) {
        all.weights.insert(all.weights.end(), g.weights.begin(), g.weights.end());
        all.values.insert(all.values.end(), g.values.begin(), g.values.end());
    }
    mfem::Vector direct;
    const double direct_volume = DirectPacked(all, direct);

    // Group by group, then combined (an empty region in between must be skipped)
    std::vector<double> volumes;
    std::vector<mfem::Vector> packed;
    for (size_t g = 0; g < groups.size(); g++) {
        mfem::Vector p;
        volumes.push_back(DirectPacked(groups[g], p));
        packed.push_back(p);
        if (g == 0) {
            volumes.push_back(0.0);
            packed.push_back(
                mfem::Vector(exaconstit::StatisticsDataSize(num_components, num_bins)));
        }
    }
    mfem::Vector combined;
    const double volume = exaconstit::CombineStatistics(
        volumes, packed, num_components, num_bins, combined);

    EXPECT_NEAR(volume, direct_volume, 1e-12 * direct_volume);
    ASSERT_EQ(combined.Size(), direct.Size());
    for (int i = 0; i < direct.Size(); i++) {
        EXPECT_NEAR(combined[i], direct[i], 1e-10 * (1.0 + std::abs(direct[i])))
            << "packed entry " << i;
    }

    // Statistics from the packed result against their definitions
    for (int c = 0; c < num_components; c++) {
        const double mean = direct[c];
        const double variance = direct[num_components + c] / direct_volume;
        const double skewness = direct[2 * num_components + c] / direct_volume /
                                std::pow(variance, 1.5);
        const double kurtosis = direct[3 * num_components + c] / direct_volume /
                                    (variance * variance) -
                                3.0;
        const auto check = [&](const std::string& name, const double expected) {
            const auto row = exaconstit::StatisticRow(combined, volume, num_components, name);
            EXPECT_NEAR(row[c], expected, 1e-10 * (1.0 + std::abs(expected))) << name;
        };
        check("mean", mean);
        check("variance", variance);
        check("std", std::sqrt(variance));
        check("skewness", skewness);
        check("kurtosis", kurtosis);
        check("min", direct[4 * num_components + c]);
        check("max", direct[5 * num_components + c]);
    }

    // Histogram fractions add up to 1 per component
    const auto hist = exaconstit::HistogramRow(combined, volume, num_components, num_bins);
    for (int c = 0; c < num_components; c++) {
        double total = 0.0;
        for (int b = 0; b < num_bins + 2; b++) {
            total += hist[c * (num_bins + 2) + b];
        }
        EXPECT_NEAR(total, 1.0, 1e-12);
    }
}

TEST(VolumeStatistics, ConstantFieldHasNoShape) {
    // Round-off spread of a constant: skewness and kurtosis must come out 0, not noise
    Samples s;
    for (int i = 0; i < 10; i++) {
        s.weights.push_back(0.1 + 0.01 * i);
        s.values.push_back(3.0 + 1e-17 * (i % 3));
        s.values.push_back(-1.0);
    }
    mfem::Vector packed;
    const double volume = DirectPacked(s, packed);
    for (const char* name : {"variance", "std", "skewness", "kurtosis"}) {
        const auto row = exaconstit::StatisticRow(packed, volume, num_components, name);
        EXPECT_EQ(row[0], 0.0) << name;
        EXPECT_EQ(row[1], 0.0) << name;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
