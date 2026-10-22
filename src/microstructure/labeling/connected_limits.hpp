#pragma once

/**
 * @file connected_limits.hpp
 * @brief Limits of the "connected" distance mode (labeling/connected_labeler.hpp), in a header of
 * their own so the options code can check them without including the labeler.
 */

namespace exaconstit::microstructure {

/**
 * @brief Largest accepted ConnectedLabelParameters::max_rounds (and Tessellation.
 * connected_max_rounds). Every round costs a component search (several neighbor exchanges and
 * collectives) and adds one list entry per orphan, so more rounds than this are never useful;
 * 16-64 is plenty for lattices (the rounds needed are about the number of seeds whose cells reach
 * across one void, usually a handful). Compile-time constant, usable on host and device.
 */
inline constexpr int CONNECTED_MAX_ROUNDS_LIMIT = 1000;

} // namespace exaconstit::microstructure
