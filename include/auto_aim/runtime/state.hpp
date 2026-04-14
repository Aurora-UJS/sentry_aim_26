#pragma once

#include "auto_aim/type.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace armor {

struct AimCommand {
    double yaw = 0.0;
    double pitch = 0.0;
    bool fire_enable = false;
    double distance_m = 0.0;
    double flight_time_s = 0.0;
    bool valid = false;
};

struct TransformResult {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::string source_frame = "camera";
    std::string target_frame = "aim";
};

struct PipelineTimings {
    double detect_ms = 0.0;
    double solve_ms = 0.0;
    double track_ms = 0.0;
    double select_ms = 0.0;
    double transform_ms = 0.0;
    double aim_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineStats {
    std::size_t detection_count = 0;
    std::size_t solved_count = 0;
    std::size_t tracker_count = 0;
    std::size_t tracked_candidate_count = 0;
    bool has_target = false;
    double dt_s = 0.0;
    PipelineTimings timings;
};

struct PipelineInput {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    std::string frame_id = "camera";
};

struct PipelineOutput {
    std::vector<ArmorObject> detections;
    std::vector<Armor> solved_armors;
    std::vector<Armor> tracked_targets;
    std::optional<Armor> selected_target;
    std::optional<TransformResult> transformed_target;
    std::optional<AimCommand> aim_command;
    PipelineStats stats;
};

struct PipelineFrame {
    PipelineInput input;
    PipelineOutput output;
};

struct RuntimeState {
    std::size_t frame_index = 0;
    std::chrono::steady_clock::time_point last_timestamp{};
    std::optional<Armor> last_selected_target;
    std::optional<AimCommand> last_aim_command;
};

}  // namespace armor
