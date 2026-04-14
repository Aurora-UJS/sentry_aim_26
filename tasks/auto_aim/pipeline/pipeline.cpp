#include "auto_aim/pipeline/pipeline.hpp"

#include "auto_aim/armor_detector/detector_factory.hpp"
#include "utils/logger/logger.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace armor {

namespace {

double elapsedMilliseconds(const std::chrono::steady_clock::time_point& start,
                           const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

AutoAimPipeline::AutoAimPipeline(Components components) : components_(std::move(components)) {
    if (!components_.detector || !components_.pnp_solver || !components_.tracker_manager ||
        !components_.transformer || !components_.target_selector || !components_.aim_solver) {
        throw std::invalid_argument("AutoAimPipeline components must all be initialized");
    }
}

std::unique_ptr<AutoAimPipeline> AutoAimPipeline::create(const PipelineConfig& config) {
    Components components;
    components.detector = DetectorFactory::createFromConfig(config.detector_config_path);
    if (!components.detector) {
        utils::logger()->error("[AutoAimPipeline] Failed to create detector from {}",
                               config.detector_config_path);
        return nullptr;
    }

    components.pnp_solver = std::make_unique<PnpSolver>(config.camera_config_path);
    const TrackerConfig tracker_config = TrackerConfig::fromYaml(config.tracker_config_path);
    components.tracker_manager = std::make_unique<TrackerManager>(tracker_config);

    TransformConfig transform_config = config.transform_config;
    if (!config.camera_config_path.empty() && transform_config.isIdentity()) {
        transform_config = TransformConfig::fromYaml(config.camera_config_path);
    }
    components.transformer = std::make_unique<TargetTransformer>(transform_config);
    components.target_selector = std::make_unique<TargetSelector>();

    AimSolverConfig aim_solver_config = config.aim_solver_config;
    if (aim_solver_config.bullet_speed_mps <= 0.0) {
        aim_solver_config.bullet_speed_mps = tracker_config.bullet_speed;
    }
    components.aim_solver = std::make_unique<AimSolver>(aim_solver_config);

    return std::make_unique<AutoAimPipeline>(std::move(components));
}

PipelineFrame AutoAimPipeline::process(const PipelineInput& input) {
    PipelineFrame frame;
    frame.input = input;

    const auto total_start = std::chrono::steady_clock::now();

    if (state_.last_timestamp.time_since_epoch().count() > 0) {
        frame.output.stats.dt_s =
            std::chrono::duration<double>(input.timestamp - state_.last_timestamp).count();
    }

    if (input.frame.empty()) {
        frame.output.stats.timings.total_ms = 0.0;
        return frame;
    }

    auto stage_start = std::chrono::steady_clock::now();
    frame.output.detections = components_.detector->detect(input.frame);
    auto stage_end = std::chrono::steady_clock::now();
    frame.output.stats.timings.detect_ms = elapsedMilliseconds(stage_start, stage_end);
    frame.output.stats.detection_count = frame.output.detections.size();

    stage_start = std::chrono::steady_clock::now();
    for (const auto& detection : frame.output.detections) {
        Armor armor;
        if (!components_.pnp_solver->solve(detection, armor)) {
            continue;
        }

        armor.timestamp = input.timestamp;
        frame.output.solved_armors.push_back(armor);
    }
    stage_end = std::chrono::steady_clock::now();
    frame.output.stats.timings.solve_ms = elapsedMilliseconds(stage_start, stage_end);
    frame.output.stats.solved_count = frame.output.solved_armors.size();

    stage_start = std::chrono::steady_clock::now();
    components_.tracker_manager->update(frame.output.solved_armors, input.timestamp);
    frame.output.tracked_targets = components_.tracker_manager->getTrackedTargets();
    stage_end = std::chrono::steady_clock::now();
    frame.output.stats.timings.track_ms = elapsedMilliseconds(stage_start, stage_end);
    frame.output.stats.tracker_count = components_.tracker_manager->getTrackers().size();
    frame.output.stats.tracked_candidate_count = frame.output.tracked_targets.size();

    stage_start = std::chrono::steady_clock::now();
    frame.output.selected_target = components_.target_selector->select(frame.output.tracked_targets);
    stage_end = std::chrono::steady_clock::now();
    frame.output.stats.timings.select_ms = elapsedMilliseconds(stage_start, stage_end);

    if (frame.output.selected_target.has_value()) {
        stage_start = std::chrono::steady_clock::now();
        frame.output.transformed_target =
            components_.transformer->transform(frame.output.selected_target.value());
        stage_end = std::chrono::steady_clock::now();
        frame.output.stats.timings.transform_ms = elapsedMilliseconds(stage_start, stage_end);

        stage_start = std::chrono::steady_clock::now();
        frame.output.aim_command =
            components_.aim_solver->solve(frame.output.transformed_target.value());
        stage_end = std::chrono::steady_clock::now();
        frame.output.stats.timings.aim_ms = elapsedMilliseconds(stage_start, stage_end);
    }

    frame.output.stats.has_target = frame.output.aim_command.has_value();
    frame.output.stats.timings.total_ms =
        elapsedMilliseconds(total_start, std::chrono::steady_clock::now());

    state_.frame_index++;
    state_.last_timestamp = input.timestamp;
    state_.last_selected_target = frame.output.selected_target;
    state_.last_aim_command = frame.output.aim_command;

    return frame;
}

}  // namespace armor
