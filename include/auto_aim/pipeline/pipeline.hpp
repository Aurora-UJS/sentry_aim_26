#pragma once

#include "auto_aim/aim_solver/aim_solver.hpp"
#include "auto_aim/armor_detector/detector.hpp"
#include "auto_aim/armor_solver/solver.hpp"
#include "auto_aim/armor_tracker/tracker.hpp"
#include "auto_aim/runtime/state.hpp"
#include "auto_aim/target_selector/target_selector.hpp"
#include "auto_aim/transform/transform.hpp"

#include <memory>
#include <string>

namespace armor {

struct PipelineConfig {
    std::string detector_config_path;
    std::string camera_config_path;
    std::string tracker_config_path;
    TransformConfig transform_config;
    AimSolverConfig aim_solver_config;
};

class AutoAimPipeline {
public:
    struct Components {
        std::unique_ptr<Detector> detector;
        std::unique_ptr<PnpSolver> pnp_solver;
        std::unique_ptr<TrackerManager> tracker_manager;
        std::unique_ptr<TargetTransformer> transformer;
        std::unique_ptr<TargetSelector> target_selector;
        std::unique_ptr<AimSolver> aim_solver;
    };

    explicit AutoAimPipeline(Components components);

    static std::unique_ptr<AutoAimPipeline> create(const PipelineConfig& config);

    PipelineFrame process(const PipelineInput& input);

    const RuntimeState& state() const { return state_; }

private:
    Components components_;
    RuntimeState state_;
};

}  // namespace armor
