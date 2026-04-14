#pragma once

#include "auto_aim/runtime/state.hpp"

#include <optional>

namespace armor {

struct AimSolverConfig {
    double bullet_speed_mps = -1.0;
    double system_latency_s = 0.0;
    double pitch_offset_rad = 0.0;
};

class AimSolver {
public:
    explicit AimSolver(AimSolverConfig config = {});

    std::optional<AimCommand> solve(const TransformResult& target) const;

private:
    AimSolverConfig config_;
};

}  // namespace armor
