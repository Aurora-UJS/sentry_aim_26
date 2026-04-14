#include "auto_aim/aim_solver/aim_solver.hpp"

#include <cmath>

namespace armor {

AimSolver::AimSolver(AimSolverConfig config) : config_(config) {}

std::optional<AimCommand> AimSolver::solve(const TransformResult& target) const {
    const double distance = target.position.norm();
    if (distance <= 0.0) {
        return std::nullopt;
    }

    AimCommand command;
    command.distance_m = distance;
    command.flight_time_s =
        (config_.bullet_speed_mps > 0.0 ? distance / config_.bullet_speed_mps : 0.0) +
        config_.system_latency_s;
    command.yaw = std::atan2(target.position.x(), target.position.z());
    command.pitch =
        std::atan2(-target.position.y(), std::hypot(target.position.x(), target.position.z())) +
        config_.pitch_offset_rad;
    command.valid = std::isfinite(command.yaw) && std::isfinite(command.pitch);
    command.fire_enable = command.valid;

    if (!command.valid) {
        return std::nullopt;
    }

    return command;
}

}  // namespace armor
