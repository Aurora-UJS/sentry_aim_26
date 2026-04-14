#include "auto_aim/target_selector/target_selector.hpp"

#include <limits>

namespace armor {

std::optional<Armor> TargetSelector::select(const std::vector<Armor>& tracked_targets) const {
    std::optional<Armor> best_target;
    double min_distance = std::numeric_limits<double>::max();

    for (const auto& target : tracked_targets) {
        if (!target.is_ok) {
            continue;
        }

        const double distance = target.pos.norm();
        if (distance < min_distance) {
            min_distance = distance;
            best_target = target;
        }
    }

    return best_target;
}

}  // namespace armor
