#pragma once

#include "auto_aim/type.hpp"

#include <optional>
#include <vector>

namespace armor {

class TargetSelector {
public:
    std::optional<Armor> select(const std::vector<Armor>& tracked_targets) const;
};

}  // namespace armor
