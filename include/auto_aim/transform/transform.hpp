#pragma once

#include "auto_aim/runtime/state.hpp"

#include <string>

#include <Eigen/Geometry>

namespace armor {

struct TransformConfig {
    std::string source_frame = "camera";
    std::string target_frame = "aim";
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    [[nodiscard]] bool isIdentity() const;

    static TransformConfig fromYaml(const std::string& yaml_path);
};

class TargetTransformer {
public:
    explicit TargetTransformer(TransformConfig config = {});

    TransformResult transform(const Armor& target) const;

private:
    TransformConfig config_;
};

}  // namespace armor
