#include "auto_aim/transform/transform.hpp"

#include "utils/logger/logger.hpp"

#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace armor {

namespace {

Eigen::Vector3d parseVector3(const YAML::Node& node, const Eigen::Vector3d& fallback) {
    if (!node || !node.IsSequence() || node.size() != 3) {
        return fallback;
    }

    return Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

Eigen::Quaterniond parseQuaternionWxyz(const YAML::Node& node,
                                       const Eigen::Quaterniond& fallback) {
    if (!node || !node.IsSequence() || node.size() != 4) {
        return fallback;
    }

    Eigen::Quaterniond q(node[0].as<double>(), node[1].as<double>(), node[2].as<double>(),
                         node[3].as<double>());
    if (q.norm() <= 1e-12) {
        return fallback;
    }
    q.normalize();
    return q;
}

Eigen::Quaterniond parseRpyDeg(const YAML::Node& node, const Eigen::Quaterniond& fallback) {
    if (!node || !node.IsSequence() || node.size() != 3) {
        return fallback;
    }

    constexpr double kDegToRad = M_PI / 180.0;
    const double roll = node[0].as<double>() * kDegToRad;
    const double pitch = node[1].as<double>() * kDegToRad;
    const double yaw = node[2].as<double>() * kDegToRad;

    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    q.normalize();
    return q;
}

}  // namespace

bool TransformConfig::isIdentity() const {
    return source_frame == "camera" && target_frame == "aim" &&
           rotation.isApprox(Eigen::Quaterniond::Identity()) &&
           translation.isApprox(Eigen::Vector3d::Zero());
}

TransformConfig TransformConfig::fromYaml(const std::string& yaml_path) {
    TransformConfig config;

    try {
        const YAML::Node root = YAML::LoadFile(yaml_path);
        const YAML::Node tf_node = root["target_transform"];
        if (!tf_node) {
            utils::logger()->info(
                "[TransformConfig] No 'target_transform' node found in {}, using identity extrinsic",
                yaml_path);
            return config;
        }

        if (tf_node["source_frame"]) {
            config.source_frame = tf_node["source_frame"].as<std::string>();
        }
        if (tf_node["target_frame"]) {
            config.target_frame = tf_node["target_frame"].as<std::string>();
        }
        if (tf_node["translation_m"]) {
            config.translation = parseVector3(tf_node["translation_m"], config.translation);
        }
        if (tf_node["rotation_quat_wxyz"]) {
            config.rotation = parseQuaternionWxyz(tf_node["rotation_quat_wxyz"], config.rotation);
        } else if (tf_node["rotation_rpy_deg"]) {
            config.rotation = parseRpyDeg(tf_node["rotation_rpy_deg"], config.rotation);
        }

        utils::logger()->info(
            "[TransformConfig] Loaded {} -> {} from {} | t=({:.3f}, {:.3f}, {:.3f})",
            config.source_frame, config.target_frame, yaml_path, config.translation.x(),
            config.translation.y(), config.translation.z());
    } catch (const YAML::Exception& e) {
        utils::logger()->error("[TransformConfig] YAML parse error in {}: {}", yaml_path,
                               e.what());
    } catch (const std::exception& e) {
        utils::logger()->error("[TransformConfig] Failed to load {}: {}", yaml_path, e.what());
    }

    return config;
}

TargetTransformer::TargetTransformer(TransformConfig config) : config_(std::move(config)) {
    if (config_.rotation.norm() <= 1e-12) {
        config_.rotation = Eigen::Quaterniond::Identity();
    } else {
        config_.rotation.normalize();
    }
}

TransformResult TargetTransformer::transform(const Armor& target) const {
    TransformResult result;
    result.position = config_.rotation * target.pos + config_.translation;
    result.source_frame = config_.source_frame;
    result.target_frame = config_.target_frame;
    return result;
}

}  // namespace armor
