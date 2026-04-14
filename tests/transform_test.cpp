#include "auto_aim/transform/transform.hpp"

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

namespace {

armor::Armor makeArmorAt(const Eigen::Vector3d& position) {
    armor::Armor armor;
    armor.pos = position;
    armor.is_ok = true;
    return armor;
}

}  // namespace

TEST_CASE("TargetTransformer keeps identity transform unchanged", "[transform]") {
    const armor::TargetTransformer transformer;
    const armor::Armor armor = makeArmorAt(Eigen::Vector3d(0.2, -0.3, 1.5));

    const auto result = transformer.transform(armor);

    CHECK_THAT(result.position.x(), WithinAbs(0.2, 1e-9));
    CHECK_THAT(result.position.y(), WithinAbs(-0.3, 1e-9));
    CHECK_THAT(result.position.z(), WithinAbs(1.5, 1e-9));
    CHECK(result.source_frame == "camera");
    CHECK(result.target_frame == "aim");
}

TEST_CASE("TargetTransformer applies rotation and translation extrinsic", "[transform]") {
    armor::TransformConfig config;
    config.source_frame = "camera";
    config.target_frame = "gimbal";
    config.translation = Eigen::Vector3d(1.0, 2.0, 3.0);
    config.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));

    const armor::TargetTransformer transformer(config);
    const armor::Armor armor = makeArmorAt(Eigen::Vector3d(1.0, 0.0, 0.0));

    const auto result = transformer.transform(armor);

    CHECK_THAT(result.position.x(), WithinAbs(1.0, 1e-9));
    CHECK_THAT(result.position.y(), WithinAbs(3.0, 1e-9));
    CHECK_THAT(result.position.z(), WithinAbs(3.0, 1e-9));
    CHECK(result.source_frame == "camera");
    CHECK(result.target_frame == "gimbal");
}

TEST_CASE("TransformConfig can load extrinsic from YAML", "[transform]") {
    const auto config = armor::TransformConfig::fromYaml("config/camera_info.yaml");

    CHECK(config.source_frame == "camera");
    CHECK(config.target_frame == "gimbal");
    CHECK_THAT(config.translation.x(), WithinAbs(0.02, 1e-9));
    CHECK_THAT(config.translation.y(), WithinAbs(-0.01, 1e-9));
    CHECK_THAT(config.translation.z(), WithinAbs(0.08, 1e-9));

    const Eigen::Vector3d rotated = config.rotation * Eigen::Vector3d::UnitX();
    CHECK_THAT(rotated.x(), WithinAbs(0.0, 1e-9));
    CHECK_THAT(rotated.y(), WithinAbs(1.0, 1e-9));
    CHECK_THAT(rotated.z(), WithinAbs(0.0, 1e-9));
}
