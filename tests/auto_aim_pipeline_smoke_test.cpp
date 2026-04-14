/**
 ************************************************************************
 *
 * @file auto_aim_pipeline_smoke_test.cpp
 * @brief 非交互式流水线冒烟测试
 *
 ************************************************************************
 */

#include "auto_aim/pipeline/pipeline.hpp"
#include "utils/logger/logger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kFx = 1807.12121;
constexpr double kFy = 1806.46896;
constexpr double kCx = 711.11997;
constexpr double kCy = 562.49495;

const cv::Mat kTestCameraMatrix =
    (cv::Mat_<double>(3, 3) << kFx, 0, kCx, 0, kFy, kCy, 0, 0, 1);
const cv::Mat kTestDistCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

std::vector<cv::Point2f> generateSyntheticImagePoints(const std::vector<cv::Point3f>& object_points,
                                                      const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, tvec, kTestCameraMatrix, kTestDistCoeffs, image_points);
    return image_points;
}

armor::ArmorObject makeSyntheticDetection() {
    const cv::Mat true_rvec = (cv::Mat_<double>(3, 1) << 0.0, 0.05, 0.0);
    const cv::Mat true_tvec = (cv::Mat_<double>(3, 1) << 0.1, -0.05, 1.2);

    const std::vector<cv::Point3f> object_points = {
        {-static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2),
         0.0f},
        {-static_cast<float>(SMALL_ARMOR_WIDTH / 2), static_cast<float>(SMALL_ARMOR_HEIGHT / 2),
         0.0f},
        {static_cast<float>(SMALL_ARMOR_WIDTH / 2), static_cast<float>(SMALL_ARMOR_HEIGHT / 2),
         0.0f},
        {static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2),
         0.0f}};

    armor::ArmorObject detection;
    detection.color = armor::ArmorColor::BLUE;
    detection.number = armor::ArmorNumber::NO3;
    detection.type = armor::ArmorType::SMALL;
    detection.pts = generateSyntheticImagePoints(object_points, true_rvec, true_tvec);
    detection.center = (detection.pts[0] + detection.pts[2]) * 0.5f;
    detection.box = cv::boundingRect(detection.pts);
    detection.prob = 0.95f;
    detection.class_prob = 0.98f;
    return detection;
}

class FakeDetector final : public armor::Detector {
public:
    explicit FakeDetector(std::vector<armor::ArmorObject> detections)
        : detections_(std::move(detections)) {}

    std::vector<armor::ArmorObject> detect(const cv::Mat&) override { return detections_; }

    void setParams(const armor::DetectorParams&) override {}

private:
    std::vector<armor::ArmorObject> detections_;
};

}  // namespace

TEST_CASE("AutoAimPipeline smoke path produces a target and aim command", "[pipeline]") {
    utils::logger()->set_level(spdlog::level::warn);

    armor::TrackerConfig tracker_config;
    tracker_config.tracking_threshold = 1;
    tracker_config.lost_time_threshold = 3;
    tracker_config.bullet_speed = 30.0;

    armor::AimSolverConfig aim_solver_config;
    aim_solver_config.bullet_speed_mps = tracker_config.bullet_speed;

    armor::AutoAimPipeline::Components components;
    components.detector = std::make_unique<FakeDetector>(
        std::vector<armor::ArmorObject>{makeSyntheticDetection()});
    components.pnp_solver = std::make_unique<armor::PnpSolver>("config/camera_info.yaml");
    components.tracker_manager = std::make_unique<armor::TrackerManager>(tracker_config);
    components.transformer = std::make_unique<armor::TargetTransformer>();
    components.target_selector = std::make_unique<armor::TargetSelector>();
    components.aim_solver = std::make_unique<armor::AimSolver>(aim_solver_config);

    armor::AutoAimPipeline pipeline(std::move(components));

    armor::PipelineInput input;
    input.frame = cv::Mat::zeros(1024, 1280, CV_8UC3);
    input.timestamp = std::chrono::steady_clock::now();

    const auto result = pipeline.process(input);
    const auto& transformed_target = result.output.transformed_target;
    const auto& aim_command = result.output.aim_command;

    REQUIRE(result.output.stats.detection_count == 1);
    REQUIRE(result.output.stats.solved_count == 1);
    REQUIRE(result.output.stats.tracker_count == 1);
    REQUIRE(result.output.stats.tracked_candidate_count == 1);
    REQUIRE(result.output.selected_target.has_value());
    REQUIRE(transformed_target.has_value());
    REQUIRE(aim_command.has_value());
    REQUIRE(aim_command->valid);
    REQUIRE(aim_command->fire_enable);
    CHECK(result.output.stats.has_target);
    CHECK_THAT(result.output.selected_target->pos.x(), WithinAbs(0.1, 0.02));
    CHECK_THAT(result.output.selected_target->pos.y(), WithinAbs(-0.05, 0.02));
    CHECK_THAT(result.output.selected_target->pos.z(), WithinAbs(1.2, 0.02));
    CHECK_THAT(aim_command->yaw,
               WithinAbs(std::atan2(transformed_target->position.x(), transformed_target->position.z()),
                         1e-6));
    CHECK_THAT(
        aim_command->pitch,
        WithinAbs(std::atan2(-transformed_target->position.y(),
                             std::hypot(transformed_target->position.x(), transformed_target->position.z())),
                      1e-6));
    CHECK_THAT(aim_command->distance_m, WithinAbs(transformed_target->position.norm(), 1e-6));
    CHECK_THAT(aim_command->flight_time_s,
               WithinAbs(transformed_target->position.norm() / tracker_config.bullet_speed, 1e-6));
    CHECK(pipeline.state().frame_index == 1);
    REQUIRE(pipeline.state().last_aim_command.has_value());
    CHECK(pipeline.state().last_aim_command->fire_enable);
}
