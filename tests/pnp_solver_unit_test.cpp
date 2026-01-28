/**
 ************************************************************************
 *
 * @file pnp_solver_unit_test.cpp
 * @author Neomelt
 * @brief PnP求解器单元测试（使用Catch2）
 *
 ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 ************************************************************************
 */

#include "auto_aim/armor_solver/solver.hpp"
#include "auto_aim/type.hpp"
#include "utils/logger/logger.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <opencv2/opencv.hpp>

using Catch::Matchers::WithinAbs;

// 测试用的相机参数
const cv::Mat TEST_CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 
    1807.12121, 0, 711.11997,
    0, 1806.46896, 562.49495,
    0, 0, 1);

const cv::Mat TEST_DIST_COEFFS = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

// 创建临时的相机配置文件用于测试
void createTestCameraConfig(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << TEST_CAMERA_MATRIX;
    fs << "distortion_coefficients" << TEST_DIST_COEFFS;
    fs.release();
}

// 生成合成的装甲板图像点（用于测试）
std::vector<cv::Point2f> generateSyntheticImagePoints(
    const std::vector<cv::Point3f>& object_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {
    
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    return image_points;
}

TEST_CASE("PnpSolver初始化测试", "[solver]") {
    const std::string test_config = "/home/neomelt/sentry_aim_26/config/camera_info.yaml";
    
    SECTION("使用有效的配置文件初始化") {
        REQUIRE_NOTHROW(armor::PnpSolver(test_config));
    }
    
    SECTION("使用不存在的配置文件") {
        // 应该能构造但会输出错误日志
        REQUIRE_NOTHROW(armor::PnpSolver("non_existent.yaml"));
    }
}

TEST_CASE("PnP求解 - 小装甲板", "[solver]") {
    const std::string test_config = "/home/neomelt/sentry_aim_26/config/camera_info.yaml";
    armor::PnpSolver solver(test_config);
    
    SECTION("正面小装甲板 - 距离1米") {
        // 设置真实位姿：正前方1米
        cv::Mat true_rvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);  // 无旋转
        cv::Mat true_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 1.0);  // Z方向1米
        
        // 生成小装甲板的3D点
        std::vector<cv::Point3f> object_points = {
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0}
        };
        
        // 使用实际的相机参数投影到图像
        cv::FileStorage fs(test_config, cv::FileStorage::READ);
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"]["data"] >> camera_matrix;
        camera_matrix = camera_matrix.clone().reshape(1, 3);  // 使用clone()确保连续性
        fs["distortion_coefficients"]["data"] >> dist_coeffs;
        dist_coeffs = dist_coeffs.clone().reshape(1, 1);
        fs.release();
        
        auto image_points = generateSyntheticImagePoints(
            object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs);
        
        // 构造ArmorObject
        armor::ArmorObject obj;
        obj.pts = image_points;
        obj.type = armor::ArmorType::SMALL;
        obj.number = armor::ArmorNumber::NO3;
        obj.color = armor::ArmorColor::BLUE;
        
        // 求解
        armor::Armor result;
        REQUIRE(solver.solve(obj, result));
        
        // 验证结果
        CHECK(result.is_ok);
        CHECK(result.type == "small");
        CHECK(result.number == armor::ArmorNumber::NO3);
        
        // 验证位置（允许小误差）
        CHECK_THAT(result.pos.x(), WithinAbs(0.0, 0.001));
        CHECK_THAT(result.pos.y(), WithinAbs(0.0, 0.001));
        CHECK_THAT(result.pos.z(), WithinAbs(1.0, 0.001));
        
        // 验证距离
        double distance = result.pos.norm();
        CHECK_THAT(distance, WithinAbs(1.0, 0.001));
    }
    
    SECTION("偏右小装甲板 - 距离2米") {
        // 设置真实位姿：右像0.5米，前方2米
        cv::Mat true_rvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::Mat true_tvec = (cv::Mat_<double>(3, 1) << 0.5, 0, 2.0);
            
        std::vector<cv::Point3f> object_points = {
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0}
        };
            
        cv::FileStorage fs(test_config, cv::FileStorage::READ);
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"]["data"] >> camera_matrix;
        camera_matrix = camera_matrix.clone().reshape(1, 3);
        fs["distortion_coefficients"]["data"] >> dist_coeffs;
        dist_coeffs = dist_coeffs.clone().reshape(1, 1);
        fs.release();
            
        auto image_points = generateSyntheticImagePoints(
            object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs);
        
        armor::ArmorObject obj;
        obj.pts = image_points;
        obj.type = armor::ArmorType::SMALL;
        obj.number = armor::ArmorNumber::NO4;
        
        armor::Armor result;
        REQUIRE(solver.solve(obj, result));
        
        CHECK_THAT(result.pos.x(), WithinAbs(0.5, 0.001));
        CHECK_THAT(result.pos.y(), WithinAbs(0.0, 0.001));
        CHECK_THAT(result.pos.z(), WithinAbs(2.0, 0.001));
    }
}

TEST_CASE("PnP求解 - 大装甲板", "[solver]") {
    const std::string test_config = "/home/neomelt/sentry_aim_26/config/camera_info.yaml";
    armor::PnpSolver solver(test_config);
    
    SECTION("正面大装甲板 - 距离3米") {
        cv::Mat true_rvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
        cv::Mat true_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 3.0);
        
        std::vector<cv::Point3f> object_points = {
            {-static_cast<float>(LARGE_ARMOR_WIDTH / 2), -static_cast<float>(LARGE_ARMOR_HEIGHT / 2), 0},
            {-static_cast<float>(LARGE_ARMOR_WIDTH / 2),  static_cast<float>(LARGE_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(LARGE_ARMOR_WIDTH / 2),  static_cast<float>(LARGE_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(LARGE_ARMOR_WIDTH / 2), -static_cast<float>(LARGE_ARMOR_HEIGHT / 2), 0}
        };
        
        cv::FileStorage fs(test_config, cv::FileStorage::READ);
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"]["data"] >> camera_matrix;
        camera_matrix = camera_matrix.clone().reshape(1, 3);
        fs["distortion_coefficients"]["data"] >> dist_coeffs;
        dist_coeffs = dist_coeffs.clone().reshape(1, 1);
        fs.release();
        
        auto image_points = generateSyntheticImagePoints(
            object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs);
        
        armor::ArmorObject obj;
        obj.pts = image_points;
        obj.type = armor::ArmorType::LARGE;
        obj.number = armor::ArmorNumber::SENTRY;
        
        armor::Armor result;
        REQUIRE(solver.solve(obj, result));
        
        CHECK(result.type == "large");
        CHECK_THAT(result.pos.z(), WithinAbs(3.0, 0.001));
    }
}

TEST_CASE("PnP求解 - 异常情况", "[solver]") {
    const std::string test_config = "/home/neomelt/sentry_aim_26/config/camera_info.yaml";
    armor::PnpSolver solver(test_config);
    
    SECTION("关键点数量不足") {
        armor::ArmorObject obj;
        obj.pts = {{100, 100}, {200, 100}};  // 只有2个点
        obj.type = armor::ArmorType::SMALL;
        
        armor::Armor result;
        REQUIRE_FALSE(solver.solve(obj, result));
    }
    
    SECTION("关键点数量过多") {
        armor::ArmorObject obj;
        obj.pts = {{100, 100}, {200, 100}, {200, 200}, {100, 200}, {150, 150}};  // 5个点
        obj.type = armor::ArmorType::SMALL;
        
        armor::Armor result;
        REQUIRE_FALSE(solver.solve(obj, result));
    }
}

TEST_CASE("PnP求解 - 姿态测试", "[solver]") {
    const std::string test_config = "/home/neomelt/sentry_aim_26/config/camera_info.yaml";
    armor::PnpSolver solver(test_config);
    
    SECTION("装甲板有旋转") {
        // 设置一个有旋转的位姿
        cv::Mat true_rvec = (cv::Mat_<double>(3, 1) << 0, 0.1, 0);  // 绕Y轴旋转
        cv::Mat true_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 1.5);
        
        std::vector<cv::Point3f> object_points = {
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            {-static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2),  static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0},
            { static_cast<float>(SMALL_ARMOR_WIDTH / 2), -static_cast<float>(SMALL_ARMOR_HEIGHT / 2), 0}
        };
        
        cv::FileStorage fs(test_config, cv::FileStorage::READ);
        cv::Mat camera_matrix, dist_coeffs;
        fs["camera_matrix"]["data"] >> camera_matrix;
        camera_matrix = camera_matrix.clone().reshape(1, 3);
        fs["distortion_coefficients"]["data"] >> dist_coeffs;
        dist_coeffs = dist_coeffs.clone().reshape(1, 1);
        fs.release();
        
        auto image_points = generateSyntheticImagePoints(
            object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs);
        
        armor::ArmorObject obj;
        obj.pts = image_points;
        obj.type = armor::ArmorType::SMALL;
        obj.number = armor::ArmorNumber::NO5;
        
        armor::Armor result;
        REQUIRE(solver.solve(obj, result));
        
        // 验证四元数非空（至少有一个分量不为0）
        bool quaternion_is_valid = (result.ori.w() != 0 || result.ori.x() != 0 || 
                                    result.ori.y() != 0 || result.ori.z() != 0);
        CHECK(quaternion_is_valid);
        
        // 验证四元数是单位四元数
        double quat_norm = std::sqrt(
            result.ori.w() * result.ori.w() + 
            result.ori.x() * result.ori.x() + 
            result.ori.y() * result.ori.y() + 
            result.ori.z() * result.ori.z());
        CHECK_THAT(quat_norm, WithinAbs(1.0, 0.001));
    }
}
