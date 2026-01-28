/**
 ************************************************************************
 *
 * @file solve.hpp
 * @author Neomelt
 * @brief Solve functions for armor tracking
 *
 * ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 * ************************************************************************
 */

#pragma once

#include "auto_aim/type.hpp"
#include <opencv2/core.hpp>
#include <string>

namespace armor {

class PnpSolver {
public:
    explicit PnpSolver(const std::string& yaml_path);
    virtual ~PnpSolver() = default;

    /**
     * @brief 求解装甲板 3D 位姿
     * @param obj 检测器输出的 ArmorObject
     * @param result 输出的 Armor 结果（包含 pos 和 ori）
     * @return 是否解算成功
     */
    bool solve(const ArmorObject& obj, Armor& result);

private:
    cv::Mat camera_matrix_;      // 相机内参
    cv::Mat dist_coeffs_;       // 畸变系数

    // 预定义的 3D 模型点 (单位: m)
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

} // namespace armor