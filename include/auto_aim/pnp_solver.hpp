#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 引用定义的装甲板数据结构
#include "auto_aim/type.hpp"

namespace armor {

class PnpSolver {
public:
    // 构造函数：传入 yaml 路径加载相机内参
    explicit PnpSolver(const std::string& yaml_path);

    // 核心接口：输入 2D 的 ArmorObject，输出 3D 的 Armor 结果
    bool solve(const ArmorObject& obj, Armor& result);

private:
    cv::Mat camera_matrix_;    // 相机内参矩阵
    cv::Mat dist_coeffs_;      // 畸变系数矩阵

    // 存储大小装甲板的 3D 物理模型点
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

} // namespace armor
