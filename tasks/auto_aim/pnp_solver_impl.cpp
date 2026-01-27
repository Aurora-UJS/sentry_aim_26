#include "auto_aim/pnp_solver.hpp"
#include "io/camera.hpp"

#include <iostream>

namespace armor {

// 构造函数：初始化时加载相机内参和装甲板尺寸
PnpSolver::PnpSolver(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);

        // 1. 从 YAML 加载相机矩阵 (Camera Matrix)
        std::vector<double> camera_data = config["camera_matrix"]["data"].as<std::vector<double>>();
        camera_matrix_ = cv::Mat(3, 3, CV_64F, camera_data.data()).clone();

        // 2. 加载畸变系数 (Distortion Coefficients)
        std::vector<double> dist_data =
            config["distortion_coefficients"]["data"].as<std::vector<double>>();
        dist_coeffs_ = cv::Mat(1, 5, CV_64F, dist_data.data()).clone();

        // 3. 定义装甲板 3D 物理坐标 (单位: mm)
        // 坐标系习惯：装甲板中心为原点，四角顺时针或逆时针定义
        float sm_w = 135.0f;  // 小装甲板宽度
        float sm_h = 55.0f;   // 小装甲板高度
        small_armor_points_ = {{-sm_w / 2, sm_h / 2, 0},
                               {sm_w / 2, sm_h / 2, 0},
                               {sm_w / 2, -sm_h / 2, 0},
                               {-sm_w / 2, -sm_h / 2, 0}};

    } catch (const std::exception& e) {
        std::cerr << "PnP Solver 初始化失败: " << e.what() << std::endl;
    }
}

// 核心解算逻辑
bool PnpSolver::solve(const ArmorObject& obj, Armor& result) {
    // 1. 准备 2D 像素点 (来自检测器)
    std::vector<cv::Point2f> image_points;
    for (const auto& pt : obj.pts) {
        image_points.emplace_back(pt.x, pt.y);
    }

    // 2. 选择对应的 3D 模型点
    auto object_points = (obj.type == ArmorType::SMALL) ? small_armor_points_ : large_armor_points_;

    // 3. 调用 OpenCV PnP 算法
    cv::Mat rvec, tvec;  // 旋转向量和平移向量
    bool success = cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_, rvec,
                                tvec, false, cv::SOLVEPNP_IPPE);

    if (cv::solvePnP(object_points, obj.pts, camera_matrix_, dist_coeffs_, rvec, tvec)) {
        // 1. 适配 Eigen::Vector3d pos
        // 注意：tvec 出来的通常是 double，直接赋值给 Eigen 向量
        result.pos.x() = tvec.at<double>(0);
        result.pos.y() = tvec.at<double>(1);
        result.pos.z() = tvec.at<double>(2);

        // 2. 这里的 distance 成员在 Armor 中不存在，我们直接计算并打印，或者赋值给
        // distance_to_image_center（如果是业务需要） 根据你的 type.hpp，Armor 没有 distance 成员
        // 如果你只是想记录距离，可以使用 Eigen 自带的 norm() 函数
        double dist = result.pos.norm();

        // 3. 填充 Armor 结构体中的其他必要字段
        result.is_ok = true;

        return true;
    }
}
}  // namespace armor
