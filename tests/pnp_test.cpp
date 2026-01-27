#include "auto_aim/pnp_solver.hpp"
#include "auto_aim/type.hpp"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int main() {
    // 1. 初始化 Solver（确保路径与你创建的 config 对应）
    armor::PnpSolver solver("/home/ubuntu/auto_aim_26/sentry_aim_26/config/camera_params.yaml");

    // 2. 构造模拟装甲板对象
    armor::ArmorObject mock_obj;
    mock_obj.type = armor::ArmorType::SMALL;

    // --- 【必须添加的部分】这是解决崩溃的关键！ ---
    // OpenCV 需要至少 4 个点（左下, 左上, 右上, 右下）
    mock_obj.pts = {
        {600.0f, 505.0f},  // 点 1
        {600.0f, 450.0f},  // 点 2
        {735.0f, 450.0f},  // 点 3
        {735.0f, 505.0f}   // 点 4
    };
    // ------------------------------------------

    armor::Armor result;
    std::cout << ">>> 准备进行 PnP 解算测试..." << std::endl;

    // 3. 执行解算
    if (solver.solve(mock_obj, result)) {
        std::cout << "--- 解算成功！ ---" << std::endl;
        // 根据你的 type.hpp，Eigen 向量访问用 x(), y(), z()
        double distance =
            std::sqrt(result.pos.x() * result.pos.x() + result.pos.y() * result.pos.y() +
                      result.pos.z() * result.pos.z());

        std::cout << "目标 3D 坐标: "
                  << "X: " << result.pos.x() << ", "
                  << "Y: " << result.pos.y() << ", "
                  << "Z: " << result.pos.z() << std::endl;
        std::cout << "目标距离 (mm): " << distance << std::endl;
    } else {
        std::cerr << "--- 解算失败：请检查点顺序或相机参数 ---" << std::endl;
    }

    return 0;
}