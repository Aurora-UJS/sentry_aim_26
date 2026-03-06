#include "auto_aim/armor_solver/solver.hpp"

#include "utils/logger/logger.hpp"

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

namespace armor {

PnpSolver::PnpSolver(const std::string& yaml_path) {
    // 1. 加载相机参数
    utils::logger()->info("[PnpSolver] 正在加载相机配置: {}", yaml_path);

    try {
        // 使用yaml-cpp加载YAML文件
        YAML::Node config = YAML::LoadFile(yaml_path);

        // 读取camera_matrix
        if (!config["camera_matrix"]) {
            utils::logger()->error("[PnpSolver] 配置文件中没有camera_matrix字段");
            return;
        }

        auto cm_node = config["camera_matrix"];
        int cm_rows = cm_node["rows"].as<int>();
        int cm_cols = cm_node["cols"].as<int>();
        auto cm_data = cm_node["data"].as<std::vector<double>>();

        if (cm_data.size() != cm_rows * cm_cols) {
            utils::logger()->error("[PnpSolver] camera_matrix数据大小不匹配");
            return;
        }

        camera_matrix_ = cv::Mat(cm_rows, cm_cols, CV_64F);
        for (int i = 0; i < cm_rows; i++) {
            for (int j = 0; j < cm_cols; j++) {
                camera_matrix_.at<double>(i, j) = cm_data[i * cm_cols + j];
            }
        }

        // 读取distortion_coefficients
        if (!config["distortion_coefficients"]) {
            utils::logger()->error("[PnpSolver] 配置文件中没有distortion_coefficients字段");
            return;
        }

        auto dc_node = config["distortion_coefficients"];
        int dc_rows = dc_node["rows"].as<int>();
        int dc_cols = dc_node["cols"].as<int>();
        auto dc_data = dc_node["data"].as<std::vector<double>>();

        if (dc_data.size() != dc_rows * dc_cols) {
            utils::logger()->error("[PnpSolver] distortion_coefficients数据大小不匹配");
            return;
        }

        dist_coeffs_ = cv::Mat(dc_rows, dc_cols, CV_64F);
        for (int i = 0; i < dc_rows; i++) {
            for (int j = 0; j < dc_cols; j++) {
                dist_coeffs_.at<double>(i, j) = dc_data[i * dc_cols + j];
            }
        }

    } catch (const YAML::Exception& e) {
        utils::logger()->error("[PnpSolver] YAML解析错误: {}", e.what());
        return;
    } catch (const std::exception& e) {
        utils::logger()->error("[PnpSolver] 加载配置文件失败: {}", e.what());
        return;
    }

    if (camera_matrix_.empty() || dist_coeffs_.empty()) {
        utils::logger()->error("[PnpSolver] 相机参数为空");
        return;
    }

    utils::logger()->info("[PnpSolver] 相机参数加载成功");
    utils::logger()->debug("[PnpSolver] 相机内参: fx={}, fy={}, cx={}, cy={}",
                           camera_matrix_.at<double>(0, 0), camera_matrix_.at<double>(1, 1),
                           camera_matrix_.at<double>(0, 2), camera_matrix_.at<double>(1, 2));

    // 2. 初始化 3D 模型点
    // 根据 README 和模型输出，四点顺序为: 左上(TL), 左下(BL), 右下(BR), 右上(TR)
    // OpenCV 相机坐标系: X右, Y下, Z前 (朝向目标)
    // 装甲板坐标系: 中心为原点, Z=0 平面上的矩形
    // 注意：如果图示中Z轴是“垂直装甲板向里”，需要调整符号
    auto init_points = [](double w, double h) {
        return std::vector<cv::Point3f>{
            {-static_cast<float>(w / 2), -static_cast<float>(h / 2), 0.0f},  // 0: 左上 (Top-Left)
            {-static_cast<float>(w / 2), static_cast<float>(h / 2), 0.0f},  // 1: 左下 (Bottom-Left)
            {static_cast<float>(w / 2), static_cast<float>(h / 2), 0.0f},  // 2: 右下 (Bottom-Right)
            {static_cast<float>(w / 2), -static_cast<float>(h / 2), 0.0f}  // 3: 右上 (Top-Right)
        };
    };

    small_armor_points_ = init_points(SMALL_ARMOR_WIDTH, SMALL_ARMOR_HEIGHT);
    large_armor_points_ = init_points(LARGE_ARMOR_WIDTH, LARGE_ARMOR_HEIGHT);

    utils::logger()->debug("[PnpSolver] 小装甲板尺寸: {}x{} m", SMALL_ARMOR_WIDTH,
                           SMALL_ARMOR_HEIGHT);
    utils::logger()->debug("[PnpSolver] 大装甲板尺寸: {}x{} m", LARGE_ARMOR_WIDTH,
                           LARGE_ARMOR_HEIGHT);
}

bool PnpSolver::solve(const ArmorObject& obj, Armor& result) {
    // 1. 检查关键点数量
    if (obj.pts.size() != 4) {
        utils::logger()->warn("[PnpSolver] 关键点数量不等于4: {}", obj.pts.size());
        return false;
    }

    // 2. 检查相机参数
    if (camera_matrix_.empty() || dist_coeffs_.empty()) {
        utils::logger()->error("[PnpSolver] 相机参数未初始化");
        return false;
    }

    // 3. 选择对应的 3D 模型点
    const auto& object_points =
        (obj.type == ArmorType::LARGE) ? large_armor_points_ : small_armor_points_;

    // 4. 执行 PnP 解算
    cv::Mat rvec, tvec;
    // 使用 SOLVEPNP_IPPE: 适用于平面四点，效果好、速度快、稳定性强
    bool success = cv::solvePnP(object_points, obj.pts, camera_matrix_, dist_coeffs_, rvec, tvec,
                                false, cv::SOLVEPNP_IPPE);

    if (!success) {
        utils::logger()->warn("[PnpSolver] PnP 解算失败");
        return false;
    }

    // 5. 填充位置 (tvec 是相机坐标系下的平移向量)
    result.pos.x() = tvec.at<double>(0);
    result.pos.y() = tvec.at<double>(1);
    result.pos.z() = tvec.at<double>(2);

    // 计算距离
    double distance = result.pos.norm();

    // 6. 填充姿态 (将旋转向量 rvec 转为四元数)
    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);
    Eigen::Matrix3d eigen_rot;
    cv::cv2eigen(rot_mat, eigen_rot);
    result.ori = Eigen::Quaterniond(eigen_rot);

    // 7. 补充元数据
    result.number = obj.number;
    result.type = armorTypeToString(obj.type);
    result.timestamp = std::chrono::steady_clock::now();
    result.is_ok = true;

    // 8. 记录详细信息
    utils::logger()->debug(
        "[PnpSolver] 解算成功 - 类型: {}, 位置: ({:.3f}, {:.3f}, {:.3f}) m, 距离: {:.3f} m",
        result.type, result.pos.x(), result.pos.y(), result.pos.z(), distance);

    return true;
}

}  // namespace armor