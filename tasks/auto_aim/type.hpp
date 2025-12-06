#pragma once

#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

// 定义装甲板的物理尺寸（单位：米）
constexpr double SMALL_ARMOR_WIDTH = 135.0 / 1000.0;
constexpr double SMALL_ARMOR_HEIGHT = 55.0 / 1000.0;
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 55.0 / 1000.0;

namespace armor {

// -----------------------------------------------------------------------------
// 枚举定义
// -----------------------------------------------------------------------------

enum class ArmorColor { BLUE = 0, RED, NONE, PURPLE };

inline int formArmorColor(ArmorColor color) {
    switch (color) {
        case ArmorColor::RED:
            return 0;
        case ArmorColor::BLUE:
            return 1;
        case ArmorColor::NONE:
            return 2;
        case ArmorColor::PURPLE:
            return 3;
    }
    return 2;
}

enum class ArmorNumber { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };

inline std::ostream& operator<<(std::ostream& os, ArmorNumber number) {
    switch (number) {
        case ArmorNumber::SENTRY:
            return os << "SENTRY";
        case ArmorNumber::NO1:
            return os << "NO1";
        case ArmorNumber::NO2:
            return os << "NO2";
        case ArmorNumber::NO3:
            return os << "NO3";
        case ArmorNumber::NO4:
            return os << "NO4";
        case ArmorNumber::NO5:
            return os << "NO5";
        case ArmorNumber::OUTPOST:
            return os << "OUTPOST";
        case ArmorNumber::BASE:
            return os << "BASE";
        case ArmorNumber::UNKNOWN:
            return os << "UNKNOWN";
        default:
            return os << "Invalid(" << static_cast<int>(number) << ")";
    }
}

inline int formArmorNumber(ArmorNumber number) {
    switch (number) {
        case ArmorNumber::SENTRY:
            return 0;
        case ArmorNumber::NO1:
            return 1;
        case ArmorNumber::NO2:
            return 2;
        case ArmorNumber::NO3:
            return 3;
        case ArmorNumber::NO4:
            return 4;
        case ArmorNumber::NO5:
            return 5;
        case ArmorNumber::OUTPOST:
            return 6;
        case ArmorNumber::BASE:
            return 7;
        case ArmorNumber::UNKNOWN:
            return 8;
    }
    return 8;
}

inline std::string armorNumberToString(ArmorNumber num) {
    switch (num) {
        case ArmorNumber::SENTRY:
            return "SENTRY";
        case ArmorNumber::BASE:
            return "BASE";
        case ArmorNumber::OUTPOST:
            return "OUTPOST";
        case ArmorNumber::NO1:
            return "NO1";
        case ArmorNumber::NO2:
            return "NO2";
        case ArmorNumber::NO3:
            return "NO3";
        case ArmorNumber::NO4:
            return "NO4";
        case ArmorNumber::NO5:
            return "NO5";
        default:
            return "UNKNOWN";
    }
}

inline int retypetotracker(ArmorNumber a) {
    static std::unordered_map<std::string, int> armor_map;
    static bool loaded = false;

    if (!loaded) {
        try {
            // 确保 AUTO_AIM_CONFIG 宏已定义，或者使用默认路径
#ifdef AUTO_AIM_CONFIG
            YAML::Node config = YAML::LoadFile(AUTO_AIM_CONFIG)["armor_map"];
            for (auto it = config.begin(); it != config.end(); ++it) {
                armor_map[it->first.as<std::string>()] = it->second.as<int>();
            }
#endif
            loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "[retypetotracker] Failed to load armor_map: " << e.what() << std::endl;
        }
    }

    std::string key = armorNumberToString(a);
    auto it = armor_map.find(key);
    if (it != armor_map.end())
        return it->second;

    // 默认映射
    switch (a) {
        case ArmorNumber::SENTRY:
            return 4;
        case ArmorNumber::NO1:
            return 2;
        case ArmorNumber::NO2:
            return 3;
        case ArmorNumber::NO3:
            return 4;
        case ArmorNumber::NO4:
            return 5;
        case ArmorNumber::NO5:
            return 9;  // 假设
        case ArmorNumber::OUTPOST:
            return 6;
        case ArmorNumber::BASE:
            return 7;
        default:
            return -1;
    }
}

inline bool isSameTarget(ArmorNumber a, ArmorNumber b) {
    return retypetotracker(a) == retypetotracker(b);
}

enum class ArmorType { SMALL, LARGE, INVALID };

inline std::string armorTypeToString(const ArmorType& type) {
    switch (type) {
        case ArmorType::SMALL:
            return "small";
        case ArmorType::LARGE:
            return "large";
        default:
            return "invalid";
    }
}

// -----------------------------------------------------------------------------
// 核心结构体
// -----------------------------------------------------------------------------

/**
 * @brief 深度学习模型检测到的装甲板对象
 */
struct ArmorObject {
    ArmorColor color = ArmorColor::NONE;
    ArmorNumber number = ArmorNumber::UNKNOWN;
    ArmorType type = ArmorType::SMALL;

    float prob = 0.f;        // 整体置信度
    float class_prob = 0.f;  // 类别置信度

    // 关键点：建议存储顺序为 [左下, 左上, 右上, 右下] 以适配 PnP
    // 如果模型输出是 [左上, 左下, 右下, 右上]，请在赋值时调整顺序
    std::vector<cv::Point2f> pts;

    cv::Rect box;        // 包围盒
    cv::Point2f center;  // 图像中心点

    ArmorObject() = default;

    /**
     * @brief 获取用于 PnP 解算的关键点
     * @return 顺序必须符合 buildObjectPoints 定义的 3D 点顺序
     *         通常为: 左下(0), 左上(1), 右上(2), 右下(3)
     */
    std::vector<cv::Point2f> landmarks() const { return pts; }

    /**
     * @brief 构建 3D 物体坐标点 (用于 solvePnP)
     * @param w 装甲板宽
     * @param h 装甲板高
     * @return 3D 点集，顺序需与 landmarks() 返回的 2D 点一致
     */
    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double& w,
                                                           const double& h) noexcept {
        // 顺序: 左下, 左上, 右上, 右下 (物体坐标系: 中心为原点)
        // 注意：这里的坐标系定义需要与你的 PnP 解算器一致
        // 假设: x右, y下, z前 (OpenCV 默认) -> 物体平面 z=0
        // 或者: x前, y左, z上 (ROS 默认)

        // 这里沿用之前的逻辑：
        // 0: 左下 (-w/2, h/2)
        // 1: 左上 (-w/2, -h/2)
        // 2: 右上 (w/2, -h/2)
        // 3: 右下 (w/2, h/2)
        // 注意：具体正负号取决于你的坐标系定义，这里仅作示例
        return {
            PointType(0, w / 2, -h / 2),  // 左下? 需确认
            PointType(0, w / 2, h / 2),   // 左上?
            PointType(0, -w / 2, h / 2),  // 右上?
            PointType(0, -w / 2, -h / 2)  // 右下?
        };
    }
};

constexpr const char* K_ARMOR_NAMES[] = {"sentry", "1", "2", "3", "4", "5", "outpost", "base"};

/**
 * @brief 解算后的装甲板信息 (3D)
 */
struct Armor {
    ArmorNumber number;
    std::string type;
    Eigen::Vector3d pos;
    Eigen::Quaterniond ori;
    Eigen::Vector3d target_pos;
    Eigen::Quaterniond target_ori;
    float distance_to_image_center;
    float yaw;
    std::chrono::steady_clock::time_point timestamp;
    bool is_ok = false;
    bool is_none_purple = false;
    int id = -1;

    // 用于调试的反投影函数
    std::vector<cv::Point2f> toPtsDebug(const cv::Mat& camera_intrinsic,
                                        const cv::Mat& camera_distortion) {
        std::vector<cv::Point2f> image_points;
        const std::vector<cv::Point3f>* model_points;

        // 3D 点定义 (需与 PnP 解算时使用的模型一致)
        // 顺序: 左上前, 左下前, 右下前, 右上前...
        static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_BLOCK = {
            {0, 0.025, -0.066},       // 左上前
            {0, -0.025, -0.066},      // 左下前
            {0, -0.025, 0.066},       // 右下前
            {0, 0.025, 0.066},        // 右上前
            {0.015, 0.025, -0.066},   // 左上后
            {0.015, -0.025, -0.066},  // 左下后
            {0.015, -0.025, 0.066},   // 右下后
            {0.015, 0.025, 0.066},    // 右上后
        };

        static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_BLOCK = {
            {0, 0.025, -0.1125},     {0, -0.025, -0.1125},    {0, -0.025, 0.1125},
            {0, 0.025, 0.1125},      {0.015, 0.025, -0.1125}, {0.015, -0.025, -0.1125},
            {0.015, -0.025, 0.1125}, {0.015, 0.025, 0.1125},
        };

        if (type == "large") {
            model_points = &BIG_ARMOR_3D_POINTS_BLOCK;
        } else {
            model_points = &SMALL_ARMOR_3D_POINTS_BLOCK;
        }

        Eigen::Matrix3d tf_rot = target_ori.toRotationMatrix();
        cv::Mat rot_mat =
            (cv::Mat_<double>(3, 3) << tf_rot(0, 0), tf_rot(0, 1), tf_rot(0, 2), tf_rot(1, 0),
             tf_rot(1, 1), tf_rot(1, 2), tf_rot(2, 0), tf_rot(2, 1), tf_rot(2, 2));

        cv::Mat rvec, tvec;
        cv::Rodrigues(rot_mat, rvec);
        tvec = (cv::Mat_<double>(3, 1) << target_pos.x(), target_pos.y(), target_pos.z());

        cv::projectPoints(*model_points, rvec, tvec, camera_intrinsic, camera_distortion,
                          image_points);
        return image_points;
    }
};

struct Armors {
    std::vector<Armor> armors;
    std::chrono::steady_clock::time_point timestamp;
    std::string frame_id;
    int id;
    Eigen::Vector3d v;
};

// 前哨站相关常量
static constexpr double outpost_v_yaw = 0.8 * M_PI;
static constexpr double DZ_1 = 0.1;
static constexpr double DZ_2 = -0.1;
static constexpr double DZ_3 = 0.2;
static constexpr double DZ_4 = -0.2;
static constexpr std::array<double, 4> outpostDZ = {DZ_1, DZ_2, DZ_3, DZ_4};

inline double outpost_diff_from_id(int id) {
    switch (id) {
        case 1:
            return DZ_1;
        case 2:
            return DZ_2;
        case 3:
            return DZ_3;
        case 4:
            return DZ_4;
        default:
            return 0.0;
    }
}

inline int quantize_outpost_diff(double dz) {
    static constexpr double candidates[] = {DZ_1, DZ_2, DZ_3, DZ_4};
    int best_id = 1;
    double min_diff = std::abs(dz - candidates[0]);
    for (int i = 1; i < 4; ++i) {
        double diff = std::abs(dz - candidates[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_id = i + 1;
        }
    }
    return best_id;
}

}  // namespace armor