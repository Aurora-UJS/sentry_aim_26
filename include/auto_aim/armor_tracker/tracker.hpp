/**
 ************************************************************************
 *
 * @file tracker.hpp
 * @author Neomelt
 * @brief Armor tracking interface with configurable Kalman filter
 *
 ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 ************************************************************************
 */

#pragma once

#include "auto_aim/type.hpp"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace armor {

/**
 * @brief 跟踪器配置参数
 */
struct TrackerConfig {
    // 滤波器选择
    std::string filter_type = "ekf";  // "ekf", "ukf", "aekf", "esekf"
    
    // 运动模型参数
    double max_match_distance = 0.5;     // 最大匹配距离（米）
    double max_match_yaw_diff = 1.0;     // 最大匹配yaw角差（弧度）
    int lost_time_threshold = 5;         // 丢失帧数阈值
    int tracking_threshold = 3;          // 跟踪稳定阈值
    
    // 过程噪声（线性平移 + 旋转）
    double position_noise = 0.1;         // 位置过程噪声
    double velocity_noise = 1.0;         // 速度过程噪声
    double yaw_noise = 0.1;              // yaw角过程噪声
    double yaw_velocity_noise = 0.5;     // yaw角速度过程噪声
    
    // 测量噪声
    double measurement_position_noise = 0.05;  // 位置测量噪声
    double measurement_yaw_noise = 0.1;        // yaw角测量噪声
    
    // 预测参数
    double bullet_speed = 28.0;          // 子弹速度（m/s）
    
    // 加载配置文件
    static TrackerConfig fromYaml(const std::string& yaml_path);
};

/**
 * @brief 单目标跟踪器（单个装甲板）
 * 状态向量: [x, y, z, vx, vy, vz, yaw, vyaw]
 * - x,y,z: 3D位置
 * - vx,vy,vz: 线性速度
 * - yaw: 装甲板yaw角
 * - vyaw: yaw角速度
 */
class ArmorTracker {
public:
    ArmorTracker(const TrackerConfig& config, const Armor& initial_armor);
    ~ArmorTracker();  // 需要在cpp中定义，因为FilterInterface是不完整类型

    /**
     * @brief 预测下一时刻状态
     * @param dt 时间间隔（秒）
     */
    void predict(double dt);

    /**
     * @brief 更新状态（测量更新）
     * @param armor 测量到的装甲板
     * @return 是否更新成功
     */
    bool update(const Armor& armor);

    /**
     * @brief 获取预测的装甲板状态
     * @param dt 预测提前量（秒，考虑子弹飞行时间）
     * @return 预测的装甲板
     */
    Armor getPredictedArmor(double dt = 0.0) const;

    /**
     * @brief 计算与测量的匹配度
     * @param armor 测量到的装甲板
     * @return 匹配距离
     */
    double computeMatchDistance(const Armor& armor) const;

    /**
     * @brief 判断跟踪器是否稳定
     */
    bool isTracking() const { return tracking_count_ >= config_.tracking_threshold; }

    /**
     * @brief 判断跟踪器是否丢失
     */
    bool isLost() const { return lost_count_ >= config_.lost_time_threshold; }

    /**
     * @brief 增加丢失计数
     */
    void incrementLostCount() { lost_count_++; }

    /**
     * @brief 重置丢失计数
     */
    void resetLostCount() { lost_count_ = 0; }

    /**
     * @brief 获取跟踪的装甲板ID
     */
    ArmorNumber getArmorNumber() const { return armor_number_; }

    /**
     * @brief 获取最后更新时间
     */
    std::chrono::steady_clock::time_point getLastUpdateTime() const { return last_update_time_; }

    // 滤波器抽象接口（public以便在tracker.cpp中实现具体滤波器）
    class FilterInterface;

private:
    TrackerConfig config_;
    std::unique_ptr<FilterInterface> filter_;
    
    // 跟踪状态
    ArmorNumber armor_number_;
    int tracking_count_ = 0;
    int lost_count_ = 0;
    std::chrono::steady_clock::time_point last_update_time_;
    
    // 辅助函数
    void initFilter(const Armor& initial_armor);
    Armor stateToArmor() const;
};

/**
 * @brief 多目标跟踪管理器
 */
class TrackerManager {
public:
    explicit TrackerManager(const TrackerConfig& config);
    ~TrackerManager() = default;

    /**
     * @brief 更新跟踪器（匹配 + 更新 + 创建新跟踪器）
     * @param armors 检测到的装甲板列表
     * @param timestamp 当前时间戳
     */
    void update(const std::vector<Armor>& armors, 
                std::chrono::steady_clock::time_point timestamp);

    /**
     * @brief 获取最佳跟踪目标（距离图像中心最近的稳定跟踪）
     * @return 预测的装甲板，如果没有有效跟踪返回nullopt
     */
    std::optional<Armor> getBestTarget() const;

    /**
     * @brief 获取所有活跃的跟踪器
     */
    const std::vector<std::unique_ptr<ArmorTracker>>& getTrackers() const { return trackers_; }

private:
    TrackerConfig config_;
    std::vector<std::unique_ptr<ArmorTracker>> trackers_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    // 数据关联
    void matchArmorsToTrackers(const std::vector<Armor>& armors,
                              std::vector<int>& matched_tracker_ids,
                              std::vector<int>& matched_armor_ids);
    
    // 清理丢失的跟踪器
    void removeLostedTrackers();
};

} // namespace armor
