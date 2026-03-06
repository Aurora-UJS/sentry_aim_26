#include "auto_aim/armor_tracker/tracker.hpp"

#include "math/filter/filter_lib.hpp"
#include "utils/logger/logger.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

namespace armor {

// ============================================================================
// TrackerConfig
// ============================================================================

TrackerConfig TrackerConfig::fromYaml(const std::string& yaml_path) {
    TrackerConfig config;

    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        auto tracker_node = root["armor_tracker"];

        if (!tracker_node) {
            utils::logger()->warn("[TrackerConfig] No 'armor_tracker' node found, using defaults");
            return config;
        }

        // 滤波器类型
        if (tracker_node["filter_type"]) {
            config.filter_type = tracker_node["filter_type"].as<std::string>();
        }

        // 匹配参数
        if (tracker_node["max_match_distance"]) {
            config.max_match_distance = tracker_node["max_match_distance"].as<double>();
        }
        if (tracker_node["max_match_yaw_diff"]) {
            config.max_match_yaw_diff = tracker_node["max_match_yaw_diff"].as<double>();
        }
        if (tracker_node["lost_time_threshold"]) {
            config.lost_time_threshold = tracker_node["lost_time_threshold"].as<int>();
        }
        if (tracker_node["tracking_threshold"]) {
            config.tracking_threshold = tracker_node["tracking_threshold"].as<int>();
        }

        // 过程噪声
        if (tracker_node["position_noise"]) {
            config.position_noise = tracker_node["position_noise"].as<double>();
        }
        if (tracker_node["velocity_noise"]) {
            config.velocity_noise = tracker_node["velocity_noise"].as<double>();
        }
        if (tracker_node["yaw_noise"]) {
            config.yaw_noise = tracker_node["yaw_noise"].as<double>();
        }
        if (tracker_node["yaw_velocity_noise"]) {
            config.yaw_velocity_noise = tracker_node["yaw_velocity_noise"].as<double>();
        }

        // 测量噪声
        if (tracker_node["measurement_position_noise"]) {
            config.measurement_position_noise =
                tracker_node["measurement_position_noise"].as<double>();
        }
        if (tracker_node["measurement_yaw_noise"]) {
            config.measurement_yaw_noise = tracker_node["measurement_yaw_noise"].as<double>();
        }

        // 子弹速度
        if (tracker_node["bullet_speed"]) {
            config.bullet_speed = tracker_node["bullet_speed"].as<double>();
        }

        utils::logger()->info("[TrackerConfig] Loaded from {}: filter={}, max_dist={:.2f}m",
                              yaml_path, config.filter_type, config.max_match_distance);

    } catch (const YAML::Exception& e) {
        utils::logger()->error("[TrackerConfig] YAML parse error: {}", e.what());
    }

    return config;
}

// ============================================================================
// FilterInterface - 滤波器抽象基类
// ============================================================================

class ArmorTracker::FilterInterface {
public:
    using State = Eigen::Matrix<double, 8, 1>;  // [x, y, z, vx, vy, vz, yaw, vyaw]

    virtual ~FilterInterface() = default;
    virtual void predict(double dt) = 0;
    virtual void update(const Eigen::Vector4d& measurement) = 0;  // [x, y, z, yaw]
    virtual State getState() const = 0;
    virtual void setState(const State& state) = 0;
};

// ============================================================================
// EKF 实现
// ============================================================================

class EKFFilter : public ArmorTracker::FilterInterface {
public:
    static constexpr int N_X = 8;  // 状态维度
    static constexpr int N_Z = 4;  // 测量维度

    // 过程模型 functor: 接受 (Jet[N_X] input, Jet[N_X] output)
    struct ProcessModel {
        double dt;  // 时间间隔

        template <typename T>
        void operator()(const T* x_in, T* x_out) const {
            // 匀速模型: [x, y, z, vx, vy, vz, yaw, vyaw]
            x_out[0] = x_in[0] + x_in[3] * T(dt);  // x = x + vx * dt
            x_out[1] = x_in[1] + x_in[4] * T(dt);  // y = y + vy * dt
            x_out[2] = x_in[2] + x_in[5] * T(dt);  // z = z + vz * dt
            x_out[3] = x_in[3];                    // vx
            x_out[4] = x_in[4];                    // vy
            x_out[5] = x_in[5];                    // vz
            x_out[6] = x_in[6] + x_in[7] * T(dt);  // yaw = yaw + vyaw * dt
            x_out[7] = x_in[7];                    // vyaw
        }
    };

    // 测量模型 functor: 接受 (Jet[N_X] input, Jet[N_Z] output)
    struct MeasurementModel {
        template <typename T>
        void operator()(const T* x, T* z) const {
            // 测量 = [x, y, z, yaw]
            z[0] = x[0];
            z[1] = x[1];
            z[2] = x[2];
            z[3] = x[6];
        }
    };

    using EKF = kalman_lib::ExtendedKalmanFilter<N_X, N_Z, ProcessModel, MeasurementModel>;

    EKFFilter(const TrackerConfig& config, const State& initial_state)
        : config_(config), process_model_{0.0} {
        MeasurementModel measurement_model;

        // 过程噪声矩阵 Q
        auto update_Q = [this]() -> Eigen::Matrix<double, N_X, N_X> {
            Eigen::Matrix<double, N_X, N_X> Q = Eigen::Matrix<double, N_X, N_X>::Zero();
            Q.diagonal() << config_.position_noise, config_.position_noise, config_.position_noise,
                config_.velocity_noise, config_.velocity_noise, config_.velocity_noise,
                config_.yaw_noise, config_.yaw_velocity_noise;
            return Q;
        };

        // 测量噪声矩阵 R
        auto update_R = [this](const Eigen::Vector4d&) -> Eigen::Matrix<double, N_Z, N_Z> {
            Eigen::Matrix<double, N_Z, N_Z> R = Eigen::Matrix<double, N_Z, N_Z>::Zero();
            R.diagonal() << config_.measurement_position_noise, config_.measurement_position_noise,
                config_.measurement_position_noise, config_.measurement_yaw_noise;
            return R;
        };

        // 初始协方差
        Eigen::Matrix<double, N_X, N_X> P0 = Eigen::Matrix<double, N_X, N_X>::Identity() * 1.0;

        ekf_ = std::make_unique<EKF>(process_model_, measurement_model, update_Q, update_R, P0);
        ekf_->setState(initial_state);

        // 设置角度残差处理（yaw角需要归一化到[-pi, pi]）
        ekf_->setResidualFunc(
            [](const Eigen::Vector4d& z_pred, const Eigen::Vector4d& z_meas) -> Eigen::Vector4d {
                Eigen::Vector4d residual = z_meas - z_pred;
                // 归一化yaw角（索引3）
                while (residual(3) > M_PI)
                    residual(3) -= 2.0 * M_PI;
                while (residual(3) < -M_PI)
                    residual(3) += 2.0 * M_PI;
                return residual;
            });
    }

    void predict(double dt) override {
        // 更新过程模型的dt
        process_model_.dt = dt;
        ekf_->predict();

        // 归一化yaw角
        State x = ekf_->getState();
        while (x(6) > M_PI)
            x(6) -= 2.0 * M_PI;
        while (x(6) < -M_PI)
            x(6) += 2.0 * M_PI;
        ekf_->setStatePost(x);
    }

    void update(const Eigen::Vector4d& measurement) override { ekf_->update(measurement); }

    State getState() const override { return ekf_->getState(); }

    void setState(const State& state) override { ekf_->setState(state); }

private:
    TrackerConfig config_;
    ProcessModel process_model_;
    std::unique_ptr<EKF> ekf_;
};

// ============================================================================
// ArmorTracker
// ============================================================================

ArmorTracker::ArmorTracker(const TrackerConfig& config, const Armor& initial_armor)
    : config_(config),
      armor_number_(initial_armor.number),
      last_update_time_(initial_armor.timestamp) {
    initFilter(initial_armor);
    tracking_count_ = 1;
    lost_count_ = 0;

    utils::logger()->debug("[ArmorTracker] Created for armor {}",
                           armorNumberToString(armor_number_));
}

ArmorTracker::~ArmorTracker() = default;

void ArmorTracker::initFilter(const Armor& initial_armor) {
    // 从装甲板姿态提取yaw角
    Eigen::Vector3d euler_angles = initial_armor.ori.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler_angles(0);  // ZYX欧拉角的Z分量

    // 初始化状态：位置已知，速度和角速度为0
    FilterInterface::State initial_state;
    initial_state << initial_armor.pos.x(), initial_armor.pos.y(), initial_armor.pos.z(), 0.0, 0.0,
        0.0,       // 速度初始化为0
        yaw, 0.0;  // yaw和角速度

    // 根据配置创建滤波器
    if (config_.filter_type == "ekf") {
        filter_ = std::make_unique<EKFFilter>(config_, initial_state);
        utils::logger()->debug("[ArmorTracker] Using EKF filter");
    } else {
        // 其他滤波器类型的实现可以后续添加
        utils::logger()->warn("[ArmorTracker] Unsupported filter type '{}', using EKF",
                              config_.filter_type);
        filter_ = std::make_unique<EKFFilter>(config_, initial_state);
    }
}

void ArmorTracker::predict(double dt) {
    if (filter_) {
        filter_->predict(dt);
    }
}

bool ArmorTracker::update(const Armor& armor) {
    // 提取测量值
    Eigen::Vector3d euler_angles = armor.ori.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler_angles(0);

    Eigen::Vector4d measurement;
    measurement << armor.pos.x(), armor.pos.y(), armor.pos.z(), yaw;

    // 更新滤波器
    if (filter_) {
        filter_->update(measurement);
    }

    // 更新跟踪状态
    tracking_count_++;
    lost_count_ = 0;
    last_update_time_ = armor.timestamp;

    return true;
}

Armor ArmorTracker::getPredictedArmor(double dt) const {
    Armor predicted;

    if (!filter_) {
        return predicted;
    }

    FilterInterface::State state = filter_->getState();

    // 预测未来dt时刻的状态
    if (dt > 0) {
        state(0) += state(3) * dt;  // x
        state(1) += state(4) * dt;  // y
        state(2) += state(5) * dt;  // z
        state(6) += state(7) * dt;  // yaw

        // 归一化yaw
        while (state(6) > M_PI)
            state(6) -= 2.0 * M_PI;
        while (state(6) < -M_PI)
            state(6) += 2.0 * M_PI;
    }

    // 填充装甲板信息
    predicted.number = armor_number_;
    predicted.pos << state(0), state(1), state(2);

    // 从yaw角构造四元数（简化：只考虑yaw旋转）
    predicted.ori = Eigen::AngleAxisd(state(6), Eigen::Vector3d::UnitZ());

    predicted.timestamp = last_update_time_;
    predicted.is_ok = isTracking();

    return predicted;
}

double ArmorTracker::computeMatchDistance(const Armor& armor) const {
    if (!filter_) {
        return std::numeric_limits<double>::max();
    }

    FilterInterface::State state = filter_->getState();

    // 位置距离
    Eigen::Vector3d pos_diff = armor.pos - Eigen::Vector3d(state(0), state(1), state(2));
    double pos_dist = pos_diff.norm();

    // yaw角差异
    Eigen::Vector3d euler_angles = armor.ori.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler_angles(0);
    double yaw_diff = std::abs(yaw - state(6));
    while (yaw_diff > M_PI)
        yaw_diff = 2.0 * M_PI - yaw_diff;

    // 综合距离（加权）
    return pos_dist + yaw_diff * 0.5;  // yaw权重可配置
}

// ============================================================================
// TrackerManager
// ============================================================================

TrackerManager::TrackerManager(const TrackerConfig& config) : config_(config) {
    utils::logger()->info("[TrackerManager] Initialized with filter_type={}", config_.filter_type);
}

void TrackerManager::update(const std::vector<Armor>& armors,
                            std::chrono::steady_clock::time_point timestamp) {
    // 计算时间间隔
    double dt = 0.0;
    if (last_update_time_.time_since_epoch().count() > 0) {
        dt = std::chrono::duration<double>(timestamp - last_update_time_).count();
    }
    last_update_time_ = timestamp;

    // 预测所有跟踪器
    for (auto& tracker : trackers_) {
        tracker->predict(dt);
    }

    // 数据关联
    std::vector<int> matched_tracker_ids;
    std::vector<int> matched_armor_ids;
    matchArmorsToTrackers(armors, matched_tracker_ids, matched_armor_ids);

    // 更新匹配的跟踪器
    for (size_t i = 0; i < matched_tracker_ids.size(); ++i) {
        int tracker_id = matched_tracker_ids[i];
        int armor_id = matched_armor_ids[i];

        if (tracker_id >= 0 && armor_id >= 0) {
            trackers_[tracker_id]->update(armors[armor_id]);
            trackers_[tracker_id]->resetLostCount();
        }
    }

    // 标记未匹配的跟踪器为丢失
    std::vector<bool> tracker_matched(trackers_.size(), false);
    for (int id : matched_tracker_ids) {
        if (id >= 0)
            tracker_matched[id] = true;
    }

    for (size_t i = 0; i < trackers_.size(); ++i) {
        if (!tracker_matched[i]) {
            trackers_[i]->incrementLostCount();
        }
    }

    // 创建新的跟踪器（未匹配的装甲板）
    std::vector<bool> armor_matched(armors.size(), false);
    for (int id : matched_armor_ids) {
        if (id >= 0)
            armor_matched[id] = true;
    }

    for (size_t i = 0; i < armors.size(); ++i) {
        if (!armor_matched[i] && armors[i].is_ok) {
            trackers_.push_back(std::make_unique<ArmorTracker>(config_, armors[i]));
            utils::logger()->debug("[TrackerManager] Created new tracker for armor {}",
                                   armorNumberToString(armors[i].number));
        }
    }

    // 清理丢失的跟踪器
    removeLostedTrackers();
}

void TrackerManager::matchArmorsToTrackers(const std::vector<Armor>& armors,
                                           std::vector<int>& matched_tracker_ids,
                                           std::vector<int>& matched_armor_ids) {
    matched_tracker_ids.clear();
    matched_armor_ids.clear();

    if (trackers_.empty() || armors.empty()) {
        return;
    }

    // 计算距离矩阵
    std::vector<std::vector<double>> distance_matrix(trackers_.size(),
                                                     std::vector<double>(armors.size()));

    for (size_t i = 0; i < trackers_.size(); ++i) {
        for (size_t j = 0; j < armors.size(); ++j) {
            // 检查装甲板号是否匹配
            if (isSameTarget(trackers_[i]->getArmorNumber(), armors[j].number)) {
                distance_matrix[i][j] = trackers_[i]->computeMatchDistance(armors[j]);
            } else {
                distance_matrix[i][j] = std::numeric_limits<double>::max();
            }
        }
    }

    // 简单的贪心匹配算法
    std::vector<bool> tracker_used(trackers_.size(), false);
    std::vector<bool> armor_used(armors.size(), false);

    for (size_t iter = 0; iter < std::min(trackers_.size(), armors.size()); ++iter) {
        // 找到最小距离
        double min_dist = std::numeric_limits<double>::max();
        int best_tracker = -1;
        int best_armor = -1;

        for (size_t i = 0; i < trackers_.size(); ++i) {
            if (tracker_used[i])
                continue;
            for (size_t j = 0; j < armors.size(); ++j) {
                if (armor_used[j])
                    continue;
                if (distance_matrix[i][j] < min_dist) {
                    min_dist = distance_matrix[i][j];
                    best_tracker = static_cast<int>(i);
                    best_armor = static_cast<int>(j);
                }
            }
        }

        // 检查距离阈值
        if (best_tracker >= 0 && best_armor >= 0 && min_dist < config_.max_match_distance) {
            matched_tracker_ids.push_back(best_tracker);
            matched_armor_ids.push_back(best_armor);
            tracker_used[best_tracker] = true;
            armor_used[best_armor] = true;
        } else {
            break;  // 没有更多有效匹配
        }
    }
}

void TrackerManager::removeLostedTrackers() {
    trackers_.erase(std::remove_if(trackers_.begin(), trackers_.end(),
                                   [](const std::unique_ptr<ArmorTracker>& tracker) {
                                       return tracker->isLost();
                                   }),
                    trackers_.end());
}

std::optional<Armor> TrackerManager::getBestTarget() const {
    // 找到距离最近的稳定跟踪
    std::optional<Armor> best_target;
    double min_distance = std::numeric_limits<double>::max();

    for (const auto& tracker : trackers_) {
        if (!tracker->isTracking())
            continue;

        // 计算子弹飞行时间（简化：假设直线飞行）
        Armor predicted = tracker->getPredictedArmor(0.0);
        double distance = predicted.pos.norm();
        double flight_time = distance / config_.bullet_speed;

        // 预测子弹到达时的位置
        predicted = tracker->getPredictedArmor(flight_time);

        if (distance < min_distance) {
            min_distance = distance;
            best_target = predicted;
        }
    }

    return best_target;
}

}  // namespace armor
