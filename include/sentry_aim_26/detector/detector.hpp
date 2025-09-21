// 修复 include/sentry_aim_26/detector/detector.hpp
/**
 * @file detector.hpp
 * @brief 检测器接口和YOLO实现（支持角点 + color/class 解耦）
 * @author xlqmu
 * @date 2025-07-15
 * @version 1.1
 *
 * @copyright Copyright (c) 2025 Team SentryAim
 */

#pragma once
#include <Eigen/Dense>
#include <expected>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <vector>
#include "sentry_aim_26/core/config.hpp"

// 运行时配置存储
struct DetectionConfig {
    int input_w = 640;
    int input_h = 640;
    int num_classes = 8;
    int num_colors = 4;
    int topk = 128;
    float nms_thresh = 0.3f;
    float bbox_conf_thresh = 0.3f;
    float merge_conf_error = 0.3f;
    float merge_min_iou = 0.5f;
    
    // 从TOML配置更新
    void updateFromToml(const toml::Config& config) {
        input_w = config.model.input_width;
        input_h = config.model.input_height;
        num_classes = config.detection.num_classes;
        num_colors = config.detection.num_colors;
        nms_thresh = config.model.nms_threshold;
        bbox_conf_thresh = config.model.confidence_threshold;
        topk = config.model.max_detections;
    }
};

// 全局配置实例
extern DetectionConfig g_detection_config;

// 结构体定义
struct ArmorObject {
    std::array<cv::Point2f, 4> apex;
    std::vector<cv::Point2f> pts;
    cv::Rect rect;
    int cls;
    int color;
    float prob;
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

// 函数声明
constexpr auto argmax(std::span<const float> data) -> int;
auto scaledResize(cv::Mat& img, Eigen::Matrix<float, 3, 3>& transform_matrix) -> cv::Mat;
auto generate_grids_and_stride(int target_w, int target_h,
                               std::span<const int> strides) -> std::vector<GridAndStride>;
auto generateYoloxProposals(std::span<const GridAndStride> grid_strides,
                            std::span<const float> feat_data,
                            const Eigen::Matrix<float, 3, 3>& transform_matrix,
                            float prob_threshold) -> std::vector<ArmorObject>;
constexpr auto intersection_area(const ArmorObject& a, const ArmorObject& b) -> float;
auto nms_sorted_bboxes(std::vector<ArmorObject>& objects, float nms_threshold) -> std::vector<int>;
auto decodeOutputs(std::span<const float> prob, const Eigen::Matrix<float, 3, 3>& transform_matrix,
                   int img_w, int img_h) -> std::vector<ArmorObject>;

// 装甲板检测器类
class ArmorDetector {
  public:
    ArmorDetector() = default;
    ~ArmorDetector() = default;

    // 删除拷贝构造和赋值
    ArmorDetector(const ArmorDetector&) = delete;
    ArmorDetector& operator=(ArmorDetector&) = delete;

    // 允许移动构造和赋值
    ArmorDetector(ArmorDetector&&) = default;
    ArmorDetector& operator=(ArmorDetector&&) = default;

    auto initModel(std::string_view model_path) -> std::expected<void, std::string>;
    auto detect(cv::Mat& img) -> std::expected<std::vector<ArmorObject>, std::string>;

  private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ArmorDetector"};
    Ort::MemoryInfo memory_info_{"Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    std::unique_ptr<Ort::Session> session_;
    std::optional<Ort::AllocatedStringPtr> input_name_; // 使用 optional 包装
    Ort::AllocatorWithDefaultOptions allocator_;
    bool model_initialized_ = false;
    
    // 时序稳定性相关
    std::vector<ArmorObject> previous_detections_;
    static constexpr float STABILITY_DISTANCE_THRESHOLD = 50.0f; // 像素距离阈值
    static constexpr float STABILITY_CONFIDENCE_FACTOR = 0.1f;   // 稳定性置信度加成
    
    // 时序稳定性函数
    auto applyTemporalStability(std::vector<ArmorObject>& current_detections) -> void;
};