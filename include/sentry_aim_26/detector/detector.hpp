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

// 常量定义
namespace config {
constexpr int INPUT_W = 416;
constexpr int INPUT_H = 416;
constexpr int NUM_CLASSES = 8;
constexpr int NUM_COLORS = 4;
constexpr int TOPK = 128;
constexpr float NMS_THRESH = 0.3f;
constexpr float BBOX_CONF_THRESH = 0.75f;
constexpr float MERGE_CONF_ERROR = 0.15f;
constexpr float MERGE_MIN_IOU = 0.9f;
} // namespace config

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
    ArmorDetector& operator=(const ArmorDetector&) = delete;

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
};