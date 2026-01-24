/**
 ************************************************************************
 *
 * @file onnxruntime_detector.hpp
 * @author Xlqmu
 * @brief ONNX Runtime 装甲板检测器
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#pragma once

#include "auto_aim/type.hpp"
#include "detector.hpp"
#include "utils/logger/logger.hpp"

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

namespace armor {

/**
 * @brief onnxruntime 检测器实现类
 * 支持直接输出装甲板 4 点坐标和类别 ID
 */
class Onnxruntime_Detector : public Detector {
public:
    Onnxruntime_Detector();
    ~Onnxruntime_Detector() override = default;

    /**
     * @brief 初始化模型
     * @param model_path ONNX 模型路径
     */
    bool init(const std::string& model_path);

    std::vector<ArmorObject> detect(const cv::Mat& image) override;
    void setParams(const DetectorParams& params) override;
    cv::Mat getDebugImage() const override { return debug_image_; }

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;   // 保持字符串生命周期
    std::vector<std::string> output_names_str_;  // 保持字符串生命周期

    // 参数
    DetectorParams params_;

    // 调试
    cv::Mat debug_image_;

    // 内部处理函数
    cv::Mat preProcess(const cv::Mat& image, float& scale) const;
    std::vector<ArmorObject> postProcess(const float* data, int rows, int dimensions, float scale,
                                         const cv::Mat& origin_img) const;
};

}  // namespace armor
