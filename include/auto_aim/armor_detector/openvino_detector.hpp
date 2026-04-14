/**
 ************************************************************************
 *
 * @file openvino_detector.hpp
 * @author Xlqmu
 * @brief OpenVINO 装甲板检测器
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

#include <openvino/openvino.hpp>

namespace armor {

/**
 * @brief OpenVINO 检测器实现类
 * 支持直接输出装甲板 4 点坐标和类别 ID
 * 支持 Intel CPU/GPU/NPU 加速
 */
class OpenVINODetector : public Detector {
public:
    OpenVINODetector();
    ~OpenVINODetector() override = default;

    /**
     * @brief 初始化模型
     * @param model_path ONNX 模型路径
     * @param device 推理设备 ("CPU", "GPU", "NPU", "AUTO")
     */
    bool init(const std::string& model_path, const std::string& device = "AUTO");

    std::vector<ArmorObject> detect(const cv::Mat& image) override;
    void setParams(const DetectorParams& params) override;
    cv::Mat getDebugImage() const override { return debug_image_; }

private:
    // OpenVINO
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    std::string input_name_;
    std::string output_name_;
    ov::Shape input_shape_;
    ov::Shape output_shape_;

    // 设备
    std::string device_;

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
