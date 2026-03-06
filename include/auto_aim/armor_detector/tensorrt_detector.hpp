/**
 ************************************************************************
 *
 * @file tensorrt_detector.hpp
 * @brief TensorRT 装甲板检测器（阶段2骨架）
 *
 * ************************************************************************
 */

#pragma once

#include "detector.hpp"

#include <opencv2/opencv.hpp>

namespace armor {

class TensorRTDetector : public Detector {
public:
    TensorRTDetector() = default;
    ~TensorRTDetector() override = default;

    bool init(const std::string& model_path, const std::string& device = "GPU");

    std::vector<ArmorObject> detect(const cv::Mat& image) override;
    void setParams(const DetectorParams& params) override;
    cv::Mat getDebugImage() const override { return debug_image_; }

private:
    DetectorParams params_;
    cv::Mat debug_image_;
    bool initialized_ = false;
    std::string device_ = "GPU";
};

}  // namespace armor
