/**
 ************************************************************************
 *
 * @file ncnn_detector.hpp
 * @brief NCNN 装甲板检测器（阶段2骨架）
 *
 * ************************************************************************
 */

#pragma once

#include "detector.hpp"

#include <opencv2/opencv.hpp>

namespace armor {

class NCNNDetector : public Detector {
public:
    NCNNDetector() = default;
    ~NCNNDetector() override = default;

    bool init(const std::string& model_path, const std::string& device = "CPU");

    std::vector<ArmorObject> detect(const cv::Mat& image) override;
    void setParams(const DetectorParams& params) override;
    cv::Mat getDebugImage() const override { return debug_image_; }

private:
    DetectorParams params_;
    cv::Mat debug_image_;
    bool initialized_ = false;
    std::string device_ = "CPU";
};

}  // namespace armor
