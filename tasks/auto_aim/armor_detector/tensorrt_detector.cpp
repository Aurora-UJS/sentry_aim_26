#include "auto_aim/armor_detector/tensorrt_detector.hpp"

#include "utils/logger/logger.hpp"

namespace armor {

bool TensorRTDetector::init(const std::string& model_path, const std::string& device) {
    (void)model_path;
    device_ = device;
    initialized_ = true;
    utils::logger()->warn(
        "[TensorRTDetector] 当前为阶段2骨架实现，尚未接入真实 TensorRT engine；请后续补 engine "
        "加载与推理。");
    return true;
}

std::vector<ArmorObject> TensorRTDetector::detect(const cv::Mat& image) {
    if (!initialized_) {
        utils::logger()->error("[TensorRTDetector] 尚未初始化");
        return {};
    }
    if (params_.enable_debug) {
        debug_image_ = image.clone();
    }
    return {};
}

void TensorRTDetector::setParams(const DetectorParams& params) {
    params_ = params;
    if (!params_.model_path.empty() && !initialized_) {
        init(params_.model_path, device_);
    }
}

}  // namespace armor
