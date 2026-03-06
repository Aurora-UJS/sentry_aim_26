#include "auto_aim/armor_detector/ncnn_detector.hpp"

#include "utils/logger/logger.hpp"

namespace armor {

bool NCNNDetector::init(const std::string& model_path, const std::string& device) {
    (void)model_path;
    device_ = device;
    initialized_ = true;
    utils::logger()->warn(
        "[NCNNDetector] 当前为阶段2骨架实现，尚未接入真实 NCNN 推理图；请后续补模型加载与推理。");
    return true;
}

std::vector<ArmorObject> NCNNDetector::detect(const cv::Mat& image) {
    if (!initialized_) {
        utils::logger()->error("[NCNNDetector] 尚未初始化");
        return {};
    }
    if (params_.enable_debug) {
        debug_image_ = image.clone();
    }
    return {};
}

void NCNNDetector::setParams(const DetectorParams& params) {
    params_ = params;
    if (!params_.model_path.empty() && !initialized_) {
        init(params_.model_path, device_);
    }
}

}  // namespace armor
