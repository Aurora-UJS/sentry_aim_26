/**
 ************************************************************************
 *
 * @file detector.hpp
 * @author Xlqmu
 * @brief 装甲板检测器抽象接口
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

namespace armor {

// 前向声明
struct ArmorObject;

/**
 * @brief 检测器参数配置
 */
struct DetectorParams {
    // --- 神经网络参数 ---
    std::string model_path;            // 模型路径 (.onnx)
    float conf_threshold = 0.5;        // 置信度阈值
    float nms_threshold = 0.45;        // NMS 阈值
    cv::Size input_size = {416, 416};  // 模型输入尺寸

    bool enable_debug = false;
};

/**
 * @brief 装甲板检测器抽象接口
 *
 * 实现类需要完成：
 * 1. 装甲板四点坐标输出
 * 2. 装甲板类别输出
 * 3. 装甲板颜色输出
 */
class Detector {
public:
    virtual ~Detector() = default;

    /**
     * @brief 检测图像中的装甲板
     * @param image 输入图像（BGR 格式）
     * @return 检测到的装甲板列表
     */
    virtual std::vector<ArmorObject> detect(const cv::Mat& image) = 0;

    /**
     * @brief 设置检测器参数
     * @param params 参数配置
     */
    virtual void setParams(const DetectorParams& params) = 0;

    /**
     * @brief 获取调试图像（可选实现）
     * @return 标注了检测结果的图像
     */
    // nodiscard 表示调用者应使用返回值，避免忽略
    [[nodiscard]] virtual cv::Mat getDebugImage() const { return {}; }
};

}  // namespace armor
