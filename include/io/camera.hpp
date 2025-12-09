/**
 ************************************************************************
 *
 * @file camera.hpp
 * @author Aurora Vision Team
 * @brief 工业相机抽象接口
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace io {

/**
 * @brief 相机参数
 */
struct CameraParams {
    int width = 1280;
    int height = 1024;
    int fps = 100;
    int exposure = 5000;  // us
    int gain = 10;
    std::string config_path;  // 配置文件路径
};

/**
 * @brief 工业相机抽象接口
 */
class Camera {
public:
    virtual ~Camera() = default;

    /**
     * @brief 打开相机
     * @return 是否成功
     */
    virtual bool open() = 0;

    /**
     * @brief 关闭相机
     */
    virtual void close() = 0;

    /**
     * @brief 获取一帧图像
     * @param image 输出图像
     * @return 是否成功获取
     */
    virtual bool getFrame(cv::Mat& image) = 0;

    /**
     * @brief 设置参数
     * @param params 相机参数
     */
    virtual void setParams(const CameraParams& params) = 0;

    /**
     * @brief 检查相机是否打开
     */
    virtual bool isOpen() const = 0;
};

}  // namespace io
