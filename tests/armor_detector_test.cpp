/**
 ************************************************************************
 *
 * @file armor_detector_test.cpp
 * @author Xlqmu
 * @brief 装甲板检测器测试程序
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#include "auto_aim/armor_detector/detector_factory.hpp"
#include "utils/logger/logger.hpp"
#include "utils/plotjuggler_udp.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
    // 从配置文件创建检测器
    auto detector = armor::DetectorFactory::createFromConfig("config/detector_config.yaml");
    
    if (!detector) {
        utils::logger()->error("[Test] 检测器创建失败");
        return -1;
    }
    
    // 初始化 PlotJuggler
    utils::PJStreamer pj;

    utils::logger()->info("[Test] 装甲板检测器测试开始");

    // 测试视频路径
    std::string video_path = "video/blue/v3.avi";
    if (argc > 1) {
        video_path = argv[1];
    }

    // 读取测试视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        utils::logger()->error("[Test] 无法读取视频: {}", video_path);
        return -1;
    }

    utils::logger()->info("[Test] 视频已打开: {}", video_path);
    utils::logger()->info("[Test] 按 'q' 退出");

    cv::Mat image;
    while (true) {
        cap >> image;
        if (image.empty()) {
            break;
        }

        // 检测（包含预处理+推理+后处理）
        auto start_total = std::chrono::high_resolution_clock::now();
        auto armors = detector->detect(image);
        auto end_total = std::chrono::high_resolution_clock::now();
        
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total)
                .count();
        // 将数据发送到 PlotJuggler
        pj.send_map({
            {"detect_duration_ms", static_cast<double>(duration)},
            {"armor_count", static_cast<double>(armors.size())},
            {"fps", 1000.0 / static_cast<double>(duration + 1)}
        });

        // 显示结果
        cv::Mat debug_img = detector->getDebugImage();
        if (debug_img.empty()) {
            debug_img = image.clone();
        }

        std::string info = "FPS: " + std::to_string(1000.0 / static_cast<double>(duration + 1)) +
                           " Armors: " + std::to_string(armors.size());
        cv::putText(debug_img, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);

        cv::imshow("Armor Detector", debug_img);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) {
            break;
        }
    }

    utils::logger()->info("[Test] 测试完成");
    return 0;
}
