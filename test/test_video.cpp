// 修改 test/test_video.cpp
#include "sentry_aim_26/detector/detector.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

auto processVideo(const std::string& input_path, const std::string& output_path,
                  const std::string& model_path,
                  bool realtime_display) -> std::expected<void, std::string> {

    ArmorDetector detector;
    auto init_result = detector.initModel(model_path);
    if (!init_result) {
        return std::unexpected("初始化模型失败: " + init_result.error());
    }

    std::cout << "模型加载成功" << std::endl;

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        return std::unexpected("无法打开视频: " + input_path);
    }

    // 获取视频信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "视频信息: " << frame_width << "x" << frame_height << " @ " << fps
              << " FPS, 总帧数: " << total_frames << std::endl;

    // 创建视频写入器
    cv::VideoWriter writer;
    if (!output_path.empty()) {
        writer.open(output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps,
                    cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            return std::unexpected("无法创建输出视频: " + output_path);
        }
    }

    cv::Mat frame;
    int frame_count = 0;
    int detection_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();

        // 检测
        auto objects = detector.detect(frame);

        if (objects) {
            if (!objects->empty()) {
                detection_count++;
                std::cout << "第 " << frame_count << " 帧: 检测到 " << objects->size() << " 个目标"
                          << std::endl;
            }

            // 绘制检测结果
            for (const auto& obj : *objects) {
                cv::polylines(frame, obj.pts, true, cv::Scalar(0, 255, 0), 2);
                cv::rectangle(frame, obj.rect, cv::Scalar(255, 0, 0), 1);

                std::string label = "cls:" + std::to_string(obj.cls) +
                                    " color:" + std::to_string(obj.color) +
                                    " conf:" + std::to_string(obj.prob).substr(0, 4);

                cv::putText(frame, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 0), 1);
            }
        }

        // 显示进度和FPS
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
        if (duration.count() > 0) {
            float current_fps = 1000.0f / duration.count();
            cv::putText(frame, "FPS: " + std::to_string(current_fps).substr(0, 4),
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255),
                        2);
        }

        std::string progress =
            "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames);
        cv::putText(frame, progress, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);

        last_time = current_time;

        // 实时显示
        if (realtime_display) {
            cv::imshow("Video Detection", frame);
            char key = cv::waitKey(1);
            if (key == 27)
                break; // ESC键退出
        }

        // 写入输出视频
        if (writer.isOpened()) {
            writer.write(frame);
        }

        // 每10帧输出一次进度
        if (frame_count % 10 == 0) {
            std::cout << "处理进度: " << (frame_count * 100 / total_frames) << "%" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n=== 处理完成 ===" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "有检测结果的帧数: " << detection_count << std::endl;
    std::cout << "检测率: " << (detection_count * 100.0 / frame_count) << "%" << std::endl;
    std::cout << "总耗时: " << total_duration.count() << "ms" << std::endl;
    std::cout << "平均处理速度: " << (frame_count * 1000.0 / total_duration.count()) << " FPS"
              << std::endl;

    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    if (realtime_display) {
        cv::destroyAllWindows();
    }

    if (!output_path.empty()) {
        std::cout << "视频输出保存至: " << output_path << std::endl;
    }

    return {};
}
