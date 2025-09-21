#include <iostream>
#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

auto processVideo(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string> {
    // 更新全局检测配置
    g_detection_config.updateFromToml(config);
    
    ArmorDetector detector;
    auto init_result = detector.init(config.model.model_path);
    if (!init_result.has_value()) {
        return std::unexpected("Failed to initialize detector: " + init_result.error());
    }

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        return std::unexpected("Failed to open video: " + input_path);
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "视频信息:" << std::endl;
    std::cout << "  总帧数: " << total_frames << std::endl;
    std::cout << "  帧率: " << fps << " FPS" << std::endl;

    // 设置输出视频
    cv::VideoWriter writer;
    if (config.output.save_video && !config.output.output_path.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::Size frame_size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                           static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        writer.open(config.output.output_path, fourcc, fps, frame_size);
        
        if (!writer.isOpened()) {
            std::cerr << "Warning: Failed to open output video writer" << std::endl;
        } else {
            std::cout << "输出视频将保存至: " << config.output.output_path << std::endl;
        }
    }

    cv::Mat frame;
    int frame_count = 0;
    int detection_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;

    while (cap.read(frame)) {
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();

        // 检测
        auto result = detector.detect(frame);
        if (result.has_value()) {
            auto objects = result.value();
            if (!objects.empty()) {
                detection_count++;
            }
            
            // 绘制检测结果
            for (const auto& obj : objects) {
                cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2);
                
                std::string label = "cls:" + std::to_string(obj.cls) + 
                                   " color:" + std::to_string(obj.color) + 
                                   " conf:" + std::to_string(obj.prob).substr(0, 4);
                
                cv::putText(frame, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 0), 1);
                
                if (config.display.show_detection_polygons && !obj.pts.empty()) {
                    std::vector<cv::Point> int_pts;
                    for (const auto& pt : obj.pts) {
                        int_pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                    }
                    cv::polylines(frame, int_pts, true, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // 显示FPS
        if (config.display.show_fps) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
            if (duration.count() > 0) {
                float current_fps = 1000.0f / duration.count();
                cv::putText(frame, "FPS: " + std::to_string(current_fps).substr(0, 4),
                            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }
        }

        // 显示进度信息
        if (config.display.show_detection_info) {
            std::string progress = "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames);
            cv::putText(frame, progress, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);
        }

        last_time = current_time;

        // 实时显示
        if (config.display.enable_realtime_display) {
            cv::imshow(config.display.window_title, frame);
            char key = cv::waitKey(1);
            if (key == 27) break; // ESC键退出
        }

        // 保存到视频文件
        if (writer.isOpened()) {
            writer.write(frame);
        }

        // 每10帧输出一次进度
        if (frame_count % 10 == 0 && config.debug.print_detection_stats) {
            std::cout << "处理进度: " << (frame_count * 100 / total_frames) << "%" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (config.debug.print_detection_stats) {
        std::cout << "\n=== 处理完成 ===" << std::endl;
        std::cout << "总帧数: " << frame_count << std::endl;
        std::cout << "有检测结果的帧数: " << detection_count << std::endl;
        std::cout << "检测率: " << (frame_count > 0 ? (detection_count * 100.0 / frame_count) : 0) << "%" << std::endl;
        std::cout << "总耗时: " << total_duration.count() << "ms" << std::endl;
        std::cout << "平均处理速度: " << (total_duration.count() > 0 ? (frame_count * 1000.0 / total_duration.count()) : 0) << " FPS" << std::endl;
    }

    cap.release();
    if (writer.isOpened()) {
        writer.release();
        std::cout << "视频输出保存至: " << config.output.output_path << std::endl;
    }
    
    if (config.display.enable_realtime_display) {
        cv::destroyAllWindows();
    }

    return {};
}
