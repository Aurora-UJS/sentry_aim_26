// 修改 test/test_video.cpp
#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

auto processVideo(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string> {
    // 更新全局检测配置
    g_detection_config.updateFromToml(config);

    ArmorDetector detector;
    auto init_result = detector.initModel(config.model.path);
    if (!init_result) {
        return std::unexpected("初始化模型失败: " + init_result.error());
    }

    if (config.debug.print_model_info) {
        std::cout << "模型加载成功: " << config.model.path << std::endl;
    }

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

    // 创建视频写入器（如果需要保存）
    cv::VideoWriter writer;
    if (config.output.save_video) {
        // 确保输出目录存在
        std::filesystem::path output_path(config.output.output_path);
        std::filesystem::create_directories(output_path.parent_path());
        
        // 解析编码格式
        int fourcc;
        if (config.output.video_codec == "mp4v") {
            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        } else if (config.output.video_codec == "XVID") {
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        } else {
            fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // 默认
        }
        
        writer.open(config.output.output_path, fourcc, fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            return std::unexpected("无法创建输出视频: " + config.output.output_path);
        }
        std::cout << "输出视频将保存至: " << config.output.output_path << std::endl;
    } else {
        std::cout << "视频保存已禁用，仅" << (config.display.enable_realtime_display ? "实时显示" : "后台处理") << std::endl;
    }

    cv::Mat frame;
    int frame_count = 0;
    int detection_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    
    auto bbox_color = cv::Scalar(config.display.bbox_color[0], config.display.bbox_color[1], config.display.bbox_color[2]);
    auto text_color = cv::Scalar(config.display.text_color[0], config.display.text_color[1], config.display.text_color[2]);

    while (cap.read(frame)) {
        frame_count++;
        
        // 应用帧跳过
        if (frame_count % config.performance.frame_skip != 0) {
            continue;
        }
        
        // 检查最大帧数限制
        if (config.performance.max_frames > 0 && frame_count > config.performance.max_frames) {
            break;
        }
        
        auto current_time = std::chrono::high_resolution_clock::now();

        // 显示处理进度
        if (config.debug.verbose_logging || frame_count % 10 == 0) {
            int progress = (frame_count * 100) / total_frames;
            std::cout << "处理进度: " << progress << "%" << std::endl;
        }

        // 检测
        auto objects = detector.detect(frame);

        if (objects) {
            if (!objects->empty()) {
                detection_count++;
                if (config.debug.verbose_logging) {
                    std::cout << "第 " << frame_count << " 帧: 检测到 " << objects->size() << " 个目标" << std::endl;
                }
            }

            // 绘制检测结果
            for (const auto& obj : *objects) {
                if (!obj.pts.empty()) {
                    std::vector<cv::Point> int_pts;
                    for (const auto& pt : obj.pts) {
                        int_pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                    }
                    cv::polylines(frame, int_pts, true, bbox_color, 2);
                }
                cv::rectangle(frame, obj.rect, bbox_color, 1);

                if (config.display.show_detection_info) {
                    std::string class_name = (obj.cls < config.detection.class_names.size()) ? 
                        config.detection.class_names[obj.cls] : "cls" + std::to_string(obj.cls);
                    std::string color_name = (obj.color < config.detection.color_names.size()) ? 
                        config.detection.color_names[obj.color] : "color" + std::to_string(obj.color);
                    
                    std::string label = class_name + "-" + color_name + 
                                      " (" + std::to_string(obj.prob).substr(0, 4) + ")";

                    cv::putText(frame, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
                }
            }
        }

        // 显示FPS
        if (config.display.show_fps) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
            if (duration.count() > 0) {
                float fps_current = 1000.0f / duration.count();
                cv::putText(frame, "FPS: " + std::to_string(fps_current).substr(0, 4), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }
        }
        last_time = current_time;
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
