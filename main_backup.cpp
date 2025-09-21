/**
 * @file main.cpp
 * @brief 装甲板检测系统主程序 - 支持TOML配置
 */

#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

// 前向声明
auto processVideo(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string>;
auto processImage(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string>;
auto processCamera(const toml::Config& config) -> std::expected<void, std::string>;

// 处理单张图片的函数
auto processImage(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string> {
//                     frame, label,
//                     cv::Point(static_cast<int>(det.center.x), static_cast<int>(det.center.y - 10)),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//             }
//         } else {
//             std::cerr << "Detection failed: " << result.error() << std::endl;
//         }

//         // 显示
//         cv::imshow("YOLO Video Detection", frame);

//         // 保存
//         if (writer.isOpened()) {
//             writer.write(frame);
//         }

//         // ESC退出
//         if (cv::waitKey(1) == 27)
//             break;
//     }

//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto total_duration =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

//     std::cout << "Processed " << frame_count << " frames in " << total_duration.count()
//               << "ms (avg: " << (total_duration.count() / frame_count) << "ms/frame)" << std::endl;

//     cap.release();
//     writer.release();
//     cv::destroyAllWindows();

//     return 0;
// }

// 修复访问 config 的问题
/**
 * @file main.cpp
 * @brief YOLO检测器视频测试示例（无GUI版本）
 */

// 修改 main.cpp
#include "sentry_aim_26/detector/detector.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 声明视频测试函数
auto processVideo(
    const std::string& input_path, const std::string& output_path,
    const std::string& model_path = "/home/xlqmu/sentry_aim_26/assets/models/onnx/autoaim_0526.onnx",
    bool realtime_display = false) -> std::expected<void, std::string>;

// 处理单张图片的函数
auto processImage(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string> {
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

    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        return std::unexpected("无法加载图片: " + input_path);
    }

    std::cout << "输入图片尺寸: " << img.size() << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto objects = detector.detect(img);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "检测耗时: " << duration.count() << "ms" << std::endl;

    if (!objects) {
        return std::unexpected("检测失败: " + objects.error());
    }

    if (config.debug.print_detection_stats) {
        std::cout << "检测到 " << objects->size() << " 个目标" << std::endl;
    }

    // 绘制检测结果
    cv::Mat result_img = img.clone();
    auto bbox_color = cv::Scalar(config.display.bbox_color[0], config.display.bbox_color[1], config.display.bbox_color[2]);
    auto text_color = cv::Scalar(config.display.text_color[0], config.display.text_color[1], config.display.text_color[2]);
    
    for (const auto& obj : *objects) {
        if (config.debug.verbose_logging) {
            std::cout << "目标: 类别=" << obj.cls << ", 颜色=" << obj.color << ", 置信度=" << obj.prob << std::endl;
        }

        // 绘制四边形
        if (!obj.pts.empty()) {
            std::vector<cv::Point> int_pts;
            for (const auto& pt : obj.pts) {
                int_pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
            }
            cv::polylines(result_img, int_pts, true, bbox_color, 2);
        }

        // 绘制包围框
        cv::rectangle(result_img, obj.rect, bbox_color, 1);

        // 添加文本标签
        if (config.display.show_detection_info) {
            std::string class_name = (obj.cls < config.detection.class_names.size()) ? 
                config.detection.class_names[obj.cls] : "cls" + std::to_string(obj.cls);
            std::string color_name = (obj.color < config.detection.color_names.size()) ? 
                config.detection.color_names[obj.color] : "color" + std::to_string(obj.color);
            
            std::string label = class_name + "-" + color_name + 
                              " (" + std::to_string(obj.prob).substr(0, 4) + ")";

            cv::putText(result_img, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
        }
    }

    // 实时显示
    if (config.display.enable_realtime_display) {
        cv::imshow(config.display.window_title, result_img);
        std::cout << "按任意键继续..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // 保存结果图片
    if (config.output.save_images) {
        // 确保输出目录存在
        std::filesystem::create_directories(config.output.image_output_dir);
        
        std::filesystem::path input_file(input_path);
        std::string output_filename = input_file.stem().string() + "_result" + input_file.extension().string();
        std::string output_path = config.output.image_output_dir + "/" + output_filename;
        
        if (!cv::imwrite(output_path, result_img)) {
            return std::unexpected("无法保存输出图片: " + output_path);
        }
        std::cout << "图片处理完成，输出保存至: " << output_path << std::endl;
    } else {
        std::cout << "图片处理完成（仅显示，未保存）" << std::endl;
    }

    return {};
}

// 实时摄像头检测函数
auto processCamera(const toml::Config& config) -> std::expected<void, std::string> {
    // 更新全局检测配置
    g_detection_config.updateFromToml(config);
    
    ArmorDetector detector;
    auto init_result = detector.initModel(config.model.path);
    if (!init_result) {
        return std::unexpected("初始化模型失败: " + init_result.error());
    }

    if (config.debug.print_model_info) {
        std::cout << "模型加载成功，开始实时检测..." << std::endl;
    }

    cv::VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        return std::unexpected("无法打开摄像头");
    }

    cv::Mat frame;
    int frame_count = 0;
    auto last_time = std::chrono::high_resolution_clock::now();
    
    auto bbox_color = cv::Scalar(config.display.bbox_color[0], config.display.bbox_color[1], config.display.bbox_color[2]);
    auto text_color = cv::Scalar(config.display.text_color[0], config.display.text_color[1], config.display.text_color[2]);

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

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

        // 检测
        auto objects = detector.detect(frame);

        if (objects && !objects->empty()) {
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

        // 计算并显示FPS
        if (config.display.show_fps) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
            if (duration.count() > 0) {
                float fps = 1000.0f / duration.count();
                cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }
        }
        last_time = current_time;

        // 显示结果
        if (config.display.enable_realtime_display) {
            cv::imshow(config.display.window_title, frame);
        }

        // 按ESC键退出
        char key = cv::waitKey(1);
        if (key == 27) break; // ESC键
    }

    cap.release();
    if (config.display.enable_realtime_display) {
        cv::destroyAllWindows();
    }
    
    if (config.debug.print_detection_stats) {
        std::cout << "实时检测结束，总处理帧数: " << frame_count << std::endl;
    }
    return {};
}

// 新的主函数
int main(int argc, char* argv[]) {
    // 默认配置文件路径
    std::string config_path = "config/detection_config.toml";
    std::string mode = "video";
    std::string input_path;
    
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <模式> <输入路径> [配置文件路径]\n"
                  << "模式:\n"
                  << "  image <输入图片路径> [配置文件路径]\n"
                  << "  video <输入视频路径> [配置文件路径]\n"
                  << "  camera [配置文件路径]\n"
                  << "\n"
                  << "配置文件默认路径: config/detection_config.toml\n"
                  << "所有参数都在配置文件中设置，包括:\n"
                  << "- 实时显示设置\n"
                  << "- 视频/图片保存设置\n"
                  << "- 检测参数\n"
                  << "- 显示样式\n";
        return 1;
    }

    mode = argv[1];
    
    if (mode != "camera") {
        if (argc < 3) {
            std::cerr << "错误: " << mode << " 模式需要指定输入路径" << std::endl;
            return 1;
        }
        input_path = argv[2];
        
        if (argc >= 4) {
            config_path = argv[3];
        }
    } else {
        if (argc >= 3) {
            config_path = argv[2];
        }
    }

    // 加载配置文件
    auto config_opt = toml::Config::load(config_path);
    if (!config_opt) {
        std::cerr << "无法加载配置文件: " << config_path << std::endl;
        std::cerr << "请确保配置文件存在，或创建默认配置文件" << std::endl;
        return 1;
    }
    
    auto config = config_opt.value();
    
    // 打印配置信息
    config.print();

    // 根据模式执行相应的处理
    std::expected<void, std::string> result;
    
    if (mode == "image") {
        result = processImage(input_path, config);
    } else if (mode == "video") {
        result = processVideo(input_path, config);
    } else if (mode == "camera") {
        result = processCamera(config);
    } else {
        std::cerr << "无效模式: " << mode << std::endl;
        return 1;
    }

    if (!result) {
        std::cerr << "错误: " << result.error() << std::endl;
        return 1;
    }

    return 0;
}
