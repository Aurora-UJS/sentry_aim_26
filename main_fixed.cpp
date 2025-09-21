#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"

// 前向声明
auto processVideo(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string>;
auto processImage(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string>;
auto processCamera(const toml::Config& config) -> std::expected<void, std::string>;

// 处理单张图片的函数
auto processImage(const std::string& input_path, const toml::Config& config) -> std::expected<void, std::string> {
    // 更新全局检测配置
    g_detection_config.updateFromToml(config);
    
    ArmorDetector detector;
    auto init_result = detector.init(config.model.model_path);
    if (!init_result.has_value()) {
        return std::unexpected("Failed to initialize detector: " + init_result.error());
    }

    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        return std::unexpected("Failed to load image: " + input_path);
    }

    // 检测
    auto result = detector.detect(image);
    if (!result.has_value()) {
        return std::unexpected("Detection failed: " + result.error());
    }

    auto objects = result.value();
    
    // 绘制检测结果
    for (const auto& obj : objects) {
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 2);
        
        std::string label = "cls:" + std::to_string(obj.cls) + 
                           " color:" + std::to_string(obj.color) + 
                           " conf:" + std::to_string(obj.prob).substr(0, 4);
        
        cv::putText(image, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 0), 1);
        
        if (config.display.show_detection_polygons && !obj.pts.empty()) {
            std::vector<cv::Point> int_pts;
            for (const auto& pt : obj.pts) {
                int_pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
            }
            cv::polylines(image, int_pts, true, cv::Scalar(0, 255, 0), 2);
        }
    }

    // 显示结果
    if (config.display.enable_realtime_display) {
        cv::imshow(config.display.window_title, image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // 保存结果
    if (config.output.save_image && !config.output.output_path.empty()) {
        cv::imwrite(config.output.output_path, image);
        std::cout << "图片保存至: " << config.output.output_path << std::endl;
    }

    return {};
}

// 处理摄像头输入的函数
auto processCamera(const toml::Config& config) -> std::expected<void, std::string> {
    // 更新全局检测配置
    g_detection_config.updateFromToml(config);
    
    ArmorDetector detector;
    auto init_result = detector.init(config.model.model_path);
    if (!init_result.has_value()) {
        return std::unexpected("Failed to initialize detector: " + init_result.error());
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return std::unexpected("Failed to open camera");
    }

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;

    std::cout << "摄像头已启动，按ESC键退出..." << std::endl;

    while (cap.read(frame)) {
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();

        // 检测
        auto result = detector.detect(frame);
        if (result.has_value()) {
            auto objects = result.value();
            
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

        last_time = current_time;

        // 显示
        cv::imshow(config.display.window_title, frame);
        char key = cv::waitKey(1);
        if (key == 27) break; // ESC键退出
    }

    cap.release();
    cv::destroyAllWindows();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (config.debug.print_detection_stats) {
        std::cout << "\n=== 处理完成 ===" << std::endl;
        std::cout << "总帧数: " << frame_count << std::endl;
        std::cout << "平均处理速度: " << (total_duration.count() > 0 ? (frame_count * 1000.0 / total_duration.count()) : 0) << " FPS" << std::endl;
    }

    return {};
}

int main(int argc, char* argv[]) {
    // 默认配置文件路径
    std::string config_path = "config/detection_config.toml";
    
    // 解析命令行参数
    if (argc >= 2) {
        config_path = argv[1];
    }
    
    // 加载配置
    toml::Config config;
    if (!config.load(config_path)) {
        std::cerr << "Failed to load config file: " << config_path << std::endl;
        return -1;
    }
    
    std::cout << "使用配置文件: " << config_path << std::endl;
    std::cout << "模型文件: " << config.model.model_path << std::endl;
    
    // 根据命令行参数选择模式
    if (argc >= 3) {
        std::string input_path = argv[2];
        
        if (input_path == "camera") {
            // 摄像头模式
            auto result = processCamera(config);
            if (!result.has_value()) {
                std::cerr << "Camera processing failed: " << result.error() << std::endl;
                return -1;
            }
        } else if (std::filesystem::exists(input_path)) {
            // 检查是否为图片文件
            std::string ext = std::filesystem::path(input_path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                // 图片模式
                auto result = processImage(input_path, config);
                if (!result.has_value()) {
                    std::cerr << "Image processing failed: " << result.error() << std::endl;
                    return -1;
                }
            } else if (ext == ".mp4" || ext == ".avi" || ext == ".mov") {
                // 视频模式
                auto result = processVideo(input_path, config);
                if (!result.has_value()) {
                    std::cerr << "Video processing failed: " << result.error() << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "Unsupported file format: " << ext << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Input file not found: " << input_path << std::endl;
            return -1;
        }
    } else {
        // 默认使用摄像头模式
        std::cout << "默认使用摄像头模式，使用方法：" << std::endl;
        std::cout << "  " << argv[0] << " [config_file] [input_file|camera]" << std::endl;
        std::cout << "  示例：" << std::endl;
        std::cout << "    " << argv[0] << " config/detection_config.toml camera" << std::endl;
        std::cout << "    " << argv[0] << " config/detection_config.toml test.mp4" << std::endl;
        std::cout << "    " << argv[0] << " config/detection_config.toml image.jpg" << std::endl;
        
        auto result = processCamera(config);
        if (!result.has_value()) {
            std::cerr << "Camera processing failed: " << result.error() << std::endl;
            return -1;
        }
    }
    
    return 0;
}
