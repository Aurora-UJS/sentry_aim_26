// /**
//  * @file main.cpp
//  * @brief YOLO检测器视频测试示例（新版：含角点、颜色）
//  */

// #include "sentry_aim_26/detector/detector.hpp"
// #include <iostream>
// #include <opencv2/opencv.hpp>

// int main() {
//     using namespace sentry_aim;

//     // 配置YOLO
//     YoloConfig config;
//     config.model_path =
//         "/home/xlqmu/sentry_aim_26/assets/models/aim1.onnx"; // 替换为你自己的模型路径
//     config.confidence_threshold = 0.5f;
//     config.nms_threshold = 0.4f;
//     config.input_size = cv::Size(416, 416);
//     config.class_names = {"armor"};
//     config.color_names = {"blue", "red", "gray", "purple"};

//     // 创建检测器
//     YoloDetector detector(config);

//     if (!detector.isInitialized()) {
//         std::cerr << "Failed to initialize YOLO detector" << std::endl;
//         return -1;
//     }

//     std::cout << "✓ " << detector.getModelInfo() << std::endl;

//     // 打开视频
//     cv::VideoCapture cap("/home/xlqmu/sentry_aim_26/test/video/test.mp4"); // 替换为你的视频路径
//     if (!cap.isOpened()) {
//         std::cerr << "Cannot open video file" << std::endl;
//         return -1;
//     }

//     // 视频参数
//     double fps = cap.get(cv::CAP_PROP_FPS);
//     int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
//     int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
//     int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

//     std::cout << "Video info: " << frame_width << "x" << frame_height << " @ " << fps
//               << " FPS, Total: " << total_frames << " frames" << std::endl;

//     // 输出视频
//     cv::VideoWriter writer("output_result.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps,
//                            cv::Size(frame_width, frame_height));

//     // 主循环
//     cv::Mat frame;
//     int frame_count = 0;
//     auto start_time = std::chrono::high_resolution_clock::now();

//     while (true) {
//         cap >> frame;
//         if (frame.empty())
//             break;

//         frame_count++;
//         std::cout << "Processing frame " << frame_count << "/" << total_frames << std::endl;

//         auto result = detector.detect(frame);

//         if (result) {
//             auto detections = result.value();
//             std::cout << "  Found " << detections.size() << " armors" << std::endl;

//             for (const auto& det : detections) {
//                 const auto& pts = det.keypoints;
//                 if (pts.size() == 4) {
//                     for (int i = 0; i < 4; ++i) {
//                         cv::line(frame, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
//                         cv::circle(frame, pts[i], 3, cv::Scalar(0, 0, 255), -1);
//                     }
//                 } else {
//                     // 退化情况，用bbox
//                     cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 255), 2);
//                 }

//                 cv::circle(frame, det.center, 3, cv::Scalar(255, 0, 0), -1);

//                 // 构建标签
//                 std::string class_str =
//                     (det.class_id >= 0 && det.class_id < config.class_names.size())
//                         ? config.class_names[det.class_id]
//                         : "unknown";

//                 std::string color_str =
//                     (det.color_id >= 0 && det.color_id < config.color_names.size())
//                         ? config.color_names[det.color_id]
//                         : "unknown";

//                 std::string label = "[" + color_str + "] " + class_str + " (" +
//                                     std::to_string(det.confidence).substr(0, 4) + ")";

//                 cv::putText(
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
    const std::string& model_path = "/home/xlqmu/sentry_aim_26/assets/models/aim1.onnx",
    bool realtime_display = false) -> std::expected<void, std::string>;

// 处理单张图片的测试函数（带实时显示）
auto processImage(
    const std::string& input_path, const std::string& output_path,
    const std::string& model_path = "/home/xlqmu/sentry_aim_26/assets/models/aim1.onnx",
    bool realtime_display = false) -> std::expected<void, std::string> {

    ArmorDetector detector;
    auto init_result = detector.initModel(model_path);
    if (!init_result) {
        return std::unexpected("初始化模型失败: " + init_result.error());
    }

    std::cout << "模型加载成功" << std::endl;

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

    std::cout << "检测到 " << objects->size() << " 个目标" << std::endl;

    // 绘制检测结果
    cv::Mat result_img = img.clone();
    for (const auto& obj : *objects) {
        std::cout << "目标: 类别=" << obj.cls << ", 颜色=" << obj.color << ", 置信度=" << obj.prob
                  << std::endl;

        // 绘制四边形
        cv::polylines(result_img, obj.pts, true, cv::Scalar(0, 255, 0), 2);

        // 绘制包围框
        cv::rectangle(result_img, obj.rect, cv::Scalar(255, 0, 0), 1);

        // 添加文本标签
        std::string label = "cls:" + std::to_string(obj.cls) +
                            " color:" + std::to_string(obj.color) +
                            " conf:" + std::to_string(obj.prob).substr(0, 4);

        cv::putText(result_img, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 0), 1);
    }

    // 实时显示
    if (realtime_display) {
        cv::imshow("Detection Result", result_img);
        std::cout << "按任意键继续..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // 保存结果（如果提供了输出路径）
    if (!output_path.empty()) {
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
auto processCamera(
    const std::string& model_path = "/home/xlqmu/sentry_aim_26/assets/models/aim1.onnx")
    -> std::expected<void, std::string> {

    ArmorDetector detector;
    auto init_result = detector.initModel(model_path);
    if (!init_result) {
        return std::unexpected("初始化模型失败: " + init_result.error());
    }

    std::cout << "模型加载成功，开始实时检测..." << std::endl;

    cv::VideoCapture cap(0); // 打开默认摄像头
    if (!cap.isOpened()) {
        return std::unexpected("无法打开摄像头");
    }

    cv::Mat frame;
    int frame_count = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();

        // 检测
        auto objects = detector.detect(frame);

        if (objects) {
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

        // 计算并显示FPS
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
        if (duration.count() > 0) {
            float fps = 1000.0f / duration.count();
            cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
        last_time = current_time;

        // 显示结果
        cv::imshow("Real-time Detection", frame);

        // 按ESC键退出
        char key = cv::waitKey(1);
        if (key == 27)
            break; // ESC键
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "实时检测结束" << std::endl;
    return {};
}

// 修复 main.cpp 中的参数解析逻辑
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <模式> [参数...]\n"
                  << "模式:\n"
                  << "  image <输入图片> <输出图片> [模型路径] [--display]\n"
                  << "  video <输入视频> <输出视频> [模型路径] [--display]\n"
                  << "  camera [模型路径]  # 实时摄像头检测\n"
                  << "  \n"
                  << "选项:\n"
                  << "  --display: 显示实时检测结果\n"
                  << "  --no-save: 不保存输出文件（仅显示）\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string model_path = "/home/xlqmu/sentry_aim_26/assets/models/aim1.onnx";
    bool realtime_display = false;
    bool no_save = false;

    // 检查选项
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--display") {
            realtime_display = true;
        } else if (arg == "--no-save") {
            no_save = true;
        } else {
            args.push_back(arg);
        }
    }

    if (mode == "camera") {
        // 实时摄像头检测
        if (!args.empty()) {
            model_path = args[0];
        }
        auto result = processCamera(model_path);
        if (!result) {
            std::cerr << "错误: " << result.error() << std::endl;
            return 1;
        }
    } else if (mode == "image") {
        if (args.size() < 2) {
            std::cerr << "图片模式需要: image <输入图片> <输出图片> [模型路径] [选项]\n";
            return 1;
        }

        std::string input_path = args[0];
        std::string output_path = no_save ? "" : args[1];

        if (args.size() >= 3) {
            model_path = args[2];
        }

        auto result = processImage(input_path, output_path, model_path, realtime_display);
        if (!result) {
            std::cerr << "错误: " << result.error() << std::endl;
            return 1;
        }
    } else if (mode == "video") {
        if (args.size() < 2) {
            std::cerr << "视频模式需要: video <输入视频> <输出视频> [模型路径] [选项]\n";
            return 1;
        }

        std::string input_path = args[0];
        std::string output_path = no_save ? "" : args[1];

        if (args.size() >= 3) {
            model_path = args[2];
        }

        auto result = processVideo(input_path, output_path, model_path, realtime_display);
        if (!result) {
            std::cerr << "错误: " << result.error() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "无效模式: " << mode << std::endl;
        return 1;
    }

    return 0;
}
