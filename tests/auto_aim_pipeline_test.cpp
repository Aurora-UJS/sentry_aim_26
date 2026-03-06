/**
 ************************************************************************
 *
 * @file auto_aim_pipeline_test.cpp
 * @author Neomelt
 * @brief 完整的自瞄流水线测试：检测 -> 求解 -> 跟踪
 *
 ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 ************************************************************************
 */

#include "auto_aim/armor_detector/onnxruntime_detector.hpp"
#include "auto_aim/armor_solver/solver.hpp"
#include "auto_aim/armor_tracker/tracker.hpp"
#include "utils/logger/logger.hpp"
#include "utils/plotjuggler_udp.hpp"

#include <chrono>
#include <iomanip>

#include <opencv2/opencv.hpp>

using namespace armor;

// 可视化函数：绘制检测框、3D坐标轴和跟踪信息
void drawVisualization(cv::Mat& img, const std::vector<ArmorObject>& detections,
                       const std::vector<Armor>& solved_armors,
                       const std::optional<Armor>& tracked_target) {
    // 1. 绘制所有检测到的装甲板
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& obj = detections[i];

        // 绘制四点边框
        for (int j = 0; j < 4; j++) {
            cv::line(img, obj.pts[j], obj.pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::circle(img, obj.pts[j], 5, cv::Scalar(255, 0, 255), -1);
            // 显示点索引
            cv::putText(img, std::to_string(j), obj.pts[j] + cv::Point2f(-10, -10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        // 显示类别和置信度
        std::ostringstream ss;
        ss << armorTypeToString(obj.type) << " " << armorNumberToString(obj.number);
        cv::putText(img, ss.str(), obj.center + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 0), 2);
    }

    // 2. 绘制求解后的装甲板信息
    for (size_t i = 0; i < solved_armors.size() && i < detections.size(); ++i) {
        const auto& armor = solved_armors[i];
        const auto& obj = detections[i];

        if (!armor.is_ok)
            continue;

        // 显示距离和位置
        std::ostringstream ss;
        ss << "Dist: " << std::fixed << std::setprecision(2) << armor.pos.norm() << "m";
        cv::putText(img, ss.str(), obj.center + cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);

        ss.str("");
        ss << "Pos: (" << std::fixed << std::setprecision(2) << armor.pos.x() << ","
           << armor.pos.y() << "," << armor.pos.z() << ")";
        cv::putText(img, ss.str(), obj.center + cv::Point2f(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 200, 200), 1);
    }

    // 3. 高亮显示跟踪目标
    if (tracked_target.has_value()) {
        const auto& target = tracked_target.value();

        // 在图像顶部显示跟踪信息
        cv::rectangle(img, cv::Point(10, 10), cv::Point(500, 120), cv::Scalar(0, 0, 0), -1);
        cv::rectangle(img, cv::Point(10, 10), cv::Point(500, 120), cv::Scalar(0, 255, 0), 2);

        int y_offset = 30;
        std::ostringstream ss;

        ss << "TRACKING: " << armorNumberToString(target.number);
        cv::putText(img, ss.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
        y_offset += 25;

        ss.str("");
        ss << "Predicted Pos: (" << std::fixed << std::setprecision(2) << target.pos.x() << ", "
           << target.pos.y() << ", " << target.pos.z() << ")m";
        cv::putText(img, ss.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
        y_offset += 20;

        ss.str("");
        ss << "Distance: " << std::fixed << std::setprecision(2) << target.pos.norm() << "m";
        cv::putText(img, ss.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
        y_offset += 20;

        // 计算yaw角
        Eigen::Vector3d euler = target.ori.toRotationMatrix().eulerAngles(2, 1, 0);
        ss.str("");
        ss << "Yaw: " << std::fixed << std::setprecision(1) << (euler(0) * 180.0 / M_PI) << " deg";
        cv::putText(img, ss.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    utils::logger()->set_level(spdlog::level::info);
    utils::logger()->info("[Pipeline Test] Starting auto-aim pipeline test");

    // 参数检查
    if (argc < 5) {
        utils::logger()->error(
            "Usage: {} <video_path> <model_path> <camera_config> <tracker_config>", argv[0]);
        utils::logger()->info(
            "Example: {} video/demo.avi model/armor.onnx config/camera_info_demo.yaml "
            "config/tracker_config.yaml",
            argv[0]);
        return -1;
    }

    std::string video_path = argv[1];
    std::string model_path = argv[2];
    std::string camera_config = argv[3];
    std::string tracker_config = argv[4];

    // 1. 初始化检测器
    utils::logger()->info("[Pipeline Test] Initializing detector...");
    OnnxRuntimeDetector detector;
    if (!detector.init(model_path)) {
        utils::logger()->error("[Pipeline Test] Failed to initialize detector");
        return -1;
    }

    // 设置检测器参数
    DetectorParams detector_params;
    detector_params.conf_threshold = 0.6f;
    detector_params.nms_threshold = 0.5f;
    detector_params.input_size = {640, 640};
    detector_params.enable_debug = false;
    detector.setParams(detector_params);

    // 2. 初始化求解器
    utils::logger()->info("[Pipeline Test] Initializing solver...");
    PnpSolver solver(camera_config);

    // 3. 初始化跟踪器
    utils::logger()->info("[Pipeline Test] Initializing tracker...");
    auto tracker_cfg = TrackerConfig::fromYaml(tracker_config);
    TrackerManager tracker_manager(tracker_cfg);

    // 4. 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        utils::logger()->error("[Pipeline Test] Failed to open video: {}", video_path);
        return -1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    double dt = 1.0 / fps;

    utils::logger()->info("[Pipeline Test] Video info: {} frames, {:.1f} FPS", total_frames, fps);
    utils::logger()->info("[Pipeline Test] Press 'q' to quit, SPACE to pause/resume");

    // 统计信息
    int frame_count = 0;
    int detection_count = 0;
    int solve_success_count = 0;
    int tracking_count = 0;

    auto start_time = std::chrono::steady_clock::now();
    bool paused = false;

    cv::Mat frame;
    while (true) {
        if (!paused) {
            if (!cap.read(frame)) {
                utils::logger()->info("[Pipeline Test] End of video");
                break;
            }
            frame_count++;
        }

        cv::Mat display = frame.clone();
        auto frame_time = std::chrono::steady_clock::now();

        if (!paused) {
            // ========== 步骤1: 检测 ==========
            auto t1 = std::chrono::high_resolution_clock::now();
            auto detections = detector.detect(frame);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto detect_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            detection_count += detections.size();

            // ========== 步骤2: 求解 ==========
            std::vector<Armor> solved_armors;
            int solve_success = 0;

            t1 = std::chrono::high_resolution_clock::now();
            for (const auto& obj : detections) {
                Armor armor;
                if (solver.solve(obj, armor)) {
                    armor.timestamp = frame_time;
                    solved_armors.push_back(armor);
                    solve_success++;
                }
            }
            t2 = std::chrono::high_resolution_clock::now();
            auto solve_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            solve_success_count += solve_success;

            // ========== 步骤3: 跟踪 ==========
            t1 = std::chrono::high_resolution_clock::now();
            tracker_manager.update(solved_armors, frame_time);
            auto tracked_target = tracker_manager.getBestTarget();
            t2 = std::chrono::high_resolution_clock::now();
            auto track_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            if (tracked_target.has_value()) {
                tracking_count++;
            }

            utils::PJStreamer pj;

            pj.send_map(
                {{"target_distance_m",
                  tracked_target.has_value() ? tracked_target->pos.norm() : -1.0},
                 {"timestamp",
                  static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          frame_time.time_since_epoch())
                                          .count())},
                 {"detection_count", static_cast<double>(detections.size())},
                 {"solve_success", static_cast<double>(solve_success)},
                 {"tracker_count", static_cast<double>(tracker_manager.getTrackers().size())},
                 {"detect_time_ms", static_cast<double>(detect_time)},
                 {"solve_time_ms", static_cast<double>(solve_time)},
                 {"track_time_ms", static_cast<double>(track_time)},
                 {"fps", 1000.0 / static_cast<double>(detect_time + solve_time + track_time + 1)}});

            // ========== 可视化 ==========
            drawVisualization(display, detections, solved_armors, tracked_target);

            // 显示性能统计
            std::ostringstream ss;
            ss << "Frame: " << frame_count << " | Detected: " << detections.size()
               << " | Solved: " << solve_success
               << " | Tracked: " << tracker_manager.getTrackers().size();
            cv::putText(display, ss.str(), cv::Point(10, display.rows - 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            ss.str("");
            ss << "Detect: " << detect_time << "ms | Solve: " << solve_time
               << "ms | Track: " << track_time << "ms";
            cv::putText(display, ss.str(), cv::Point(10, display.rows - 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        } else {
            // 暂停状态提示
            cv::putText(display, "PAUSED", cv::Point(display.cols / 2 - 100, display.rows / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 4);
        }

        cv::imshow("Auto-Aim Pipeline Test", display);

        int key = cv::waitKey(paused ? 0 : 1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == ' ') {  // SPACE
            paused = !paused;
            utils::logger()->info("[Pipeline Test] {}", paused ? "Paused" : "Resumed");
        }
    }

    // 最终统计
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(end_time - start_time).count();

    utils::logger()->info("========== Pipeline Test Summary ==========");
    utils::logger()->info("Total frames processed: {}", frame_count);
    utils::logger()->info("Total detections: {}", detection_count);
    utils::logger()->info("Solve success: {}", solve_success_count);
    utils::logger()->info("Tracking frames: {}", tracking_count);
    utils::logger()->info("Average FPS: {:.2f}", frame_count / total_time);
    utils::logger()->info("Average detections per frame: {:.2f}",
                          static_cast<double>(detection_count) / frame_count);
    utils::logger()->info("Solve success rate: {:.1f}%",
                          100.0 * solve_success_count / std::max(1, detection_count));
    utils::logger()->info("Tracking rate: {:.1f}%", 100.0 * tracking_count / frame_count);

    cv::destroyAllWindows();
    return 0;
}
