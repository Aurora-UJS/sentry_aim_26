/**
 ************************************************************************
 *
 * @file auto_aim_pipeline_test.cpp
 * @author Neomelt
 * @brief 完整的自瞄流水线测试：通过正式 pipeline 执行检测 -> 解算 -> 跟踪 -> 选目标 -> 解算指令
 *
 ************************************************************************
 * @copyright Copyright (c) 2026 Aurora Vision
 ************************************************************************
 */

#include "auto_aim/pipeline/pipeline.hpp"
#include "utils/logger/logger.hpp"
#include "utils/plotjuggler_udp.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace armor;

namespace {

void drawVisualization(cv::Mat& img, const PipelineOutput& output) {
    for (const auto& obj : output.detections) {
        if (obj.pts.size() < 4) {
            continue;
        }

        for (int j = 0; j < 4; ++j) {
            cv::line(img, obj.pts[j], obj.pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::circle(img, obj.pts[j], 5, cv::Scalar(255, 0, 255), -1);
            cv::putText(img, std::to_string(j), obj.pts[j] + cv::Point2f(-10, -10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        std::ostringstream label;
        label << armorTypeToString(obj.type) << " " << armorNumberToString(obj.number);
        cv::putText(img, label.str(), obj.center + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 0), 2);
    }

    for (std::size_t i = 0; i < output.solved_armors.size() && i < output.detections.size(); ++i) {
        const auto& armor = output.solved_armors[i];
        const auto& obj = output.detections[i];
        if (!armor.is_ok) {
            continue;
        }

        std::ostringstream label;
        label << "Dist: " << std::fixed << std::setprecision(2) << armor.pos.norm() << "m";
        cv::putText(img, label.str(), obj.center + cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 255), 1);
    }

    if (!output.selected_target.has_value()) {
        return;
    }

    const auto& target = output.selected_target.value();
    cv::rectangle(img, cv::Point(10, 10), cv::Point(520, 145), cv::Scalar(0, 0, 0), -1);
    cv::rectangle(img, cv::Point(10, 10), cv::Point(520, 145), cv::Scalar(0, 255, 0), 2);

    int y_offset = 30;
    std::ostringstream line;

    line << "TARGET: " << armorNumberToString(target.number);
    cv::putText(img, line.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
    y_offset += 25;

    line.str("");
    line.clear();
    line << "Pos: (" << std::fixed << std::setprecision(2) << target.pos.x() << ", "
         << target.pos.y() << ", " << target.pos.z() << ")m";
    cv::putText(img, line.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
    y_offset += 20;

    line.str("");
    line.clear();
    line << "Distance: " << std::fixed << std::setprecision(2) << target.pos.norm() << "m";
    cv::putText(img, line.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
    y_offset += 20;

    if (output.aim_command.has_value()) {
        line.str("");
        line.clear();
        line << "Yaw/Pitch: " << std::fixed << std::setprecision(2) << output.aim_command->yaw
             << " / " << output.aim_command->pitch << " rad";
        cv::putText(img, line.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
        y_offset += 20;

        line.str("");
        line.clear();
        line << "Flight: " << std::fixed << std::setprecision(3)
             << output.aim_command->flight_time_s << " s";
        cv::putText(img, line.str(), cv::Point(20, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    utils::logger()->set_level(spdlog::level::info);
    utils::logger()->info("[Pipeline Test] Starting auto-aim pipeline test");

    if (argc < 5) {
        utils::logger()->error(
            "Usage: {} <video_path> <detector_config> <camera_config> <tracker_config>", argv[0]);
        utils::logger()->info(
            "Example: {} video/demo.avi config/detector_config.yaml "
            "config/camera_info_demo.yaml config/tracker_config.yaml",
            argv[0]);
        return -1;
    }

    PipelineConfig pipeline_config;
    pipeline_config.detector_config_path = argv[2];
    pipeline_config.camera_config_path = argv[3];
    pipeline_config.tracker_config_path = argv[4];

    auto pipeline = AutoAimPipeline::create(pipeline_config);
    if (!pipeline) {
        utils::logger()->error("[Pipeline Test] Failed to initialize pipeline");
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        utils::logger()->error("[Pipeline Test] Failed to open video: {}", argv[1]);
        return -1;
    }

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    utils::logger()->info("[Pipeline Test] Video info: {} frames, {:.1f} FPS", total_frames, fps);
    utils::logger()->info("[Pipeline Test] Press 'q' to quit, SPACE to pause/resume");

    int frame_count = 0;
    int detection_count = 0;
    int solve_success_count = 0;
    int tracking_count = 0;
    bool paused = false;

    auto start_time = std::chrono::steady_clock::now();
    PipelineFrame latest_frame;
    bool has_frame = false;
    cv::Mat frame;

    while (true) {
        if (!paused) {
            if (!cap.read(frame)) {
                utils::logger()->info("[Pipeline Test] End of video");
                break;
            }

            PipelineInput input;
            input.frame = frame;
            input.timestamp = std::chrono::steady_clock::now();
            input.frame_id = "camera";

            latest_frame = pipeline->process(input);
            has_frame = true;

            frame_count++;
            detection_count += static_cast<int>(latest_frame.output.stats.detection_count);
            solve_success_count += static_cast<int>(latest_frame.output.stats.solved_count);
            tracking_count += latest_frame.output.stats.has_target ? 1 : 0;

            utils::PJStreamer pj;
            pj.send_map(
                {{"target_distance_m", latest_frame.output.selected_target.has_value()
                                           ? latest_frame.output.selected_target->pos.norm()
                                           : -1.0},
                 {"timestamp",
                  static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          latest_frame.input.timestamp.time_since_epoch())
                                          .count())},
                 {"detection_count",
                  static_cast<double>(latest_frame.output.stats.detection_count)},
                 {"solve_success", static_cast<double>(latest_frame.output.stats.solved_count)},
                 {"tracker_count", static_cast<double>(latest_frame.output.stats.tracker_count)},
                 {"tracked_candidates",
                  static_cast<double>(latest_frame.output.stats.tracked_candidate_count)},
                 {"detect_time_ms", latest_frame.output.stats.timings.detect_ms},
                 {"solve_time_ms", latest_frame.output.stats.timings.solve_ms},
                 {"track_time_ms", latest_frame.output.stats.timings.track_ms},
                 {"select_time_ms", latest_frame.output.stats.timings.select_ms},
                 {"aim_time_ms", latest_frame.output.stats.timings.aim_ms},
                 {"pipeline_total_ms", latest_frame.output.stats.timings.total_ms}});
        }

        if (!has_frame) {
            continue;
        }

        cv::Mat display = latest_frame.input.frame.clone();
        drawVisualization(display, latest_frame.output);

        std::ostringstream status;
        status << "Frame: " << frame_count
               << " | Detected: " << latest_frame.output.stats.detection_count
               << " | Solved: " << latest_frame.output.stats.solved_count
               << " | Trackers: " << latest_frame.output.stats.tracker_count
               << " | Target: " << (latest_frame.output.stats.has_target ? "yes" : "no");
        cv::putText(display, status.str(), cv::Point(10, display.rows - 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        status.str("");
        status.clear();
        status << "Detect: " << static_cast<int>(latest_frame.output.stats.timings.detect_ms)
               << "ms | Solve: " << static_cast<int>(latest_frame.output.stats.timings.solve_ms)
               << "ms | Track: " << static_cast<int>(latest_frame.output.stats.timings.track_ms)
               << "ms | Aim: " << static_cast<int>(latest_frame.output.stats.timings.aim_ms)
               << "ms";
        cv::putText(display, status.str(), cv::Point(10, display.rows - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        if (paused) {
            cv::putText(display, "PAUSED", cv::Point(display.cols / 2 - 100, display.rows / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 4);
        }

        cv::imshow("Auto-Aim Pipeline Test", display);

        const int key = cv::waitKey(paused ? 0 : 1);
        if (key == 'q' || key == 27) {
            break;
        }
        if (key == ' ') {
            paused = !paused;
            utils::logger()->info("[Pipeline Test] {}", paused ? "Paused" : "Resumed");
        }
    }

    const auto end_time = std::chrono::steady_clock::now();
    const double total_time = std::chrono::duration<double>(end_time - start_time).count();

    utils::logger()->info("========== Pipeline Test Summary ==========");
    utils::logger()->info("Total frames processed: {}", frame_count);
    utils::logger()->info("Total detections: {}", detection_count);
    utils::logger()->info("Solve success: {}", solve_success_count);
    utils::logger()->info("Tracking frames: {}", tracking_count);
    if (total_time > 0.0) {
        utils::logger()->info("Average FPS: {:.2f}", frame_count / total_time);
    }
    utils::logger()->info(
        "Average detections per frame: {:.2f}",
        frame_count > 0 ? static_cast<double>(detection_count) / frame_count : 0.0);
    utils::logger()->info(
        "Solve success rate: {:.1f}%",
        detection_count > 0 ? 100.0 * solve_success_count / detection_count : 0.0);
    utils::logger()->info("Tracking rate: {:.1f}%",
                          frame_count > 0 ? 100.0 * tracking_count / frame_count : 0.0);

    cv::destroyAllWindows();
    return 0;
}
