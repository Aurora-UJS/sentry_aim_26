/**
 * @file test_video_threaded.cpp
 * @brief 多线程视频检测处理器 - 使用线程池提高性能
 * @author xlqmu
 * @date 2025-08-15
 */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <vector>
#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"

// 全局变量声明
extern DetectionConfig g_detection_config;

// 线程安全的帧队列
class FrameQueue {
private:
    std::queue<std::pair<cv::Mat, int>> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool finished_ = false;

public:
    void push(const cv::Mat& frame, int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace(frame.clone(), frame_id);
        condition_.notify_one();
    }

    bool pop(cv::Mat& frame, int& frame_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) {
            return false;
        }
        
        auto pair = queue_.front();
        queue_.pop();
        frame = pair.first;
        frame_id = pair.second;
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 线程安全的结果队列
class ResultQueue {
private:
    std::queue<std::pair<std::vector<ArmorObject>, int>> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool finished_ = false;

public:
    void push(const std::vector<ArmorObject>& objects, int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace(objects, frame_id);
        condition_.notify_one();
    }

    bool pop(std::vector<ArmorObject>& objects, int& frame_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) {
            return false;
        }
        
        auto pair = queue_.front();
        queue_.pop();
        objects = pair.first;
        frame_id = pair.second;
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 多线程视频处理器
class ThreadedVideoProcessor {
private:
    ArmorDetector detector_;
    FrameQueue frame_queue_;
    ResultQueue result_queue_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> stop_workers_{false};
    int num_threads_;
    toml::Config config_;

    void workerThread() {
        ArmorDetector thread_detector;
        if (!thread_detector.initModel(config_.model.path).has_value()) {
            std::cerr << "Worker thread failed to initialize detector" << std::endl;
            return;
        }

        cv::Mat frame;
        int frame_id;
        
        while (!stop_workers_ && frame_queue_.pop(frame, frame_id)) {
            auto result = thread_detector.detect(frame);
            std::vector<ArmorObject> objects;
            if (result.has_value()) {
                objects = result.value();
            }
            result_queue_.push(objects, frame_id);
        }
    }

    void drawDetections(cv::Mat& frame, const std::vector<ArmorObject>& objects) {
        for (const auto& obj : objects) {
            // 绘制边界框
            cv::rectangle(frame, obj.rect, 
                         cv::Scalar(config_.display.bbox_color[0], 
                                   config_.display.bbox_color[1], 
                                   config_.display.bbox_color[2]), 2);
            
            if (config_.display.show_detection_info) {
                // 显示置信度和类别信息
                std::string label = "Class:" + std::to_string(obj.cls) + 
                                   " Color:" + std::to_string(obj.color) + 
                                   " Conf:" + std::to_string(obj.prob);
                
                cv::putText(frame, label, 
                           cv::Point(obj.rect.x, obj.rect.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(config_.display.text_color[0],
                                     config_.display.text_color[1],
                                     config_.display.text_color[2]), 1);
            }
        }
    }

public:
    ThreadedVideoProcessor(int num_threads = std::thread::hardware_concurrency()) 
        : num_threads_(num_threads) {}

    bool initialize(const std::string& config_path) {
        if (!config_.load(config_path)) {
            std::cerr << "Failed to load config from " << config_path << std::endl;
            return false;
        }
        
        g_detection_config.updateFromToml(config_);
        
        if (!detector_.initModel(config_.model.path).has_value()) {
            std::cerr << "Failed to initialize main detector" << std::endl;
            return false;
        }
        
        return true;
    }

    void startWorkers() {
        for (int i = 0; i < num_threads_; ++i) {
            worker_threads_.emplace_back(&ThreadedVideoProcessor::workerThread, this);
        }
    }

    void stopWorkers() {
        stop_workers_ = true;
        frame_queue_.finish();
        
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        result_queue_.finish();
    }

    void processVideo(const std::string& video_path, bool display = false) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "无法打开视频文件: " << video_path << std::endl;
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frame_size(
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        
        std::cout << "视频FPS: " << fps << std::endl;
        std::cout << "分辨率: " << frame_size.width << "x" << frame_size.height << std::endl;

        // 设置输出视频写入器
        cv::VideoWriter writer;
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        
        if (config_.output.save_video) {
            writer.open(config_.output.output_path, fourcc, fps, frame_size);
            if (writer.isOpened()) {
                std::cout << "输出视频将保存至: " << config_.output.output_path << std::endl;
            } else {
                std::cerr << "无法创建输出视频文件" << std::endl;
            }
        }

        // 启动工作线程
        startWorkers();

        auto start_time = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        int processed_frames = 0;
        
        // 预读所有帧到内存 (避免多线程视频解码问题)
        std::vector<cv::Mat> all_frames;
        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame_count % config_.performance.frame_skip == 0) {
                all_frames.push_back(frame.clone());
            }
            frame_count++;
            
            if (config_.performance.max_frames > 0 && 
                frame_count >= config_.performance.max_frames) {
                break;
            }
        }
        
        std::cout << "预读了 " << all_frames.size() << " 帧到内存" << std::endl;

        // 推送帧到队列进行并行处理
        for (size_t i = 0; i < all_frames.size(); ++i) {
            frame_queue_.push(all_frames[i], static_cast<int>(i));
        }
        frame_queue_.finish();

        // 处理结果
        std::vector<ArmorObject> objects;
        int frame_id;
        std::map<int, std::vector<ArmorObject>> results_buffer;
        
        // 收集所有结果
        while (result_queue_.pop(objects, frame_id)) {
            results_buffer[frame_id] = objects;
        }

        // 按顺序处理并显示结果
        for (size_t i = 0; i < all_frames.size(); ++i) {
            cv::Mat display_frame = all_frames[i].clone();
            
            if (results_buffer.find(static_cast<int>(i)) != results_buffer.end()) {
                drawDetections(display_frame, results_buffer[static_cast<int>(i)]);
            }
            
            processed_frames++;

            // 显示FPS信息
            if (config_.display.show_fps) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                double current_fps = processed_frames * 1000.0 / elapsed.count();
                
                std::string fps_text = "FPS: " + std::to_string(current_fps);
                cv::putText(display_frame, fps_text, cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(config_.display.text_color[0],
                                     config_.display.text_color[1],
                                     config_.display.text_color[2]), 2);
            }

            // 显示或保存结果
            if (display && config_.display.enable_realtime_display) {
                cv::imshow(config_.display.window_title, display_frame);
                int wait_ms = static_cast<int>(1000.0 / fps);
                if (cv::waitKey(wait_ms) == 27) break; // ESC键退出
            }

            if (writer.isOpened()) {
                writer.write(display_frame);
            }
        }

        stopWorkers();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n性能统计:" << std::endl;
        std::cout << "总帧数: " << frame_count << std::endl;
        std::cout << "处理帧数: " << processed_frames << std::endl;
        std::cout << "总时间: " << duration.count() << " ms" << std::endl;
        std::cout << "平均帧率: " << processed_frames * 1000.0 / duration.count() << " FPS" << std::endl;
        std::cout << "工作线程数: " << num_threads_ << std::endl;
        std::cout << "跳帧间隔: " << config_.performance.frame_skip << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <config_file> <video_file> [--display] [--threads N]" << std::endl;
        return -1;
    }

    std::string config_file = argv[1];
    std::string video_file = argv[2];
    bool display = false;
    int num_threads = std::thread::hardware_concurrency();

    // 解析命令行参数
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--display") {
            display = true;
        } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        }
    }

    std::cout << "使用 " << num_threads << " 个工作线程" << std::endl;

    ThreadedVideoProcessor processor(num_threads);
    
    if (!processor.initialize(config_file)) {
        return -1;
    }
    
    processor.processVideo(video_file, display);
    
    return 0;
}
