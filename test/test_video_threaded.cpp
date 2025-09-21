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

// 全局变量和函数声明
extern DetectionConfig g_detection_config;

// 配置加载函数
bool loadConfig(const std::string& config_path, toml::Config& config) {
    return config.load(config_path);
}

// 检测器初始化函数
bool initializeDetector(ArmorDetector& detector, const toml::Config& config) {
    g_detection_config.updateFromToml(config);
    auto result = detector.initModel(config.model.path);
    return result.has_value();
}

// 检测函数包装器
std::vector<ArmorObject> detect(ArmorDetector& detector, cv::Mat& frame) {
    auto result = detector.detect(frame);
    if (result.has_value()) {
        return result.value();
    }
    return {};
}

// 帧数据结构
struct FrameData {
    cv::Mat frame;
    int frame_id;
    std::chrono::high_resolution_clock::time_point timestamp;
};

// 检测结果结构
struct DetectionResult {
    std::vector<ArmorObject> objects;
    int frame_id;
    std::chrono::high_resolution_clock::time_point process_start;
    std::chrono::high_resolution_clock::time_point process_end;
};

// 线程安全的帧队列
class FrameQueue {
private:
    std::queue<FrameData> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> finished_{false};
    size_t max_size_;

public:
    FrameQueue(size_t max_size = 10) : max_size_(max_size) {}

    bool push(const FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            return false; // 队列满，丢弃帧
        }
        queue_.push(frame);
        condition_.notify_one();
        return true;
    }

    bool pop(FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !finished_) {
            condition_.wait(lock);
        }
        if (queue_.empty()) {
            return false;
        }
        frame = queue_.front();
        queue_.pop();
        return true;
    }

    void finish() {
        finished_ = true;
        condition_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 线程安全的结果队列
class ResultQueue {
private:
    std::queue<DetectionResult> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;

public:
    void push(const DetectionResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(result);
        condition_.notify_one();
    }

    bool pop(DetectionResult& result, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (condition_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            result = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 工作线程函数
void workerThread(int thread_id, FrameQueue& frame_queue, ResultQueue& result_queue) {
    std::cout << "Worker thread " << thread_id << " started" << std::endl;
    
    FrameData frame_data;
    while (frame_queue.pop(frame_data)) {
        auto process_start = std::chrono::high_resolution_clock::now();
        
        // 执行检测
        auto objects = detect(frame_data.frame);
        
        auto process_end = std::chrono::high_resolution_clock::now();
        
        // 将结果放入结果队列
        DetectionResult result;
        result.objects = std::move(objects);
        result.frame_id = frame_data.frame_id;
        result.process_start = process_start;
        result.process_end = process_end;
        
        result_queue.push(result);
    }
    
    std::cout << "Worker thread " << thread_id << " finished" << std::endl;
}

// 绘制检测结果
void drawDetections(cv::Mat& frame, const std::vector<ArmorObject>& objects, 
                   const DetectionConfig& config) {
    for (const auto& obj : objects) {
        // 绘制装甲板轮廓
        std::vector<cv::Point> pts;
        for (const auto& pt : obj.pts) {
            pts.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }
        
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{pts}, true, 
                     cv::Scalar(config.display.bbox_color[0], 
                               config.display.bbox_color[1], 
                               config.display.bbox_color[2]), 2);
        
        // 绘制置信度和类别信息
        if (config.display.show_detection_info) {
            std::string label = "Class:" + std::to_string(obj.cls) + 
                               " Color:" + std::to_string(obj.color) + 
                               " Conf:" + std::to_string(obj.prob).substr(0, 4);
            
            cv::putText(frame, label, 
                       cv::Point(obj.rect.x, obj.rect.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(config.display.text_color[0],
                                 config.display.text_color[1],
                                 config.display.text_color[2]), 1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <video_file> [--display]" << std::endl;
        return -1;
    }

    // 加载配置
    std::string config_file = argv[1];
    std::string video_file = argv[2];
    bool display = (argc > 3 && std::string(argv[3]) == "--display");
    
    if (!loadConfig(config_file, g_detection_config)) {
        std::cerr << "Failed to load config from " << config_file << std::endl;
        return -1;
    }

    std::cout << "使用配置文件: " << config_file << std::endl;
    std::cout << "模型文件: " << g_detection_config.model.path << std::endl;

    // 初始化检测器
    if (!initializeDetector(g_detection_config.model.path)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }

    // 打开视频
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << video_file << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "视频信息:" << std::endl;
    std::cout << "  总帧数: " << total_frames << std::endl;
    std::cout << "  帧率: " << fps << " FPS" << std::endl;

    // 设置输出视频
    cv::VideoWriter writer;
    if (g_detection_config.output.save_video) {
        cv::Size frame_size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                           static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(g_detection_config.output.output_path, fourcc, fps, frame_size);
        if (writer.isOpened()) {
            std::cout << "输出视频将保存至: " << g_detection_config.output.output_path << std::endl;
        }
    }

    // 创建线程池
    const int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    std::cout << "使用 " << num_threads << " 个工作线程" << std::endl;
    
    FrameQueue frame_queue(20); // 最大缓存20帧
    ResultQueue result_queue;
    
    // 启动工作线程
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(workerThread, i, std::ref(frame_queue), std::ref(result_queue));
    }

    // 统计变量
    int frame_count = 0;
    int processed_frame_count = 0;
    int detection_count = 0;
    std::map<int, cv::Mat> frame_buffer; // 存储等待处理结果的帧
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_progress_time = start_time;

    // 主循环：读取和显示
    cv::Mat frame;
    bool reading_finished = false;
    
    while (true) {
        // 读取新帧并放入队列
        if (!reading_finished && cap.read(frame)) {
            frame_count++;
            
            // 跳帧处理
            if (frame_count % g_detection_config.performance.frame_skip == 0) {
                FrameData frame_data;
                frame_data.frame = frame.clone();
                frame_data.frame_id = frame_count;
                frame_data.timestamp = std::chrono::high_resolution_clock::now();
                
                if (frame_queue.push(frame_data)) {
                    frame_buffer[frame_count] = frame.clone();
                    processed_frame_count++;
                }
            }
        } else if (!reading_finished) {
            reading_finished = true;
            frame_queue.finish();
            std::cout << "视频读取完成，等待处理结果..." << std::endl;
        }

        // 处理检测结果
        DetectionResult result;
        if (result_queue.pop(result, std::chrono::milliseconds(10))) {
            auto it = frame_buffer.find(result.frame_id);
            if (it != frame_buffer.end()) {
                cv::Mat display_frame = it->second.clone();
                
                if (!result.objects.empty()) {
                    detection_count++;
                    drawDetections(display_frame, result.objects, g_detection_config);
                }

                // 显示FPS
                if (g_detection_config.display.show_fps) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        result.process_end - result.process_start);
                    float processing_fps = 1000.0f / duration.count();
                    cv::putText(display_frame, "Processing FPS: " + std::to_string(processing_fps).substr(0, 5),
                               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                               cv::Scalar(g_detection_config.display.text_color[0],
                                         g_detection_config.display.text_color[1],
                                         g_detection_config.display.text_color[2]), 2);
                }

                // 实时显示
                if (display && g_detection_config.display.enable_realtime_display) {
                    cv::imshow(g_detection_config.display.window_title, display_frame);
                    int wait_ms = static_cast<int>(1000.0 / fps);
                    char key = cv::waitKey(wait_ms);
                    if (key == 27) break; // ESC键退出
                }

                // 保存到视频文件
                if (writer.isOpened()) {
                    writer.write(display_frame);
                }

                frame_buffer.erase(it);
            }
        }

        // 显示进度
        auto current_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_progress_time).count() > 1000) {
            if (total_frames > 0) {
                int progress = (frame_count * 100) / total_frames;
                std::cout << "处理进度: " << progress << "%, 队列大小: " << frame_queue.size() 
                         << ", 结果队列: " << result_queue.size() << std::endl;
            }
            last_progress_time = current_time;
        }

        // 检查是否完成
        if (reading_finished && frame_buffer.empty() && result_queue.size() == 0) {
            break;
        }
    }

    // 等待所有工作线程完成
    for (auto& worker : workers) {
        worker.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 输出统计信息
    std::cout << "\n=== 处理完成 ===" << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "处理帧数: " << processed_frame_count << std::endl;
    std::cout << "跳帧间隔: " << g_detection_config.performance.frame_skip << std::endl;
    std::cout << "有检测结果的帧数: " << detection_count << std::endl;
    std::cout << "检测率: " << (processed_frame_count > 0 ? (detection_count * 100.0 / processed_frame_count) : 0) << "%" << std::endl;
    std::cout << "总耗时: " << total_duration.count() << "ms" << std::endl;
    std::cout << "平均处理速度: " << (total_duration.count() > 0 ? (processed_frame_count * 1000.0 / total_duration.count()) : 0) << " FPS" << std::endl;
    std::cout << "使用线程数: " << num_threads << std::endl;

    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();

    return 0;
}
