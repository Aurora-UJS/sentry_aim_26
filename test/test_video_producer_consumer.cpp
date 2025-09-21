/**
 * @file test_video_producer_consumer.cpp
 * @brief 真正的生产者-消费者模式：一个线程读取，多个线程检测
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
#include <vector>
#include "sentry_aim_26/detector/detector.hpp"
#include "sentry_aim_26/core/config.hpp"

// 全局变量声明
extern DetectionConfig g_detection_config;

// 帧数据结构
struct FrameData {
    cv::Mat frame;
    int frame_id;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    FrameData() = default;
    FrameData(const cv::Mat& f, int id) 
        : frame(f.clone()), frame_id(id), timestamp(std::chrono::high_resolution_clock::now()) {}
};

// 检测结果结构
struct DetectionResult {
    std::vector<ArmorObject> objects;
    int frame_id;
    std::chrono::high_resolution_clock::time_point detection_time;
    
    DetectionResult() = default;
    DetectionResult(const std::vector<ArmorObject>& objs, int id)
        : objects(objs), frame_id(id), detection_time(std::chrono::high_resolution_clock::now()) {}
};

// 线程安全的帧队列（生产者-消费者）
class ThreadSafeFrameQueue {
private:
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> finished_{false};
    size_t max_size_;

public:
    ThreadSafeFrameQueue(size_t max_size = 10) : max_size_(max_size) {}

    bool push(const FrameData& frame_data) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 当队列满时等待，而不是丢弃帧
        condition_.wait(lock, [this] { return queue_.size() < max_size_ || finished_; });
        
        if (finished_) return false;
        
        queue_.push(frame_data);
        condition_.notify_one();
        return true;
    }

    bool pop(FrameData& frame_data) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) return false;
        
        frame_data = queue_.front();
        queue_.pop();
        
        // 通知生产者队列有空间了
        condition_.notify_all();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 线程安全的结果队列
class ThreadSafeResultQueue {
private:
    std::queue<DetectionResult> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> finished_{false};

public:
    void push(const DetectionResult& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!finished_) {
            queue_.push(result);
            condition_.notify_one();
        }
    }

    bool pop(DetectionResult& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) return false;
        
        result = queue_.front();
        queue_.pop();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 生产者-消费者视频处理器
class ProducerConsumerProcessor {
private:
    toml::Config config_;
    ThreadSafeFrameQueue frame_queue_;
    ThreadSafeResultQueue result_queue_;
    std::vector<std::thread> detector_threads_;
    std::thread producer_thread_;
    std::thread consumer_thread_;
    std::atomic<bool> stop_processing_{false};
    std::atomic<int> frames_read_{0};
    std::atomic<int> frames_processed_{0};
    int num_detector_threads_;

    // 消费者：检测装甲板
    void detector_worker(int worker_id) {
        // 每个工作线程都有自己的检测器实例
        ArmorDetector detector;
        if (!detector.initModel(config_.model.path).has_value()) {
            std::cerr << "检测器工作线程 " << worker_id << " 初始化失败" << std::endl;
            return;
        }

        std::cout << "检测器工作线程 " << worker_id << " 启动" << std::endl;
        
        FrameData frame_data;
        while (frame_queue_.pop(frame_data)) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 执行检测
            auto result = detector.detect(frame_data.frame);
            std::vector<ArmorObject> objects;
            if (result.has_value()) {
                objects = result.value();
            }
            
            // 将结果推送到结果队列
            DetectionResult detection_result(objects, frame_data.frame_id);
            result_queue_.push(detection_result);
            
            frames_processed_++;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto detection_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            if (frames_processed_ % 100 == 0) {
                std::cout << "工作线程 " << worker_id << " 处理第 " << frames_processed_ 
                         << " 帧，检测时间: " << detection_time.count() / 1000.0 << " ms" << std::endl;
            }
        }
        
        std::cout << "检测器工作线程 " << worker_id << " 结束" << std::endl;
    }

    // 消费者：处理检测结果并显示
    void result_consumer(bool display, cv::VideoWriter& writer, double fps, 
                        const std::vector<cv::Mat>& frame_cache) {
        std::cout << "结果消费者线程启动..." << std::endl;
        
        DetectionResult result;
        std::map<int, DetectionResult> result_buffer;
        int next_frame_to_display = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        int displayed_frames = 0;
        
        while (result_queue_.pop(result)) {
            result_buffer[result.frame_id] = result;
            
            // 按顺序处理结果
            while (result_buffer.find(next_frame_to_display) != result_buffer.end()) {
                auto& current_result = result_buffer[next_frame_to_display];
                
                // 获取对应的原始帧
                if (next_frame_to_display < static_cast<int>(frame_cache.size())) {
                    cv::Mat display_frame = frame_cache[next_frame_to_display].clone();
                    
                    // 绘制检测结果
                    drawDetections(display_frame, current_result.objects);
                    
                    // 显示FPS信息
                    if (config_.display.show_fps) {
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                        if (elapsed.count() > 0) {
                            double current_fps = displayed_frames * 1000.0 / elapsed.count();
                            std::string fps_text = "Real-time FPS: " + std::to_string(current_fps).substr(0, 6);
                            cv::putText(display_frame, fps_text, cv::Point(10, 30),
                                       cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                       cv::Scalar(config_.display.text_color[0],
                                                 config_.display.text_color[1],
                                                 config_.display.text_color[2]), 2);
                        }
                    }
                    
                    // 显示检测信息
                    if (config_.display.show_detection_info && !current_result.objects.empty()) {
                        std::string detection_info = "Detections: " + std::to_string(current_result.objects.size());
                        cv::putText(display_frame, detection_info, cv::Point(10, 70),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                   cv::Scalar(config_.display.text_color[0],
                                             config_.display.text_color[1],
                                             config_.display.text_color[2]), 2);
                    }
                    
                    // 实时显示
                    if (display && config_.display.enable_realtime_display) {
                        cv::imshow(config_.display.window_title, display_frame);
                        // 按原视频帧率播放，确保正常播放速度
                        int wait_ms = std::max(1, static_cast<int>(1000.0 / fps));
                        if (cv::waitKey(wait_ms) == 27) break; // ESC键退出
                    }
                    
                    // 保存视频
                    if (writer.isOpened()) {
                        writer.write(display_frame);
                    }
                    
                    displayed_frames++;
                }
                
                result_buffer.erase(next_frame_to_display);
                next_frame_to_display++;
            }
        }
        
        std::cout << "结果消费者线程结束，显示了 " << displayed_frames << " 帧" << std::endl;
    }

    void drawDetections(cv::Mat& frame, const std::vector<ArmorObject>& objects) {
        for (const auto& obj : objects) {
            // 对于YOLO-pose模型，优先显示关键点连线
            if (!obj.pts.empty()) {
                // 绘制装甲板的4个角点连线
                std::vector<cv::Point> int_pts;
                for (const auto& pt : obj.pts) {
                    int_pts.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
                }
                cv::polylines(frame, int_pts, true, 
                             cv::Scalar(config_.display.bbox_color[0], 
                                       config_.display.bbox_color[1], 
                                       config_.display.bbox_color[2]), 2);
                
                // 在关键点上绘制小圆点
                for (const auto& pt : obj.pts) {
                    cv::circle(frame, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 
                              3, cv::Scalar(config_.display.keypoint_color[0],
                                           config_.display.keypoint_color[1],
                                           config_.display.keypoint_color[2]), -1);
                }
            } else {
                // 如果没有关键点，则显示矩形框作为备选
                cv::rectangle(frame, obj.rect, 
                             cv::Scalar(config_.display.bbox_color[0], 
                                       config_.display.bbox_color[1], 
                                       config_.display.bbox_color[2]), 2);
            }
            
            if (config_.display.show_detection_info) {
                // 生成简化标签：颜色字母 + 类别名称 + 置信度
                std::vector<std::string> color_symbols = {"R", "B", "G", "P"}; // 红蓝灰紫
                std::vector<std::string> class_names = {"G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"};
                
                std::string color_str = (obj.color >= 0 && obj.color < color_symbols.size()) ? 
                                       color_symbols[obj.color] : "?";
                std::string class_str = (obj.cls >= 0 && obj.cls < class_names.size()) ? 
                                       class_names[obj.cls] : "?";
                
                std::string label = color_str + class_str + " " + 
                                   std::to_string(obj.prob).substr(0, 4);
                
                // 标签显示在第一个关键点附近，如果没有关键点则显示在矩形框上
                cv::Point label_pos = !obj.pts.empty() ? 
                    cv::Point(static_cast<int>(obj.pts[0].x), static_cast<int>(obj.pts[0].y - 10)) :
                    cv::Point(obj.rect.x, obj.rect.y - 10);
                
                cv::putText(frame, label, label_pos,
                           cv::FONT_HERSHEY_SIMPLEX, 0.5,
                           cv::Scalar(config_.display.text_color[0],
                                     config_.display.text_color[1],
                                     config_.display.text_color[2]), 1);
            }
        }
    }

public:
    ProducerConsumerProcessor(int num_detector_threads = 2) 
        : frame_queue_(20), num_detector_threads_(num_detector_threads) {}

    bool initialize(const std::string& config_path) {
        if (!config_.load(config_path)) {
            std::cerr << "无法加载配置文件: " << config_path << std::endl;
            return false;
        }
        
        g_detection_config.updateFromToml(config_);
        std::cout << "配置加载成功，检测器线程数: " << num_detector_threads_ << std::endl;
        return true;
    }

    void process(const std::string& video_path, bool display = false) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 获取视频信息
        cv::VideoCapture temp_cap(video_path);
        double fps = temp_cap.get(cv::CAP_PROP_FPS);
        cv::Size frame_size(
            static_cast<int>(temp_cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        temp_cap.release();
        
        std::cout << "视频FPS: " << fps << ", 分辨率: " << frame_size << std::endl;

        // 设置输出视频
        cv::VideoWriter writer;
        if (config_.output.save_video) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(config_.output.output_path, fourcc, fps, frame_size);
            if (writer.isOpened()) {
                std::cout << "输出视频将保存至: " << config_.output.output_path << std::endl;
            }
        }

        // 预读视频帧到缓存（用于显示）
        cv::VideoCapture cap(video_path);
        std::vector<cv::Mat> frame_cache;
        cv::Mat frame;
        int frame_count = 0;
        
        while (cap.read(frame)) {
            // frame_skip为0表示不跳帧，处理每一帧
            // frame_skip为1表示处理每一帧，frame_skip为2表示每隔一帧处理一次，以此类推
            if (config_.performance.frame_skip == 0 || frame_count % config_.performance.frame_skip == 0) {
                frame_cache.push_back(frame.clone());
            }
            frame_count++;
            
            if (config_.performance.max_frames > 0 && 
                frame_count >= config_.performance.max_frames) {
                break;
            }
        }
        cap.release();
        
        std::cout << "预读了 " << frame_cache.size() << " 帧到缓存" << std::endl;

        // 启动生产者线程（推送缓存的帧到检测队列）
        producer_thread_ = std::thread([this, &frame_cache]() {
            std::cout << "生产者线程启动，开始推送帧到检测队列..." << std::endl;
            
            for (size_t i = 0; i < frame_cache.size() && !stop_processing_; ++i) {
                FrameData frame_data(frame_cache[i], static_cast<int>(i));
                if (!frame_queue_.push(frame_data)) {
                    break;
                }
                frames_read_++;
                
                // 控制推送速度，避免队列过满
                if (frame_queue_.size() > 15) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            
            frame_queue_.finish();
            std::cout << "生产者线程结束，共推送 " << frames_read_ << " 帧" << std::endl;
        });

        // 启动检测器工作线程
        for (int i = 0; i < num_detector_threads_; ++i) {
            detector_threads_.emplace_back(&ProducerConsumerProcessor::detector_worker, this, i);
        }

        // 启动结果消费者线程（带帧缓存）
        consumer_thread_ = std::thread(&ProducerConsumerProcessor::result_consumer, this, 
                                      display, std::ref(writer), fps, std::cref(frame_cache));

        // 等待生产者完成
        if (producer_thread_.joinable()) {
            producer_thread_.join();
        }

        // 等待所有检测器完成
        for (auto& thread : detector_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // 关闭结果队列并等待消费者完成
        result_queue_.finish();
        if (consumer_thread_.joinable()) {
            consumer_thread_.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 输出性能统计
        std::cout << "\n=== 性能统计 ===" << std::endl;
        std::cout << "读取帧数: " << frames_read_ << std::endl;
        std::cout << "处理帧数: " << frames_processed_ << std::endl;
        std::cout << "总时间: " << duration.count() << " ms" << std::endl;
        std::cout << "平均处理帧率: " << frames_processed_ * 1000.0 / duration.count() << " FPS" << std::endl;
        std::cout << "检测器线程数: " << num_detector_threads_ << std::endl;
        std::cout << "跳帧间隔: " << config_.performance.frame_skip << std::endl;
        std::cout << "队列大小统计 - 帧队列: " << frame_queue_.size() << ", 结果队列: " << result_queue_.size() << std::endl;
        
        if (display) {
            cv::destroyAllWindows();
        }
    }

    void stop() {
        stop_processing_ = true;
        frame_queue_.finish();
        result_queue_.finish();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <config_file> <video_file> [--display] [--threads N]" << std::endl;
        std::cout << "示例: " << argv[0] << " config/detection_config.toml test/video/test.mp4 --display --threads 3" << std::endl;
        return -1;
    }

    std::string config_file = argv[1];
    std::string video_file = argv[2];
    bool display = false;
    int num_detector_threads = 2; // 默认2个检测线程

    // 解析命令行参数
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--display") {
            display = true;
        } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            num_detector_threads = std::atoi(argv[++i]);
        }
    }

    std::cout << "=== 生产者-消费者模式视频处理器 ===" << std::endl;
    std::cout << "配置文件: " << config_file << std::endl;
    std::cout << "视频文件: " << video_file << std::endl;
    std::cout << "检测器线程数: " << num_detector_threads << std::endl;
    std::cout << "显示模式: " << (display ? "开启" : "关闭") << std::endl;

    ProducerConsumerProcessor processor(num_detector_threads);
    
    if (!processor.initialize(config_file)) {
        return -1;
    }
    
    processor.process(video_file, display);
    
    return 0;
}
