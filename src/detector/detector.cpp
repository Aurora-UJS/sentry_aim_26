/**
 * @file detector.cpp
 * @brief 神经网络推理的装甲板检测器实现
 * @author xlqmu
 * @date 2025-07-15
 * @version 1.0
 *
 * @copyright Copyright (c) 2025 Team SentryAim
 */

#include "sentry_aim_26/detector/detector.hpp"
#include <algorithm>
#include <chrono>
#include <ranges>
#include <limits>

// 全局配置实例定义
DetectionConfig g_detection_config;

// 工具函数：找到最大值索引
constexpr auto argmax(std::span<const float> data) -> int {
    return std::ranges::distance(data.begin(), std::ranges::max_element(data, std::less<>{}));
}

// 图像预处理：Letterbox 缩放
auto scaledResize(cv::Mat& img, Eigen::Matrix<float, 3, 3>& transform_matrix) -> cv::Mat {
    float r = std::min(static_cast<float>(g_detection_config.input_w) / img.cols,
                       static_cast<float>(g_detection_config.input_h) / img.rows);
    int unpad_w = static_cast<int>(r * img.cols);
    int unpad_h = static_cast<int>(r * img.rows);
    int dw = (g_detection_config.input_w - unpad_w) / 2;
    int dh = (g_detection_config.input_h - unpad_h) / 2;

    transform_matrix << 1.0f / r, 0.0f, -static_cast<float>(dw) / r, 0.0f, 1.0f / r,
        -static_cast<float>(dh) / r, 0.0f, 0.0f, 1.0f;

    cv::Mat resized, padded;
    cv::resize(img, resized, cv::Size(unpad_w, unpad_h));
    
    // 确保 padding 后的尺寸正确
    int pad_top = dh;
    int pad_bottom = g_detection_config.input_h - unpad_h - pad_top;
    int pad_left = dw;
    int pad_right = g_detection_config.input_w - unpad_w - pad_left;
    
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

// 生成网格和步幅
auto generate_grids_and_stride(int target_w, int target_h,
                               std::span<const int> strides) -> std::vector<GridAndStride> {
    std::vector<GridAndStride> grid_strides;
    for (int stride : strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 : std::views::iota(0, num_grid_h)) {
            for (int g0 : std::views::iota(0, num_grid_w)) {
                grid_strides.push_back({g0, g1, stride});
            }
        }
    }
    return grid_strides;
}

// 解析已解码的检测结果 - 适配[1, 25200, 22]输出格式
auto parseDecodedDetections(std::span<const float> feat_data,
                           const Eigen::Matrix<float, 3, 3>& transform_matrix,
                           float prob_threshold,
                           int num_detections = 25200) -> std::vector<ArmorObject> {
    std::vector<ArmorObject> objects;
    const int output_dim = 22; // 固定22维输出: 8坐标 + 1置信度 + 4颜色 + 9数字类别

    std::cout << "开始解析 " << num_detections << " 个检测候选框" << std::endl;

    for (int i = 0; i < num_detections; ++i) {
        size_t basic_pos = i * output_dim;

        // 提取置信度 (index 8) 并应用sigmoid
        float raw_confidence = feat_data[basic_pos + 8];
        float prob = 1.0f / (1.0f + std::exp(-raw_confidence));

        // 调试信息：每10000个检测输出一次置信度
        if (i % 10000 == 0) {
            std::cout << "检测 " << i << ": 原始置信度=" << raw_confidence 
                      << ", Sigmoid后=" << prob << std::endl;
        }

        if (prob >= prob_threshold) {
            // 提取8个关键点坐标 (0-7) - 这些应该已经是像素坐标了
            std::array<float, 8> coords;
            for (int j = 0; j < 8; ++j) {
                coords[j] = feat_data[basic_pos + j];
            }

            // 提取颜色分类 (9-12: 红蓝灰紫)
            int color = argmax(feat_data.subspan(basic_pos + 9, 4));
            
            // 提取数字类别 (13-21: G/1/2/3/4/5/O/Bs/Bb)
            int cls = argmax(feat_data.subspan(basic_pos + 13, 9));

            ArmorObject obj;
            
            // 构建关键点矩阵 (4个角点) - 直接使用坐标，通过变换矩阵转换到原图
            Eigen::Matrix<float, 3, 4> apex_norm;
            apex_norm << coords[0], coords[2], coords[4], coords[6], 
                         coords[1], coords[3], coords[5], coords[7], 
                         1.0f, 1.0f, 1.0f, 1.0f;
            auto apex_dst = transform_matrix * apex_norm;

            // 设置角点坐标
            for (int j = 0; j < 4; ++j) {
                obj.apex[j] = cv::Point2f(apex_dst(0, j), apex_dst(1, j));
                obj.pts.push_back(obj.apex[j]);
            }
            
            obj.rect = cv::boundingRect(obj.pts);
            obj.cls = cls;
            obj.color = color;
            obj.prob = prob;
            
            // 调试信息：检测到目标
            std::cout << "检测到目标: 类别=" << cls << " 颜色=" << color 
                      << " 置信度=" << prob << " 位置=" << obj.rect << std::endl;
                      
            objects.push_back(obj);
        }
    }
    
    std::cout << "解析完成，找到 " << objects.size() << " 个有效检测" << std::endl;
    return objects;
}

// 计算交集面积
constexpr auto intersection_area(const ArmorObject& a, const ArmorObject& b) -> float {
    return (a.rect & b.rect).area();
}

// NMS
auto nms_sorted_bboxes(std::vector<ArmorObject>& objects, float nms_threshold) -> std::vector<int> {
    std::ranges::sort(objects, std::greater<>{}, &ArmorObject::prob);

    std::vector<int> picked;
    std::vector<float> areas(objects.size());
    for (size_t i : std::views::iota(size_t{0}, objects.size())) {
        areas[i] = objects[i].rect.area();
    }

    for (size_t i : std::views::iota(size_t{0}, objects.size())) {
        const auto& a = objects[i];
        bool keep = true;
        for (int j : picked) {
            float inter_area = intersection_area(a, objects[j]);
            float union_area = areas[i] + areas[j] - inter_area;
            float iou = union_area > 0 ? inter_area / union_area : 0;
            if (iou > nms_threshold || std::isnan(iou)) {
                keep = false;
                if (iou > g_detection_config.merge_min_iou &&
                    std::abs(a.prob - objects[j].prob) < g_detection_config.merge_conf_error &&
                    a.cls == objects[j].cls && a.color == objects[j].color) {
                    objects[j].pts.insert(objects[j].pts.end(), a.pts.begin(), a.pts.end());
                }
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
    return picked;
}

// 解码模型输出 - 适配[1, 25200, 22]格式
auto decodeOutputs(std::span<const float> prob, const Eigen::Matrix<float, 3, 3>& transform_matrix,
                   int img_w, int img_h) -> std::vector<ArmorObject> {
    
    // 新格式：直接解析已解码的检测结果，不需要网格生成
    const int num_detections = 25200;  // 根据您的模型输出 [1, 25200, 22]
    
    std::cout << "使用新解码方法处理 [1, 25200, 22] 格式输出" << std::endl;
    
    auto proposals = parseDecodedDetections(prob, transform_matrix, g_detection_config.bbox_conf_thresh, num_detections);

    std::cout << "解析后得到 " << proposals.size() << " 个候选框" << std::endl;

    if (proposals.size() > g_detection_config.topk) {
        std::cout << "限制检测数量从 " << proposals.size() << " 到 " << g_detection_config.topk << std::endl;
        proposals.resize(g_detection_config.topk);
    }

    auto picked = nms_sorted_bboxes(proposals, g_detection_config.nms_thresh);
    std::cout << "NMS后保留 " << picked.size() << " 个检测" << std::endl;
    
    std::vector<ArmorObject> objects(picked.size());
    for (size_t i : std::views::iota(size_t{0}, picked.size())) {
        objects[i] = std::move(proposals[picked[i]]);
    }
    return objects;
}

// 装甲板检测器实现
auto ArmorDetector::initModel(std::string_view model_path) -> std::expected<void, std::string> {
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.data(), Ort::SessionOptions{});
        auto input_count = session_->GetInputCount();
        if (input_count != 1) {
            return std::unexpected("模型必须只有一个输入");
        }

        // 修复：使用 optional 包装
        input_name_ = session_->GetInputNameAllocated(0, allocator_);
        model_initialized_ = true;
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(e.what());
    }
}

auto ArmorDetector::detect(cv::Mat& img) -> std::expected<std::vector<ArmorObject>, std::string> {
    if (!session_ || !model_initialized_ || !input_name_) {
        return std::unexpected("模型未初始化");
    }

    std::cout << "原始图像尺寸: " << img.cols << "x" << img.rows << std::endl;

    Eigen::Matrix<float, 3, 3> transform_matrix;
    cv::Mat input_img = scaledResize(img, transform_matrix);

    std::cout << "预处理后图像尺寸: " << input_img.cols << "x" << input_img.rows << std::endl;
    std::cout << "预期输入尺寸: " << g_detection_config.input_w << "x" << g_detection_config.input_h << std::endl;

    std::vector<Ort::Float16_t> input_data(1 * 3 * g_detection_config.input_w * g_detection_config.input_h);
    for (int y : std::views::iota(0, g_detection_config.input_h)) {
        for (int x : std::views::iota(0, g_detection_config.input_w)) {
            auto pixel = input_img.at<cv::Vec3b>(y, x);
            // CHW 格式：[通道][高度][宽度] - 转换为 float16
            input_data[0 * g_detection_config.input_h * g_detection_config.input_w + y * g_detection_config.input_w + x] = Ort::Float16_t(pixel[2] / 255.0f); // R
            input_data[1 * g_detection_config.input_h * g_detection_config.input_w + y * g_detection_config.input_w + x] = Ort::Float16_t(pixel[1] / 255.0f); // G
            input_data[2 * g_detection_config.input_h * g_detection_config.input_w + y * g_detection_config.input_w + x] = Ort::Float16_t(pixel[0] / 255.0f); // B
        }
    }

    std::vector<int64_t> input_shape = {1, 3, g_detection_config.input_h, g_detection_config.input_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    try {
        // 修复：使用 optional 的 value()
        std::vector<const char*> input_names = {input_name_->get()};
        std::vector<const char*> output_names = {"output"};
        
        std::cout << "开始推理..." << std::endl;
        
        auto outputs = session_->Run(Ort::RunOptions{}, input_names.data(), &input_tensor, 1,
                                     output_names.data(), 1);
        
        std::cout << "推理完成，开始解析输出..." << std::endl;
        
        // 获取输出张量信息
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto* prob = outputs[0].GetTensorMutableData<float>();
        std::span<const float> prob_span(prob, output_info.GetElementCount());
        
        auto detection_result = decodeOutputs(prob_span, transform_matrix, img.cols, img.rows);
        // 应用时序稳定性
        applyTemporalStability(detection_result);
        return detection_result;
    } catch (const std::exception& e) {
        return std::unexpected(e.what());
    }
}

// 时序稳定性函数实现
auto ArmorDetector::applyTemporalStability(std::vector<ArmorObject>& current_detections) -> void {
    if (previous_detections_.empty()) {
        // 首次检测，直接保存
        previous_detections_ = current_detections;
        return;
    }
    
    // 为当前检测结果寻找最近的历史匹配
    for (auto& current : current_detections) {
        float min_distance = std::numeric_limits<float>::max();
        ArmorObject* best_match = nullptr;
        
        for (auto& previous : previous_detections_) {
            // 只匹配相同类别和颜色的目标
            if (previous.cls == current.cls && previous.color == current.color) {
                // 计算中心点距离
                cv::Point2f prev_center(previous.rect.x + previous.rect.width/2.0f, 
                                       previous.rect.y + previous.rect.height/2.0f);
                cv::Point2f curr_center(current.rect.x + current.rect.width/2.0f, 
                                       current.rect.y + current.rect.height/2.0f);
                
                float distance = cv::norm(prev_center - curr_center);
                
                if (distance < min_distance && distance < STABILITY_DISTANCE_THRESHOLD) {
                    min_distance = distance;
                    best_match = &previous;
                }
            }
        }
        
        // 如果找到匹配的历史目标，增加置信度稳定性
        if (best_match != nullptr) {
            // 对稳定目标给予置信度加成
            current.prob = std::min(1.0f, current.prob + STABILITY_CONFIDENCE_FACTOR);
            
            // 位置平滑：使用加权平均减少抖动
            constexpr float smooth_factor = 0.7f; // 当前帧权重
            cv::Point2f curr_center(current.rect.x + current.rect.width/2.0f, 
                                   current.rect.y + current.rect.height/2.0f);
            cv::Point2f prev_center(best_match->rect.x + best_match->rect.width/2.0f, 
                                   best_match->rect.y + best_match->rect.height/2.0f);
            
            cv::Point2f smoothed_center = smooth_factor * curr_center + (1.0f - smooth_factor) * prev_center;
            
            // 更新rect位置
            int new_x = static_cast<int>(smoothed_center.x - current.rect.width/2.0f);
            int new_y = static_cast<int>(smoothed_center.y - current.rect.height/2.0f);
            current.rect.x = new_x;
            current.rect.y = new_y;
            
            // 同时平滑关键点坐标
            cv::Point2f offset = smoothed_center - curr_center;
            for (auto& apex : current.apex) {
                apex += offset;
            }
            for (auto& pt : current.pts) {
                pt += offset;
            }
        }
    }
    
    // 更新历史检测结果
    previous_detections_ = current_detections;
}