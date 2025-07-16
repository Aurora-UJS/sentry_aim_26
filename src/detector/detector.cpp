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
#include <ranges>

// 工具函数：找到最大值索引
constexpr auto argmax(std::span<const float> data) -> int {
    return std::ranges::distance(data.begin(), std::ranges::max_element(data, std::less<>{}));
}

// 图像预处理：Letterbox 缩放
auto scaledResize(cv::Mat& img, Eigen::Matrix<float, 3, 3>& transform_matrix) -> cv::Mat {
    float r = std::min(static_cast<float>(config::INPUT_W) / img.cols,
                       static_cast<float>(config::INPUT_H) / img.rows);
    int unpad_w = static_cast<int>(r * img.cols);
    int unpad_h = static_cast<int>(r * img.rows);
    int dw = (config::INPUT_W - unpad_w) / 2;
    int dh = (config::INPUT_H - unpad_h) / 2;

    transform_matrix << 1.0f / r, 0.0f, -static_cast<float>(dw) / r, 0.0f, 1.0f / r,
        -static_cast<float>(dh) / r, 0.0f, 0.0f, 1.0f;

    cv::Mat resized, padded;
    cv::resize(img, resized, cv::Size(unpad_w, unpad_h));
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT);
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

// 生成 YOLOX 候选框
auto generateYoloxProposals(std::span<const GridAndStride> grid_strides,
                            std::span<const float> feat_data,
                            const Eigen::Matrix<float, 3, 3>& transform_matrix,
                            float prob_threshold) -> std::vector<ArmorObject> {
    std::vector<ArmorObject> objects;
    constexpr int output_dim = 9 + config::NUM_COLORS + config::NUM_CLASSES;

    for (size_t i : std::views::iota(size_t{0}, grid_strides.size())) {
        const auto& gs = grid_strides[i];
        size_t basic_pos = i * output_dim;

        std::array<float, 8> coords;
        for (int j : std::views::iota(0, 8)) {
            coords[j] = (feat_data[basic_pos + j] + (j % 2 == 0 ? gs.grid0 : gs.grid1)) * gs.stride;
        }

        int color = argmax(feat_data.subspan(basic_pos + 9, config::NUM_COLORS));
        int cls =
            argmax(feat_data.subspan(basic_pos + 9 + config::NUM_COLORS, config::NUM_CLASSES));
        float prob = feat_data[basic_pos + 8];

        if (prob >= prob_threshold) {
            ArmorObject obj;
            Eigen::Matrix<float, 3, 4> apex_norm;
            apex_norm << coords[0], coords[2], coords[4], coords[6], coords[1], coords[3],
                coords[5], coords[7], 1.0f, 1.0f, 1.0f, 1.0f;
            auto apex_dst = transform_matrix * apex_norm;

            for (int j : std::views::iota(0, 4)) {
                obj.apex[j] = cv::Point2f(apex_dst(0, j), apex_dst(1, j));
                obj.pts.push_back(obj.apex[j]);
            }
            obj.rect = cv::boundingRect(obj.pts);
            obj.cls = cls;
            obj.color = color;
            obj.prob = prob;
            objects.push_back(obj);
        }
    }
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
                if (iou > config::MERGE_MIN_IOU &&
                    std::abs(a.prob - objects[j].prob) < config::MERGE_CONF_ERROR &&
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

// 解码模型输出
auto decodeOutputs(std::span<const float> prob, const Eigen::Matrix<float, 3, 3>& transform_matrix,
                   int img_w, int img_h) -> std::vector<ArmorObject> {
    std::vector<int> strides = {8, 16, 32};
    auto grid_strides = generate_grids_and_stride(config::INPUT_W, config::INPUT_H, strides);
    auto proposals =
        generateYoloxProposals(grid_strides, prob, transform_matrix, config::BBOX_CONF_THRESH);

    if (proposals.size() > config::TOPK) {
        proposals.resize(config::TOPK);
    }

    auto picked = nms_sorted_bboxes(proposals, config::NMS_THRESH);
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

    Eigen::Matrix<float, 3, 3> transform_matrix;
    cv::Mat input_img = scaledResize(img, transform_matrix);

    std::vector<float> input_data(1 * 3 * config::INPUT_W * config::INPUT_H);
    for (int y : std::views::iota(0, config::INPUT_H)) {
        for (int x : std::views::iota(0, config::INPUT_W)) {
            auto pixel = input_img.at<cv::Vec3b>(y, x);
            input_data[y * config::INPUT_W * 3 + x * 3 + 0] = pixel[2] / 255.0f; // R
            input_data[y * config::INPUT_W * 3 + x * 3 + 1] = pixel[1] / 255.0f; // G
            input_data[y * config::INPUT_W * 3 + x * 3 + 2] = pixel[0] / 255.0f; // B
        }
    }

    std::vector<int64_t> input_shape = {1, 3, config::INPUT_H, config::INPUT_W};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    try {
        // 修复：使用 optional 的 value()
        std::vector<const char*> input_names = {input_name_->get()};
        std::vector<const char*> output_names = {"output"};
        auto outputs = session_->Run(Ort::RunOptions{}, input_names.data(), &input_tensor, 1,
                                     output_names.data(), 1);
        auto* prob = outputs[0].GetTensorMutableData<float>();
        std::span<const float> prob_span(prob,
                                         outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());
        return decodeOutputs(prob_span, transform_matrix, img.cols, img.rows);
    } catch (const std::exception& e) {
        return std::unexpected(e.what());
    }
}