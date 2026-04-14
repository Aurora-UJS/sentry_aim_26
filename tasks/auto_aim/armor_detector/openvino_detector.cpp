/**
 ************************************************************************
 *
 * @file openvino_detector.cpp
 * @author Xlqmu
 * @brief 基于 YOLO 的纯神经网络装甲板检测器实现 (OpenVINO)
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#include "auto_aim/armor_detector/openvino_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace armor {

OpenVINODetector::OpenVINODetector() {
    utils::logger()->info("[OpenVINODetector] 创建实例");

    // 打印可用设备
    auto devices = core_.get_available_devices();
    std::string dev_list;
    for (const auto& dev : devices) {
        if (!dev_list.empty())
            dev_list += ", ";
        dev_list += dev;
    }
    utils::logger()->info("[OpenVINODetector] 可用设备: {}", dev_list);
}

bool OpenVINODetector::init(const std::string& model_path, const std::string& device) {
    try {
        device_ = device;

        // 读取模型
        utils::logger()->info("[OpenVINODetector] 正在加载模型: {}", model_path);
        model_ = core_.read_model(model_path);

        // 获取输入输出信息
        auto inputs = model_->inputs();
        auto outputs = model_->outputs();

        if (inputs.empty() || outputs.empty()) {
            utils::logger()->error("[OpenVINODetector] 模型输入/输出为空");
            return false;
        }

        input_name_ = inputs[0].get_any_name();
        output_name_ = outputs[0].get_any_name();
        input_shape_ = inputs[0].get_shape();

        utils::logger()->info("[OpenVINODetector] 输入名: {}, 输出名: {}", input_name_,
                              output_name_);
        utils::logger()->info("[OpenVINODetector] 输入形状: [{}, {}, {}, {}]", input_shape_[0],
                              input_shape_[1], input_shape_[2], input_shape_[3]);

        // 使用预处理 API 自动处理精度转换（支持 FP16 模型）
        ov::preprocess::PrePostProcessor ppp(model_);

        // 配置输入：始终使用 FP32 + NCHW，让 OpenVINO 自动转换
        ppp.input().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
        ppp.input().model().set_layout("NCHW");

        // 配置输出：输出转换为 FP32
        ppp.output().tensor().set_element_type(ov::element::f32);

        model_ = ppp.build();

        // 配置并编译模型
        ov::AnyMap config;

        // 针对不同设备的优化配置
        if (device == "CPU") {
            // CPU 性能优化
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
            config[ov::hint::num_requests.name()] = 1;
        } else if (device == "GPU") {
            // GPU 性能优化
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        } else if (device == "NPU") {
            // NPU 性能优化
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        // AUTO 模式会自动选择最佳设备

        utils::logger()->info("[OpenVINODetector] 正在编译模型到设备: {}", device);
        compiled_model_ = core_.compile_model(model_, device, config);

        // 创建推理请求
        infer_request_ = compiled_model_.create_infer_request();

        // 获取实际输出形状
        auto output_tensor = infer_request_.get_output_tensor();
        output_shape_ = output_tensor.get_shape();
        utils::logger()->info("[OpenVINODetector] 输出形状: [{}, {}, {}]", output_shape_[0],
                              output_shape_[1], output_shape_[2]);

        utils::logger()->info("[OpenVINODetector] 模型加载成功, 设备: {}", device);
        return true;
    } catch (const ov::Exception& e) {
        utils::logger()->error("[OpenVINODetector] OpenVINO 错误: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        utils::logger()->error("[OpenVINODetector] 初始化异常: {}", e.what());
        return false;
    }
}

void OpenVINODetector::setParams(const DetectorParams& params) {
    params_ = params;
    if (!params_.model_path.empty()) {
        init(params_.model_path);
    }
}

std::vector<ArmorObject> OpenVINODetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    // 1. 预处理
    float scale = 1.0F;
    cv::Mat blob = preProcess(image, scale);

    // 2. 设置输入
    ov::Tensor input_tensor(ov::element::f32, {1, 3, static_cast<size_t>(params_.input_size.height),
                                               static_cast<size_t>(params_.input_size.width)});

    // 复制数据到输入张量
    float* input_data = input_tensor.data<float>();
    std::memcpy(input_data, blob.data, blob.total() * sizeof(float));

    infer_request_.set_input_tensor(input_tensor);

    // 3. 推理
    infer_request_.infer();

    // 4. 获取输出
    ov::Tensor output_tensor = infer_request_.get_output_tensor();
    const float* output_data = output_tensor.data<const float>();
    auto out_shape = output_tensor.get_shape();

    int rows = static_cast<int>(out_shape[1]);
    int dimensions = static_cast<int>(out_shape[2]);

    auto armors = postProcess(output_data, rows, dimensions, scale, image);

    // 5. 调试绘图
    if (params_.enable_debug) {
        debug_image_ = image.clone();
        for (const auto& armor : armors) {
            // 绘制 4 点
            for (int i = 0; i < 4; i++) {
                cv::line(debug_image_, armor.pts[i], armor.pts[(i + 1) % 4], cv::Scalar(0, 255, 0),
                         2);
            }
            // 绘制类别和置信度
            std::string label = armorNumberToString(armor.number) + " " +
                                std::to_string(static_cast<int>(armor.prob * 100)) + "%";
            cv::putText(debug_image_, label, armor.pts[1], cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 0), 2);
        }
    }

    return armors;
}

cv::Mat OpenVINODetector::preProcess(const cv::Mat& image, float& scale) const {
    // 计算缩放比例 (保持长宽比)
    int w = image.cols;
    int h = image.rows;
    float r = std::min(static_cast<float>(params_.input_size.width) / static_cast<float>(w),
                       static_cast<float>(params_.input_size.height) / static_cast<float>(h));

    int new_unpad_w = static_cast<int>(static_cast<float>(w) * r);
    int new_unpad_h = static_cast<int>(static_cast<float>(h) * r);

    scale = r;  // 记录缩放比例用于还原坐标

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

    // Padding 到 input_size
    cv::Mat padded = cv::Mat::zeros(params_.input_size, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, new_unpad_w, new_unpad_h)));

    // 归一化 [0, 255] -> [0, 1] 并转换为 NCHW
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), false,
                           false);

    return blob;
}

std::vector<ArmorObject> OpenVINODetector::postProcess(
    const float* data, int rows, int dimensions, float scale,
    [[maybe_unused]] const cv::Mat& origin_img) const {
    std::vector<ArmorObject> armors;

    if (dimensions < 22) {
        utils::logger()->warn("[OpenVINODetector] 模型输出维度不足: {}", dimensions);
        return {};
    }

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> landmarks_list;
    std::vector<ArmorColor> colors;

    // data 指针遍历
    const float* p_data = data;

    for (int i = 0; i < rows; ++i) {
        // Objectness (8)
        float obj_score = p_data[8];
        obj_score = 1.0F / (1.0F + std::exp(-obj_score));

        // 颜色置信度 (9-12)
        std::array<float, 4> color_scores{};
        {
            const float* src = p_data + 9;
            float max_v = src[0];
            for (int j = 1; j < 4; j++) {
                if (src[j] > max_v) {
                    max_v = src[j];
                }
            }
            float sum = 0;
            for (int j = 0; j < 4; j++) {
                color_scores.at(j) = std::exp(src[j] - max_v);
                sum += color_scores.at(j);
            }
            for (int j = 0; j < 4; j++) {
                color_scores.at(j) /= sum;
            }
        }

        int color_id = 0;
        float max_color_score = color_scores[0];
        for (int c = 1; c < 4; ++c) {
            if (color_scores.at(c) > max_color_score) {
                max_color_score = color_scores.at(c);
                color_id = c;
            }
        }

        // 数字置信度 (13-21)
        std::array<float, 9> num_scores{};
        {
            const float* src = p_data + 13;
            float max_v = src[0];
            for (int j = 1; j < 9; j++) {
                if (src[j] > max_v) {
                    max_v = src[j];
                }
            }
            float sum = 0;
            for (int j = 0; j < 9; j++) {
                num_scores.at(j) = std::exp(src[j] - max_v);
                sum += num_scores.at(j);
            }
            for (int j = 0; j < 9; j++) {
                num_scores.at(j) /= sum;
            }
        }

        int num_id = 0;
        float max_num_score = num_scores[0];
        for (int n = 1; n < 9; ++n) {
            if (num_scores.at(n) > max_num_score) {
                max_num_score = num_scores.at(n);
                num_id = n;
            }
        }

        // 综合置信度
        float confidence = obj_score * max_color_score * max_num_score;

        if (confidence >= params_.conf_threshold) {
            // 解析关键点 (0-8)
            std::vector<cv::Point2f> landmarks;
            float min_x = 1e5F;
            float min_y = 1e5F;
            float max_x = -1e5F;
            float max_y = -1e5F;

            for (int k = 0; k < 4; k++) {
                float px = p_data[static_cast<ptrdiff_t>(k) * 2] / scale;
                float py = p_data[static_cast<ptrdiff_t>(k) * 2 + 1] / scale;
                landmarks.push_back(cv::Point2f(px, py));

                min_x = std::min(min_x, px);
                min_y = std::min(min_y, py);
                max_x = std::max(max_x, px);
                max_y = std::max(max_y, py);
            }

            // 构建包围盒
            int left = static_cast<int>(min_x);
            int top = static_cast<int>(min_y);
            int width = static_cast<int>(max_x - min_x);
            int height = static_cast<int>(max_y - min_y);

            confidences.push_back(confidence);
            class_ids.push_back(num_id);
            boxes.push_back(cv::Rect(left, top, width, height));
            landmarks_list.push_back(landmarks);

            // 颜色映射: 0:Blue, 1:Red, 2:Grey(None), 3:Purple
            switch (color_id) {
                case 0:
                    colors.push_back(ArmorColor::BLUE);
                    break;
                case 1:
                    colors.push_back(ArmorColor::RED);
                    break;
                case 2:
                    colors.push_back(ArmorColor::NONE);
                    break;
                case 3:
                    colors.push_back(ArmorColor::PURPLE);
                    break;
                default:
                    colors.push_back(ArmorColor::NONE);
                    break;
            }
        }
        p_data += dimensions;
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, params_.conf_threshold, params_.nms_threshold, indices);

    utils::logger()->debug("[OpenVINODetector] NMS 前: {}, NMS 后: {}", boxes.size(),
                           indices.size());

    for (int idx : indices) {
        ArmorObject armor;
        armor.box = boxes[idx];
        armor.prob = confidences[idx];
        armor.color = colors[idx];
        armor.pts = landmarks_list[idx];

        // 数字映射: G, 1, 2, 3, 4, 5, O, Bs, Bb
        int cid = class_ids[idx];
        switch (cid) {
            case 0:
                armor.number = ArmorNumber::SENTRY;
                break;
            case 1:
                armor.number = ArmorNumber::NO1;
                break;
            case 2:
                armor.number = ArmorNumber::NO2;
                break;
            case 3:
                armor.number = ArmorNumber::NO3;
                break;
            case 4:
                armor.number = ArmorNumber::NO4;
                break;
            case 5:
                armor.number = ArmorNumber::NO5;
                break;
            case 6:
                armor.number = ArmorNumber::OUTPOST;
                break;
            case 7:
            case 8:
                armor.number = ArmorNumber::BASE;
                break;  // Base Small & Big
            default:
                armor.number = ArmorNumber::UNKNOWN;
                break;
        }

        // 判断大小装甲板 (根据类别)
        if (cid == 0 || cid == 1 || cid == 8) {  // Sentry, 1, Base Big -> Large
            armor.type = ArmorType::LARGE;
        } else {
            armor.type = ArmorType::SMALL;
        }

        // 计算中心
        cv::Point2f center(0, 0);
        for (const auto& p : armor.pts) {
            center += p;
        }
        armor.center = center / 4.0F;

        armors.push_back(armor);
    }

    return armors;
}

}  // namespace armor
