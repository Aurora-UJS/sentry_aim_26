/**
 ************************************************************************
 *
 * @file yolo_detector.cpp
 * @author Xlqmu
 * @brief 基于 YOLO 的纯神经网络装甲板检测器实现 (ONNX Runtime)
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#include "auto_aim/armor_detector/onnxruntime_detector.hpp"

#include <algorithm>
#include <cmath>

namespace armor {

OnnxRuntimeDetector::OnnxRuntimeDetector() {
    utils::logger()->info("[OnnxRuntimeDetector] 创建实例");
}

bool OnnxRuntimeDetector::init(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntimeDetector");
        session_options_ = std::make_unique<Ort::SessionOptions>();

        // 设置线程数
        session_options_->SetIntraOpNumThreads(4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 创建会话
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // 获取输入输出信息
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();

        input_names_.clear();
        input_names_str_.clear();
        output_names_.clear();
        output_names_str_.clear();

        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator_);
            input_names_str_.push_back(input_name.get());
            input_names_.push_back(input_names_str_.back().c_str());
        }

        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_.push_back(output_name.get());
            output_names_.push_back(output_names_str_.back().c_str());
        }

        utils::logger()->info("[OnnxRuntimeDetector] 模型加载成功: {}", model_path);
        return true;
    } catch (const Ort::Exception& e) {
        utils::logger()->error("[OnnxRuntimeDetector] ONNX Runtime 错误: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        utils::logger()->error("[OnnxRuntimeDetector] 初始化异常: {}", e.what());
        return false;
    }
}

void OnnxRuntimeDetector::setParams(const DetectorParams& params) {
    params_ = params;
    if (!params_.model_path.empty()) {
        init(params_.model_path);
    }
}

std::vector<ArmorObject> OnnxRuntimeDetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    if (!session_) {
        utils::logger()->error("[OnnxRuntimeDetector] 会话未初始化");
        return {};
    }

    // 1. 预处理
    float scale = 1.0F;
    cv::Mat blob = preProcess(image, scale);

    // 2. 准备输入 Tensor
    std::vector<int64_t> input_shape = {1, 3, params_.input_size.height, params_.input_size.width};
    size_t input_tensor_size =
        static_cast<size_t>(1) * 3 * params_.input_size.height * params_.input_size.width;

    // 检查模型输入类型
    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_type = input_type_info.GetTensorTypeAndShapeInfo().GetElementType();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 使用 vector 存储 input_tensor 以避免默认构造问题
    std::vector<Ort::Value> input_tensors;
    cv::Mat blob_fp16;  // 必须在此处声明以保持数据在 Run 期间有效

    if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // FP32 模型
        float* input_data = reinterpret_cast<float*>(blob.data);
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_data, input_tensor_size, input_shape.data(), input_shape.size()));
    } else if (input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // FP16 模型 - 需要转换
        blob.convertTo(blob_fp16, CV_16F);

        // CV_16F 数据本质上是 uint16_t (half float)
        Ort::Float16_t* input_data = reinterpret_cast<Ort::Float16_t*>(blob_fp16.data);
        input_tensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info, input_data, input_tensor_size, input_shape.data(), input_shape.size()));
    } else {
        utils::logger()->error("[OnnxRuntimeDetector] 不支持的模型输入类型: {}",
                               static_cast<int>(input_tensor_type));
        return {};
    }

    // 3. 推理
    auto output_tensors =
        session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(), 1,
                      output_names_.data(), output_names_.size());

    // 4. 后处理
    // 假设只有一个输出
    float* output_data = nullptr;
    std::vector<float> output_data_fp32;  // 如果需要转换，用于存储 FP32 数据

    auto output_type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_tensor_type = output_type_info.GetElementType();
    auto output_shape = output_type_info.GetShape();

    int rows = static_cast<int>(output_shape[1]);
    int dimensions = static_cast<int>(output_shape[2]);
    size_t output_size = static_cast<size_t>(rows) * dimensions;

    if (output_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        output_data = output_tensors[0].GetTensorMutableData<float>();
    } else if (output_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // 输出也是 FP16，需要转回 FP32 供后处理使用
        const Ort::Float16_t* raw_output = output_tensors[0].GetTensorData<Ort::Float16_t>();
        output_data_fp32.resize(output_size);

        // 使用 OpenCV 进行批量转换 (利用 Mat)
        cv::Mat output_mat_fp16(
            1, static_cast<int>(output_size), CV_16F,
            const_cast<void*>(reinterpret_cast<const void*>(raw_output)));  // NOLINT
        cv::Mat output_mat_fp32;
        output_mat_fp16.convertTo(output_mat_fp32, CV_32F);

        memcpy(output_data_fp32.data(), output_mat_fp32.data, output_size * sizeof(float));
        output_data = output_data_fp32.data();
    } else {
        utils::logger()->error("[OnnxRuntimeDetector] 不支持的模型输出类型: {}",
                               static_cast<int>(output_tensor_type));
        return {};
    }

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

cv::Mat OnnxRuntimeDetector::preProcess(const cv::Mat& image, float& scale) const {
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

std::vector<ArmorObject> OnnxRuntimeDetector::postProcess(
    const float* data, int rows, int dimensions, float scale,
    [[maybe_unused]] const cv::Mat& origin_img) const {
    std::vector<ArmorObject> armors;

    if (dimensions < 22) {
        utils::logger()->warn("[OnnxRuntimeDetector] 模型输出维度不足: {}", dimensions);
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

    utils::logger()->debug("[OnnxRuntimeDetector] NMS 前: {}, NMS 后: {}", boxes.size(),
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
