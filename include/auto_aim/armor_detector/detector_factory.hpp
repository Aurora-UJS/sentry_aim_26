/**
 ************************************************************************
 *
 * @file detector_factory.hpp
 * @author Xlqmu
 * @brief 装甲板检测器工厂 - 运行时后端选择
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Aurora Vision
 * ************************************************************************
 */

#pragma once

#include "detector.hpp"
#include "onnxruntime_detector.hpp"
#ifdef HAVE_OPENVINO
#include "openvino_detector.hpp"
#endif

#include "utils/logger/logger.hpp"

#include <filesystem>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

namespace armor {

/**
 * @brief 推理后端类型
 */
enum class InferenceBackend {
    ONNXRUNTIME,
    OPENVINO,
    TENSORRT,
    NCNN,
    AUTO  // 自动选择：优先 OpenVINO，否则 OnnxRuntime
};

/**
 * @brief 检测器工厂配置
 */
struct DetectorConfig {
    InferenceBackend backend = InferenceBackend::ONNXRUNTIME;
    std::string model_path;
    std::string onnxruntime_device = "CPU";
    std::string openvino_device = "AUTO";
    DetectorParams params;

    /**
     * @brief 从 YAML 文件加载配置
     */
    static DetectorConfig fromYaml(const std::string& config_path) {
        DetectorConfig config;

        try {
            YAML::Node yaml = YAML::LoadFile(config_path);
            auto detector_node = yaml["armor_detector"];

            if (!detector_node) {
                utils::logger()->warn("[DetectorFactory] 配置文件中未找到 armor_detector 节点");
                return config;
            }

            // 解析后端类型
            std::string backend_str = detector_node["backend"].as<std::string>("onnxruntime");
            if (backend_str == "openvino") {
                config.backend = InferenceBackend::OPENVINO;
            } else if (backend_str == "tensorrt" || backend_str == "trt") {
                config.backend = InferenceBackend::TENSORRT;
            } else if (backend_str == "ncnn") {
                config.backend = InferenceBackend::NCNN;
            } else if (backend_str == "auto") {
                config.backend = InferenceBackend::AUTO;
            } else {
                config.backend = InferenceBackend::ONNXRUNTIME;
            }

            // 模型路径
            config.model_path = detector_node["model_path"].as<std::string>("");

            // 设备配置
            if (detector_node["device"]) {
                config.onnxruntime_device =
                    detector_node["device"]["onnxruntime"].as<std::string>("CPU");
                config.openvino_device =
                    detector_node["device"]["openvino"].as<std::string>("AUTO");
            }

            // 检测参数
            config.params.model_path = config.model_path;
            config.params.conf_threshold = detector_node["conf_threshold"].as<float>(0.5f);
            config.params.nms_threshold = detector_node["nms_threshold"].as<float>(0.45f);

            int input_w = detector_node["input_width"].as<int>(416);
            int input_h = detector_node["input_height"].as<int>(416);
            config.params.input_size = cv::Size(input_w, input_h);

            config.params.enable_debug = detector_node["enable_debug"].as<bool>(false);

            utils::logger()->info("[DetectorFactory] 配置加载成功: backend={}, model={}",
                                  backend_str, config.model_path);

        } catch (const YAML::Exception& e) {
            utils::logger()->error("[DetectorFactory] YAML 解析错误: {}", e.what());
        }

        return config;
    }
};

/**
 * @brief 检测器工厂类
 */
class DetectorFactory {
public:
    /**
     * @brief 从配置文件创建检测器
     * @param config_path 配置文件路径
     * @return 检测器智能指针
     */
    static std::unique_ptr<Detector> createFromConfig(const std::string& config_path) {
        auto config = DetectorConfig::fromYaml(config_path);
        return create(config);
    }

    /**
     * @brief 从配置创建检测器
     * @param config 检测器配置
     * @return 检测器智能指针
     */
    static std::unique_ptr<Detector> create(const DetectorConfig& config) {
        InferenceBackend backend = config.backend;

        // AUTO 模式自动选择
        if (backend == InferenceBackend::AUTO) {
#ifdef HAVE_OPENVINO
            backend = InferenceBackend::OPENVINO;
            utils::logger()->info("[DetectorFactory] AUTO 模式选择 OpenVINO");
#else
            backend = InferenceBackend::ONNXRUNTIME;
            utils::logger()->info("[DetectorFactory] AUTO 模式选择 OnnxRuntime");
#endif
        }

        std::unique_ptr<Detector> detector;

        switch (backend) {
            case InferenceBackend::OPENVINO: {
#ifdef HAVE_OPENVINO
                auto ov_detector = std::make_unique<OpenVINODetector>();
                if (ov_detector->init(config.model_path, config.openvino_device)) {
                    ov_detector->setParams(config.params);
                    detector = std::move(ov_detector);
                    utils::logger()->info("[DetectorFactory] 创建 OpenVINO 检测器, 设备: {}",
                                          config.openvino_device);
                } else {
                    utils::logger()->error(
                        "[DetectorFactory] OpenVINO 初始化失败，回退到 OnnxRuntime");
                    // 回退到 OnnxRuntime
                    auto ort_detector = std::make_unique<OnnxRuntimeDetector>();
                    ort_detector->init(config.model_path);
                    ort_detector->setParams(config.params);
                    detector = std::move(ort_detector);
                }
#else
                utils::logger()->warn("[DetectorFactory] OpenVINO 未编译，使用 OnnxRuntime");
                auto ort_detector = std::make_unique<OnnxRuntimeDetector>();
                ort_detector->init(config.model_path);
                ort_detector->setParams(config.params);
                detector = std::move(ort_detector);
#endif
                break;
            }

            case InferenceBackend::TENSORRT: {
                utils::logger()->warn(
                    "[DetectorFactory] TensorRT 后端暂未编译集成，回退到 OnnxRuntime");
                auto ort_detector = std::make_unique<OnnxRuntimeDetector>();
                if (ort_detector->init(config.model_path)) {
                    ort_detector->setParams(config.params);
                    detector = std::move(ort_detector);
                    utils::logger()->info(
                        "[DetectorFactory] 创建 OnnxRuntime 检测器(来自 TensorRT 回退)");
                } else {
                    utils::logger()->error("[DetectorFactory] OnnxRuntime 初始化失败");
                }
                break;
            }

            case InferenceBackend::NCNN: {
                utils::logger()->warn(
                    "[DetectorFactory] NCNN 后端暂未编译集成，回退到 OnnxRuntime");
                auto ort_detector = std::make_unique<OnnxRuntimeDetector>();
                if (ort_detector->init(config.model_path)) {
                    ort_detector->setParams(config.params);
                    detector = std::move(ort_detector);
                    utils::logger()->info(
                        "[DetectorFactory] 创建 OnnxRuntime 检测器(来自 NCNN 回退)");
                } else {
                    utils::logger()->error("[DetectorFactory] OnnxRuntime 初始化失败");
                }
                break;
            }

            case InferenceBackend::ONNXRUNTIME:
            default: {
                auto ort_detector = std::make_unique<OnnxRuntimeDetector>();
                if (ort_detector->init(config.model_path)) {
                    ort_detector->setParams(config.params);
                    detector = std::move(ort_detector);
                    utils::logger()->info("[DetectorFactory] 创建 OnnxRuntime 检测器");
                } else {
                    utils::logger()->error("[DetectorFactory] OnnxRuntime 初始化失败");
                }
                break;
            }
        }

        return detector;
    }

    /**
     * @brief 快速创建检测器（使用后端名称字符串）
     * @param backend_name 后端名称: "onnxruntime", "openvino", "tensorrt", "ncnn", "auto"
     * @param model_path 模型路径
     * @param device 设备名称
     * @return 检测器智能指针
     */
    static std::unique_ptr<Detector> create(const std::string& backend_name,
                                            const std::string& model_path,
                                            const std::string& device = "AUTO") {
        DetectorConfig config;
        config.model_path = model_path;
        config.params.model_path = model_path;

        if (backend_name == "openvino") {
            config.backend = InferenceBackend::OPENVINO;
            config.openvino_device = device;
        } else if (backend_name == "tensorrt" || backend_name == "trt") {
            config.backend = InferenceBackend::TENSORRT;
        } else if (backend_name == "ncnn") {
            config.backend = InferenceBackend::NCNN;
        } else if (backend_name == "auto") {
            config.backend = InferenceBackend::AUTO;
            config.openvino_device = device;
            config.onnxruntime_device = device;
        } else {
            config.backend = InferenceBackend::ONNXRUNTIME;
            config.onnxruntime_device = device;
        }

        return create(config);
    }
};

/**
 * @brief 后端名称转字符串
 */
inline std::string backendToString(InferenceBackend backend) {
    switch (backend) {
        case InferenceBackend::ONNXRUNTIME:
            return "onnxruntime";
        case InferenceBackend::OPENVINO:
            return "openvino";
        case InferenceBackend::TENSORRT:
            return "tensorrt";
        case InferenceBackend::NCNN:
            return "ncnn";
        case InferenceBackend::AUTO:
            return "auto";
        default:
            return "unknown";
    }
}

}  // namespace armor
