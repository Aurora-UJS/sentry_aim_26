#!/usr/bin/env python3
"""
调试模型输出格式的脚本
用于验证模型的实际输出格式和数值范围
"""

import onnxruntime as ort
import numpy as np
import cv2
import os


def preprocess_static_fp16(image_path, input_size=(640, 640)):
    """预处理函数 - 与您的Python代码完全一致"""
    original_image = cv2.imread(image_path)
    img_height, img_width = original_image.shape[:2]
    input_h, input_w = input_size
    scale = min(input_h / img_height, input_w / img_width)
    resized_h, resized_w = int(img_height * scale), int(img_width * scale)
    resized_img = cv2.resize(original_image, (resized_w, resized_h))
    padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded_img[
        (input_h - resized_h) // 2 : (input_h + resized_h) // 2,
        (input_w - resized_w) // 2 : (input_w + resized_w) // 2,
    ] = resized_img
    padded_img = padded_img[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR->RGB
    padded_img = np.expand_dims(padded_img, axis=0).astype(np.float16)
    return padded_img, original_image, scale


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


def debug_model_output():
    """调试模型输出格式"""
    # 使用您的模型路径
    MODEL_PATH = "assets/models/onnx/autoaim_0708.onnx"
    TEST_IMAGE = "27.jpg"  # 使用工作区中的测试图像

    print("=== 模型输出调试 ===")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试图像: {TEST_IMAGE}")

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return
    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 测试图像不存在 {TEST_IMAGE}")
        return

    try:
        # 创建会话 - 使用CPU以保证兼容性
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(MODEL_PATH, providers=providers)

        print(f"✓ 模型加载成功")
        print(f"✓ 使用Provider: {session.get_providers()}")

        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]

        print(f"\n=== 模型信息 ===")
        print(f"输入名称: {input_info.name}")
        print(f"输入形状: {input_info.shape}")
        print(f"输入类型: {input_info.type}")
        print(f"输出名称: {output_info.name}")
        print(f"输出形状: {output_info.shape}")
        print(f"输出类型: {output_info.type}")

        # 预处理图像
        print(f"\n=== 图像预处理 ===")
        input_data, original_img, scale = preprocess_static_fp16(TEST_IMAGE, (640, 640))
        print(f"预处理后输入形状: {input_data.shape}")
        print(f"输入数据类型: {input_data.dtype}")
        print(f"输入数值范围: [{input_data.min():.4f}, {input_data.max():.4f}]")

        # 推理
        print(f"\n=== 模型推理 ===")
        result = session.run([output_info.name], {input_info.name: input_data})
        output = result[0]

        print(f"输出形状: {output.shape}")
        print(f"输出数据类型: {output.dtype}")
        print(f"输出数值范围: [{output.min():.4f}, {output.max():.4f}]")

        # 分析输出结构
        print(f"\n=== 输出结构分析 ===")
        if len(output.shape) == 3:
            batch, num_detections, features = output.shape
            print(f"批次大小: {batch}")
            print(f"检测数量: {num_detections}")
            print(f"特征维度: {features}")

            # 分析第一个检测的特征分布
            first_detection = output[0, 0, :]  # 第一个检测
            print(f"\n=== 第一个检测的特征分析 ===")
            print(f"坐标 (0-7): {first_detection[0:8]}")
            print(f"置信度原始值 (8): {first_detection[8]:.4f}")
            print(f"置信度Sigmoid后: {sigmoid(first_detection[8]):.4f}")
            print(f"颜色特征 (9-12): {first_detection[9:13]}")
            print(f"类别特征 (13-21): {first_detection[13:22]}")

            # 统计有多少检测超过不同置信度阈值
            confidences = output[0, :, 8]  # 所有检测的置信度
            sigmoid_confidences = sigmoid(confidences)

            print(f"\n=== 置信度统计 ===")
            for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                count = np.sum(sigmoid_confidences > threshold)
                print(f"置信度 > {threshold}: {count} 个检测")

            # 找到置信度最高的几个检测
            top_indices = np.argsort(sigmoid_confidences)[-5:][::-1]
            print(f"\n=== 置信度最高的5个检测 ===")
            for i, idx in enumerate(top_indices):
                conf = sigmoid_confidences[idx]
                color_id = np.argmax(output[0, idx, 9:13])
                class_id = np.argmax(output[0, idx, 13:22])
                print(
                    f"#{i+1}: 索引={idx}, 置信度={conf:.4f}, 颜色={color_id}, 类别={class_id}"
                )

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_model_output()
