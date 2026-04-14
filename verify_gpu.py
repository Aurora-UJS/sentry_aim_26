import onnxruntime as ort
import numpy as np
import time
import os

# 1. 强制设定稳定性环境变量，防止 MIOpen 寻优崩溃
os.environ["MIOPEN_FIND_MODE"] = "1"
os.environ["MIGRAPHX_ENABLE_SESSIONS_CACHE"] = "1"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

def test_migraphx_performance():
    # 打印当前可用的提供者
    providers = ort.get_available_providers()
    print(f"[*] 可用的推理后端: {providers}")

    if 'MIGraphXExecutionProvider' not in providers:
        print("[!] 错误: 未检测到 MIGraphX，请检查库路径 LD_LIBRARY_PATH")
        return

    # 2. 准备一个简单的随机模型 (这里如果你有实际模型路径可以替换它)
    # 如果没有模型文件，脚本会提示你使用 sentry_aim_26 的模型
    model_path = "models/0526.onnx" # 请修改为你自瞄模型的实际路径
    if not os.path.exists(model_path):
        print(f"[!] 未找到 {model_path}，请将你的 .onnx 模型文件放在当前目录下并改名。")
        return

    # 3. 配置 MIGraphX 推理选项
    # 开启 FP16 是 780M 性能翻倍的核心
    provider_options = [{
        'device_id': 0,
        'migraphx_fp16_enable': True, 
    }]

    print("[*] 正在载入模型并进行算子编译 (MIGraphX FP16)...")
    start_load = time.time()
    try:
        session = ort.InferenceSession(model_path, providers=['MIGraphXExecutionProvider'], provider_options=provider_options)
    except Exception as e:
        print(f"[!] 编译失败: {e}")
        return
    
    print(f"[*] 模型载入成功，耗时: {time.time() - start_load:.2f}s")

    # 4. 准备输入数据
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # 处理动态 BatchSize
    if isinstance(input_shape[0], str) or input_shape[0] is None:
        input_shape[0] = 1
    
    dummy_input = np.random.random(input_shape).astype(np.float32)

    # 5. 热身 (Warm up) - 这一步非常关键
    print("[*] 正在进行热身推理...")
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # 6. 正式压力测试
    print("[*] 开始性能测试 (循环 100 次)...")
    latencies = []
    for _ in range(100):
        t1 = time.time()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.time() - t1) * 1000)

    print("-" * 30)
    print(f"📊 780M MIGraphX 测试报告:")
    print(f"平均延迟 (Avg Latency): {np.mean(latencies):.2f} ms")
    print(f"最大延迟 (Max Latency): {np.max(latencies):.2f} ms")
    print(f"预估每秒帧数 (FPS): {1000 / np.mean(latencies):.1f}")
    print("-" * 30)

if __name__ == "__main__":
    test_migraphx_performance()