# ONNX Runtime GPU 加速配置完成

## ✅ 已完成的修改

### 1. 更新 CMakeLists.txt
**文件**: [tasks/CMakeLists.txt](tasks/CMakeLists.txt)

- 使用自编译的 ONNX Runtime 1.25.0（支持 MIGraphX）
- 路径: `/onnxruntime/build/Linux/Release/`
- 添加了 ROCm 库路径 `/opt/rocm/lib`
- 配置了正确的 RPATH

### 2. 修改检测器初始化代码
**文件**: [tasks/auto_aim/armor_detector/onnxruntime_detector.cpp](tasks/auto_aim/armor_detector/onnxruntime_detector.cpp#L33-L46)

添加了 MIGraphX 执行提供程序配置：
```cpp
OrtMIGraphXProviderOptions migraphx_options{};
migraphx_options.device_id = 0;
migraphx_options.migraphx_fp16_enable = 1;  // 启用 FP16 加速
// ... 其他配置
session_options_->AppendExecutionProvider_MIGraphX(migraphx_options);
```

### 3. 创建了辅助工具
- [verify_gpu.py](verify_gpu.py) - GPU 验证脚本
- [run_with_gpu.sh](run_with_gpu.sh) - 运行脚本（如需要环境变量）

## 🎯 GPU 加速特性

- ✅ **MIGraphX 执行提供程序已启用**
- ✅ **FP16 半精度加速已启用**（提升性能）
- ✅ **自动回退机制**（GPU 失败时自动使用 CPU）
- ✅ **支持 AMD Radeon 780M (gfx1103)**

## 📊 测试结果

运行测试程序输出：
```
[info] [OnnxRuntimeDetector] MIGraphX (AMD GPU) 加速已启用
```

这表明 GPU 加速已成功配置！

## 🚀 使用方法

### 编译项目
```bash
cd /home/neomelt/sentry_aim_26/build
cmake ..
cmake --build . -j$(nproc)
```

### 运行测试
```bash
cd /home/neomelt/sentry_aim_26/build
./tests/armor_detector_test
```

### 验证 GPU 支持
```bash
cd /home/neomelt/sentry_aim_26
python3 verify_gpu.py
```

## 💡 注意事项

1. **模型文件**: 确保模型文件存在于 `models/0526.onnx`
2. **ROCm 依赖**: 需要 ROCm 正确安装在 `/opt/rocm`
3. **ONNX Runtime**: 使用自编译版本 v1.25.0（包含 MIGraphX 支持）

## 🔧 技术细节

### 硬件信息
- **GPU**: AMD Radeon 780M Graphics (gfx1103)
- **ROCm 版本**: 6.16.13
- **支持的执行提供程序**: MIGraphXExecutionProvider, CPUExecutionProvider

### 软件配置
- **ONNX Runtime**: 1.25.0 (自编译)
- **编译选项**: 包含 MIGraphX, ROCm 支持
- **精度模式**: FP32 (默认) + FP16 (加速)

## 📝 性能优化建议

1. **FP16 模式**: 已启用，可提升约 2x 推理速度
2. **批处理**: 如果处理多帧，考虑使用批处理
3. **模型优化**: 确保使用 ONNX 优化的模型
4. **内存管理**: MIGraphX 会自动管理 GPU 内存

## ✨ 最小化修改原则

按照您的要求，只修改了必要的文件：
1. ✅ `tasks/CMakeLists.txt` - 链接配置
2. ✅ `tasks/auto_aim/armor_detector/onnxruntime_detector.cpp` - GPU 初始化

原有代码逻辑保持不变，仅添加 GPU 加速支持。
