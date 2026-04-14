#!/bin/bash

# ONNX Runtime 性能基准测试脚本

echo "=========================================="
echo "  ONNX Runtime 性能基准测试"
echo "=========================================="
echo ""

# 检查构建目录
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "错误: 构建目录不存在，请先编译项目"
    echo "运行: cmake -B build && cmake --build build"
    exit 1
fi

# 检查测试程序
TEST_PROGRAM="$BUILD_DIR/tests/armor_detector_test"
if [ ! -f "$TEST_PROGRAM" ]; then
    echo "错误: 测试程序不存在: $TEST_PROGRAM"
    exit 1
fi

# 检查测试视频
TEST_VIDEO="video/blue/v3.avi"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "警告: 测试视频不存在: $TEST_VIDEO"
    echo "将使用摄像头或第一个可用视频"
fi

echo "1. 检查 OpenCL 支持情况"
echo "=========================================="
echo ""

# 检查 ONNX Runtime 是否链接了 OpenCL
echo "检查 libonnxruntime.so 的 OpenCL 依赖:"
if ldd 3rdparty/onnxruntime/lib/libonnxruntime.so.* 2>/dev/null | grep -i opencl; then
    echo "✅ 找到 OpenCL 依赖"
    HAS_OPENCL_LIB=1
else
    echo "❌ 未找到 OpenCL 依赖"
    echo "   建议: 使用支持 OpenCL 的 ONNX Runtime 版本"
    HAS_OPENCL_LIB=0
fi
echo ""

# 检查系统 OpenCL 环境
echo "检查系统 OpenCL 运行时:"
if command -v clinfo &> /dev/null; then
    echo "✅ 找到 clinfo 工具"
    echo ""
    echo "OpenCL 平台和设备:"
    clinfo -l 2>/dev/null | grep -E "(Platform|Device)" || echo "无法获取 OpenCL 设备信息"
    HAS_OPENCL_RT=1
else
    echo "❌ 未找到 clinfo 工具"
    echo "   安装: sudo apt install clinfo"
    HAS_OPENCL_RT=0
fi
echo ""

echo "2. 系统信息"
echo "=========================================="
echo ""

# CPU 信息
echo "CPU 型号:"
lscpu | grep "Model name" || echo "无法获取 CPU 信息"
echo ""

echo "CPU 核心数:"
nproc
echo ""

# GPU 信息（如果有）
echo "GPU 信息:"
if command -v lspci &> /dev/null; then
    lspci | grep -i vga || lspci | grep -i 3d || echo "无法检测 GPU"
else
    echo "无法检测 GPU (lspci 不可用)"
fi
echo ""

echo "3. 运行性能测试"
echo "=========================================="
echo ""

echo "提示: 测试将运行几秒钟，请等待..."
echo ""

# 运行测试并记录时间
echo "开始测试..."
if [ -f "$TEST_VIDEO" ]; then
    echo "使用测试视频: $TEST_VIDEO"
    "$TEST_PROGRAM" "$TEST_VIDEO" 2>&1 | tee /tmp/onnx_perf_test.log
else
    echo "使用默认输入"
    "$TEST_PROGRAM" 2>&1 | tee /tmp/onnx_perf_test.log
fi

echo ""
echo "4. 性能分析"
echo "=========================================="
echo ""

# 从日志中提取 FPS 信息
if [ -f /tmp/onnx_perf_test.log ]; then
    echo "从日志中提取性能数据..."
    
    # 查找 FPS 相关信息
    if grep -i "fps\|帧率" /tmp/onnx_perf_test.log > /dev/null; then
        echo ""
        echo "FPS 统计:"
        grep -i "fps\|帧率" /tmp/onnx_perf_test.log | tail -n 5
    fi
    
    # 查找 OpenCL 相关信息
    if grep -i "opencl" /tmp/onnx_perf_test.log > /dev/null; then
        echo ""
        echo "OpenCL 状态:"
        grep -i "opencl" /tmp/onnx_perf_test.log
    fi
    
    # 查找推理时间信息
    if grep -i "inference\|推理" /tmp/onnx_perf_test.log > /dev/null; then
        echo ""
        echo "推理时间:"
        grep -i "inference\|推理" /tmp/onnx_perf_test.log | tail -n 5
    fi
fi

echo ""
echo "5. 优化建议"
echo "=========================================="
echo ""

if [ $HAS_OPENCL_LIB -eq 0 ]; then
    echo "⚠️  ONNX Runtime 未编译 OpenCL 支持"
    echo "   建议: 从官方下载支持 OpenCL 的版本"
    echo "   链接: https://github.com/microsoft/onnxruntime/releases"
    echo ""
fi

if [ $HAS_OPENCL_RT -eq 0 ]; then
    echo "⚠️  系统缺少 OpenCL 运行时"
    echo "   AMD GPU 用户安装: sudo apt install mesa-opencl-icd"
    echo "   Intel GPU 用户安装: sudo apt install intel-opencl-icd"
    echo ""
fi

echo "✅ 已应用的优化:"
echo "   - 线程数优化（默认 1 个线程）"
echo "   - 扩展图优化"
echo "   - 内存模式优化"
echo "   - 顺序执行模式"
if [ $HAS_OPENCL_LIB -eq 1 ] && [ $HAS_OPENCL_RT -eq 1 ]; then
    echo "   - OpenCL GPU 加速（如果可用）"
fi
echo ""

echo "📖 更多优化信息请参考:"
echo "   docs/ONNX_PERFORMANCE_TUNING.md"
echo ""

echo "测试完成！"
