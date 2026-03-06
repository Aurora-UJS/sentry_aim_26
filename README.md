# AURORA_AIM

## tool chain

clang19以上（用最新） + cmake 3.28以上 + ninja 1.13.1 + vcpkg + git + clang-format-19

代码cpp标准采用c++20

## 框架核心

为了保持代码清晰、易维护，即使独自开发，也需要遵循以下约定：

1. 接口驱动：所有功能实现必须继承 include/ 中的抽象类（如 Detector、PnpSolver）。
2. 模块化：将功能分为检测、PNP、跟踪、串口，分别实现，存放在 tasks/auto_aim/ 或 src/io/。
3. 配置驱动：参数（如检测阈值、串口波特率）通过 yaml/toml 文件（configs/）加载。
   - 检测器后端支持通过 `config/detector_config.yaml` 的 `armor_detector.backend` 切换：`onnxruntime` / `openvino` / `tensorrt` / `ncnn` / `auto`。
4. 调试友好：使用 spdlog 记录日志，OpenCV 可视化结果，Catch2 编写单元测试。
5. 版本控制：即使独自开发，使用 git 提交到分支（如 feature/detector），便于回滚和记录。

## 开发流程

以下是基于接口的开发步骤，分为实验、实现、测试、集成和优化，参考中南大学（模块分离）和同济（分层架构）。

### 3.1 项目结构

确保项目结构如下：

``` zsh
.
├── env/
│   └── setup.sh
├── configs/
│   └── auto_aim.yaml
├── include/
│   └── auto_aim/
│       ├── auto_aim.hpp
│       └── config.hpp
├── src/
│   ├── io/
│   │   └── serial_io.cpp
│   └── main.cpp
├── tasks/
│   └── auto_aim/
│       ├── playground/
│       │   └── sample_detector.cpp
│       ├── detector_impl.cpp
│       ├── pnp_solver_impl.cpp
│       └── tracker_impl.cpp
├── tools/
│   ├── ekf_tracker.hpp
│   ├── serial_protocol.hpp
│   └── yaml_parser.hpp
├── tests/
│   ├── detector_test.cpp
│   └── serial_test.cpp
├── build/
└── CMakeLists.txt
```

### 3.2 开发步骤

#### 3.2.1 实验（Playground）

通过新建git branch尝试开发新算法，验证思路，避免直接修改核心代码。

#### 3.2.2 实现（接口实现）

验证实验有效后，实现接口，迁移到如 tasks/auto_aim/。

#### 3.2.3 测试

为每个模块编写 Catch2 测试，存放在 tests/。

#### 3.2.4 集成

将模块集成到主程序 src/main.cpp，通过 AutoAim 类调用。

#### 3.2.5 优化

1. 性能：使用 std::chrono 测量模块耗时：

``` cpp
auto start = std::chrono::high_resolution_clock::now();
auto result = detector_->detect(image);
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now() - start);
spdlog::info("Detection took {} us", duration.count());
```

2. 串口优化：实现异步接收（参考中南大学异步设计）：

``` cpp
// src/io/serial_io.cpp (部分)
std::optional<std::future<size_t>> SerialDriverImpl::async_receive(std::span<uint8_t> buffer, int timeout_ms) {
    auto promise = std::make_shared<std::promise<size_t>>();
    boost::asio::async_read(port_, boost::asio::buffer(buffer.data(), buffer.size()),
        [promise](const boost::system::error_code& ec, size_t bytes) {
            if (!ec) promise->set_value(bytes);
            else promise->set_exception(std::make_exception_ptr(std::runtime_error(ec.message())));
        });
    return promise->get_future();
}
```


## 代码风格

参考中南大学（简洁、日志丰富）和 Google C++ Style Guide：

1. 命名：

- 文件：snake_case（e.g., detector_impl.cpp）。
- 文件：snake_case（例如，detector_impl.cpp）。

- 类：CamelCase（e.g., ContourDetector）。
- 类：CamelCase（例如，ContourDetector）。

- 函数/变量：snake_case（e.g., detect_targets）。
- 函数/变量：snake_case（例如，detect_targets）。

- 常量：UPPER_SNAKE_CASE（e.g., MIN_AREA）。
- 常量：UPPER_SNAKE_CASE（例如，MIN_AREA）。

2. 格式：

- 缩进：4 空格。
- 头文件：#pragma once。
注释：Doxygen 风格：

``` cpp
/// @brief Detects armor plates in the image.
/// @param image Input BGR image.
/// @return Detected points or nullopt if none.
std::optional<std::vector<Eigen::Vector2d>> detect(const cv::Mat& image) override;
```

3. 日志：spdlog 记录关键信息：

``` cpp
spdlog::info("Detected {} targets in {} us", points.size(), duration.count());
```

4. 错误处理：优先 std::optional，异常仅用于致命错误。

## PR 规范

- 在本地或 GitHub 记录变更（即使不正式 PR）。

提交前：

运行测试：ninja && ./tests/test_auto_aim。
格式化：clang-format -i **/*.cpp **/*.hpp。

``` zsh
git checkout -b feature/new_detector
git add tasks/auto_aim/detector_impl.cpp
git commit -m "[Detector] Add contour-based detection"
git push origin feature/new_detector
```
## 编译命令

``` zsh
rm -rf build
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build
```

## 推理后端切换

在 `config/detector_config.yaml` 中修改：

```yaml
armor_detector:
  backend: "openvino"   # onnxruntime | openvino | tensorrt | ncnn | auto
```

说明：
- `onnxruntime`：稳定默认后端。
- `openvino`：编译时启用 `USE_OPENVINO=ON` 且系统可找到 OpenVINO 时生效。
- `tensorrt` / `ncnn`：当前版本已开放配置入口；若未集成对应后端实现，将自动回退到 `onnxruntime` 并输出日志提示。
- `auto`：优先 OpenVINO（可用时），否则回退 OnnxRuntime。

关于模型

直接采用szu的基于yolov5魔改的网络

1. 模型输入 (Input)格式：NCHW (Batch, Channels, Height, Width)。尺寸：$640 \times 640$（由 params_.input_size 定义）。

    预处理：缩放：采用 letterbox 方式（保持长宽比，缺失部分补黑边）。

    归一化：像素值从 [0, 255] 缩放到 [0, 1]。
    
    类型：支持 FP32 或 FP16。

2. 模型输出 (Output) —— 22 维向量拆解对于每一个检测到的候选框，其对应的 22 维数据定义如下：

    | 索引 (Index) | 含义 |处理方式 |
    |:---:|:---:|:---:|
    | 0, 1 | 第 1 个关键点 | $(x_1, y_1)$直接除以 scale 还原到原图| 
    |2, 3|第 2 个关键点| $(x_2, y_2)$直接除以 scale 还原到原图|
    |4, 5|第 3 个关键点| $(x_3, y_3)$直接除以 scale 还原到原图|
    |6, 7|第 4 个关键点| $(x_4, y_4)$直接除以 scale 还原到原图|
    |8|Objectness (置信度)|通过 Sigmoid 函数激活|
    |9 - 12|颜色分类 (4 类)|通过 Softmax 归一化|
    |13 - 21|数字/类型分类 (9 类)|通过 Softmax 归一化|

A. 颜色映射 (Index 9-12)代码中定义的映射逻辑如下：

    0: 蓝色 (BLUE)

    1: 红色 (RED)

    2: 灰色/无 (NONE)

    3: 紫色 (PURPLE)

B. 数字/类型映射 (Index 13-21)代码中定义的映射逻辑如下：

    0: 哨兵 (Sentry)

    1: 1 号 (Hero)

    2: 2 号 (Engineer)

    3: 3 号 (Infantry)

    4: 4 号 (Infantry)

    5: 5 号 (Infantry)

    6: 前哨站 (Outpost)

    7: 基地小装甲 (Base Small)

    8: 基地大装甲 (Base Big)