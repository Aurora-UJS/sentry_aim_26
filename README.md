# AURORA_AIM

## tool chain

clang19以上（用最新） + cmake 3.28以上 + ninja 1.13.1 + vcpkg + git

代码cpp标准采用c++20

## 框架核心

为了保持代码清晰、易维护，即使独自开发，也需要遵循以下约定：

1. 接口驱动：所有功能实现必须继承 include/ 中的抽象类（如 Detector、PnpSolver）。
2. 模块化：将功能分为检测、PNP、跟踪、串口，分别实现，存放在 tasks/auto_aim/ 或 src/io/。
3. 配置驱动：参数（如检测阈值、串口波特率）通过 yaml/toml 文件（configs/）加载。
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

在 tasks/auto_aim/playground/ 中尝试新算法，验证思路，避免直接修改核心代码。

#### 3.2.2 实现（接口实现）

验证实验有效后，实现接口，迁移到 tasks/auto_aim/。

#### 3.2.3 测试

为每个模块编写 Catch2 测试，存放在 tests/。

#### 3.2.4 集成

将模块集成到主程序 src/main.cpp，通过 AutoAim 类调用。

#### 3.2.5 优化

1. 性能：使用 std::chrono 测量模块耗时：

``` zsh
auto start = std::chrono::high_resolution_clock::now();
auto result = detector_->detect(image);
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now() - start);
spdlog::info("Detection took {} us", duration.count());
```

2. 串口优化：实现异步接收（参考中南大学异步设计）：

``` zsh
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

``` zsh
/// @brief Detects armor plates in the image.
/// @param image Input BGR image.
/// @return Detected points or nullopt if none.
std::optional<std::vector<Eigen::Vector2d>> detect(const cv::Mat& image) override;
```

3. 日志：spdlog 记录关键信息：

``` zsh
spdlog::info("Detected {} targets in {} us", points.size(), duration.count());
```

4. 错误处理：优先 std::optional，异常仅用于致命错误。

## PR 规范

- 在本地或 GitHub 记录变更（即使不正式 PR）。

提交前：

运行测试：ninja && ./tests/test_auto_aim。
格式化：clang-format -i **/*.cpp **/*.hpp。

``` ZSH
git checkout -b feature/new_detector
git add tasks/auto_aim/detector_impl.cpp
git commit -m "[Detector] Add contour-based detection"
git push origin feature/new_detector
```
## 编译命令

``` zsh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ \ 
      -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```