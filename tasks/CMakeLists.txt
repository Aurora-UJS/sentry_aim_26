# 推理后端支持开关
option(USE_OPENVINO "Enable OpenVINO inference backend" ON)
option(USE_TENSORRT "Enable TensorRT inference backend (skeleton in stage2)" OFF)
option(USE_NCNN "Enable NCNN inference backend (skeleton in stage2)" OFF)

# 基础源文件
set(TASKS_SOURCES
    auto_aim/armor_detector/onnxruntime_detector.cpp
    auto_aim/armor_solver/solver.cpp
    auto_aim/armor_tracker/tracker.cpp
)

# OpenVINO 检测器
if(USE_OPENVINO)
    find_package(OpenVINO QUIET)
    if(OpenVINO_FOUND)
        list(APPEND TASKS_SOURCES auto_aim/armor_detector/openvino_detector.cpp)
        message(STATUS "OpenVINO found, version: ${OpenVINO_VERSION}")
    else()
        message(WARNING "OpenVINO not found, disabling OpenVINO support")
        set(USE_OPENVINO OFF)
    endif()
endif()

# TensorRT / NCNN 阶段2骨架实现
if(USE_TENSORRT)
    list(APPEND TASKS_SOURCES auto_aim/armor_detector/tensorrt_detector.cpp)
    message(STATUS "TensorRT skeleton backend enabled")
endif()

if(USE_NCNN)
    list(APPEND TASKS_SOURCES auto_aim/armor_detector/ncnn_detector.cpp)
    message(STATUS "NCNN skeleton backend enabled")
endif()

add_library(tasks ${TASKS_SOURCES})

# ONNX Runtime - 使用 3rdparty 目录
set(CUSTOM_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/onnxruntime")
set(CUSTOM_ORT_LIB "${CUSTOM_ORT_ROOT}/lib")
target_include_directories(tasks PUBLIC 
    ${CUSTOM_ORT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../base  # 添加base目录以访问滤波器库
)

# 添加 ROCm 库路径
link_directories(/opt/rocm/lib)

target_link_libraries(tasks
    PUBLIC 
    aurora_vision
    ${OpenCV_LIBS}
    opencv_calib3d
    yaml-cpp
    ${CUSTOM_ORT_LIB}/libonnxruntime.so
)

# OpenVINO 链接
if(USE_OPENVINO AND OpenVINO_FOUND)
    target_link_libraries(tasks PUBLIC openvino::runtime)
    target_compile_definitions(tasks PUBLIC HAVE_OPENVINO)
    message(STATUS "OpenVINO backend enabled")
endif()

if(USE_TENSORRT)
    target_compile_definitions(tasks PUBLIC HAVE_TENSORRT)
endif()

if(USE_NCNN)
    target_compile_definitions(tasks PUBLIC HAVE_NCNN)
endif()

# 设置 RPATH 以便运行时能找到库（包括 ROCm 和 ONNX Runtime）
set_target_properties(tasks PROPERTIES
    INSTALL_RPATH "${CUSTOM_ORT_LIB}:/opt/rocm/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)