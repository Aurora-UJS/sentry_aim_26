add_library(tasks
    auto_aim/armor_detector/onnxruntime_detector.cpp
    auto_aim/armor_solver/solver.cpp
    auto_aim/armor_tracker/tracker.cpp
)

# ONNX Runtime
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/onnxruntime")
target_include_directories(tasks PUBLIC 
    ${ORT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../base  # 添加base目录以访问滤波器库
)
target_link_libraries(tasks
    PUBLIC 
    aurora_vision
    ${OpenCV_LIBS}
    opencv_calib3d
    ${ORT_ROOT}/lib/libonnxruntime.so.1.18.0
)