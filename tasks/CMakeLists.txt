add_library(tasks
    auto_aim/armor_detector/yolo_detector.cpp
)

# ONNX Runtime
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/onnxruntime")
target_include_directories(tasks PUBLIC ${ORT_ROOT}/include)
target_link_libraries(tasks
    PUBLIC 
    aurora_vision
    ${OpenCV_LIBS}
    ${ORT_ROOT}/lib/libonnxruntime.so.1.18.0
)