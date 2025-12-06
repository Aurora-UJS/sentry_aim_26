add_library(tasks
    auto_aim/armor_detector/detector_impl.cpp

    # armor_detector
    # onnxruntime
    # auto_aim/armor_detector/onnxruntime/armor_detector_onnx.cpp
    # tensorRT
    # auto_aim/armor_detector/tensorrt/armor_detector_tensorrt.cpp
)
target_include_directories(tasks
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(tasks
    PUBLIC 
    aurora_vision
    ${OpenCV_LIBS})