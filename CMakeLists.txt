cmake_minimum_required(VERSION 3.28)
project(aurora_vision LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

# Set onnxruntime_DIR if using onnxruntime
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime")

# Fix TBB linking
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--copy-dt-needed-entries")

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Ceres REQUIRED)

# Core library
add_library(aurora_vision
    src/utils/logger/logger.cpp

    # src/core/auto_aim.cpp # Uncomment when implemented
    # src/io/serial_io.cpp  # Uncomment when implemented
)

# Include directories
target_include_directories(aurora_vision
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/task
)

# Link dependencies
target_link_libraries(aurora_vision
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_highgui
        Eigen3::Eigen
        spdlog::spdlog
        fmt::fmt
        yaml-cpp
        TBB::tbb
        Boost::system
        Ceres::ceres
)

# Test executable
add_executable(test_aurora_vision
    tests/logger_module_test.cpp
    # tests/detector_test.cpp  # Uncomment when implemented
    # tests/serial_test.cpp    # Uncomment when implemented
    # tests/auto_aim_test.cpp  # Uncomment when implemented
)
target_link_libraries(test_aurora_vision PRIVATE aurora_vision Catch2::Catch2WithMain)

# Subdirectories
add_subdirectory(src/utils)
add_subdirectory(tasks)
add_subdirectory(tests)

# Installation
install(TARGETS aurora_vision utils
        EXPORT aurora_visionTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT aurora_visionTargets
        FILE aurora_visionTargets.cmake
        NAMESPACE aurora_vision::
        DESTINATION lib/cmake/aurora_vision)