cmake_minimum_required(VERSION 3.28)
project(sp_vision_refactoring LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

# Fix TBB linking
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--copy-dt-needed-entries")

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Ceres REQUIRED)

# Core library
add_library(sp_vision_refactoring
    src/utils/logger/logger.cpp

    # src/core/auto_aim.cpp # Uncomment when implemented
    # src/io/serial_io.cpp  # Uncomment when implemented
)

# Include directories
target_include_directories(sp_vision_refactoring
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/task
)

# Link dependencies
target_link_libraries(sp_vision_refactoring
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
add_executable(test_sp_vision
    tests/logger_module_test.cpp
    # tests/detector_test.cpp  # Uncomment when implemented
    # tests/serial_test.cpp    # Uncomment when implemented
    # tests/auto_aim_test.cpp  # Uncomment when implemented
)
target_link_libraries(test_sp_vision PRIVATE sp_vision_refactoring Catch2::Catch2WithMain)

# Subdirectories
add_subdirectory(src/utils)
add_subdirectory(task)
add_subdirectory(tests)

# Installation
install(TARGETS sp_vision_refactoring utils
        EXPORT sp_vision_refactoringTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT sp_vision_refactoringTargets
        FILE sp_vision_refactoringTargets.cmake
        NAMESPACE sp_vision_refactoring::
        DESTINATION lib/cmake/sp_vision_refactoring)