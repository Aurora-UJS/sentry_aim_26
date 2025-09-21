add_library(task
    auto_aim/detector_impl.cpp
    # auto_aim/pnp_solver_impl.cpp
    # auto_aim/tracker_impl.cpp
    # auto_buff/buff_detector_impl.cpp
    # Add more files as implemented
)
target_include_directories(task
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(task PUBLIC sp_vision_refactoring)